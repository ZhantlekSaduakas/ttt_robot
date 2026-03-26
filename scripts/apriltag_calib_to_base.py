#!/usr/bin/env python3
import os
import time
import json
import threading

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

import tf2_ros
import tf2_geometry_msgs  # noqa: needed for PoseStamped transform
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ==============================
# Defaults (overridable via ROS params)
# ==============================
TAG_SIZE = 0.03
FAMILY = "tag36h11"

BOARD_MAP = {
    0: (0, 0), 1: (0, 1), 2: (0, 2),
    3: (1, 0), 4: (1, 1), 5: (1, 2),
    6: (2, 0), 7: (2, 1), 8: (2, 2),
}

TARGET_SAMPLES_PER_TAG = 60
MIN_SAMPLES_PER_TAG = 25
OUTLIER_POS_THRESH_M = 0.015
PRINT_EVERY_S = 1.0

OUTPUT_JSON_DEFAULT = "apriltag_poses_camera.json"

# ==============================
# AprilTag 3D Model (square)
# ==============================
def make_object_points(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    return np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

def quat_average(quats: np.ndarray) -> np.ndarray:
    if len(quats) == 0:
        raise ValueError("No quaternions to average")
    Q = quats.copy()
    ref = Q[0]
    for i in range(len(Q)):
        if np.dot(ref, Q[i]) < 0:
            Q[i] = -Q[i]
    A = np.zeros((4, 4), dtype=np.float64)
    for q in Q:
        A += np.outer(q, q)
    A /= len(Q)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]
    q_avg = q_avg / np.linalg.norm(q_avg)
    return q_avg.astype(np.float32)

def robust_pose_from_samples(pos_samples, quat_samples):
    P = np.array(pos_samples, dtype=np.float32)
    pos_med = np.median(P, axis=0)
    Q = np.array(quat_samples, dtype=np.float32)
    quat_avg = quat_average(Q)
    return pos_med, quat_avg

def pos_outlier(pos: np.ndarray, current_median: np.ndarray, thresh_m: float) -> bool:
    return np.linalg.norm(pos - current_median) > thresh_m

# ==============================
# ROS Node
# ==============================
class AprilTagCalibToBase:
    def __init__(self):
        # Params
        self.tag_size = float(rospy.get_param("~tag_size", TAG_SIZE))
        self.family = str(rospy.get_param("~family", FAMILY))

        self.base_frame = str(rospy.get_param("~base_frame", "panda_link0"))
        # IMPORTANT: Use the optical frame for solvePnP consistency
        self.camera_frame = str(rospy.get_param("~camera_frame", "camera_color_optical_frame"))

        self.image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw"))
        self.camera_info_topic = str(rospy.get_param("~camera_info_topic", "/camera/color/camera_info"))

        self.target_samples = int(rospy.get_param("~target_samples_per_tag", TARGET_SAMPLES_PER_TAG))
        self.min_samples = int(rospy.get_param("~min_samples_per_tag", MIN_SAMPLES_PER_TAG))
        self.outlier_thresh = float(rospy.get_param("~outlier_pos_thresh_m", OUTLIER_POS_THRESH_M))
        self.print_every = float(rospy.get_param("~print_every_s", PRINT_EVERY_S))

        self.output_json = str(rospy.get_param("~output_json", OUTPUT_JSON_DEFAULT))

        # State
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.K = None
        self.D = None
        self.dist_model = None
        self.have_cam_info = False

        self.object_points = make_object_points(self.tag_size)

        self.detector = Detector(
            families=self.family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25
        )

        self.pos_samples = {tid: [] for tid in BOARD_MAP.keys()}
        self.quat_samples = {tid: [] for tid in BOARD_MAP.keys()}
        self.running_median = {}

        self.last_print = 0.0

        # TF2
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        self.sub_info = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.cb_info, queue_size=1)
        self.sub_img = rospy.Subscriber(self.image_topic, Image, self.cb_img, queue_size=1)

        rospy.loginfo("AprilTagCalibToBase running.")
        rospy.loginfo("Image: %s", self.image_topic)
        rospy.loginfo("CameraInfo: %s", self.camera_info_topic)
        rospy.loginfo("TF: base_frame=%s camera_frame=%s", self.base_frame, self.camera_frame)
        rospy.loginfo("Press 's' to save, 'q' to quit (OpenCV window).")

    def cb_info(self, msg: CameraInfo):
        with self.lock:
            # K is 3x3 row-major in msg.K
            self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)

            # Distortion handling:
            # - OpenCV solvePnP typically expects 4,5,8,12,14 depending on model.
            # - RealSense often uses 5 coeffs for plumb_bob.
            self.dist_model = msg.distortion_model
            D = np.array(msg.D, dtype=np.float32).flatten()
            if D.size >= 5:
                self.D = D[:5].reshape(5, 1)
            elif D.size == 4:
                # Rare; still acceptable
                self.D = D.reshape(4, 1)
            else:
                self.D = np.zeros((5, 1), dtype=np.float32)

            self.have_cam_info = True

    def save_results_json(self, filename: str):
        results = {}
        for tid in sorted(BOARD_MAP.keys()):
            if len(self.pos_samples[tid]) < self.min_samples:
                continue

            pos_med, quat_avg = robust_pose_from_samples(self.pos_samples[tid], self.quat_samples[tid])
            r, c = BOARD_MAP[tid]
            key = f"cell_{r}_{c}"

            results[key] = {
                "tag_id": int(tid),
                "row": int(r),
                "col": int(c),
                "frame_id": self.base_frame,
                "position_m": {"x": float(pos_med[0]), "y": float(pos_med[1]), "z": float(pos_med[2])},
                "quaternion_xyzw": {"x": float(quat_avg[0]), "y": float(quat_avg[1]), "z": float(quat_avg[2]), "w": float(quat_avg[3])},
                "num_samples": int(len(self.pos_samples[tid]))
            }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def all_done(self):
        return all(len(self.pos_samples[tid]) >= self.target_samples for tid in BOARD_MAP.keys())

    def transform_pose_to_base(self, pose_cam: PoseStamped) -> PoseStamped:
        # Look up transform at image time
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                pose_cam.header.frame_id,
                pose_cam.header.stamp,
                rospy.Duration(0.2)
            )
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, tf)
            return pose_base
        except Exception as e:
            rospy.logwarn_throttle(1.0, "TF lookup/transform failed (%s -> %s): %s",
                                   pose_cam.header.frame_id, self.base_frame, str(e))
            return None

    def cb_img(self, msg: Image):
        if not self.have_cam_info:
            rospy.logwarn_throttle(2.0, "Waiting for CameraInfo...")
            return

        # Convert to OpenCV
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, "cv_bridge failed: %s", str(e))
            return

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        # Copy intrinsics safely
        with self.lock:
            K = self.K.copy()
            D = self.D.copy()

        # Collect
        for det in detections:
            tid = int(det.tag_id)
            if tid not in BOARD_MAP:
                continue

            corners = det.corners.astype(np.float32)

            # solvePnP returns pose of the tag in the camera frame:
            # X_cam = R * X_obj + t
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                corners,
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                continue

            # Build PoseStamped in camera optical frame
            pose_cam = PoseStamped()
            pose_cam.header.stamp = msg.header.stamp
            # IMPORTANT: solvePnP matches optical convention (x right, y down, z forward)
            pose_cam.header.frame_id = self.camera_frame

            t = tvec.flatten().astype(np.float32)
            q = R.from_rotvec(rvec.flatten()).as_quat().astype(np.float32)  # x y z w

            pose_cam.pose.position.x = float(t[0])
            pose_cam.pose.position.y = float(t[1])
            pose_cam.pose.position.z = float(t[2])
            pose_cam.pose.orientation.x = float(q[0])
            pose_cam.pose.orientation.y = float(q[1])
            pose_cam.pose.orientation.z = float(q[2])
            pose_cam.pose.orientation.w = float(q[3])

            # Transform to base frame
            pose_base = self.transform_pose_to_base(pose_cam)
            if pose_base is None:
                continue

            pos = np.array([
                pose_base.pose.position.x,
                pose_base.pose.position.y,
                pose_base.pose.position.z
            ], dtype=np.float32)

            quat = np.array([
                pose_base.pose.orientation.x,
                pose_base.pose.orientation.y,
                pose_base.pose.orientation.z,
                pose_base.pose.orientation.w
            ], dtype=np.float32)

            # Outlier rejection in BASE frame (more meaningful)
            if tid in self.running_median and len(self.pos_samples[tid]) >= 8:
                if pos_outlier(pos, self.running_median[tid], self.outlier_thresh):
                    continue

            self.pos_samples[tid].append(pos)
            self.quat_samples[tid].append(quat)

            if len(self.pos_samples[tid]) >= 8 and (len(self.pos_samples[tid]) % 5 == 0):
                self.running_median[tid] = np.median(np.array(self.pos_samples[tid], dtype=np.float32), axis=0)

            # Draw overlay
            corners_int = corners.astype(int)
            for i in range(4):
                cv2.line(color_image, tuple(corners_int[i]), tuple(corners_int[(i + 1) % 4]), (0, 255, 0), 2)

            r, c = BOARD_MAP[tid]
            txt = f"ID{tid} ({r},{c}) n={len(self.pos_samples[tid])}"
            cv2.putText(color_image, txt, (corners_int[0][0], corners_int[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Status print
        now = time.time()
        if now - self.last_print > self.print_every:
            os.system("clear")
            print("[INFO] Sampling AprilTag poses (BASE frame)")
            for tid in sorted(BOARD_MAP.keys()):
                r, c = BOARD_MAP[tid]
                n = len(self.pos_samples[tid])
                print(f"  Tag {tid} -> cell({r},{c}) : {n}/{self.target_samples}")
            print(f"\nBase frame: {self.base_frame}")
            print(f"Camera frame used for solvePnP: {self.camera_frame}")
            print("\nPress 's' to save (needs >= min samples per tag).")
            print("Press 'q' to quit.")
            self.last_print = now

        cv2.imshow("AprilTag Calibration (to base)", color_image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            rospy.signal_shutdown("User quit")
        if k == ord("s"):
            results = self.save_results_json(self.output_json)
            rospy.loginfo("Saved %d cells to %s", len(results), self.output_json)

        if self.all_done():
            results = self.save_results_json(self.output_json)
            rospy.loginfo("DONE: all tags reached %d samples. Saved %d cells to %s",
                          self.target_samples, len(results), self.output_json)
            rospy.signal_shutdown("Completed")

def main():
    rospy.init_node("apriltag_calib_to_base", anonymous=False)
    node = AprilTagCalibToBase()
    rospy.spin()

if __name__ == "__main__":
    main()
