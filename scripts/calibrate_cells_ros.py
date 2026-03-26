#!/usr/bin/env python3
import os
import json
import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs  # noqa
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

# ------------------------------
# Params
# ------------------------------
TAG_SIZE_M = float(rospy.get_param("~tag_size_m", 0.035))
TAG_FAMILY = rospy.get_param("~tag_family", "tag36h11")

IMAGE_TOPIC = rospy.get_param("~image_topic", "/camera/color/image_raw")
CAMINFO_TOPIC = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

BASE_FRAME = rospy.get_param("~base_frame", "panda_link0")
OPTICAL_FRAME = rospy.get_param("~optical_frame", "camera_color_optical_frame")
BOARD_FRAME = rospy.get_param("~board_frame", "board_frame")

CELL_SIZE_M = float(rospy.get_param("~cell_size_m", 0.05))  # 5 cm per cell step
MAX_REPROJ_ERR_PX = float(rospy.get_param("~max_reproj_err_px", 5.0))

# Assumed ID layout (row-major)
TAG_IDS = [0,1,2,3,4,5,6,7,8]

# board axes:
# +X_board points from tag0 -> tag2 direction
# +Y_board points from tag0 -> tag6 direction
# +Z_board is XxY (right-hand), flipped so it points toward camera (optional but stabilizes)
FLIP_Z_TOWARD_CAMERA = bool(rospy.get_param("~flip_z_toward_camera", True))

OUT_JSON = rospy.get_param(
    "~output_json",
    os.path.expanduser("~/ttt_ws/src/ttt_robot/scripts/board_calib_9tags.json")
)

# Tag model points (corners in tag frame): TL, TR, BR, BL
half = TAG_SIZE_M / 2.0
OBJECT_POINTS = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0],
], dtype=np.float32)


def order_corners_clockwise_tl(corners: np.ndarray) -> np.ndarray:
    corners = corners.astype(np.float32)
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    idx = np.argsort(angles)
    ccw = corners[idx]

    s = ccw.sum(axis=1)
    start = int(np.argmin(s))
    ccw = np.roll(ccw, -start, axis=0)

    tl = ccw[0]
    bl = ccw[1]
    br = ccw[2]
    tr = ccw[3]
    cw = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)
    return cw


def pose_from_rvec_tvec(rvec, tvec, frame_id: str, stamp: rospy.Time) -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame_id
    ps.pose.position.x = float(tvec[0])
    ps.pose.position.y = float(tvec[1])
    ps.pose.position.z = float(tvec[2])
    q = R.from_rotvec(rvec).as_quat()  # xyzw
    ps.pose.orientation.x = float(q[0])
    ps.pose.orientation.y = float(q[1])
    ps.pose.orientation.z = float(q[2])
    ps.pose.orientation.w = float(q[3])
    return ps


def best_ippe_pose(pts2d_ud, K):
    """Return (rvec, tvec, reproj_err) using solvePnPGeneric(IPPE_SQUARE)."""
    D0 = np.zeros((4, 1), dtype=np.float32)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        OBJECT_POINTS,
        pts2d_ud.reshape(-1, 1, 2),
        K,
        D0,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or rvecs is None or len(rvecs) == 0:
        return None

    best = None
    best_err = 1e18
    for rv, tv, er in zip(rvecs, tvecs, errs):
        tv = tv.reshape(3)
        z = float(tv[2])
        if z <= 0:
            continue
        proj, _ = cv2.projectPoints(OBJECT_POINTS, rv, tv, K, D0)
        proj = proj.reshape(-1, 2)
        reproj = float(np.mean(np.linalg.norm(proj - pts2d_ud, axis=1)))
        if reproj < best_err:
            best_err = reproj
            best = (rv.reshape(3), tv.reshape(3))

    if best is None:
        return None
    rv, tv = best
    return rv.astype(np.float32), tv.astype(np.float32), best_err


def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def build_board_frame_from_tags(tag_positions_base, camera_pos_base=None):
    """
    tag_positions_base: dict {tid: np.array([x,y,z])} in BASE_FRAME
    Returns (R_base_board, t_base_board) where:
      p_base = R_base_board * p_board + t_base_board
    """
    # Need at least these for stable axes
    need = [0,2,6,8,4]
    for tid in need:
        if tid not in tag_positions_base:
            return None

    p0 = tag_positions_base[0]
    p2 = tag_positions_base[2]
    p6 = tag_positions_base[6]
    p4 = tag_positions_base[4]

    x_dir = normalize(p2 - p0)
    y_dir = normalize(p6 - p0)
    if x_dir is None or y_dir is None:
        return None

    # Orthonormalize y w.r.t x (Gram-Schmidt)
    y_dir = y_dir - np.dot(y_dir, x_dir) * x_dir
    y_dir = normalize(y_dir)
    if y_dir is None:
        return None

    z_dir = np.cross(x_dir, y_dir)
    z_dir = normalize(z_dir)
    if z_dir is None:
        return None

    # Optional: flip Z to point toward camera (for consistent orientation)
    if FLIP_Z_TOWARD_CAMERA and camera_pos_base is not None:
        # vector from board to camera
        to_cam = normalize(camera_pos_base - p4)
        if to_cam is not None and np.dot(z_dir, to_cam) < 0:
            z_dir = -z_dir
            # keep right-hand: flip y to preserve x cross y = z
            y_dir = -y_dir

    R_base_board = np.stack([x_dir, y_dir, z_dir], axis=1)  # columns are axes
    t_base_board = p4  # origin at center tag 4
    return R_base_board, t_base_board


class Node:
    def __init__(self):
        self.bridge = CvBridge()
        self.detector = Detector(families=TAG_FAMILY, refine_edges=True)

        self.have_info = False
        self.K = None
        self.D = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tfb = tf2_ros.TransformBroadcaster()

        rospy.Subscriber(CAMINFO_TOPIC, CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber(IMAGE_TOPIC, Image, self.cb_img, queue_size=1)

        rospy.loginfo("apriltag_board_frame_9tags running")
        rospy.loginfo("Image: %s | Info: %s", IMAGE_TOPIC, CAMINFO_TOPIC)
        rospy.loginfo("Frames: %s -> %s, publishing %s", OPTICAL_FRAME, BASE_FRAME, BOARD_FRAME)
        rospy.loginfo("Assumed IDs row-major: %s", TAG_IDS)
        rospy.loginfo("Controls: 's' save JSON, 'q' quit")

    def cb_info(self, msg: CameraInfo):
        if self.have_info:
            return
        self.K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
        self.D = np.array(msg.D, dtype=np.float32).reshape(-1, 1)
        self.have_info = True

        # confirm TF exists (base<-optical)
        try:
            self.tf_buffer.lookup_transform(BASE_FRAME, OPTICAL_FRAME, rospy.Time(0), timeout=rospy.Duration(5.0))
            rospy.loginfo("TF OK: %s <- %s", BASE_FRAME, OPTICAL_FRAME)
        except Exception as e:
            rospy.logerr("TF missing (%s <- %s): %s", BASE_FRAME, OPTICAL_FRAME, str(e))
            raise

    def cb_img(self, msg: Image):
        if not self.have_info:
            return

        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = self.detector.detect(gray)

        # For saving results
        tag_base_positions = {}
        tag_base_poses = {}

        # For drawing
        for det in dets:
            tid = int(det.tag_id)
            if tid not in TAG_IDS:
                continue

            corners_raw = det.corners.astype(np.float32)
            corners = order_corners_clockwise_tl(corners_raw)

            # undistort to pixel coords using K
            corners_ud = cv2.undistortPoints(
                corners.reshape(-1, 1, 2), self.K, self.D, P=self.K
            ).reshape(-1, 2).astype(np.float32)

            sol = best_ippe_pose(corners_ud, self.K)
            if sol is None:
                continue
            rvec, tvec, reproj = sol
            if reproj > MAX_REPROJ_ERR_PX:
                continue

            pose_opt = pose_from_rvec_tvec(rvec, tvec, OPTICAL_FRAME, stamp)

            # transform to base
            try:
                pose_base = self.tf_buffer.transform(pose_opt, BASE_FRAME, timeout=rospy.Duration(0.3))
            except Exception:
                continue

            pB = pose_base.pose.position
            tag_base_positions[tid] = np.array([pB.x, pB.y, pB.z], dtype=np.float32)
            tag_base_poses[tid] = pose_base

            # draw
            pts = corners_raw.astype(int)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                img, f"ID:{tid} rpj:{reproj:.2f}",
                (pts[0][0], pts[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2
            )

        # Try to get camera position in base (for optional Z flipping)
        cam_pos_base = None
        try:
            Tbc = self.tf_buffer.lookup_transform(BASE_FRAME, OPTICAL_FRAME, rospy.Time(0), timeout=rospy.Duration(0.1))
            cam_pos_base = np.array([Tbc.transform.translation.x,
                                    Tbc.transform.translation.y,
                                    Tbc.transform.translation.z], dtype=np.float32)
        except Exception:
            pass

        board_sol = build_board_frame_from_tags(tag_base_positions, camera_pos_base=cam_pos_base)

        status_line = f"Seen {len(tag_base_positions)}/9 tags"
        if board_sol is None:
            cv2.putText(img, status_line + " | board_frame: NOT READY", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            Rbb, tbb = board_sol

            # publish TF base -> board_frame
            T = TransformStamped()
            T.header.stamp = stamp
            T.header.frame_id = BASE_FRAME
            T.child_frame_id = BOARD_FRAME
            T.transform.translation.x = float(tbb[0])
            T.transform.translation.y = float(tbb[1])
            T.transform.translation.z = float(tbb[2])
            q = R.from_matrix(Rbb).as_quat()  # xyzw
            T.transform.rotation.x = float(q[0])
            T.transform.rotation.y = float(q[1])
            T.transform.rotation.z = float(q[2])
            T.transform.rotation.w = float(q[3])
            self.tfb.sendTransform(T)

            # show board origin
            cv2.putText(img, status_line + f" | board_frame OK (origin=tag4)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # optional: print some board-frame coordinates for sanity
            # Example: compute tag positions in board coords (p_board = R^T (p_base - t))
            for tid in [0,4,8]:
                if tid in tag_base_positions:
                    pb = tag_base_positions[tid]
                    p_board = (Rbb.T @ (pb - tbb)).reshape(3)
                    cv2.putText(img, f"ID{tid} in board: x={p_board[0]:.3f} y={p_board[1]:.3f}",
                                (10, 55 + 18 * [0,4,8].index(tid)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.putText(img, "Press 's' save JSON | 'q' quit", (10, img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow("9 AprilTags -> board_frame (TF)", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            self.save_json(tag_base_positions, tag_base_poses, board_sol, stamp)
        elif key == ord('q'):
            rospy.signal_shutdown("User quit")

    def save_json(self, tag_base_positions, tag_base_poses, board_sol, stamp):
        out = {
            "stamp": float(stamp.to_sec()) if stamp != rospy.Time(0) else float(rospy.Time.now().to_sec()),
            "base_frame": BASE_FRAME,
            "optical_frame": OPTICAL_FRAME,
            "board_frame": BOARD_FRAME,
            "tag_family": TAG_FAMILY,
            "tag_size_m": TAG_SIZE_M,
            "cell_size_m": CELL_SIZE_M,
            "assumed_tag_layout_row_major": TAG_IDS,
            "tags_in_base": {},
            "board_in_base": None,
        }

        for tid in sorted(tag_base_positions.keys()):
            p = tag_base_positions[tid]
            pose = tag_base_poses[tid].pose
            out["tags_in_base"][str(tid)] = {
                "position_m": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                "quaternion_xyzw": {
                    "x": float(pose.orientation.x),
                    "y": float(pose.orientation.y),
                    "z": float(pose.orientation.z),
                    "w": float(pose.orientation.w),
                }
            }

        if board_sol is not None:
            Rbb, tbb = board_sol
            q = R.from_matrix(Rbb).as_quat()
            out["board_in_base"] = {
                "translation_m": {"x": float(tbb[0]), "y": float(tbb[1]), "z": float(tbb[2])},
                "quaternion_xyzw": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
                "R_base_board_cols": Rbb.tolist(),
            }

        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        rospy.loginfo("Saved JSON to %s (tags=%d, board=%s)",
                      OUT_JSON, len(tag_base_positions), "OK" if board_sol else "NONE")


def main():
    rospy.init_node("apriltag_board_frame_9tags", anonymous=True)
    Node()
    rospy.spin()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
