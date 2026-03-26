#!/usr/bin/env python3
import os
import json
import math
import time
import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector

# ----------------------------
# Config (ROS params supported)
# ----------------------------
DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_INFO_TOPIC  = "/camera/color/camera_info"

DEFAULT_TAG_FAMILY  = "tag36h11"
DEFAULT_TAG_SIZE_M  = 0.035

# Default: 9 tags with IDs 0..8
DEFAULT_TAG_IDS     = list(range(9))

DEFAULT_CAM_FRAME   = "camera_color_optical_frame"
DEFAULT_OUTPUT_JSON = os.path.expanduser("~/ttt_ws/src/ttt_robot/scripts/apriltag_poses_camera.json")

# Drawing
AXIS_LEN_M = 0.03  # length of drawn axes in meters (visual only)

# Filtering
MIN_MASK_AREA_PX = 0  # not used here; left for extension


def solve_tag_pose_ippe_square(corners_px, K, tag_size_m):
    """
    Robust pose for planar square using OpenCV IPPE.
    corners_px: (4,2) pixel corners (order from detector)
    K: 3x3 intrinsics
    Returns (R_cam_tag, t_cam_tag, reproj_err) or None
    """
    s = tag_size_m / 2.0

    # Object points in tag frame plane z=0.
    # NOTE: Assumes detector corner order matches this order.
    obj = np.array([
        [-s, -s, 0.0],
        [ s, -s, 0.0],
        [ s,  s, 0.0],
        [-s,  s, 0.0],
    ], dtype=np.float64)

    img = np.array(corners_px, dtype=np.float64).reshape(4, 1, 2)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or len(rvecs) == 0:
        return None

    best_idx = None
    best_err = 1e18
    for i, (rv, tv, er) in enumerate(zip(rvecs, tvecs, errs)):
        z = float(tv[2])
        e = float(er) if np.size(er) == 1 else float(er[0])
        if z <= 0:
            continue
        if e < best_err:
            best_err = e
            best_idx = i

    if best_idx is None:
        return None

    rvec = rvecs[best_idx]
    tvec = tvecs[best_idx].reshape(3)
    R_cam_tag, _ = cv2.Rodrigues(rvec)
    return R_cam_tag, tvec, float(best_err)


def draw_axes(img, K, R_cam_tag, t_cam_tag, axis_len_m=0.03):
    """Draw tag axes (x=red, y=green, z=blue) projected onto the image."""
    # 3D points in tag frame
    pts_tag = np.array([
        [0, 0, 0],
        [axis_len_m, 0, 0],
        [0, axis_len_m, 0],
        [0, 0, axis_len_m],
    ], dtype=np.float64)

    # Transform to camera: p_cam = R * p_tag + t
    pts_cam = (R_cam_tag @ pts_tag.T).T + t_cam_tag.reshape(1, 3)

    # Project
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    def proj(P):
        X, Y, Z = P
        Z = max(Z, 1e-6)
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return int(round(u)), int(round(v))

    o = proj(pts_cam[0])
    x = proj(pts_cam[1])
    y = proj(pts_cam[2])
    z = proj(pts_cam[3])

    cv2.line(img, o, x, (0, 0, 255), 3)   # X red (BGR)
    cv2.line(img, o, y, (0, 255, 0), 3)   # Y green
    cv2.line(img, o, z, (255, 0, 0), 3)   # Z blue


def pose_dict_from_Rt(R_cam_tag, t_cam_tag):
    q = R.from_matrix(R_cam_tag).as_quat()  # x,y,z,w
    return {
        "position_m": {"x": float(t_cam_tag[0]), "y": float(t_cam_tag[1]), "z": float(t_cam_tag[2])},
        "quaternion_xyzw": {"x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3])},
    }


def main():
    rospy.init_node("save_apriltag_coords")

    image_topic = rospy.get_param("~image_topic", DEFAULT_IMAGE_TOPIC)
    info_topic  = rospy.get_param("~info_topic", DEFAULT_INFO_TOPIC)

    tag_family  = rospy.get_param("~tag_family", DEFAULT_TAG_FAMILY)
    tag_size_m  = float(rospy.get_param("~tag_size_m", DEFAULT_TAG_SIZE_M))

    tag_ids_param = rospy.get_param("~tag_ids", DEFAULT_TAG_IDS)  # can be list or string
    if isinstance(tag_ids_param, str):
        # allow "0,1,2,3,4,5,6,7,8"
        tag_ids = [int(x.strip()) for x in tag_ids_param.split(",") if x.strip() != ""]
    else:
        tag_ids = [int(x) for x in tag_ids_param]
    tag_ids = sorted(tag_ids)

    cam_frame = rospy.get_param("~cam_frame", DEFAULT_CAM_FRAME)
    out_json  = rospy.get_param("~output_json", DEFAULT_OUTPUT_JSON)

    bridge = CvBridge()

    latest_img = {"msg": None}
    latest_info = {"msg": None}

    def cb_img(msg): latest_img["msg"] = msg
    def cb_info(msg): latest_info["msg"] = msg

    rospy.Subscriber(image_topic, Image, cb_img, queue_size=1)
    rospy.Subscriber(info_topic, CameraInfo, cb_info, queue_size=1)

    detector = Detector(
        families=tag_family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )

    rospy.loginfo("save_apriltag_coords running.")
    rospy.loginfo("  image_topic=%s", image_topic)
    rospy.loginfo("  info_topic=%s", info_topic)
    rospy.loginfo("  tag_family=%s tag_size_m=%.4f", tag_family, tag_size_m)
    rospy.loginfo("  tag_ids=%s", tag_ids)
    rospy.loginfo("Controls: [s]=save JSON, [q]=quit")

    # Store last good pose per tag_id
    tag_poses = {}  # id -> dict with pose + meta

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        if latest_img["msg"] is None or latest_info["msg"] is None:
            rate.sleep()
            continue

        img_msg = latest_img["msg"]
        info_msg = latest_info["msg"]

        try:
            bgr = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "cv_bridge failed: %s", str(e))
            rate.sleep()
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        K = np.array(info_msg.K, dtype=np.float64).reshape(3, 3)

        tags = detector.detect(gray, estimate_tag_pose=False)

        # Draw and update poses
        for t in tags:
            tid = int(t.tag_id)
            if tid not in tag_ids:
                continue

            sol = solve_tag_pose_ippe_square(t.corners, K, tag_size_m)
            if sol is None:
                continue
            R_cam_tag, t_cam_tag, err = sol

            # Save in memory
            tag_poses[tid] = {
                "tag_id": tid,
                "frame_id": cam_frame,
                "reproj_err_px": float(err),
                "corners_px": [[float(x), float(y)] for (x, y) in t.corners],
                **pose_dict_from_Rt(R_cam_tag, t_cam_tag),
                "stamp": float(img_msg.header.stamp.to_sec()) if img_msg.header.stamp != rospy.Time(0) else time.time(),
            }

            # Draw corners, ID, axes
            corners_i = np.array(t.corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr, [corners_i], True, (0, 255, 255), 2)

            cxy = t.center
            cx, cy = int(cxy[0]), int(cxy[1])
            cv2.circle(bgr, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(bgr, f"ID {tid} err {err:.2f}px",
                        (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            draw_axes(bgr, K, R_cam_tag, t_cam_tag, axis_len_m=AXIS_LEN_M)

        # Status overlay
        seen = [i for i in tag_ids if i in tag_poses]
        missing = [i for i in tag_ids if i not in tag_poses]
        cv2.putText(bgr, f"Seen {len(seen)}/{len(tag_ids)} tags. Missing: {missing}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(bgr, "Press 's' to save JSON, 'q' to quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("AprilTag Capture + Save", bgr)
        key = cv2.waitKey(1) & 0xFF

        # Save if user presses 's' OR if all tags are captured
        if key == ord('s') or (len(seen) == len(tag_ids) and len(tag_ids) > 0):
            out = {
                "camera_frame": cam_frame,
                "image_topic": image_topic,
                "camera_info_topic": info_topic,
                "tag_family": tag_family,
                "tag_size_m": tag_size_m,
                "tag_ids_expected": tag_ids,
                "tags": {str(tid): tag_poses[tid] for tid in sorted(tag_poses.keys())}
            }
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w") as f:
                json.dump(out, f, indent=2)
            rospy.loginfo("Saved %d tag poses to %s", len(tag_poses), out_json)

            # If auto-saved because all tags found, don’t spam saves repeatedly
            if len(seen) == len(tag_ids) and key != ord('s'):
                cv2.putText(bgr, "AUTO-SAVED (all tags found). Press 'q' to quit.",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("AprilTag Capture + Save", bgr)
                cv2.waitKey(50)

        if key == ord('q'):
            break

        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
