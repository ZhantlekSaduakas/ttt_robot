#!/usr/bin/env python3
"""
vision_yolo_token_pick_node3.py  (KEEP YAW LOGIC + STABLE POSITION + GREEN RECT ONLY FOR YAW)

Keeps from FIRST code:
- Yaw logic using segmentation contour + cv2.minAreaRect:
    angle_fixed = angle; if w<h: angle_fixed += 90
    angle_fixed = _norm_0_180(angle_fixed)
    yaw = _norm_0_180(angle_fixed - 90)
  NOTE: _norm_0_180 matches your FIRST code exactly (including >179 -> 0 clamp).

Improves from SECOND code:
- Uses BOX CENTER (cx,cy) for 3D position (stable).
- Uses GREEN rotated minAreaRect ONLY for yaw estimation (decoupled from position).
- Optional top-face depth flatness filtering:
    ~min_valid_depth_px (default 300)
    ~max_depth_sigma_m  (default 0.007)
    ~bbox_shrink        (default 0.65)

Selection rule (same as before):
- Select best detection of requested token by XY camera-frame distance:
    score = x^2 + y^2   (smaller = better)

Publishes (CONTINUOUSLY):
  /ttt/token_best_pose_cam   PoseStamped (camera optical frame)
  /ttt/token_best_pose_base  PoseStamped (base frame) [tf2_geometry_msgs if available]
  /ttt/token_best_yaw_deg    Float32     (camera-plane yaw, degrees)
"""

import os
import threading
import numpy as np
import cv2
import rospy
import rospkg

from ultralytics import YOLO

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Float32, String

import tf2_ros

try:
    import tf2_geometry_msgs  # noqa
except Exception as e:
    tf2_geometry_msgs = None
    _TF2_GEOM_ERR = str(e)


def token_from_name(raw_name: str):
    s = raw_name.strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if s.startswith("x") or " x" in f" {s} ":
        return "X"
    if s.startswith("o") or " o" in f" {s} ":
        return "O"
    return None


# ---- KEEP EXACTLY FROM YOUR FIRST CODE (incl. >179 -> 0 behavior) ----
def _norm_0_180(deg: float) -> float:
    a = float(deg) % 180.0
    if a < 0:
        a += 180.0
    if a > 179.0:
        a = 0.0
    return a


# ---- KEEP YAW COMPUTATION LOGIC FROM YOUR FIRST CODE ----
def yaw_from_contour_minarearect(mask_bool: np.ndarray):
    """
    contour -> minAreaRect angle as yaw, then subtract 90deg:
      yaw = normalize( rect_angle_fixed - 90deg ) in [0,180)

    Returns:
      yaw_deg, contour_box(4,2), rect_angle_dbg
    """
    m = (mask_bool.astype(np.uint8) * 255)
    if m.ndim != 2:
        m = m[:, :, 0]

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, None, None

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 50.0:
        return 0.0, None, None

    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h), angle)
    (_rcx, _rcy), (w, h), angle = rect
    w = float(w)
    h = float(h)
    angle = float(angle)

    angle_fixed = angle
    # OpenCV convention: angle in [-90,0), use long-side direction consistently
    if w < h:
        angle_fixed += 90.0

    angle_fixed = _norm_0_180(angle_fixed)
    yaw = _norm_0_180(angle_fixed - 90.0)

    box = cv2.boxPoints(rect)
    box = np.int32(np.round(box))

    return float(yaw), box, float(angle_fixed)


class VisionYoloTokenPick:
    def __init__(self):
        rp = rospkg.RosPack()
        pkg = rp.get_path("ttt_robot")
        default_weights = os.path.join(pkg, "models", "best.pt")

        self.weights = rospy.get_param("~yolo_weights", default_weights)
        self.conf = float(rospy.get_param("~conf_thresh", 0.5))

        self.token = rospy.get_param("~token_type", "X").strip().upper()
        if self.token not in ("X", "O"):
            self.token = "X"

        self.target_token_topic = rospy.get_param("~target_token_topic", "/ttt/vision_target_token")

        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")
        self.base_frame = rospy.get_param("~base_frame", "panda_link0")

        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        self.crop_left_px = int(rospy.get_param("~crop_left_px", 50))
        self.depth_patch_r = int(rospy.get_param("~depth_patch_r", 3))

        # --- Top-face filter params (from your SECOND code) ---
        self.min_valid_depth_px = int(rospy.get_param("~min_valid_depth_px", 300))
        self.max_depth_sigma_m = float(rospy.get_param("~max_depth_sigma_m", 0.007))
        self.bbox_shrink = float(rospy.get_param("~bbox_shrink", 0.65))

        self.rate_hz = float(rospy.get_param("~rate_hz", 10.0))
        self.show_debug = bool(rospy.get_param("~show_debug", True))
        self.debug_window = rospy.get_param("~debug_window", "YOLO Token Pick Debug")

        self.pub_cam = rospy.Publisher("/ttt/token_best_pose_cam", PoseStamped, queue_size=1, latch=True)
        self.pub_base = rospy.Publisher("/ttt/token_best_pose_base", PoseStamped, queue_size=1, latch=True)
        self.pub_yaw = rospy.Publisher("/ttt/token_best_yaw_deg", Float32, queue_size=1, latch=True)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.color = None
        self.depth_raw = None
        self.stamp = None

        self.fx = self.fy = self.cx = self.cy = None

        rospy.Subscriber(self.color_topic, Image, self.cb_color, queue_size=1)
        rospy.Subscriber(self.depth_topic, Image, self.cb_depth, queue_size=1)
        rospy.Subscriber(self.info_topic, CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber(self.target_token_topic, String, self.cb_target_token, queue_size=1)

        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        rospy.loginfo("[yolo_token_pick3] Loading YOLO weights: %s", self.weights)
        self.model = YOLO(self.weights)

        if tf2_geometry_msgs is None:
            rospy.logwarn("[yolo_token_pick3] tf2_geometry_msgs NOT available (PyKDL missing).")
            rospy.logwarn("  Error: %s", _TF2_GEOM_ERR)
            rospy.logwarn("  Base pose publish will be disabled until fixed.")
            self.tf2_ok = False
        else:
            self.tf2_ok = True

        rospy.loginfo("[yolo_token_pick3] READY")
        rospy.loginfo("  token(default)=%s conf=%.2f crop_left_px=%d", self.token, self.conf, self.crop_left_px)
        rospy.loginfo("  target_token_topic=%s", self.target_token_topic)
        rospy.loginfo("  topics: color=%s depth=%s info=%s", self.color_topic, self.depth_topic, self.info_topic)
        rospy.loginfo("  frames: camera_frame=%s base_frame=%s", self.camera_frame, self.base_frame)
        rospy.loginfo("  top-face filter: sigma<=%.4fm, min_px=%d, bbox_shrink=%.2f",
                      self.max_depth_sigma_m, self.min_valid_depth_px, self.bbox_shrink)

    def cb_target_token(self, msg: String):
        s = (msg.data or "").strip().upper()
        if s in ("X", "O"):
            if s != self.token:
                self.token = s
                rospy.loginfo("[yolo_token_pick3] Target token set to: %s", self.token)

    def cb_info(self, msg: CameraInfo):
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])

    def cb_color(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        with self.lock:
            self.color = img
            self.stamp = msg.header.stamp

    def cb_depth(self, msg: Image):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return
        with self.lock:
            self.depth_raw = np.asarray(d)

    def _depth_to_meters(self, depth_arr: np.ndarray) -> np.ndarray:
        da = depth_arr.astype(np.float32)
        mx = float(np.nanmax(da)) if da.size else 0.0
        if mx > 20.0:
            return da / 1000.0
        return da

    def _median_depth_m(self, depth_m: np.ndarray, u: int, v: int, r: int) -> float:
        h, w = depth_m.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        x0, x1 = max(u - r, 0), min(u + r + 1, w)
        y0, y1 = max(v - r, 0), min(v + r + 1, h)
        patch = depth_m[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0]
        if patch.size == 0:
            return 0.0
        return float(np.median(patch))

    def _depth_flat_enough(self, depth_m: np.ndarray, roi_mask: np.ndarray) -> bool:
        if roi_mask is None:
            return False
        z = depth_m[roi_mask]
        z = z[np.isfinite(z)]
        z = z[z > 0]
        if z.size < self.min_valid_depth_px:
            return False
        z_med = float(np.median(z))
        mad = float(np.median(np.abs(z - z_med)))
        sigma = 1.4826 * mad
        return sigma <= self.max_depth_sigma_m

    def _bbox_inner_mask(self, h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        x1 = int(np.clip(x1, 0, w - 1)); x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1)); y2 = int(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = (x2 - x1) * self.bbox_shrink
        bh = (y2 - y1) * self.bbox_shrink
        ix1 = int(np.clip(cx - 0.5 * bw, 0, w - 1))
        ix2 = int(np.clip(cx + 0.5 * bw, 0, w - 1))
        iy1 = int(np.clip(cy - 0.5 * bh, 0, h - 1))
        iy2 = int(np.clip(cy + 0.5 * bh, 0, h - 1))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        m = np.zeros((h, w), dtype=bool)
        m[iy1:iy2, ix1:ix2] = True
        return m

    def _transform_pose_to_base(self, pose_cam: PoseStamped):
        if not self.tf2_ok:
            return None
        try:
            tf = self.tf_buf.lookup_transform(
                self.base_frame,
                pose_cam.header.frame_id,
                pose_cam.header.stamp,
                rospy.Duration(0.2),
            )
            return tf2_geometry_msgs.do_transform_pose(pose_cam, tf)
        except Exception as e:
            rospy.logwarn_throttle(
                1.0,
                "[yolo_token_pick3] TF lookup/transform failed (%s -> %s): %s",
                pose_cam.header.frame_id, self.base_frame, str(e),
            )
            return None

    def _yolo_infer(self, bgr_img):
        try:
            out = self.model.predict(source=bgr_img, conf=self.conf, verbose=False, device="cpu")
            if out is None or len(out) == 0:
                return None
            return out[0]
        except Exception as e:
            rospy.logerr_throttle(2.0, "[yolo_token_pick3] YOLO inference failed: %s", str(e))
            return None

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                color = None if self.color is None else self.color.copy()
                depth_raw = None if self.depth_raw is None else self.depth_raw.copy()
                stamp = self.stamp

            if color is None or depth_raw is None or stamp is None or self.fx is None:
                rate.sleep()
                continue

            ch, cw = color.shape[:2]
            if depth_raw.shape[:2] != (ch, cw):
                depth_raw = cv2.resize(depth_raw, (cw, ch), interpolation=cv2.INTER_NEAREST)

            depth_m = self._depth_to_meters(depth_raw)

            if self.crop_left_px > 0:
                color[:, : self.crop_left_px] = 0

            res = self._yolo_infer(color)
            if res is None:
                if self.show_debug:
                    cv2.imshow(self.debug_window, color)
                    cv2.waitKey(1)
                rate.sleep()
                continue

            dbg = color
            try:
                plotted = res.plot()
                if plotted is not None:
                    dbg = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
            except Exception:
                dbg = color

            best = None
            boxes_ok = (res.boxes is not None) and (len(res.boxes) > 0)
            masks_ok = (res.masks is not None) and (getattr(res.masks, "data", None) is not None)

            # --- Preferred: masks exist -> yaw from minAreaRect (green), position from bbox center ---
            if boxes_ok and masks_ok:
                masks = res.masks.data.cpu().numpy()
                for i, cls in enumerate(res.boxes.cls.cpu().numpy()):
                    raw = res.names.get(int(cls), str(int(cls)))
                    tok = token_from_name(raw)
                    if tok != self.token:
                        continue

                    conf = float(res.boxes.conf[i].cpu().numpy())

                    # bbox center for position (stable)
                    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # mask for flatness + yaw (green rotated box)
                    mask = masks[i].astype(np.uint8)
                    if mask.shape[:2] != (ch, cw):
                        mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask.astype(bool)
                    if mask_bool.sum() < 50:
                        continue

                    yaw, contour_box, rect_dbg = yaw_from_contour_minarearect(mask_bool)

                    yaw = yaw % 90.0

                    # top-face gate
                    if not self._depth_flat_enough(depth_m, mask_bool):
                        continue

                    d = self._median_depth_m(depth_m, cx, cy, self.depth_patch_r)
                    if d <= 0:
                        continue

                    x_cam = (cx - self.cx) * d / self.fx
                    y_cam = (cy - self.cy) * d / self.fy
                    score_xy = float(x_cam * x_cam + y_cam * y_cam)

                    det = {
                        "cx": cx, "cy": cy, "d": d,
                        "yaw": float(yaw),
                        "conf": conf,
                        "score_xy": score_xy,
                        "contour_box": contour_box,
                        "rect_dbg": rect_dbg,
                    }
                    if best is None or det["score_xy"] < best["score_xy"]:
                        best = det

            # --- Fallback: no masks -> yaw=0, position from bbox center, flatness on inner bbox ---
            if best is None and boxes_ok:
                for i, cls in enumerate(res.boxes.cls.cpu().numpy()):
                    raw = res.names.get(int(cls), str(int(cls)))
                    tok = token_from_name(raw)
                    if tok != self.token:
                        continue

                    conf = float(res.boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    inner = self._bbox_inner_mask(ch, cw, x1, y1, x2, y2)
                    if inner is None or not self._depth_flat_enough(depth_m, inner):
                        continue

                    d = self._median_depth_m(depth_m, cx, cy, self.depth_patch_r)
                    if d <= 0:
                        continue

                    x_cam = (cx - self.cx) * d / self.fx
                    y_cam = (cy - self.cy) * d / self.fy
                    score_xy = float(x_cam * x_cam + y_cam * y_cam)

                    det = {
                        "cx": cx, "cy": cy, "d": d,
                        "yaw": 0.0,
                        "conf": conf,
                        "score_xy": score_xy,
                        "contour_box": None,
                        "rect_dbg": None,
                    }
                    if best is None or det["score_xy"] < best["score_xy"]:
                        best = det

            if best is None:
                if self.show_debug:
                    cv2.putText(dbg, f"{self.token}: no valid detection", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(self.debug_window, dbg)
                    cv2.waitKey(1)
                rate.sleep()
                continue

            u, v, z = best["cx"], best["cy"], best["d"]
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            self.pub_yaw.publish(Float32(best["yaw"]))

            if self.show_debug:
                # green yaw rectangle (from mask)
                if best.get("contour_box") is not None:
                    cv2.polylines(dbg, [best["contour_box"]], True, (0, 255, 0), 2)
                # red dot = bbox-center position
                cv2.circle(dbg, (u, v), 6, (0, 0, 255), -1)
                rect_dbg = best.get("rect_dbg", None)
                rect_txt = f" rect_fix={rect_dbg:.1f}" if rect_dbg is not None else ""
                cv2.putText(
                    dbg,
                    f"{self.token} cam=({x:.3f},{y:.3f},{z:.3f}) yaw={best['yaw']:.1f}{rect_txt} score={best['score_xy']:.4f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
                )
                cv2.imshow(self.debug_window, dbg)
                cv2.waitKey(1)

            pose_cam = PoseStamped()
            pose_cam.header.stamp = stamp
            pose_cam.header.frame_id = self.camera_frame
            pose_cam.pose.position.x = float(x)
            pose_cam.pose.position.y = float(y)
            pose_cam.pose.position.z = float(z)
            pose_cam.pose.orientation.w = 1.0  # yaw is published separately
            self.pub_cam.publish(pose_cam)

            pose_base = self._transform_pose_to_base(pose_cam)
            if pose_base is not None:
                self.pub_base.publish(pose_base)

            rate.sleep()


def main():
    rospy.init_node("vision_yolo_token_pick_node3", anonymous=False)
    node = VisionYoloTokenPick()
    try:
        node.spin()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()