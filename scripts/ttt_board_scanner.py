#!/usr/bin/env python3
import os
import json
import numpy as np
import cv2
import rospy
import rospkg

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Empty

from ultralytics import YOLO

from ttt_robot.msg import BoardGrid


def token_from_name(raw: str):
    s = (raw or "").strip().lower().replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if s.startswith("x") or " x" in f" {s} ":
        return "X"
    if s.startswith("o") or " o" in f" {s} ":
        return "O"
    return None


class BoardScanner:
    def __init__(self):
        rp = rospkg.RosPack()
        pkg = rp.get_path("ttt_robot")

        # Model + calibration
        self.weights = rospy.get_param("~yolo_weights", os.path.join(pkg, "models", "best.pt"))
        self.cell_json = rospy.get_param(
            "~cell_json",
            os.path.join(pkg, "scripts", "cell_poses_camera_frame.json"),
        )

        # Detection / timing
        self.conf = float(rospy.get_param("~conf_thresh", 0.7))
        self.rate_hz = float(rospy.get_param("~rate_hz", 5.0))
        self.max_assign_dist_px = float(rospy.get_param("~max_assign_dist_px", 120.0))

        # Debug view
        self.show_debug = bool(rospy.get_param("~show_debug", False))
        self.debug_window = rospy.get_param("~debug_window", "TTT Board Scanner Debug")

        # Topics
        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")

        self.out_board_topic = rospy.get_param("~out_board_topic", "/ttt/board_state")
        self.out_csv_topic = rospy.get_param("~out_csv_topic", "/ttt/board_state_csv")
        self.publish_csv_debug = bool(rospy.get_param("~publish_csv_debug", True))

        # --- Scan gating: ONLY scan/publish when robot is at scan pose ---
        self.scan_ready_topic = rospy.get_param("~scan_ready_topic", "/ttt/scan_ready")
        self.scan_window_s = float(rospy.get_param("~scan_window_s", 1.2))
        self.publish_once_per_scan = bool(rospy.get_param("~publish_once_per_scan", True))
        # ---------------------------------------------------------------

        # Buffers
        self.bridge = CvBridge()
        self.image = None

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # Load cells in camera frame -> project to pixels once intrinsics known
        self.cells_cam = self.load_cells_camera(self.cell_json)  # dict (r,c)->np.array([X,Y,Z])
        if len(self.cells_cam) != 9:
            rospy.logwarn("[board_scanner] Expected 9 cells in JSON, got %d", len(self.cells_cam))
        self.cells_px = None  # dict (r,c)->(u,v)

        # Gating state
        self._scan_active = False
        self._scan_active_until = rospy.Time(0)
        self._scan_buf_cells9 = []  # list of cells9 snapshots

        # YOLO
        rospy.loginfo("[board_scanner] Loading YOLO weights: %s", self.weights)
        self.model = YOLO(self.weights)
        rospy.loginfo("[board_scanner] READY")

        # ROS I/O
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.info_topic, CameraInfo, self.cb_info, queue_size=1)
        rospy.Subscriber(self.scan_ready_topic, Empty, self.cb_scan_ready, queue_size=1)

        self.pub_board = rospy.Publisher(self.out_board_topic, BoardGrid, queue_size=1)
        self.pub_csv = rospy.Publisher(self.out_csv_topic, String, queue_size=1) if self.publish_csv_debug else None

    def load_cells_camera(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        cells = {}
        for _, v in data.items():
            r = int(v["row"])
            c = int(v["col"])
            p = v["position_m"]
            cells[(r, c)] = np.array([float(p["x"]), float(p["y"]), float(p["z"])], dtype=np.float32)
        return cells

    def cb_info(self, msg: CameraInfo):
        self.fx = float(msg.K[0])
        self.fy = float(msg.K[4])
        self.cx = float(msg.K[2])
        self.cy = float(msg.K[5])

        # Precompute pixel centers for each cell
        self.cells_px = {}
        for (r, c), P in self.cells_cam.items():
            u, v = self.project_point(P)
            self.cells_px[(r, c)] = (u, v)

    def cb_image(self, msg: Image):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

    def cb_scan_ready(self, _msg: Empty):
        now = rospy.Time.now()
        self._scan_active = True
        self._scan_active_until = now + rospy.Duration.from_sec(max(0.1, self.scan_window_s))
        self._scan_buf_cells9 = []
        rospy.loginfo("[board_scanner] scan_ready: scanning for %.2fs, then publishing ONE BoardGrid", self.scan_window_s)

    def project_point(self, Pxyz):
        X, Y, Z = float(Pxyz[0]), float(Pxyz[1]), float(Pxyz[2])
        Z = max(Z, 1e-6)
        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy
        return int(round(u)), int(round(v))

    def nearest_cell(self, u, v):
        if not self.cells_px:
            return None, None
        best_rc = None
        best_d = 1e9
        for (r, c), (cu, cv) in self.cells_px.items():
            d = float(np.hypot(u - cu, v - cv))
            if d < best_d:
                best_d = d
                best_rc = (r, c)
        if best_rc is None or best_d > self.max_assign_dist_px:
            return None, best_d
        return best_rc, best_d

    @staticmethod
    def board_to_cells9(board3):
        return [board3[r][c] for r in range(3) for c in range(3)]

    @staticmethod
    def _majority_vote_cells9(buf):
        if not buf:
            return [""] * 9
        out = []
        for i in range(9):
            xs = sum(1 for b in buf if b[i] == "X")
            os = sum(1 for b in buf if b[i] == "O")
            if xs == 0 and os == 0:
                out.append("")
            elif xs >= os:
                out.append("X")
            else:
                out.append("O")
        return out

    def _publish_board(self, cells9):
        msg = BoardGrid()
        msg.header.stamp = rospy.Time.now()
        msg.cells = list(cells9)
        self.pub_board.publish(msg)
        if self.pub_csv is not None:
            csv = ",".join([c if c else "" for c in msg.cells])
            self.pub_csv.publish(String(data=csv))

    def spin(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.image is None or self.fx is None or self.cells_px is None:
                rate.sleep()
                continue

            now = rospy.Time.now()

            # Not in scan window -> do nothing (optionally show idle debug)
            if not self._scan_active:
                if self.show_debug and self.image is not None:
                    img_idle = self.image.copy()
                    for (r, c), (u, v) in (self.cells_px or {}).items():
                        cv2.circle(img_idle, (u, v), 6, (0, 255, 255), -1)
                        cv2.putText(img_idle, f"{r},{c}", (u + 8, v - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(img_idle, "IDLE (waiting /ttt/scan_ready)", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(self.debug_window, img_idle)
                    cv2.waitKey(1)
                rate.sleep()
                continue

            # Scan window expired -> publish voted snapshot once, then stop scanning
            if now > self._scan_active_until:
                voted = self._majority_vote_cells9(self._scan_buf_cells9)
                self._publish_board(voted)
                rospy.loginfo("[board_scanner] published gated board_state (voted over %d frames)", len(self._scan_buf_cells9))
                self._scan_active = False
                self._scan_buf_cells9 = []
                rate.sleep()
                continue

            # --- Active scan window: run detection and collect snapshots ---
            img = self.image.copy()
            board3 = [["" for _ in range(3)] for _ in range(3)]
            best_for_cell = {}  # (r,c)->(token, conf, bbox, center, dist)

            if self.show_debug:
                for (r, c), (u, v) in self.cells_px.items():
                    cv2.circle(img, (u, v), 6, (0, 255, 255), -1)
                    cv2.putText(img, f"{r},{c}", (u + 8, v - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            try:
                results = self.model(img, conf=self.conf, verbose=False)
                res = results[0]
            except Exception as e:
                rospy.logerr_throttle(2.0, "[board_scanner] YOLO failed: %s", str(e))
                rate.sleep()
                continue

            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls = int(box.cls.cpu())
                    raw = res.names.get(cls, str(cls))
                    token = token_from_name(raw)
                    if token is None:
                        continue

                    conf = float(box.conf.cpu())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    rc, dist = self.nearest_cell(cx, cy)
                    if rc is None:
                        continue
                    r, c = rc

                    prev = best_for_cell.get((r, c))
                    if prev is None or conf > prev[1]:
                        best_for_cell[(r, c)] = (token, conf, (x1, y1, x2, y2), (cx, cy), dist)

            for (r, c), (token, conf, bbox, center, dist) in best_for_cell.items():
                board3[r][c] = token
                if self.show_debug:
                    x1, y1, x2, y2 = bbox
                    cx, cy = center
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.putText(img, f"{token} {conf:.2f} d={dist:.0f}",
                                (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cells9 = self.board_to_cells9(board3)
            self._scan_buf_cells9.append(cells9)
            if len(self._scan_buf_cells9) > 100:
                self._scan_buf_cells9.pop(0)

            # Optional: publish continuously during scan window (debug)
            if not self.publish_once_per_scan:
                self._publish_board(cells9)

            if self.show_debug:
                cv2.putText(img, "SCANNING (gated)", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(self.debug_window, img)
                cv2.waitKey(1)

            rate.sleep()


def main():
    rospy.init_node("ttt_board_scanner", anonymous=False)
    node = BoardScanner()
    try:
        node.spin()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
