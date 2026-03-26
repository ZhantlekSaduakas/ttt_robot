#!/usr/bin/env python3
"""
ttt_board_gui.py

Terminal GUI for Tic-Tac-Toe board state.

- Subscribes to /ttt/board_state (ttt_robot/BoardGrid) published by ttt_board_scanner.py.
- Optionally listens to /ttt/scan_ready and /ttt/robot_move_done to annotate updates.

Behavior:
- Always shows the latest board.
- Marks a board as "FRESH" if it arrived after the most recent /ttt/scan_ready.
- Clears and redraws the terminal on each meaningful update.

No external deps; pure stdout ANSI.
"""

import sys
import rospy
from std_msgs.msg import Empty
from ttt_robot.msg import BoardGrid


def _sanitize(c: str) -> str:
    s = (c or "").strip().upper()
    return s if s in ("X", "O") else ""


def _render_board(board9):
    def cell(i):
        return board9[i] if board9[i] else " "
    rows = [
        f" {cell(0)} | {cell(1)} | {cell(2)} ",
        "---+---+---",
        f" {cell(3)} | {cell(4)} | {cell(5)} ",
        "---+---+---",
        f" {cell(6)} | {cell(7)} | {cell(8)} ",
    ]
    return "\n".join(rows)


class TTTBoardGUI:
    def __init__(self):
        self.board_topic = rospy.get_param("~board_topic", "/ttt/board_state")
        self.scan_ready_topic = rospy.get_param("~scan_ready_topic", "/ttt/scan_ready")
        self.robot_done_topic = rospy.get_param("~robot_done_topic", "/ttt/robot_move_done")

        self.title = rospy.get_param("~title", "Tic-Tac-Toe Board")
        self.show_help = bool(rospy.get_param("~show_help", True))

        self.latest_board9 = [""] * 9
        self.latest_stamp = rospy.Time(0)

        self.last_scan_ready = rospy.Time(0)
        self.last_robot_done = rospy.Time(0)

        self._last_render_key = None  # (board tuple, fresh_flag, scan_seq, done_seq)
        self._scan_seq = 0
        self._done_seq = 0

        rospy.Subscriber(self.board_topic, BoardGrid, self.cb_board, queue_size=5)
        rospy.Subscriber(self.scan_ready_topic, Empty, self.cb_scan_ready, queue_size=5)
        rospy.Subscriber(self.robot_done_topic, Empty, self.cb_robot_done, queue_size=5)

        # Draw once at start
        self._draw(extra=["Waiting for /ttt/board_state ..."])

    def cb_scan_ready(self, _msg: Empty):
        self.last_scan_ready = rospy.Time.now()
        self._scan_seq += 1
        self._draw(extra=[f"Scan ready received. Waiting for fresh board ... (scan #{self._scan_seq})"])

    def cb_robot_done(self, _msg: Empty):
        self.last_robot_done = rospy.Time.now()
        self._done_seq += 1
        self._draw(extra=[f"Robot move done received. (done #{self._done_seq})"])

    def cb_board(self, msg: BoardGrid):
        if len(msg.cells) != 9:
            rospy.logwarn_throttle(2.0, "[board_gui] BoardGrid length != 9 (%d)", len(msg.cells))
            return
        b = [_sanitize(c) for c in msg.cells]
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

        self.latest_board9 = b
        self.latest_stamp = stamp
        self._draw()

    def _draw(self, extra=None):
        extra = extra or []

        fresh = (self.latest_stamp >= self.last_scan_ready) if self.last_scan_ready != rospy.Time(0) else False

        key = (tuple(self.latest_board9), bool(fresh), self._scan_seq, self._done_seq)
        if key == self._last_render_key and not extra:
            return
        self._last_render_key = key

        # Clear screen + home cursor
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.write(f"{self.title}\n")
        sys.stdout.write(f"board_topic: {self.board_topic}\n\n")

        sys.stdout.write(_render_board(self.latest_board9) + "\n\n")

        status = []
        if self.last_scan_ready != rospy.Time(0):
            status.append(f"last_scan_ready: {self.last_scan_ready.to_sec():.3f}")
        if self.latest_stamp != rospy.Time(0):
            status.append(f"last_board_stamp: {self.latest_stamp.to_sec():.3f}")
        if fresh:
            status.append("FRESH_AFTER_SCAN: YES")
        elif self.last_scan_ready != rospy.Time(0):
            status.append("FRESH_AFTER_SCAN: no (waiting new board)")
        if self.last_robot_done != rospy.Time(0):
            status.append(f"last_robot_done: {self.last_robot_done.to_sec():.3f}")

        if status:
            sys.stdout.write(" | ".join(status) + "\n")

        for ln in extra:
            sys.stdout.write(ln + "\n")

        if self.show_help:
            sys.stdout.write("\nNotes:\n")
            sys.stdout.write("- Board updates come from ttt_board_scanner.py publishing BoardGrid.\n")
            sys.stdout.write("- \"FRESH_AFTER_SCAN\" becomes YES when a board message stamp is after /ttt/scan_ready.\n")

        sys.stdout.flush()


def main():
    rospy.init_node("ttt_board_gui", anonymous=False)
    TTTBoardGUI()
    rospy.spin()


if __name__ == "__main__":
    main()
