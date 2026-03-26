#!/usr/bin/env python3
import sys
import rospy
from std_msgs.msg import Empty, Bool, String
from ttt_robot.msg import RobotMove, BoardGrid

WINS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]

def winner(board9):
    for a, b, c in WINS:
        if board9[a] and board9[a] == board9[b] == board9[c]:
            return board9[a]
    if all(x in ("X", "O") for x in board9):
        return "D"
    return None

def minimax(board9, player_to_move, robot_token):
    w = winner(board9)
    if w == robot_token:
        return (10, None)
    if w == "D":
        return (0, None)
    if w and w != robot_token:
        return (-10, None)

    human_token = "O" if robot_token == "X" else "X"

    def next_player(p):
        return human_token if p == robot_token else robot_token

    if player_to_move == robot_token:
        best_score, best_move = -10**9, None
        for i in range(9):
            if board9[i] != "":
                continue
            b2 = board9[:]
            b2[i] = player_to_move
            score, _ = minimax(b2, next_player(player_to_move), robot_token)
            if score > best_score:
                best_score, best_move = score, i
        return best_score, best_move
    else:
        best_score, best_move = 10**9, None
        for i in range(9):
            if board9[i] != "":
                continue
            b2 = board9[:]
            b2[i] = player_to_move
            score, _ = minimax(b2, next_player(player_to_move), robot_token)
            if score < best_score:
                best_score, best_move = score, i
        return best_score, best_move


class TTTGameManagerSM:
    """
    Professional flow:

    Human finishes move -> presses N
      -> request scan
      -> wait scan_ready
      -> read fresh board_state (BoardGrid)
      -> compute robot move
      -> publish RobotMove
      -> wait robot_move_done
      -> back to wait human + N
    """

    # States
    WAIT_HUMAN_N = "WAIT_HUMAN_N"
    REQUEST_SCAN = "REQUEST_SCAN"
    WAIT_SCAN_READY = "WAIT_SCAN_READY"
    WAIT_FRESH_BOARD = "WAIT_FRESH_BOARD"
    DECIDE_AND_CMD = "DECIDE_AND_CMD"
    WAIT_ROBOT_DONE = "WAIT_ROBOT_DONE"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"

    def __init__(self):
        # ------------------------------
        # Token selection (human vs robot)
        # ------------------------------
        # Preferred flow:
        #  - If ~human_token is provided ("X" or "O"), use it.
        #  - Else, if ~ask_human_token is true AND stdin is interactive, ask once in the terminal.
        #  - Else, fall back to ~default_human_token (default "X").
        #
        # Robot token is always the opposite of human token.
        ask = bool(rospy.get_param("~ask_human_token", True))
        human = rospy.get_param("~human_token", "").strip().upper()
        default_human = rospy.get_param("~default_human_token", "X").strip().upper()
        if default_human not in ("X", "O"):
            default_human = "X"

        if human not in ("X", "O"):
            human = ""
            if ask and sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                try:
                    ans = input("[ttt_game_manager] Choose HUMAN token (X/O) then press Enter [default=%s]: " % default_human)
                    ans = (ans or "").strip().upper()
                    if ans in ("X", "O"):
                        human = ans
                except Exception:
                    # stdin might not be available when launched under roslaunch
                    human = ""

        if human not in ("X", "O"):
            human = default_human

        self.human_token = human
        self.robot_token = "O" if self.human_token == "X" else "X"

        self.abort = False

        # Latest board from scanner (continuous)
        self.latest_board9 = [""] * 9
        self.latest_board_stamp = rospy.Time(0)

        # Synchronization stamps
        self.last_scan_ready_stamp = rospy.Time(0)

        # State
        self.state = self.WAIT_HUMAN_N

        # If True: we are scanning to VERIFY the robot's own move (do NOT compute a new move)
        self._post_robot_verify_scan = False

        # Pub/Sub
        self.pub_scan_req = rospy.Publisher("/ttt/scan_request", Empty, queue_size=1)
        self.pub_go_home = rospy.Publisher("/ttt/go_home", Empty, queue_size=1)
        self.pub_move = rospy.Publisher("/ttt/robot_move_cmd", RobotMove, queue_size=1)
        self.pub_status = rospy.Publisher("/ttt/game_status", String, queue_size=1, latch=True)
        self.pub_vision_token = rospy.Publisher("/ttt/vision_target_token", String, queue_size=1, latch=True)

        rospy.Subscriber("/ttt/abort", Bool, self.cb_abort, queue_size=1)
        rospy.Subscriber("/ttt/next_turn", Empty, self.cb_next_turn, queue_size=1)
        rospy.Subscriber("/ttt/scan_ready", Empty, self.cb_scan_ready, queue_size=1)
        rospy.Subscriber("/ttt/robot_move_done", Empty, self.cb_robot_done, queue_size=1)
        rospy.Subscriber("/ttt/board_state", BoardGrid, self.cb_board, queue_size=1)

        self._set_status(
            f"READY. Robot={self.robot_token}, Human={self.human_token}. "
            f"Human plays physically, then press N+Enter."
        )

        # Inform vision node which token the robot should PICK.
        # Vision node3 can subscribe to /ttt/vision_target_token (latched).
        self.pub_vision_token.publish(String(data=self.robot_token))

        # Main loop
        self.rate = rospy.Rate(float(rospy.get_param("~rate_hz", 10.0)))
        self.scan_ready_timeout = float(rospy.get_param("~scan_ready_timeout_s", 10.0))
        self.board_fresh_timeout = float(rospy.get_param("~board_fresh_timeout_s", 4.0))
        self.robot_done_timeout = float(rospy.get_param("~robot_done_timeout_s", 90.0))

        self._t_state_enter = rospy.Time.now()

    def _set_status(self, s: str):
        rospy.loginfo(f"[game] {self.state}: {s}")
        self.pub_status.publish(String(data=f"{self.state}: {s}"))

    def _goto(self, new_state: str, msg: str = ""):
        self.state = new_state
        self._t_state_enter = rospy.Time.now()
        if msg:
            self._set_status(msg)

    def cb_abort(self, msg: Bool):
        if msg.data:
            self.abort = True
            self._goto(self.ABORTED, "ABORT received. Game manager halted.")

    def cb_board(self, msg: BoardGrid):
        if self.abort:
            return
        if len(msg.cells) != 9:
            rospy.logwarn_throttle(2.0, "[game] BoardGrid length != 9 (%d)", len(msg.cells))
            return

        b = []
        for c in msg.cells:
            c = (c or "").strip().upper()
            b.append(c if c in ("X", "O") else "")
        self.latest_board9 = b
        self.latest_board_stamp = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

        w = winner(self.latest_board9)
        if w is not None and self.state not in (self.FINISHED, self.ABORTED):
            if w == "D":
                self._goto(self.FINISHED, "Game finished: DRAW.")
            else:
                self._goto(self.FINISHED, f"Game finished: Winner={w}")

            self.pub_go_home.publish(Empty())

    def cb_next_turn(self, _msg: Empty):
        if self.abort or self.state in (self.FINISHED, self.ABORTED):
            return

        if self.state != self.WAIT_HUMAN_N:
            self._set_status("NEXT received but manager is busy; ignoring.")
            return

        # Start scan sequence
        self.pub_scan_req.publish(Empty())
        self._goto(self.WAIT_SCAN_READY, "NEXT received: requested scan. Moving robot to SCAN pose...")

    def cb_scan_ready(self, _msg: Empty):
        if self.abort or self.state in (self.FINISHED, self.ABORTED):
            return

        self.last_scan_ready_stamp = rospy.Time.now()
        if self.state == self.WAIT_SCAN_READY:
            self._goto(self.WAIT_FRESH_BOARD, "Scan ready. Waiting for fresh board_state...")

    def cb_robot_done(self, _msg: Empty):
        if self.abort or self.state in (self.FINISHED, self.ABORTED):
            return
        if self.state == self.WAIT_ROBOT_DONE:
            # NEW: after robot finishes, request a scan to verify board state
            self._post_robot_verify_scan = True
            self.pub_scan_req.publish(Empty())
            self._goto(self.WAIT_SCAN_READY, "Robot finished move: requested verification scan. Moving robot to SCAN pose...")

    def spin(self):
        while not rospy.is_shutdown():
            if self.abort:
                self.rate.sleep()
                continue

            if self.state == self.WAIT_SCAN_READY:
                dt = (rospy.Time.now() - self._t_state_enter).to_sec()
                if dt > self.scan_ready_timeout:
                    self._goto(self.WAIT_HUMAN_N, "Timeout waiting scan_ready. Press N again after checking motion node.")
                self.rate.sleep()
                continue

            if self.state == self.WAIT_FRESH_BOARD:
                # We define "fresh" as a board message that arrived AFTER scan_ready moment.
                dt = (rospy.Time.now() - self._t_state_enter).to_sec()
                is_fresh = (self.latest_board_stamp >= self.last_scan_ready_stamp)

                if is_fresh:
                    if self._post_robot_verify_scan:
                        self._post_robot_verify_scan = False
                        # NEW: after verification scan, go HOME
                        self.pub_go_home.publish(Empty())
                        self._goto(self.WAIT_HUMAN_N, "Board verified after robot move. Going HOME. Human turn: play then press N.")
                    else:
                        self._goto(self.DECIDE_AND_CMD, "Fresh board received. Computing robot move...")
                elif dt > self.board_fresh_timeout:
                    if self._post_robot_verify_scan:
                        self._post_robot_verify_scan = False
                        # NEW: go HOME even if board freshness timed out
                        self.pub_go_home.publish(Empty())
                        self._goto(self.WAIT_HUMAN_N, "Post-robot scan not fresh in time; using latest. Going HOME. Human turn: play then press N.")
                    else:
                        self._goto(self.DECIDE_AND_CMD, "Board not fresh in time. Using latest board anyway (check scanner).")

            if self.state == self.DECIDE_AND_CMD:
                w = winner(self.latest_board9)
                if w is not None:
                    self._goto(self.FINISHED, f"Game finished before robot move: {w}")
                    self.rate.sleep()
                    continue

                _, mv = minimax(self.latest_board9[:], self.robot_token, self.robot_token)
                if mv is None:
                    self._goto(self.WAIT_HUMAN_N, "No valid robot move found (board full?).")
                    self.rate.sleep()
                    continue

                rm = RobotMove()
                rm.header.stamp = rospy.Time.now()
                rm.index = int(mv)
                rm.row = int(mv // 3)
                rm.col = int(mv % 3)
                rm.token = self.robot_token

                self.pub_move.publish(rm)
                self._goto(self.WAIT_ROBOT_DONE, f"Published RobotMove: index={rm.index} (r={rm.row},c={rm.col}). Executing...")
                self.rate.sleep()
                continue

            if self.state == self.WAIT_ROBOT_DONE:
                dt = (rospy.Time.now() - self._t_state_enter).to_sec()
                if dt > self.robot_done_timeout:
                    self._goto(self.WAIT_HUMAN_N, "Timeout waiting robot_move_done. Check motion executor.")
                self.rate.sleep()
                continue

            # FINISHED / ABORTED just idle
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("ttt_game_manager", anonymous=False)
    node = TTTGameManagerSM()
    node.spin()