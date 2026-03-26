#!/usr/bin/env python3
import os
import json
import math
import rospy
import actionlib
import moveit_commander

from std_msgs.msg import Float32, Bool, String, Empty
from geometry_msgs.msg import PoseStamped
from ttt_robot.msg import RobotMove

from franka_gripper.msg import GraspAction, GraspGoal
from franka_gripper.msg import MoveAction, MoveGoal

from tf.transformations import (
    quaternion_matrix,
    quaternion_inverse,
    quaternion_multiply,
    euler_from_matrix,
)

def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def limited_wrist_delta(delta_rad: float, limit_deg: float = 60.0) -> float:
    """Clamp wrist rotation (joint7) delta to +/- limit_deg (radians)."""
    lim = math.radians(float(limit_deg))
    if delta_rad > lim:
        return lim
    if delta_rad < -lim:
        return -lim
    return float(delta_rad)


def plan_and_execute(group):
    plan_result = group.plan()
    success = False
    traj = None

    if isinstance(plan_result, (tuple, list)):
        if len(plan_result) >= 2:
            success = bool(plan_result[0])
            traj = plan_result[1]
    else:
        traj = plan_result
        try:
            success = traj is not None and len(traj.joint_trajectory.points) > 0
        except Exception:
            success = traj is not None

    if not success or traj is None:
        rospy.logerr("[motion] Planning failed.")
        return False

    ok = group.execute(traj, wait=True)
    group.stop()
    return bool(ok)

def make_pose_stamped(frame_id, x, y, z, qx, qy, qz, qw):
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp = rospy.Time.now()
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)
    ps.pose.orientation.x = float(qx)
    ps.pose.orientation.y = float(qy)
    ps.pose.orientation.z = float(qz)
    ps.pose.orientation.w = float(qw)
    return ps

def go_to_pose(group, ee_link, pose_stamped, label=""):
    if label:
        rospy.loginfo(label)
    group.clear_pose_targets()
    group.set_start_state_to_current_state()
    group.set_pose_target(pose_stamped, end_effector_link=ee_link)
    ok = plan_and_execute(group)
    group.clear_pose_targets()
    return ok

def load_cells(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    cells = []
    for _, entry in data.items():
        row = int(entry["row"])
        col = int(entry["col"])
        pos = entry["position_m"]
        quat = entry.get("quaternion_xyzw", {"x": 0, "y": 0, "z": 0, "w": 1})
        cells.append({
            "row": row, "col": col,
            "x": float(pos["x"]), "y": float(pos["y"]), "z": float(pos["z"]),
            "qx": float(quat.get("x", 0.0)),
            "qy": float(quat.get("y", 0.0)),
            "qz": float(quat.get("z", 0.0)),
            "qw": float(quat.get("w", 1.0)),
        })
    cells.sort(key=lambda c: (c["row"], c["col"]))
    return cells

def compute_tool_z_yaw_delta(q_current_xyzw, q_desired_xyzw):
    q_rel = quaternion_multiply(quaternion_inverse(q_current_xyzw), q_desired_xyzw)
    R_rel = quaternion_matrix(q_rel)
    roll, pitch, yaw = euler_from_matrix(R_rel, axes="sxyz")
    return wrap_to_pi(yaw), roll, pitch, yaw

class FrankaGripper:
    def __init__(self):
        self.move_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)

        rospy.loginfo("[gripper] Waiting /franka_gripper/move ...")
        if not self.move_client.wait_for_server(rospy.Duration(5.0)):
            raise RuntimeError("No /franka_gripper/move server")

        rospy.loginfo("[gripper] Waiting /franka_gripper/grasp ...")
        if not self.grasp_client.wait_for_server(rospy.Duration(5.0)):
            raise RuntimeError("No /franka_gripper/grasp server")

        rospy.loginfo("[gripper] READY")

    def open(self, width=0.08, speed=0.1, wait=True):
        g = MoveGoal(width=float(width), speed=float(speed))
        self.move_client.send_goal(g)
        if wait:
            self.move_client.wait_for_result(rospy.Duration(5.0))
        return True

    def grasp(self, width, force=6.0, speed=0.05, eps_in=0.005, eps_out=0.005, wait=True):
        g = GraspGoal()
        g.width = float(width)
        g.epsilon.inner = float(eps_in)
        g.epsilon.outer = float(eps_out)
        g.speed = float(speed)
        g.force = float(force)
        self.grasp_client.send_goal(g)
        if wait:
            self.grasp_client.wait_for_result(rospy.Duration(5.0))
        res = self.grasp_client.get_result()
        if res is not None and hasattr(res, "success"):
            return bool(res.success)
        return True

class TTTMotionExecutorSM:
    """
    Motion responsibilities:
      - Scan: on /ttt/scan_request, go to SCAN joints, publish /ttt/scan_ready
      - Robot move: on /ttt/robot_move_cmd, go HOME joints, pick token, place to cell, publish /ttt/robot_move_done
      - Abort: open gripper, go HOME, stop.
    """
    def __init__(self):
        self.abort = False
        self.busy = False

        # Inputs from vision pipeline
        self.token_pose_topic = rospy.get_param("~token_pose_topic", "/ttt/token_target_pose_base")

        # IMPORTANT FIX:
        # Your updated vision node publishes yaw ONLY on /ttt/token_best_yaw_deg
        self.token_yaw_topic  = rospy.get_param("~token_yaw_topic",  "/ttt/token_best_yaw_deg")

        # Cell JSON (base frame)
        self.cell_json_path = rospy.get_param(
            "~cell_json_path",
            "/home/sysgen/ttt_ws/src/ttt_robot/scripts/cell_poses_base.json"
        )

        # Motion params
        self.table_z = float(rospy.get_param("~table_z", 0.0))
        self.above_token_dz = float(rospy.get_param("~above_token_dz", 0.20))
        self.grasp_z_above_table = float(rospy.get_param("~grasp_z_above_table", 0.10))
        self.min_above_token = float(rospy.get_param("~min_above_token", 0.05))
        self.z_offset = float(rospy.get_param("~z_offset", 0.015))
        self.post_grasp_lift = float(rospy.get_param("~post_grasp_lift", 0.05))

        self.place_xy_dx = float(rospy.get_param("~place_xy_dx", 0.0))
        self.place_xy_dy = float(rospy.get_param("~place_xy_dy", 0.0))
        self.place_z_dz  = float(rospy.get_param("~place_z_dz", 0.03))

        self.ee_link = rospy.get_param("~ee_link", "panda_hand")

        # Enforced 'down' orientation for token pickup (applied at ABOVE_TOKEN).
        # Quaternion order: (x,y,z,w). Default set to (1,0,0,0) as requested.
        qd = rospy.get_param("~q_down", [1.000, 0.0, 0.0, 0.0])
        if not isinstance(qd, (list, tuple)) or len(qd) != 4:
            rospy.logwarn("[motion] ~q_down must be [x,y,z,w]. Using default (1,0,0,0).")
            qd = [1.0, 0.0, 0.0, 0.0]
        self.q_down = (float(qd[0]), float(qd[1]), float(qd[2]), float(qd[3]))
        self.wrist_limit_deg = float(rospy.get_param("~wrist_limit_deg", 45.0))

        # Gripper params
        self.gripper_open_width = float(rospy.get_param("~gripper_open_width", 0.08))
        self.gripper_open_speed = float(rospy.get_param("~gripper_open_speed", 0.10))

        self.grasp_width = float(rospy.get_param("~grasp_width", 0.05))
        self.grasp_force = float(rospy.get_param("~grasp_force", 6.0))
        self.grasp_speed = float(rospy.get_param("~grasp_speed", 0.05))
        self.grasp_eps_in = float(rospy.get_param("~grasp_eps_inner", 0.005))
        self.grasp_eps_out = float(rospy.get_param("~grasp_eps_outer", 0.005))

        # Joint targets (from your values)
        self.home_joints = rospy.get_param("~home_joints", [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.scan_joints = rospy.get_param("~scan_joints", [-1.8494, -0.4886, 2.4845, -1.8686, 0.3700, 2.2311, 1.2460])

        # MoveIt
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.group.set_max_velocity_scaling_factor(rospy.get_param("~vel_scale", 0.15))
        self.group.set_max_acceleration_scaling_factor(rospy.get_param("~acc_scale", 0.03))
        self.group.set_planning_time(rospy.get_param("~planning_time", 10.0))
        self.group.set_num_planning_attempts(int(rospy.get_param("~planning_attempts", 10)))
        try:
            self.group.allow_replanning(True)
        except Exception:
            pass
        self.group.set_goal_position_tolerance(rospy.get_param("~pos_tol", 0.003))
        self.group.set_goal_orientation_tolerance(rospy.get_param("~ori_tol", 0.04))
        self.group.set_goal_joint_tolerance(rospy.get_param("~joint_tol", 1e-4))

        # Gripper
        self.gripper = FrankaGripper()

        # Publishers
        self.pub_status = rospy.Publisher("/ttt/motion_status", String, queue_size=1, latch=True)
        self.pub_scan_ready = rospy.Publisher("/ttt/scan_ready", Empty, queue_size=1)
        self.pub_done = rospy.Publisher("/ttt/robot_move_done", Empty, queue_size=1)

        # Subscribers
        rospy.Subscriber("/ttt/abort", Bool, self.cb_abort, queue_size=1)
        rospy.Subscriber("/ttt/scan_request", Empty, self.cb_scan_request, queue_size=1)
        rospy.Subscriber("/ttt/robot_move_cmd", RobotMove, self.cb_robot_move, queue_size=1)
        rospy.Subscriber("/ttt/go_home", Empty, self.cb_go_home, queue_size=1)

        rospy.loginfo("[motion] READY (SM). HOME joints=%s | SCAN joints=%s",
                      ", ".join([f"{x:.3f}" for x in self.home_joints]),
                      ", ".join([f"{x:.3f}" for x in self.scan_joints]))
        rospy.loginfo("[motion] token_pose_topic=%s", self.token_pose_topic)
        rospy.loginfo("[motion] token_yaw_topic =%s", self.token_yaw_topic)

    def _status(self, s: str):
        rospy.loginfo(f"[motion] {s}")
        self.pub_status.publish(String(data=s))

    def cb_abort(self, msg: Bool):
        if not msg.data:
            return
        self.abort = True
        self._status("ABORT received: opening gripper + going HOME.")
        try:
            self.gripper.open(width=self.gripper_open_width, speed=self.gripper_open_speed, wait=False)
        except Exception:
            pass
        try:
            self.go_home()
        except Exception:
            pass
        self.pub_done.publish(Empty())

    def cb_scan_request(self, _msg: Empty):
        if self.abort:
            self._status("Ignoring scan_request (abort latched).")
            return
        if self.busy:
            self._status("Ignoring scan_request (busy).")
            return

        self.busy = True
        try:
            self._status("SCAN_REQUEST: moving to SCAN joints...")
            ok = self.go_scan()
            if ok:
                self._status("SCAN_READY published.")
                self.pub_scan_ready.publish(Empty())
            else:
                self._status("SCAN failed (planning/execution).")
        finally:
            self.busy = False
    
    def cb_go_home(self, _msg: Empty):
        if self.abort:
            self._status("Ignoring go_home (abort latched).")
            return
        if self.busy:
            self._status("Ignoring go_home (busy).")
            return

        self.busy = True
        try:
            self._status("GO_HOME: moving to HOME joints (game finished).")
            ok = self.go_home()
            if ok:
                self._status("GO_HOME: reached HOME.")
            else:
                self._status("GO_HOME: failed (planning/execution).")
        finally:
            self.busy = False
    

    def cb_robot_move(self, msg: RobotMove):
        if self.abort:
            self._status("Ignoring robot_move_cmd (abort latched).")
            return
        if self.busy:
            self._status("Ignoring robot_move_cmd (busy).")
            return

        self.busy = True
        try:
            self.execute_pick_place(int(msg.index))
        except Exception as e:
            rospy.logerr(f"[motion] execute_pick_place failed: {e}")
        finally:
            self.busy = False
            self.pub_done.publish(Empty())
            self._status("robot_move_done published.")

    def go_home(self):
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(self.home_joints)
        return plan_and_execute(self.group)

    def go_scan(self):
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(self.scan_joints)
        return plan_and_execute(self.group)

    def execute_pick_place(self, cell_idx: int):
        self._status(f"ROBOT_MOVE: target cell index={cell_idx}")

        self._status("Going HOME (for token pickup/idle)...")
        if not self.go_home():
            self._status("Failed to reach HOME.")
            return
        if self.abort:
            return

        self.gripper.open(width=self.gripper_open_width, speed=self.gripper_open_speed, wait=True)
        if self.abort:
            return

        self._status("Waiting token pose + yaw...")
        token_pose: PoseStamped = rospy.wait_for_message(self.token_pose_topic, PoseStamped, timeout=10.0)
        yaw_msg: Float32 = rospy.wait_for_message(self.token_yaw_topic, Float32, timeout=10.0)

        yaw_target_rad = math.radians(float(yaw_msg.data))

        target_frame = token_pose.header.frame_id if token_pose.header.frame_id else "panda_link0"
        try:
            self.group.set_pose_reference_frame(target_frame)
        except Exception:
            pass

        x = token_pose.pose.position.x
        y = token_pose.pose.position.y
        z_token = token_pose.pose.position.z + self.z_offset

        z_stage1 = max(z_token + self.above_token_dz, z_token + 0.15)
        z_stage3_nom = self.table_z + self.grasp_z_above_table
        z_stage3 = max(z_stage3_nom, z_token + self.min_above_token)
        z_stage1 = max(z_stage1, z_stage3 + 0.05)

        # Stage 1: go ABOVE token with enforced 'down' orientation (start pose for pickup)
        qx, qy, qz, qw = self.q_down
        pose_stage1 = make_pose_stamped(target_frame, x, y, z_stage1, qx, qy, qz, qw)
        self._status("Stage 1: above token (ENFORCE DOWN ORI)")
        if not go_to_pose(self.group, self.ee_link, pose_stage1):
            self._status("Stage 1 failed.")
            return
        if self.abort:
            return

        # Stage 2: descend with the SAME enforced 'down' orientation
        self._status("Stage 2: descend (KEEP DOWN ORI)")
        pose_stage2 = make_pose_stamped(target_frame, x, y, z_stage3, qx, qy, qz, qw)
        if not go_to_pose(self.group, self.ee_link, pose_stage2):
            self._status("Stage 2 failed.")
            return
        if self.abort:
            return

        # Stage 3: wrist-only yaw align (joint7), but LIMIT rotation (pickup only)
        self._status("Stage 3: wrist yaw align (joint7, LIMITED)")
        joints = self.group.get_current_joint_values()
        if len(joints) < 7:
            raise RuntimeError("MoveIt returned <7 joints")

        yaw_delta = wrap_to_pi(yaw_target_rad)  # yaw_target_rad = deg->rad from /ttt/token_best_yaw_deg
        yaw_delta_limited = limited_wrist_delta(yaw_delta, limit_deg=self.wrist_limit_deg)
        rospy.loginfo("[motion] token_yaw=%.2f deg => delta=%.2f deg => limited=%.2f deg (limit=%.1f)",
                      math.degrees(yaw_target_rad), math.degrees(yaw_delta), math.degrees(yaw_delta_limited),
                      self.wrist_limit_deg)

        joints_target = list(joints)
        joints_target[6] = wrap_to_pi(joints[6] + yaw_delta_limited)
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(joints_target)
        if not plan_and_execute(self.group):
            self._status("Stage 3 failed.")
            return
        if self.abort:
            return

        self._status("Stage 4: grasp + lift")

        self.gripper.grasp(width=self.grasp_width, force=self.grasp_force, speed=self.grasp_speed,
                           eps_in=self.grasp_eps_in, eps_out=self.grasp_eps_out, wait=True)
        if self.abort:
            return

        cur_pose_after_grasp = self.group.get_current_pose(end_effector_link=self.ee_link).pose
        q_hold = (cur_pose_after_grasp.orientation.x, cur_pose_after_grasp.orientation.y,
                  cur_pose_after_grasp.orientation.z, cur_pose_after_grasp.orientation.w)

        z_lift = cur_pose_after_grasp.position.z + self.post_grasp_lift
        pose_lift = make_pose_stamped(
            target_frame,
            cur_pose_after_grasp.position.x,
            cur_pose_after_grasp.position.y,
            z_lift,
            *q_hold
        )
        if not go_to_pose(self.group, self.ee_link, pose_lift):
            self._status("Lift failed.")
            return
        if self.abort:
            return

        self._status("Transit: going HOME with token...")
        if not self.go_home():
            self._status("Failed to return HOME.")
            return
        if self.abort:
            return

        if not os.path.exists(self.cell_json_path):
            raise RuntimeError(f"Cell JSON not found: {self.cell_json_path}")
        cells = load_cells(self.cell_json_path)
        if not (0 <= cell_idx < len(cells)):
            raise RuntimeError(f"Cell index out of range: {cell_idx}")

        chosen = cells[cell_idx]
        cell_x = chosen["x"] + self.place_xy_dx
        cell_y = chosen["y"] + self.place_xy_dy
        cell_z = chosen["z"] + self.place_z_dz
        cell_q = (chosen["qx"], chosen["qy"], chosen["qz"], chosen["qw"])

        cur_pose_home = self.group.get_current_pose(end_effector_link=self.ee_link).pose
        q_keep = (cur_pose_home.orientation.x, cur_pose_home.orientation.y,
                  cur_pose_home.orientation.z, cur_pose_home.orientation.w)

        self._status("Stage 7: above target cell")
        pose_stage7 = make_pose_stamped(target_frame, cell_x, cell_y, cell_z, *q_keep)
        if not go_to_pose(self.group, self.ee_link, pose_stage7):
            self._status("Stage 7 failed.")
            return
        if self.abort:
            return

        self._status("Stage 8: wrist-only yaw to cell orientation")
        cur_pose_cell = self.group.get_current_pose(end_effector_link=self.ee_link).pose
        q_now = (cur_pose_cell.orientation.x, cur_pose_cell.orientation.y,
                 cur_pose_cell.orientation.z, cur_pose_cell.orientation.w)

        yaw_delta_cell, _, _, _ = compute_tool_z_yaw_delta(q_now, cell_q)
        joints = self.group.get_current_joint_values()
        joints_target = list(joints)
        joints_target[6] = wrap_to_pi(joints[6] + yaw_delta_cell)
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(joints_target)
        if not plan_and_execute(self.group):
            self._status("Stage 8 failed.")
            return
        if self.abort:
            return

        self._status("Stage 9: release")
        self.gripper.open(width=self.gripper_open_width, speed=self.gripper_open_speed, wait=True)

        # self._status("Returning HOME idle...")
        # self.go_home()
        # self._status("Robot move complete.")
        self._status("Robot move complete (waiting for verification scan).")

if __name__ == "__main__":
    rospy.init_node("ttt_motion_executor", anonymous=False)
    node = TTTMotionExecutorSM()
    rospy.spin()