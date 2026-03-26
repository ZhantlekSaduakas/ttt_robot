#!/usr/bin/env python3
import os
import json
import math
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped

# ROS tf quaternion helpers (available in Noetic)
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    euler_from_matrix,
    quaternion_inverse,
    quaternion_multiply,
)


def wrap_to_pi(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def plan_and_execute(group):
    """
    Plan first, execute only if planning succeeded.
    Handles common MoveIt Python return formats across versions.
    """
    rospy.loginfo("Planning...")
    plan_result = group.plan()

    success = False
    traj = None

    if isinstance(plan_result, (tuple, list)):
        if len(plan_result) >= 2:
            success = bool(plan_result[0])
            traj = plan_result[1]
        else:
            success = False
    else:
        traj = plan_result
        try:
            success = traj is not None and len(traj.joint_trajectory.points) > 0
        except Exception:
            success = traj is not None

    if not success or traj is None:
        rospy.logerr("Planning failed. Not executing.")
        return False

    rospy.loginfo("Executing trajectory...")
    ok = group.execute(traj, wait=True)
    group.stop()
    return bool(ok)


def load_cells(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    cells = []
    for key, entry in data.items():
        row = int(entry["row"])
        col = int(entry["col"])
        pos = entry["position_m"]
        quat = entry["quaternion_xyzw"]

        cells.append(
            {
                "key": key,
                "row": row,
                "col": col,
                "x": float(pos["x"]),
                "y": float(pos["y"]),
                "z": float(pos["z"]),
                "qx": float(quat["x"]),
                "qy": float(quat["y"]),
                "qz": float(quat["z"]),
                "qw": float(quat["w"]),
            }
        )

    cells.sort(key=lambda c: (c["row"], c["col"]))
    return cells


def make_pose_stamped(planning_frame, x, y, z, qx, qy, qz, qw):
    target = PoseStamped()
    target.header.frame_id = planning_frame
    target.header.stamp = rospy.Time.now()

    target.pose.position.x = x
    target.pose.position.y = y
    target.pose.position.z = z

    target.pose.orientation.x = qx
    target.pose.orientation.y = qy
    target.pose.orientation.z = qz
    target.pose.orientation.w = qw
    return target


def go_to_pose(group, ee_link, pose_stamped, label=""):
    if label:
        rospy.loginfo(label)
    group.clear_pose_targets()
    group.set_pose_target(pose_stamped, end_effector_link=ee_link)
    ok = plan_and_execute(group)
    group.clear_pose_targets()
    return ok


def compute_tool_z_yaw_delta(q_current_xyzw, q_desired_xyzw):
    """
    Compute the rotation about tool Z (in the current tool frame) needed to move
    from q_current to q_desired.

    Returns:
      yaw_delta (float): rotation about tool Z in radians
      roll, pitch, yaw (floats): full relative RPY in tool frame (for diagnostics)
    """
    # tf uses (x,y,z,w)
    qc = q_current_xyzw
    qd = q_desired_xyzw

    # Relative rotation in tool frame: q_rel = q_current^-1 * q_desired
    q_rel = quaternion_multiply(quaternion_inverse(qc), qd)

    # Convert to matrix then to RPY (sxyz => roll about X, pitch about Y, yaw about Z)
    R_rel = quaternion_matrix(q_rel)
    roll, pitch, yaw = euler_from_matrix(R_rel, axes="sxyz")

    # yaw is the component joint7 can realize (approx) as wrist rotation
    return wrap_to_pi(yaw), roll, pitch, yaw


def main():
    rospy.init_node("go_to_all_cells_wrist_only_stageB")

    moveit_commander.roscpp_initialize([])
    group = moveit_commander.MoveGroupCommander("panda_arm")

    # Safety scaling
    group.set_max_velocity_scaling_factor(rospy.get_param("~vel_scale", 0.05))
    group.set_max_acceleration_scaling_factor(rospy.get_param("~acc_scale", 0.01))

    planning_frame = group.get_planning_frame()
    ee_link = group.get_end_effector_link()
    rospy.loginfo(f"Planning frame: {planning_frame}")
    rospy.loginfo(f"End-effector link: {ee_link}")

    # JSON path default = same folder as this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    default_json = os.path.join(script_dir, "cell_poses_base.json")
    json_path = rospy.get_param("~json_path", default_json)

    if not os.path.exists(json_path):
        rospy.logerr(f"JSON file not found: {json_path}")
        rospy.logerr("Put cell_poses_base.json next to this script OR pass _json_path:=/abs/path/file.json")
        return

    cells = load_cells(json_path)
    rospy.loginfo(f"Loaded {len(cells)} cells from: {json_path}")

    # Optional offsets / options
    z_offset = float(rospy.get_param("~z_offset", 0.0))
    start_index = int(rospy.get_param("~start_index", 0))
    start_index = max(0, min(start_index, len(cells)))

    # Stage A fixed quaternion (as you currently do)
    fixed_qx = float(rospy.get_param("~fixed_qx", 1.0))
    fixed_qy = float(rospy.get_param("~fixed_qy", 0.0))
    fixed_qz = float(rospy.get_param("~fixed_qz", 0.0))
    fixed_qw = float(rospy.get_param("~fixed_qw", 0.0))

    # Diagnostic threshold: if desired requires roll/pitch beyond this, wrist-only won't match it
    rp_warn_deg = float(rospy.get_param("~rp_warn_deg", 8.0))
    rp_warn = math.radians(rp_warn_deg)

    rospy.loginfo(
        "Two-stage behavior per cell:\n"
        "  Stage A: Move to XYZ with fixed quaternion (pose goal)\n"
        "  Stage B: Wrist-only: lock joints 1..6, adjust ONLY joint7 based on tool-Z yaw delta\n"
        "Note: Stage B can only reproduce the Z-rotation component of the desired orientation."
    )

    for i in range(start_index, len(cells)):
        c = cells[i]
        x, y, z = c["x"], c["y"], c["z"] + z_offset
        file_q = (c["qx"], c["qy"], c["qz"], c["qw"])

        rospy.loginfo(
            f"\n[{i+1}/{len(cells)}] {c['key']} (row={c['row']}, col={c['col']})  "
            f"XYZ=({x:.3f},{y:.3f},{z:.3f})"
        )

        # ---- Stage A ----
        try:
            input("Press Enter for Stage A (move to XYZ with fixed quaternion)...")
        except (EOFError, KeyboardInterrupt):
            rospy.logwarn("User aborted.")
            return

        pose_A = make_pose_stamped(planning_frame, x, y, z, fixed_qx, fixed_qy, fixed_qz, fixed_qw)
        okA = go_to_pose(group, ee_link, pose_A, label="Stage A: moving to XYZ with fixed orientation...")

        if not okA:
            rospy.logerr(f"Stage A failed at {c['key']}.")
            try:
                ans = input("Enter 'c' to continue to next cell, anything else to stop: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if ans != "c":
                return
            continue

        # Read current state after Stage A
        cur_pose = group.get_current_pose().pose
        q_cur = (cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w)

        # Compute wrist-only delta around tool Z
        yaw_delta, rel_roll, rel_pitch, rel_yaw = compute_tool_z_yaw_delta(q_cur, file_q)

        rospy.loginfo(
            "Stage B target analysis (relative rotation in TOOL frame):\n"
            f"  rel RPY [deg]: roll={math.degrees(rel_roll):.2f}, pitch={math.degrees(rel_pitch):.2f}, yaw={math.degrees(rel_yaw):.2f}\n"
            f"  wrist-only will apply yaw_delta [deg]: {math.degrees(yaw_delta):.2f}"
        )

        if abs(rel_roll) > rp_warn or abs(rel_pitch) > rp_warn:
            rospy.logwarn(
                "Desired orientation needs significant roll/pitch change.\n"
                "Wrist-only joint7 rotation cannot fully match it; it will match yaw component only.\n"
                "If you truly need full quaternion, the arm must move (or you must calibrate quats to be z-only deltas)."
            )

        # ---- Stage B: JOINT GOAL (lock joints 1..6, rotate only joint7) ----
        try:
            input("Press Enter for Stage B (wrist-only: rotate joint7)...")
        except (EOFError, KeyboardInterrupt):
            rospy.logwarn("User aborted.")
            return

        joints = group.get_current_joint_values()
        if len(joints) < 7:
            rospy.logerr("Unexpected: panda_arm returned < 7 joints.")
            return

        # Lock joints 1..6 exactly; adjust joint7
        joint7_new = wrap_to_pi(joints[6] + yaw_delta)
        joints_target = list(joints)
        joints_target[6] = joint7_new

        # Joint-only goal => MoveIt should keep joints 1..6 unchanged (no trunk motion)
        group.set_start_state_to_current_state()
        group.set_joint_value_target(joints_target)

        okB = plan_and_execute(group)

        if not okB:
            rospy.logerr(f"Stage B wrist-only failed at {c['key']}.")
            try:
                ans = input("Enter 'c' to continue to next cell, anything else to stop: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return
            if ans != "c":
                return
            continue

        # Report end state
        cur_pose2 = group.get_current_pose().pose
        rospy.loginfo(
            "Done cell. Current EE pose:\n"
            f"  position: x={cur_pose2.position.x:.4f}, y={cur_pose2.position.y:.4f}, z={cur_pose2.position.z:.4f}\n"
            f"  orientation: x={cur_pose2.orientation.x:.4f}, y={cur_pose2.orientation.y:.4f}, "
            f"z={cur_pose2.orientation.z:.4f}, w={cur_pose2.orientation.w:.4f}"
        )

    rospy.loginfo("All cells completed.")


if __name__ == "__main__":
    main()
