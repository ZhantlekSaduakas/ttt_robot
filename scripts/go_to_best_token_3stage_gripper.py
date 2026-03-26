#!/usr/bin/env python3
"""
go_to_best_token_3stage_gripper.py

Input:
  /ttt/token_target_pose_base  (geometry_msgs/PoseStamped)
    - position: token location in base frame (panda_link0)
  /ttt/token_best_yaw_deg      (std_msgs/Float32)
    - token yaw in degrees (from vision)

Goal:
  Stage 1: Move GRIPPER to (x, y, z_token + above_token_dz)
           Keep current orientation (reduces weird IK flips)
  Stage 2: Wrist-only orientation: lock joints 1..6, rotate joint7 only
           Uses yaw from /ttt/token_best_yaw_deg (NOT quaternion from target pose)
  Stage 3: Descend to grasp height: max(table_z + grasp_z_above_table, z_token + min_above_token)
           Keep orientation achieved after Stage 2

Key Fix:
  Use the SAME reference frame as the incoming token pose (panda_link0),
  instead of accidentally using planning_frame with link0 coordinates.

Yaw Fix:
  Do NOT extract yaw from token quaternion (it can be rotated by camera->base extrinsics).
  Use /ttt/token_best_yaw_deg directly and apply it as tool-Z rotation (joint7).
"""

import math
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def plan_and_execute(group):
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


def main():
    rospy.init_node("go_to_best_token_3stage_gripper")

    # ---------------- Params ----------------
    topic_in = rospy.get_param("~token_pose_topic", "/ttt/token_target_pose_base")
    yaw_topic = rospy.get_param("~token_yaw_topic", "/ttt/token_best_yaw_deg")

    # Table plane z in panda_link0 (SET THIS!)
    table_z = float(rospy.get_param("~table_z", 0.0))

    # Stage heights (your requested safer values)
    above_token_dz = float(rospy.get_param("~above_token_dz", 0.20))          # Stage 1: token + 20cm
    grasp_z_above_table = float(rospy.get_param("~grasp_z_above_table", 0.10)) # Stage 3: table + 10cm

    # Safety: never descend closer than this above the token itself
    min_above_token = float(rospy.get_param("~min_above_token", 0.05))

    # Optional global z offset applied to token z
    z_offset = float(rospy.get_param("~z_offset", 0.0))

    # Force correct EE link (GRIPPER, NOT CAMERA)
    ee_link_override = rospy.get_param("~ee_link", "panda_hand")

    # ---------------- MoveIt ----------------
    moveit_commander.roscpp_initialize([])
    group = moveit_commander.MoveGroupCommander("panda_arm")

    group.set_max_velocity_scaling_factor(rospy.get_param("~vel_scale", 0.05))
    group.set_max_acceleration_scaling_factor(rospy.get_param("~acc_scale", 0.01))

    # tolerances
    group.set_goal_position_tolerance(rospy.get_param("~pos_tol", 0.003))
    group.set_goal_orientation_tolerance(rospy.get_param("~ori_tol", 0.12))
    group.set_goal_joint_tolerance(rospy.get_param("~joint_tol", 1e-4))

    planning_frame = group.get_planning_frame()
    ee_link_moveit = group.get_end_effector_link()
    ee_link = ee_link_override if ee_link_override else ee_link_moveit

    rospy.loginfo(f"[3stage] planning_frame={planning_frame}")
    rospy.loginfo(f"[3stage] MoveIt reported ee_link={ee_link_moveit} | using ee_link={ee_link}")
    rospy.loginfo(f"[3stage] listening pose: {topic_in}")
    rospy.loginfo(f"[3stage] listening yaw : {yaw_topic}")
    rospy.loginfo(f"[3stage] table_z={table_z:.3f} -> nominal grasp_z={table_z + grasp_z_above_table:.3f}")
    rospy.loginfo(f"[3stage] Stage1 target: z = z_token + {above_token_dz:.3f}")

    # ---------------- Read one token pose ----------------
    rospy.loginfo("Waiting for token pose on %s ...", topic_in)
    token_pose: PoseStamped = rospy.wait_for_message(topic_in, PoseStamped, timeout=10.0)

    # Read yaw once (Float32 degrees)
    rospy.loginfo("Waiting for token yaw on %s ...", yaw_topic)
    yaw_msg: Float32 = rospy.wait_for_message(yaw_topic, Float32, timeout=10.0)
    yaw_target_deg = float(yaw_msg.data)
    yaw_target_rad = math.radians(yaw_target_deg)

    # IMPORTANT FIX: Use token pose frame as target frame (should be panda_link0)
    target_frame = token_pose.header.frame_id if token_pose.header.frame_id else "panda_link0"

    # Set pose reference frame (optional)
    try:
        group.set_pose_reference_frame(target_frame)
    except Exception:
        pass

    if target_frame != "panda_link0":
        rospy.logwarn(
            "[3stage] Token pose frame_id is '%s' (expected panda_link0). Double-check TF.",
            target_frame,
        )

    x = token_pose.pose.position.x
    y = token_pose.pose.position.y
    z_token = token_pose.pose.position.z + z_offset

    # Stage 1: above token
    z_stage1 = z_token + above_token_dz

    # Stage 3: safe descent height
    z_stage3_nom = table_z + grasp_z_above_table
    z_stage3 = max(z_stage3_nom, z_token + min_above_token)

    # Ensure Stage1 is comfortably above Stage3
    z_stage1 = max(z_stage1, z_stage3 + 0.05)

    rospy.loginfo(
        "Token inputs received:\n"
        f"  frame_id={token_pose.header.frame_id}\n"
        f"  using target_frame={target_frame}\n"
        f"  token XYZ=({x:.4f}, {y:.4f}, {z_token:.4f})\n"
        f"  yaw_target={yaw_target_deg:.2f} deg\n"
        f"  Stage1 z={z_stage1:.4f} (token+{above_token_dz:.2f}m)\n"
        f"  Stage3 z={z_stage3:.4f} (max(table+{grasp_z_above_table:.2f}m, token+{min_above_token:.2f}m))"
    )

    # ---------------- Stage 1 ----------------
    try:
        input("Press Enter for Stage 1 (go above token, keep CURRENT orientation)...")
    except (EOFError, KeyboardInterrupt):
        rospy.logwarn("User aborted.")
        return

    cur_pose0 = group.get_current_pose(end_effector_link=ee_link).pose
    q_cur = (cur_pose0.orientation.x, cur_pose0.orientation.y, cur_pose0.orientation.z, cur_pose0.orientation.w)

    pose_stage1 = make_pose_stamped(target_frame, x, y, z_stage1, *q_cur)
    ok1 = go_to_pose(group, ee_link, pose_stage1, label="Stage 1: moving gripper above token (current orientation)...")
    if not ok1:
        rospy.logerr("Stage 1 failed.")
        return

    # ---------------- Stage 2 ----------------
    try:
        input("Press Enter for Stage 2 (wrist-only; rotate joint7 by token yaw)...")
    except (EOFError, KeyboardInterrupt):
        rospy.logwarn("User aborted.")
        return

    joints = group.get_current_joint_values()
    if len(joints) < 7:
        rospy.logerr("Unexpected: panda_arm returned < 7 joints.")
        return

    # IMPORTANT:
    # Apply yaw about tool-Z by rotating joint7.
    # If yaw_target_deg = 0 -> no rotation.
    yaw_delta = wrap_to_pi(yaw_target_rad)

    rospy.loginfo(
        f"[3stage] Stage 2: applying wrist yaw_delta={math.degrees(yaw_delta):.2f} deg "
        f"(from {yaw_topic})"
    )

    joints_target = list(joints)
    joints_target[6] = wrap_to_pi(joints[6] + yaw_delta)

    group.set_start_state_to_current_state()
    group.set_joint_value_target(joints_target)

    ok2 = plan_and_execute(group)
    if not ok2:
        rospy.logerr("Stage 2 failed.")
        return

    # ---------------- Stage 3 ----------------
    try:
        input("Press Enter for Stage 3 (descend to grasp height)...")
    except (EOFError, KeyboardInterrupt):
        rospy.logwarn("User aborted.")
        return

    # Keep orientation achieved after Stage 2
    cur_pose2 = group.get_current_pose(end_effector_link=ee_link).pose
    q_after_2 = (cur_pose2.orientation.x, cur_pose2.orientation.y, cur_pose2.orientation.z, cur_pose2.orientation.w)

    pose_stage3 = make_pose_stamped(target_frame, x, y, z_stage3, *q_after_2)
    ok3 = go_to_pose(group, ee_link, pose_stage3, label="Stage 3: descending to grasp height...")
    if not ok3:
        rospy.logerr("Stage 3 failed.")
        return

    # Final report
    final_pose = group.get_current_pose(end_effector_link=ee_link).pose
    rospy.loginfo(
        "DONE. Final EE pose:\n"
        f"  position: x={final_pose.position.x:.4f}, y={final_pose.position.y:.4f}, z={final_pose.position.z:.4f}\n"
        f"  orientation: x={final_pose.orientation.x:.4f}, y={final_pose.orientation.y:.4f}, "
        f"z={final_pose.orientation.z:.4f}, w={final_pose.orientation.w:.4f}"
    )


if __name__ == "__main__":
    main()