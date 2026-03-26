#!/usr/bin/env python3
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped


def plan_and_execute(group):
    """
    Plan first, execute only if planning succeeded.
    Handles common MoveIt Python return formats across versions.
    """
    rospy.loginfo("Planning...")
    plan_result = group.plan()

    # MoveIt returns either:
    # - RobotTrajectory
    # - tuple (success, trajectory, planning_time, error_code)
    success = False
    traj = None

    if isinstance(plan_result, tuple) or isinstance(plan_result, list):
        # Noetic often returns a tuple
        if len(plan_result) >= 2:
            success = bool(plan_result[0])
            traj = plan_result[1]
        else:
            success = False
    else:
        # Some versions return just a trajectory object
        traj = plan_result
        # Heuristic: trajectory exists and has points
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


def main():
    rospy.init_node("go_to_scan_pose_cartesian")

    moveit_commander.roscpp_initialize([])
    group = moveit_commander.MoveGroupCommander("panda_arm")

    # Safety: start slow
    group.set_max_velocity_scaling_factor(rospy.get_param("~vel_scale", 0.05))
    group.set_max_acceleration_scaling_factor(rospy.get_param("~acc_scale", 0.01))

    planning_frame = group.get_planning_frame()
    ee_link = group.get_end_effector_link()
    rospy.loginfo(f"Planning frame: {planning_frame}")
    rospy.loginfo(f"End-effector link: {ee_link}")

    # ---- Target scan pose (EDIT THESE) ----
    # Provide x,y,z in meters in the planning frame (usually panda_link0).
    # Provide a valid quaternion orientation.
    x = rospy.get_param("~x", 0.485)
    y = rospy.get_param("~y", 0.405)
    z = rospy.get_param("~z", 0.30)

    qx = rospy.get_param("~qx", 0.924)
    qy = rospy.get_param("~qy", -0.383)
    qz = rospy.get_param("~qz", 0.0)
    qw = rospy.get_param("~qw", 0.0)

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

    rospy.loginfo(
        f"Target pose in {planning_frame}: "
        f"pos=({x:.3f},{y:.3f},{z:.3f}) "
        f"quat=({qx:.3f},{qy:.3f},{qz:.3f},{qw:.3f})"
    )

    # Clear old goals, set new pose target
    group.clear_pose_targets()
    group.set_pose_target(target, end_effector_link=ee_link)

    ok = plan_and_execute(group)

    # Clear target after motion attempt
    group.clear_pose_targets()

    if not ok:
        rospy.logerr("Failed to reach scan pose.")
        return

    rospy.loginfo("Reached scan pose.")

    # Print the actual resulting pose + joint values (useful for later)
    current_pose = group.get_current_pose().pose
    rospy.loginfo(
        "Current EE pose:\n"
        f"  position: x={current_pose.position.x:.4f}, y={current_pose.position.y:.4f}, z={current_pose.position.z:.4f}\n"
        f"  orientation: x={current_pose.orientation.x:.4f}, y={current_pose.orientation.y:.4f}, "
        f"z={current_pose.orientation.z:.4f}, w={current_pose.orientation.w:.4f}"
    )

    joints = group.get_current_joint_values()
    rospy.loginfo("Current joints (radians): " + ", ".join([f"{j:.4f}" for j in joints]))


if __name__ == "__main__":
    main()
