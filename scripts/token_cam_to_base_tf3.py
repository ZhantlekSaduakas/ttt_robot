#!/usr/bin/env python3
import math
import threading
import rospy
import tf2_ros
import tf2_geometry_msgs  # noqa
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


class TokenCamToBaseTF:
    def __init__(self):
        self.base_frame = rospy.get_param("~base_frame", "panda_link0")
        self.tf_timeout_s = float(rospy.get_param("~tf_timeout_s", 0.2))
        self.use_latest_tf = bool(rospy.get_param("~use_latest_tf", True))
        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 10.0))

        self.in_pose_topic = rospy.get_param("~in_pose_topic", "/ttt/token_best_pose_cam")
        self.in_yaw_topic  = rospy.get_param("~in_yaw_topic",  "/ttt/token_best_yaw_deg")
        self.out_pose_topic = rospy.get_param("~out_pose_topic", "/ttt/token_target_pose_base")

        self._lock = threading.Lock()
        self.latest_yaw_deg = None
        self.latest_pose_cam = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pub_pose_base = rospy.Publisher(self.out_pose_topic, PoseStamped, queue_size=10)

        rospy.Subscriber(self.in_yaw_topic, Float32, self.cb_yaw, queue_size=10)
        rospy.Subscriber(self.in_pose_topic, PoseStamped, self.cb_pose, queue_size=10)

        rospy.loginfo("[token_cam_to_base_tf3] Listening pose: %s", self.in_pose_topic)
        rospy.loginfo("[token_cam_to_base_tf3] Listening yaw : %s", self.in_yaw_topic)
        rospy.loginfo("[token_cam_to_base_tf3] Publishing base pose: %s", self.out_pose_topic)
        rospy.loginfo("[token_cam_to_base_tf3] publish_rate_hz=%.1f", self.publish_rate_hz)

    def cb_yaw(self, msg: Float32):
        with self._lock:
            self.latest_yaw_deg = float(msg.data)

    def cb_pose(self, pose_cam: PoseStamped):
        if not pose_cam.header.frame_id:
            rospy.logwarn_throttle(2.0, "[token_cam_to_base_tf3] Incoming pose has empty frame_id; skipping")
            return
        with self._lock:
            self.latest_pose_cam = pose_cam

    def _compute_pose_base(self):
        with self._lock:
            pose_cam = self.latest_pose_cam
            yaw_deg = self.latest_yaw_deg

        if pose_cam is None or yaw_deg is None:
            return None

        # IMPORTANT:
        # We only want POSITION in base frame.
        # Do NOT bake yaw into quaternion here (it causes frame-axis mismatch).
        pose_cam2 = PoseStamped()
        pose_cam2.header = pose_cam.header
        pose_cam2.pose.position = pose_cam.pose.position
        pose_cam2.pose.orientation.w = 1.0  # identity

        lookup_time = rospy.Time(0) if self.use_latest_tf else pose_cam2.header.stamp

        try:
            T = self.tf_buffer.lookup_transform(
                self.base_frame,
                pose_cam2.header.frame_id,
                lookup_time,
                rospy.Duration(self.tf_timeout_s),
            )
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam2, T)
            pose_base.header.frame_id = self.base_frame
            pose_base.header.stamp = pose_cam2.header.stamp
            return pose_base

        except Exception as e:
            rospy.logwarn_throttle(
                2.0,
                "[token_cam_to_base_tf3] TF failed (%s -> %s): %s",
                pose_cam2.header.frame_id, self.base_frame, str(e),
            )
            return None

    def spin(self):
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            pose_base = self._compute_pose_base()
            if pose_base is not None:
                self.pub_pose_base.publish(pose_base)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("token_cam_to_base_tf3")
    TokenCamToBaseTF().spin()