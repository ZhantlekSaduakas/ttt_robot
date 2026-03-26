#!/usr/bin/env python3
import rospy
from std_msgs.msg import Empty, Bool

def main():
    rospy.init_node("ttt_keyboard_gate", anonymous=False)

    pub_next = rospy.Publisher("/ttt/next_turn", Empty, queue_size=1)
    pub_abort = rospy.Publisher("/ttt/abort", Bool, queue_size=1, latch=True)

    rospy.loginfo("[kbd] Ready.")
    rospy.loginfo("[kbd] Commands:")
    rospy.loginfo("  N + Enter  -> next turn (robot may move)")
    rospy.loginfo("  A + Enter  -> ABORT game (release + return home)")

    while not rospy.is_shutdown():
        try:
            s = input().strip().upper()
        except (EOFError, KeyboardInterrupt):
            rospy.logwarn("[kbd] Exiting keyboard gate.")
            break

        if s == "N":
            pub_next.publish(Empty())
            rospy.loginfo("[kbd] NEXT")
        elif s == "A":
            pub_abort.publish(Bool(data=True))
            rospy.logwarn("[kbd] ABORT")
        elif s == "":
            continue
        else:
            rospy.loginfo("[kbd] Unknown. Use N or A.")

if __name__ == "__main__":
    main()