#!/usr/bin/env python3
import __main__
import math
import os.path

import rospy

import northstar.srv
from geometry_msgs.msg import Twist

class drive_unit:
    """
    A node providing a single service to allow for rotation of the robot to be set
    Once the rotation has been set the drive unit will continue to rotate the robot
    until a different rotation is set (or none if a rotation of zero is requested)

    Implemented as a service rather than a subscriber to a topic so that the invoking
    node has at least some feedback of whether the requested rotation will occur

    Explicity implemented in terms of the turtlesim but would benefit from a more
    abstract interface supporting different simulated robot structures
    """
    def __init__(self):
        self.svc = rospy.Service("set_rotation", northstar.srv.set_rotation, lambda x: self.set_rotation(x))
        self.pub = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size = 10)

        self.timer = rospy.Timer(rospy.Duration(1), lambda x: self.handle_timer())
        self.current_rotation_speed = 0

    def set_rotation(self, msg):
        rospy.logdebug("< %s", msg)

        # Incoming message specifies the number of degrees-per-second clockwise
        # desired. The Twist message requires counter-clockwise radians so we
        # negate and convert here
        rotation_speed = -msg.cw_deg_sec * (2 * math.pi) / 360

        if rotation_speed != self.current_rotation_speed:
            self.current_rotation_speed = rotation_speed
            self.send_rotate()
            return True

        return False

    def handle_timer(self):
        if self.current_rotation_speed != 0:
            self.send_rotate()

    def send_rotate(self):
        msg = Twist()
        msg.linear.x = msg.linear.y = msg.linear.z = 0
        msg.angular.x = msg.angular.y = 0
        msg.angular.z = self.current_rotation_speed

        rospy.logdebug("> %s", msg)
        self.pub.publish(msg)
        

if __name__ == '__main__':
    node_name = os.path.basename(__main__.__file__[:-3])
    rospy.init_node(node_name)
    rospy.loginfo("Started [%s] node", node_name)

    drv = drive_unit()

    rospy.spin()
