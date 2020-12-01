#!/usr/bin/env python3

import __main__
import math
import os.path

from northstar.srv import get_orientation
from turtlesim.msg import Pose

import rospy

class orientation_service:
    """
    A turtlesim-specific service to return the orientation of the turtle with respect to
    True North. An orientation returned of zero indicates turtle is pointing directly to North.
    """
    def __init__(self):
        # Keep the received theta (rotation) of the turtle in counter-clockwise radians where
        # zero rotation indicates pointing due East.
        self.current_theta = None

        # listen to changes in pose from the turtle simulation
        sub = rospy.Subscriber('turtle1/pose', Pose, lambda x: self.handle_incoming(x))

        # provide a service to return the current clockwise rotation from the desired target
        svc = rospy.Service("get_orientation", get_orientation, lambda x: self.get_orientation(x))

    def handle_incoming(self, msg):
        if self.current_theta is None or msg.theta != self.current_theta:
            self.current_theta = msg.theta
            rospy.logdebug("< %s", msg)

    def _calc_orientation(self):
        # We want to return degrees clockwise rotation from North calculated from
        # counter-clockwise radians rotation from East
        # once converted add on 90deg due to turtle pose's theta of zero referring
        # to due East

        if self.current_theta is not None:
            return math.degrees(-self.current_theta) + 90

        return math.nan

    def get_orientation(self, request_msg):
        response_msg = self._calc_orientation()
        rospy.logdebug("> %s", response_msg)

        return response_msg

if __name__ == '__main__':
    node_name = os.path.basename(__main__.__file__[:-3])
    rospy.init_node(node_name)
    rospy.loginfo("Started [%s] node", node_name)

    svc = orientation_service()

    rospy.spin()
