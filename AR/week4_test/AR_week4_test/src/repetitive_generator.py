#!/usr/bin/env python

import __main__
import os.path
import random
import rospy
from AR_week4_test.msg import cubic_traj_params

P_MIN, P_MAX = -10, 10
V_MIN, V_MAX = -10, 10
T_MIN, T_MAX = 5, 10

node_name = os.path.basename(__main__.__file__)

pub = rospy.Publisher('AR_week4_traj_params', cubic_traj_params, queue_size = 10)
rospy.init_node(node_name, anonymous = True)

rate = rospy.Rate(1.0 / 20.0) # hz
while not rospy.is_shutdown():
    try:
        p0 = -90
        pf = 90

        v0 = 0
        vf = 0

        t0 = 0
        tf = 1

        msg = cubic_traj_params(p0, pf, v0, vf, t0, tf)

        # print(p0, pf, v0, vf, t0, tf)
        # print(msg)

        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()

    except rospy.ROSInterruptException:
        break
