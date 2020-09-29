#!/usr/bin/env python

import __main__
import datetime
import os.path
import random
import rospy

from AR_week4_test.msg import cubic_traj_coeffs
from std_msgs.msg import Float32

node_name = os.path.basename(__main__.__file__)
rospy.init_node(node_name, anonymous = True)

pub_position = rospy.Publisher('AR_week4_traj_position', Float32, queue_size = 10)
pub_velocity = rospy.Publisher('AR_week4_traj_velocity', Float32, queue_size = 10)
pub_acceleration = rospy.Publisher('AR_week4_traj_acceleration', Float32, queue_size = 10)

rate = rospy.Rate(10)  # hz

def handle_incoming(coef):
    print("<", type(coef), coef)

    # convert times into local values; although spec is that t0 is always zero
    # we calculate time to run by tf - t0
    t0 = datetime.datetime.now()
    tf = t0 + datetime.timedelta(seconds = coef.tf - coef.t0)

    now = t0
    while True:
        t = (now - t0).total_seconds()  # elapsed time since start in seconds

        pos = coef.a0 + coef.a1 * t + coef.a2 * t**2 + coef.a3 * t**3
        vel = coef.a1 + (2 * coef.a2 * t) + (3 * coef.a3 * t**2)
        acc = (2 * coef.a2) + (6 * coef.a3 * t)

        print(str(now), t, pos, vel, acc)
        pub_position.publish(pos)
        pub_velocity.publish(vel)
        pub_acceleration.publish(acc)

        if now >= tf:
            break

        rate.sleep()
        now = datetime.datetime.now()

        # make sure we publish values for the final point even though
        # they might be published a little late...
        if now > tf:
            now = tf

sub = rospy.Subscriber('AR_week4_traj_coeffs', cubic_traj_coeffs, handle_incoming)
rospy.spin()
