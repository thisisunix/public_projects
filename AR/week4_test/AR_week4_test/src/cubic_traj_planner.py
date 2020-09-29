#!/usr/bin/env python

import __main__
import os.path
import random
import rospy

from AR_week4_test.msg import cubic_traj_params, cubic_traj_coeffs
from AR_week4_test.srv import compute_cubic_traj

node_name = os.path.basename(__main__.__file__)
rospy.init_node(node_name, anonymous = True)

service_name = "compute_cubic_traj"
rospy.wait_for_service(service_name)

compute_cubic_traj = rospy.ServiceProxy(service_name, compute_cubic_traj)

pub = rospy.Publisher('AR_week4_traj_coeffs', cubic_traj_coeffs, queue_size = 10)

def handle_incoming(request_msg):
    print("<", type(request_msg), request_msg)

    y = compute_cubic_traj(request_msg)
    print("=", type(y), y)

    reply_msg = cubic_traj_coeffs(y.a0, y.a1, y.a2, y.a3, request_msg.t0, request_msg.tf)
    print(">", type(reply_msg), reply_msg)

    rospy.loginfo(reply_msg)
    pub.publish(reply_msg)

sub = rospy.Subscriber('AR_week4_traj_params', cubic_traj_params, handle_incoming)
rospy.spin()
