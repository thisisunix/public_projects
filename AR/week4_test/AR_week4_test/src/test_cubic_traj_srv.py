#!/usr/bin/env python

import __main__
import os.path

from AR_week4_test.msg import cubic_traj_params
from AR_week4_test.srv import compute_cubic_traj

import rospy

service_name = "compute_cubic_traj"
rospy.wait_for_service(service_name)

compute_cubic_traj = rospy.ServiceProxy(service_name, compute_cubic_traj)

x = cubic_traj_params(5, -5, 0, 0, 0, 1)
y = compute_cubic_traj(x)
print(type(y), y)
