#!/usr/bin/env python

import __main__
import numpy as np
import os.path

from AR_week4_test.msg import cubic_traj_params, cubic_traj_coeffs
from AR_week4_test.srv import compute_cubic_traj

import rospy

def compute_trajectory(request_msg):
    print("<", type(request_msg), request_msg)

    # incoming request message should contain in 'p' an object of type cubic_traj_params
    p0 = request_msg.p.p0
    pf = request_msg.p.pf
    v0 = request_msg.p.v0
    vf = request_msg.p.vf
    t0 = request_msg.p.t0
    tf = request_msg.p.tf

    M = np.asarray([
        [ 1,   t0,   t0**2,      t0**3 ],
        [ 0,    1,  2 * t0,  3 * t0**2 ],
        [ 1,   tf,   tf**2,      tf**3 ],
        [ 0,    1,  2 * tf,  3 * tf**2 ],
    ])

    c = np.asarray([ p0, v0, pf, vf ])

    a = np.matmul(np.linalg.inv(M), c)

    # return coefficients as a tuple
    response_msg = a[0], a[1], a[2], a[3]

    print(">", type(response_msg), response_msg)

    return response_msg

node_name = os.path.basename(__main__.__file__)
rospy.init_node(node_name)

s = rospy.Service("compute_cubic_traj", compute_cubic_traj, compute_trajectory)
rospy.spin()
