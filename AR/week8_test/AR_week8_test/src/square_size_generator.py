#!/usr/bin/env python
# Advanced Robotic Systems (ECS7004P) AR_week8_test
# Rob Clark (190821441)

import __main__
import rospy
import os.path
import random
import sys
import time

from std_msgs.msg import Float32

pub = rospy.Publisher('AR_week8_square_size', Float32, queue_size = 10)

node_name = os.path.basename(__main__.__file__)
rospy.init_node(node_name)

# sleep a little while so any listening nodes will be more likely to be ready
time.sleep(3)

rate = rospy.Rate(1.0 / 20.0) # hz
while not rospy.is_shutdown():
    try:
        sz = round(random.uniform(0.05, 0.20), 2)
        print("*** publishing square with side-length {}".format(sz))
        pub.publish(sz)
        rate.sleep()
    except rospy.ROSInterruptException:
        break
