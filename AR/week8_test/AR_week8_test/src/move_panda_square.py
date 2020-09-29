#!/usr/bin/env python
# Advanced Robotic Systems (ECS7004P) AR_week8_test
# Rob Clark (190821441)

import __main__
import copy
import datetime
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
import os.path
import sys
import time

from std_msgs.msg import Float32
from moveit_commander.conversions import pose_to_list

node_name = os.path.basename(__main__.__file__)
rospy.init_node(node_name)

moveit_commander.roscpp_initialize(sys.argv)

robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group_name = "panda_arm"
move_group = moveit_commander.MoveGroupCommander(group_name)

display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
    moveit_msgs.msg.DisplayTrajectory, queue_size = 20)

def handle_incoming(request_msg):
    print("<", type(request_msg), request_msg)

    side_length = request_msg.data
    print("*** received message for square with sides of length {} at {}".format(side_length, datetime.datetime.now()))

    print("*** moving to initial joint-space position ...")
    joint_goal = [ 0, -pi / 4, 0, -pi / 2, 0, pi / 3, 0 ]
    move_group.go(joint_goal, wait = True)
    move_group.stop()

    # Start planning our path
    print("*** planning trajectory ...")

    p = move_group.get_current_pose().pose
    waypoints = []

    # We are using the initial position of the effector as the centre of our square
    # so our first waypoint is a corner (i.e. the effector will move diagonally)
    p.position.y -= 0.5 * side_length
    p.position.x -= 0.5 * side_length
    waypoints.append(copy.deepcopy(p))

    p.position.x += side_length
    waypoints.append(copy.deepcopy(p))

    p.position.y += side_length
    waypoints.append(copy.deepcopy(p))

    p.position.x -= side_length
    waypoints.append(copy.deepcopy(p))

    p.position.y -= side_length
    waypoints.append(copy.deepcopy(p))

    # leave effector on first/final corner

    plan, fraction = move_group.compute_cartesian_path(
                                   waypoints,   # waypoints to follow
                                   0.01,        # eef_step
                                   0.0)         # jump_threshold

    # trajectory time_from_start is in a rospy.Duration object, to which we add
    # one second. If the rViz "State Display Time" is set to REALTIME then we will
    # sleep for the correct amount of time between steps
    trajectory_duration = plan.joint_trajectory.points[-1].time_from_start + rospy.Duration(1.0)
    print("    estimated trajectory duration (REALTIME) is {:.2f} seconds".format(trajectory_duration.to_sec()))

    rospy.sleep(trajectory_duration)

    print("*** displaying trajectory ...")
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    display_trajectory_publisher.publish(display_trajectory)

    rospy.sleep(trajectory_duration)

    print("*** executing trajectory ...")
    t0 = datetime.datetime.now()
    move_group.execute(plan, wait = True)
    move_group.stop()
    move_group.clear_pose_targets()

    print("*** done at {}".format(datetime.datetime.now()))

sub = rospy.Subscriber('AR_week8_square_size', Float32, handle_incoming)
rospy.spin()
