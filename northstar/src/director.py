#!/usr/bin/env python3

import __main__
import datetime
import os.path
import random
import rospy

import northstar.srv

node_name = os.path.basename(__main__.__file__[:-3])
rospy.init_node(node_name)
rospy.loginfo("Started [%s] node", node_name)

class svc:
    """
    Wrapper class for a ROS service to allow for reconnection
    Currently doesn't cleanly recognise the difference between ^C having
    caused a problem with the proxy call or a failure with the underlying
    TCP connection
    """
    def __init__(self, service_name, service_type):
        self.service_name = service_name
        self.service_type = service_type
        self.proxy = None

        self.connect()

    def connect(self, action = "connect"):
        try:
            # wait for service to become available for maximum of 2 seconds
            rospy.wait_for_service(self.service_name, timeout = rospy.Duration.from_sec(2))
        except rospy.ROSInterruptException:
            rospy.logerr("Interrupted waiting for required service [%s] on %s", self.service_name, action)
            raise
        except rospy.ROSException:
            rospy.logerr("Timeout waiting for required service [%s] on %s", self.service_name, action)
            raise

        self.proxy = rospy.ServiceProxy(self.service_name, self.service_type)

    def __call__(self, *args, **kwargs):
        try:
            res = self.proxy(*args)

        except rospy.service.ServiceException as e:
            # unfortunately this is also the exception raised if ^C has been pressed in the terminal
            # window
            rospy.logerr("Failed to call %s service, attempting to reconnect: %s", self.service_name, e)
            self.proxy = None

            for attempt in range(3):
                try:
                    self.connect(action = "reconnect")
                    if self.proxy is not None:
                        break
                except rospy.ROSInterruptException as e:
                    rospy.logerr("Interrupted reconnecting to service [%s]: %s", self.service_name, e)
                    break
                except:
                    rospy.logerr("Failed to reconnect to %s service (attempt %s)", self.service_name, attempt + 1)

        if self.proxy is None:
            rospy.logfatal("Failed to reconnect to %s service, node will exit", self.service_name)
            rospy.signal_shutdown("Failed to reconnect to {} service, node will exit".format(self.service_name))
        else:
            res = self.proxy(*args)

        return res

try:
    get_orientation = svc("get_orientation", northstar.srv.get_orientation)
    set_rotation = svc("set_rotation", northstar.srv.set_rotation)
except:
    rospy.logfatal("Exception initialising required services, node will exit")
    rospy.signal_shutdown("Exception initialising required services, node will exit")

e = 0.5  # acceptable error in degrees of orientation to target
current_rotation = None
when_settled = None

if not rospy.is_shutdown():
    if set_rotation(0):
        current_rotation = 0
    else:
        rospy.logerr("Failed to set current rotation to zero")

fast_rate = rospy.Rate(3)  # call every 333ms
current_rate = slow_rate = rospy.Rate(0.5)  # call every 2 seconds

prev_deg_cw = None
prev_datetime = None

while not rospy.is_shutdown():
    current_rate.sleep()

    res = get_orientation()

    deg_cw = res.degrees_cw
    rotation = 0

    if prev_deg_cw is not None and current_rotation != 0 and deg_cw == prev_deg_cw:
        # every time this process is started the first requested move is apparently subject to
        # some kind of warmup and so the robot appears to be unmoving. To protect against
        # these kind of timings issues we only warn when the robot should have been moving but
        # doesn't appear to have been for over a second
        duration = (datetime.datetime.now() - prev_datetime).total_seconds()
        if duration > 1:
            rospy.logwarn("No change in orientation with rotation of %s in last %s seconds, robot may be obstructed",
                      current_rotation, duration)

    if abs(deg_cw) > e:
        # use the degrees we need to rotate to control the speed: faster to
        # begin with but as we approach the correct orientation slow down to
        # facilitate a more accurate stopping point
        rotation = max(2, min(30, int(abs(deg_cw))))

        if deg_cw > 0 and deg_cw <= 180:
            rotation = -rotation

    # if we're well aligned with north mark the time this was achieved and
    # switch the sleep duration to longer to conserve resources
    elif when_settled is None:
        when_settled = datetime.datetime.now()
        current_rate = slow_rate

    # For debugging purposes this block simulates the robot being randomly reoriented
    # once it's been accurately pointing north for at least 3 seconds
    elif datetime.datetime.now() - when_settled > datetime.timedelta(seconds = 3):
        r = random.randint(-100, 100)
        rospy.loginfo("New random state, setting rotation to {}".format(r))
        set_rotation(r)
        rospy.Rate(0.4).sleep()
        set_rotation(0)
        when_settled = None

    # instruct the drive unit to rotatet and reduce our sleep duration so we can
    # have finer control over stopping
    if rotation != current_rotation and set_rotation(rotation):
        rospy.loginfo("rotation <- {}, (deg_cw: {})".format(rotation, deg_cw))
        current_rotation = rotation
        current_rate = fast_rate

    if prev_deg_cw is None or deg_cw != prev_deg_cw:
        prev_deg_cw = deg_cw
        prev_datetime = datetime.datetime.now()

rospy.loginfo("Finished [%s] node", node_name)
