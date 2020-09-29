Rob Clark 190821441

1/ create a new catkin workspace:
    $ mkdir -p catkin_ws/src

2/ from the zip file copy the AR_week4_test directory into catkin_ws/src

3/ when in the catkin_ws directory do:
    $ cd catkin_ws
    $ catkin_make
    $ . devel/setup.bash

4/ run (each in new terminal as required, each repeating the "." command from above):
    $ roscore
    $ rosrun rqt_graph rqt_graph
    $ roslaunch AR_week4_test cubic_traj_gen.launch

