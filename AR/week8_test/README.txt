Rob Clark 190821441

1/ create a new catkin workspace:
    $ mkdir -p catkin_ws/src

2/ change into newly created directory and get required third-party components
    $ cd catkin_ws/src
    $ git clone -b melodic-devel https://github.com/ros-planning/moveit_tutorials.git
    $ git clone -b melodic-devel https://github.com/ros-planning/panda_moveit_config.git

3/ from the zip file copy the AR_week8_test directory into catkin_ws/src

4/ when in the catkin_ws directory do:
    $ cd catkin_ws
    $ catkin_make
    $ . devel/setup.bash

5/ run (each in new terminal as required, each repeating the "." command from above):

    $ roslaunch panda_moveit_config demo.launch
    # n.b. once RViz is open please ensure the "Displays / Trajectory / State Display Time" is set to REALTIME

    $ rosrun rqt_graph rqt_graph
    # n.b. add all /joint_states/position[i] values for i in 0..6
    #      set x-axis to cover at least 20 seconds and y-axis to between -2 and +2

    $ rosrun AR_week8_test move_panda_square.py
    $ rosrun AR_week8_test square_size_generator.py
