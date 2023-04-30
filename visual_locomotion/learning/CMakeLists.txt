cmake_minimum_required(VERSION 3.0.0)
project(visual_learner)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  catkin_simple REQUIRED
)

catkin_python_setup()
catkin_simple()

include_directories(
${catkin_INCLUDE_DIRS}
)

cs_export()
