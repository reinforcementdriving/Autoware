cmake_minimum_required(VERSION 2.8.3)
project(mapmaker)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
#  message_generation
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mapmaker
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

