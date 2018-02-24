# !/bin/bash

rosrun pcl_ros pointcloud_to_pcd input:=/map_point_topic
rosbag record /map_point_topic
