#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from   sensor_msgs.msg import PointCloud2, NavSatFix
import std_msgs.msg
import math
import os


#all_points = []
count = 0

def publish_points():
    global all_points, rate, count
    
    pcloud = PointCloud2()
    pcloud.header = std_msgs.msg.Header()
    pcloud.header.stamp = rospy.Time.now()
    pcloud.header.frame_id = "/map"

    pcloud = pc2.create_cloud_xyz32(pcloud.header, all_points)
    map_puber.publish(pcloud)
    count +=1
    rospy.loginfo("count:%d, length:%d", count, len(pcloud.data))
    rate.sleep()

	

#save and publish all points

#convert all points into pcd  and save it [optinal]


if __name__ == "__main__":
    rospy.init_node('map_maker', anonymous=True)
    rate = rospy.Rate(1)
    
    map_puber = rospy.Publisher('/map_point_topic', PointCloud2, queue_size=1)
 
    path = "/home1/liumeng/Autoware/data-yby/YBYLaserData/PointCloud-txt-2/"
    file_list = os.listdir(path)

    all_points = []
    origin_x = 429816.185
    origin_y = 4414758.822
    origin_z = 67.386

    for data_file in file_list:
        fopen =  open(path + data_file, "r")
        lines = fopen.readlines()
        lines = lines[1:]
        #all_points = []
        for eachline in lines:
            fields = eachline.split(",")
            all_points.append([(float)(fields[0]) - origin_x, (float)(fields[1]) - origin_y, (float)(fields[2]) - origin_z])
        fopen.close()
        publish_points()

    while not rospy.is_shutdown():
        publish_points()
	rate.sleep()


