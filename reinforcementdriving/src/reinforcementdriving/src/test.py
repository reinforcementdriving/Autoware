#!/usr/bin/env python3

import rospy
import numpy as np
#import pyglet
#from   pyglet.gl import *

import time, math
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32


class CarEnv(object):
    is_collision = False
    steer = 0
    speed = 10 #km/h

    lane_pub = PointCloud2()
    border_pub = PointCloud2()
    light_pub = PointCloud2()
    lane_center_pub = PointCloud2()

    def __init__(self, discrete_action=False):

        # init global values
        lane_info_list = []
        border_info_list = []
        light_info_list = []
        lane_center_info_list = []

        ###################################################
        ###
        # loading lane data files
        lane_path   = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane.txt"
        border_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/border.txt"
        light_path  = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/light.txt"
        lane_center_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane_center.txt"


        #lane points
        lane_op = open(lane_path, "r")
        lane_lines = lane_op.readlines()
        lane_lines = lane_lines[1:]

        origin_x = 429704.997 
        origin_y = 4414368.584
        origin_z = 58.315

        for line in lane_lines:
            fields = line.split(",")
            lane_fields = fields[1].split(" ")
            points_num = len(lane_fields)/3
            for i in range(points_num):
                lane_info_list.append([(float)(lane_fields[i*3 + 0]) - origin_x, \
                (float)(lane_fields[i*3 + 1]) - origin_y, \
                (float)(lane_fields[i*3 + 2]) - origin_z])


        #lane_pointcloud = []
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.lane_pub.header = header
        #for i in range(len(gps_point_list)):
        #    lane_pointcloud.append([gps_point_list[i].x - gps_point_list[0].x, gps_point_list[i].y - gps_point_list[0].y , 0])
        self.lane_pub = pc2.create_cloud_xyz32(self.lane_pub.header, lane_info_list)

        lane_op.close()

        ###border points
        border_op = open(border_path, "r")
        border_lines = border_op.readlines()
        border_lines = border_lines[1:]

        for line in border_lines:
            fields = line.split(",")
            border_fields = fields[1].split(" ")
            points_num = len(border_fields)/3
            for i in range(points_num):
                border_info_list.append(  \
                [(float)(border_fields[i*3 + 0]) - origin_x, \
                (float)(border_fields[i*3 + 1]) - origin_y, \
                (float)(border_fields[i*3 + 2]) - origin_z])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.border_pub.header = header
        self.border_pub = pc2.create_cloud_xyz32(self.border_pub.header, border_info_list)
        border_op.close()

        ###light points 
        light_op = open(light_path, "r")
        light_lines = light_op.readlines()
        light_lines = light_lines[1:]

        for line in light_lines:
            light_fields = line.split(",")
            light_info_list.append(  \
                [(float)(light_fields[1]) - origin_x, \
                (float)(light_fields[2]) - origin_y, \
                (float)(light_fields[3]) - origin_z])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.light_pub.header = header
        self.light_pub = pc2.create_cloud_xyz32(self.light_pub.header, light_info_list)
        light_op.close()

        ###lane center points
        lane_center_op = open(lane_center_path, "r")
        lane_center_lines = lane_center_op.readlines()
        lane_center_lines = lane_center_lines[1:]

        for line in lane_center_lines:
            fields = line.split(",")
            lane_center_fields = fields[1].split(" ")
            points_num = len(lane_center_fields)/3
            for i in range(points_num):
                lane_center_info_list.append(  \
                [(float)(lane_center_fields[i*3 + 0]) - origin_x, \
                (float)(lane_center_fields[i*3 + 1]) - origin_y, \
                (float)(lane_center_fields[i*3 + 2]) - origin_z])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.lane_center_pub.header = header
        self.lane_center_pub = pc2.create_cloud_xyz32(self.lane_center_pub.header, lane_center_info_list)
        lane_center_op.close()
 
        ##################################################

        


if __name__ == '__main__':
    rospy.init_node('ruihu_rtk_navigation',anonymous=True)
    env = CarEnv()
    lane_points_pub = rospy.Publisher("lane_topic", PointCloud2, queue_size = 10)
    border_points_pub = rospy.Publisher("border_topic", PointCloud2, queue_size = 10)
    light_points_pub = rospy.Publisher("light_topic", PointCloud2, queue_size = 10)
    lane_center_points_pub = rospy.Publisher("lane_center_topic", PointCloud2, queue_size = 10)

    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        lane_points_pub.publish(env.lane_pub)
        border_points_pub.publish(env.border_pub)
        light_points_pub.publish(env.light_pub)
        lane_center_points_pub.publish(env.lane_center_pub)
        rate.sleep()

    rospy.spin()
