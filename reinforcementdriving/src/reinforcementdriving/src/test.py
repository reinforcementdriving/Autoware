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

def BLH2XYZ(lat, lon, H):
    '''B: lat  L:lon  H: height'''

    bb =  lat*3.1415926535897932/180 #lattitude
    ll =  lon #lon

    #double a,b,n,n2,a5 
    #double e1,e12,e2,N,t,t2,l,l2,nn 
    #double cosb,sinb,cosb2,x,y 
    #double l0 

    cosb = math.cos(bb) 
    cosb2 = cosb*cosb 
    sinb = math.sin(bb) 

    #84
    a=6378137.0 #/CGCS a:6378137 f:1/298.257222101 wgs84 a:6378137 f:1/298.257223563 80 a:6378140+-5 b:6356755.2882 a:1/298.257   54 a:6378245 6356863.0188 a:1/298.3
    b=6356752.31424517949756 

    n=(a-b)/(a+b) 
    n2 =n*n 
    e1 = math.sqrt(1-b*b/a/a) 
    e12= e1*e1 
    e2 = math.sqrt(a*a/b/b-1) 

    a5 = a*((1-e12/4.0-3*e12*e12/64.0-5*e12*e12*e12/256.0)*bb       \
        -(3*e12/8.0+3*e12*e12/32.0+45*e12*e12*e12/1024.0)*math.sin(2*bb) \
        +(15*e12*e12/256.0+45*e12*e12*e12/1024.0)*math.sin(4*bb)         \
        -35*e12*e12*e12/3072.0*math.sin(6*bb))
    N = a/math.sqrt(1-e1*e1*sinb*sinb) 
    nn = e2*e2*cosb2 
    t = sinb/cosb 
    t2 = t*t 

    l0 = 117 #((int)(ll+1.5))/3*3 
    #l0 = dY #liangXiang
    #l0 = 109 #sanYa

    l = ll-l0 
    l = l*3.1415926535897932/180 
    l2 = l*l 
    x = a5+0.5*t*N*cosb2*l2*(1.0+cosb2*l2/12.0*(5.0-t2+nn*(9.0+4.0*nn)+
                           cosb2*l2/30.0*(61.0-t2*(58.0-t2)+nn*(270.0-330.0*t2)+
                           1.0/56.0*cosb2*l2*(1385.0+t2*(-3111.0+t2*(543.0-t2)))))) 
    y = N*cosb*l*(1.0+1/6.0*cosb2*l2*(1.0-t2+nn+1/20.0*cosb2*l2*(5.0+t2*(-18.0+t2-58.0*nn)+14.0*nn+1/42.0*cosb2*l2*(61.0+t2*(-479.0+t2*(179.0-t2)))))) 
    y = y+500000.0 

    xxx = (float)(((x-4000000)*1000.0+0.5))/1000.0+4000000 
    yyy = (float)(y*1000+0.5)/1000.0 

    return yyy, xxx #x,y


class CarEnv(object):
    is_collision = False
    steer = 0
    speed = 10 #km/h

    lane_pub = PointCloud2()
    border_pub = PointCloud2()
    light_pub = PointCloud2()
    lane_center_pub = PointCloud2()
    goal_pub = PointCloud2()

    def __init__(self, discrete_action=False):

        # init global values
        lane_info_list = []
        border_info_list = []
        light_info_list = []
        lane_center_info_list = []
        goal_info_list = []

        ###################################################
        # loading lane data files
        lane_path   = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane.txt"
        border_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/border.txt"
        light_path  = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/light.txt"
        lane_center_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane_center.txt"
        goal_path = "/home1/liumeng/Autoware/reinforcementdriving/data/goal_point.txt"


        ### lane points
        lane_op = open(lane_path, "r")
        lane_lines = lane_op.readlines()
        lane_lines = lane_lines[1:]

        origin_x = 429748.59446449956 
        origin_y = 4414514.143970432

        #origin_x = 429704.997 
        #origin_y = 4414368.584
        origin_z = 58.315

        for line in lane_lines:
            fields = line.split(",")
            lane_fields = fields[1].split(" ")
            points_num = len(lane_fields)/3
            for i in range(int(points_num)):
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

        ### border points
        border_op = open(border_path, "r")
        border_lines = border_op.readlines()
        border_lines = border_lines[1:]

        for line in border_lines:
            fields = line.split(",")
            border_fields = fields[1].split(" ")
            points_num = len(border_fields)/3
            for i in range(int(points_num)):
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

        ### light points 
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

        ### lane center points
        lane_center_op = open(lane_center_path, "r")
        lane_center_lines = lane_center_op.readlines()
        lane_center_lines = lane_center_lines[1:]

        for line in lane_center_lines:
            fields = line.split(",")
            lane_center_fields = fields[1].split(" ")
            points_num = len(lane_center_fields)/3
            for i in range(int(points_num)):
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
 
        ### goal points
        goal_op = open(goal_path, "r")
        goal_lines = goal_op.readlines()
        goal_lines = goal_lines[1:]

        for line in goal_lines:
            fields = line.split(",")
            goal_fields = fields[1].split(" ")
            lat, lon, H = float(goal_fields[1]), float(goal_fields[2]), float(goal_fields[3])
            x, y = BLH2XYZ(lat, lon, H)
            goal_info_list.append( [x - origin_x, y - origin_y, 0])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.goal_pub.header = header
        self.goal_pub = pc2.create_cloud_xyz32(self.goal_pub.header, goal_info_list)
        goal_op.close()
 
        ##################################################



if __name__ == '__main__':
    rospy.init_node('ruihu_rtk_navigation',anonymous=True)
    env = CarEnv()
    lane_points_pub = rospy.Publisher("lane_topic", PointCloud2, queue_size = 10)
    border_points_pub = rospy.Publisher("border_topic", PointCloud2, queue_size = 10)
    light_points_pub = rospy.Publisher("light_topic", PointCloud2, queue_size = 10)
    lane_center_points_pub = rospy.Publisher("lane_center_topic", PointCloud2, queue_size = 10)
    goal_points_pub = rospy.Publisher("goal_topic", PointCloud2, queue_size = 10)

    #x, y = BLH2XYZ(39.86186573, 116.17898188, 71.37)
    #print(x,y)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        lane_points_pub.publish(env.lane_pub)
        border_points_pub.publish(env.border_pub)
        light_points_pub.publish(env.light_pub)
        lane_center_points_pub.publish(env.lane_center_pub)
        goal_points_pub.publish(env.goal_pub)
        rate.sleep()

    rospy.spin()
