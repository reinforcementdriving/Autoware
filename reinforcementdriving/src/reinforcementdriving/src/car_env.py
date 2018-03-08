#!/usr/bin/env python3

import rospy
import numpy as np
import pyglet
#from   pyglet.gl import *

import time, math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32


class CarEnv(object):
    is_collision = False
    steer = 0
    speed = 10 #km/h

    #we can use gps data read from gps device to transport configure through launch file to init car position 

    def __init__(self, discrete_action=False):
        # publish car init position and heading

        # publish a car marker

        # import and publish vector map by using pointclouds
        car_info.width = 2.3
        car_info.length = 5.0
        self.car_heading = 100 #degree
        self.car_position = np.array([init_x, init_y], dtype = np.float64)
        self.action = [-1, 0, 1]

        # init global values
        lane_info_list = []
        boder_list = []

        ###################################################
        # loading lane data files
        lane_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane.txt"
        border_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/border.txt"
        light_path = ""

        #lane points
        lane_op = open(lane_path, "r")
        lane_lines = lane_op.readlines()
        lane_lines = lane_lines[1:]

        for line in lane_lines:
            fields = line.split(",")
            lane_fields = fields[1].split(" ")
            points_num = len(lane_fields)/3
            for i in points_num:
                lane_info_list.append([lane_fields[i*3 + 0], lane_fields[i*3 + 1], lane_fields[i*3 + 2]])

        lane_pub = PointCloud2()

        #lane_pointcloud = []
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        lane_pub.header = header
        #for i in range(len(gps_point_list)):
        #    lane_pointcloud.append([gps_point_list[i].x - gps_point_list[0].x, gps_point_list[i].y - gps_point_list[0].y , 0])
        lane_pub = pc2.create_cloud_xyz32(mapPoints.header, lane_info_list)

        ###################################################
        #TD: load and pub border points
        border_op = open(border_path, "r")

        ###################################################
        #TD: load and pub light points


        lane_op.close()
        boder_op.close()
        ###################################################
        #TD: init state_info
        self.state_info = 

    def step(self, action):
        # using action/gps info to count car position
        car_info.x = 
        car_info.y = 
        car_info.heading = 
        # publish new position

    def reset(self):
        return self._get_state()

    def render(self):

        def sample_action(self):
            return a

    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s
    def collision(self):
        #lane data in car data(x-y)
        if :
            is_collision = True

if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(30)
    for ep in range(20):
        s = env.reset()
        # for t in range(100):
        while True:
            env.render()
            s, r, done, carinfo = env.step(env.sample_action())
            if done:
                break
