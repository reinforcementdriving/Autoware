#!/usr/bin/env python3

import rospy
import numpy as np
import pyglet
#from   pyglet.gl import *

import time, math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32


class Point:
    def __init__(self, xx, yy):
        self.x = xx
        self.y = yy

def vect2d(Point p1, Point p2):
    Point temp
    temp.x = (p2.x - p1.x)
    temp.y = -1 * (p2.y - p1.y)
    return temp

def pointInRectangle(Point A, Point B, Point C, Point D, Point m):

    Point AB = vect2d(A, B)
    C1 = -1 * (AB.y*A.x + AB.x*A.y)
    D1 = (AB.y*m.x + AB.x*m.y) + C1

    Point AD = vect2d(A, D)
    C2 = -1 * (AD.y*A.x + AD.x*A.y)
    D2 = (AD.y*m.x + AD.x*m.y) + C2

    Point BC = vect2d(B, C)
    C3 = -1 * (BC.y*B.x + BC.x*B.y)
    D3 = (BC.y*m.x + BC.x*m.y) + C3

    Point CD = vect2d(C, D)
    C4 = -1 * (CD.y*C.x + CD.x*C.y)
    D4 = (CD.y*m.x + CD.x*m.y) + C4

    if 0 >= D1 && 0 >= D4 && 0 <= D2 && 0 >= D3:
        return True
    else:
        return False


class CarEnv(object):
        #we can use gps data read from gps device to transport configure through launch file to init car position 
        def __init__(self, discrete_action=False):
        self.is_collision = False
        self.speed = 10/3.6 #km/h

        self.action_bound = [-1, 1]
        self.terminator = False
        self.reward = 0

        # publish a car marker based on init position and heading

        init_x = 0
        init_y = 0
        width  = 2.3
        length = 5.0
        car_heading = 192.643/180*np.pi #degree
        self.car_info = np.array([init_x, init_y, car_heading, width, length], dtype = np.float64)
        p1 = p2 = p3 = p4 = Point(0,0)
        self.action = [-1, 0, 1]

        # import and publish vector map by using pointclouds
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

        action = np.clip(action, *self.action_bound)[0]
        self.car_info[2] += action * np.pi/30
        self.car_info[:2] = self.car_info[:2] + \
                self.speed*self.dt*np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])

        state = self.car_info[:3]

        carx, cary, carheading, carwidth, carlength = self.car_info

        car_4_points = \
        [[carx + carlength/2, cary + carwidth/2],
        [carx - carlength/2, cary + carwidth/2],
        [carx - carlength/2, cary - carwidth/2],
        [carx + carlength/2, cary - carwidth/2]]

        rotate_carxys = []
        for x,y in car_4_points:
            temp_x = x - carx
            temp_y = y - cary
            rotated_x = temp_x*np.cos(car_heading) - temp_y*np.sin(car_heading)
            rotated_y = temp_x*np.sin(car_heading) + temp_y*np.cos(car_heading)
            x = rotated_x + carx
            y = rotated_y + cary
            rotate_carxys += [x,y]

        Point p1, p2, p3, p4
        p1.x = rotate_carxys[0][0]
        p1.y = rotate_carxys[0][1]

        p2.x = rotate_carxys[1][0]
        p2.y = rotate_carxys[1][1]

        p3.x = rotate_carxys[2][0]
        p3.y = rotate_carxys[2][1]

        p4.x = rotate_carxys[3][0]
        p4.y = rotate_carxys[3][1]


        #car_4_points = [p1, p2, p3, p4]

        self.reward += self.get_goal()

        is_collision = self.collision()

        if is_collision:
            self.reward = -1
            self.terminator = True
        else:
            self.reward += 1


        return state, self.reward, self.terminator, self.car_info

        # publish new position

    def reset(self):
        return self._get_state()

    def render(self):

    def sample_action(self):
        a = np.random.uniform(*self.action_bound, size = )
        return a

    def _get_state(self):
        s = self.sensor_info[:, 0].flatten()/self.sensor_max
        return s

    def get_goal(self):
        #judge if get goal and set goal which has been getten as invalid

        is_get_goal = False
        goal_info_list_general = goal_info_list[0:-1]
        for goal in goal_info_list_general:
            get_goal = pointInRectangle(car_info.p1, car_info.p2, car_info.p3, car_info.p4, Point(goal.x, goal.y))

        is_get_final_goal = False
        is_get_final_goal = pointInRectangle(car_info.p1, car_info.p2, car_info.p3, car_info.p4,  \
                            Point(goal_info_list[-1].x, goal_info_list[-1].y))
        if is_get_final_goal:
            self.terminator = True
            return 1000

        if is_get_goal:
            return 100
        else:
            return 0

    def collision(self):
        #lane data in car box data(x-y)
        mark = False
        for lane_point in lane_info_list:
            mark = pointInRectangle(car_info.p1, car_info.p2, car_info.p3, car_info.p4, lane_point)
            if mark:
                break

        if mark:
            return True
        else:
            return False


if __name__ == '__main__':
    np.random.seed(1)
    env = CarEnv()
    env.set_fps(30)
    for ep in range(20):
        s = env.reset()
        # for t in range(100):
        while not rospy.is_shutdown():
            env.render()
            s, r, done, carinfo = env.step(env.sample_action())
            if done:
                break
