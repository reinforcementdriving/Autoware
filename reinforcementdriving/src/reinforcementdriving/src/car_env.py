#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.path as mplPath
#import pyglet

import std_msgs
import time, math
import sensor_msgs.point_cloud2 as pc2
from utils.transformations import quaternion_from_euler

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32
from visualization_msgs.msg import Marker

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

class Point:
    def __init__(self, xx, yy):
        self.x = xx
        self.y = yy

def pointInRectangle( A , B , C , D , m):
    bbPath = mplPath.Path(np.array([
                           [A.x, A.y],
                           [B.x, B.y],
                           [C.x, C.y],
                           [D.x, D.y] ]))
    result = bbPath.contains_point((m.x, m.y))

    if result:
        return True
    else:
        return False


class CarEnv(object):
    dt = 0.1 #s
    lane_pub    = PointCloud2()
    border_pub  = PointCloud2()
    light_pub   = PointCloud2()
    lane_center_pub = PointCloud2()
    goal_pub    = PointCloud2()
    corners_pub = PointCloud2()

    origin_x = 429748.59446449956 
    origin_y = 4414514.143970432
    #origin_x = 429704.997 
    #origin_y = 4414368.584
    origin_z = 58.315
    origin_heading = 192.643
    origin_roll    = 0.0
    origin_pitch   = 0.0

    #we can use gps data read from gps device to transport configure through launch file to init car position 
    def __init__(self):
        self.is_collision = False
        self.speed = 30/3.6 #km/h

        self.action_bound = [-1, 1]
        self.terminator = False
        self.reward = 0
        self.start_point = [self.origin_x, self.origin_y]
        self.start_heading = self.origin_heading

        # publish a car marker based on init position and heading
        init_x = 0
        init_y = 0
        width  = 2.3
        length = 5.0
        car_heading = self.origin_heading#/180*np.pi #degree
        self.car_info = np.array([init_x, init_y, car_heading, width, length], dtype = np.float64)
        self.p1 = self.p2 = self.p3 = self.p4 = Point(0, 0)
        self.action = [-1, 0, 1]

        #pub car position
        self.marker_car = Marker()
        self.marker_car.header.frame_id = "/map"
        self.marker_car.header.stamp = rospy.Time.now()
        self.marker_car.ns = "basic_shapes"
        self.marker_car.id = 0
        self.marker_car.type = Marker.CUBE
        self.marker_car.scale.x = 4.506
        self.marker_car.scale.y = 1.841
        self.marker_car.scale.z = 1.740
        self.marker_car.action = Marker.MODIFY
        self.marker_car.pose.position.x = self.car_info[0] #- self.origin_x
        self.marker_car.pose.position.y = self.car_info[1] #- self.origin_y
        self.marker_car.pose.position.z = 0.0
        #q_x, q_y, q_z, q_w = rpy2q( 0.0, self.car_info[2]*180/np.pi, 0.0)
        quaternion = quaternion_from_euler(0.0, 0.0, self.car_info[2]*180/np.pi, axes = "sxyz")
        self.marker_car.pose.orientation.x = quaternion[0]
        self.marker_car.pose.orientation.y = quaternion[1]
        self.marker_car.pose.orientation.z = quaternion[2]
        self.marker_car.pose.orientation.w = quaternion[3]
        self.marker_car.color.r = 0.0
        self.marker_car.color.g = 1.0
        self.marker_car.color.b = 0.0
        self.marker_car.color.a = 1
        self.marker_car.lifetime = rospy.Duration(0.1)


        # import and publish vector map by using pointclouds
        # init global values
        self.lane_info_list   = []
        self.border_info_list = []
        self.light_info_list  = []
        self.lane_center_info_list = []
        self.goal_info_list = []

        ###################################################
        # loading lane data files
        lane_path   = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane.txt"
        border_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/border.txt"
        light_path  = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/light.txt"
        lane_center_path = "/home1/liumeng/Autoware/reinforcementdriving/data_yby_vector/lane_center.txt"
        goal_path   = "/home1/liumeng/Autoware/reinforcementdriving/data/goal_point.txt"

        ### lane points #######################################################################################
        lane_op = open(lane_path, "r")
        lane_lines = lane_op.readlines()
        lane_lines = lane_lines[1:]

        for line in lane_lines:
            fields = line.split(",")
            lane_fields = fields[1].split(" ")
            points_num = len(lane_fields)/3
            for i in range(int(points_num)):
                self.lane_info_list.append([(float)(lane_fields[i*3 + 0]) - self.origin_x, \
                (float)(lane_fields[i*3 + 1]) - self.origin_y, \
                (float)(lane_fields[i*3 + 2]) - self.origin_z])


        #lane_pointcloud = []
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.lane_pub.header = header
        #for i in range(len(gps_point_list)):
        #    lane_pointcloud.append([gps_point_list[i].x - gps_point_list[0].x, gps_point_list[i].y - gps_point_list[0].y , 0])
        self.lane_pub = pc2.create_cloud_xyz32(self.lane_pub.header, self.lane_info_list)

        lane_op.close()

        ### border points ######################################################################################
        border_op = open(border_path, "r")
        border_lines = border_op.readlines()
        border_lines = border_lines[1:]

        for line in border_lines:
            fields = line.split(",")
            border_fields = fields[1].split(" ")
            points_num = len(border_fields)/3
            for i in range(int(points_num)):
                self.border_info_list.append(  \
                [(float)(border_fields[i*3 + 0]) - self.origin_x, \
                (float)(border_fields[i*3 + 1]) - self.origin_y, \
                (float)(border_fields[i*3 + 2]) - self.origin_z])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.border_pub.header = header
        self.border_pub = pc2.create_cloud_xyz32(self.border_pub.header, self.border_info_list)
        border_op.close()

        ### light points  #######################################################################################
        light_op = open(light_path, "r")
        light_lines = light_op.readlines()
        light_lines = light_lines[1:]

        for line in light_lines:
            light_fields = line.split(",")
            self.light_info_list.append(  \
                [(float)(light_fields[1]) - self.origin_x, \
                (float)(light_fields[2]) - self.origin_y, \
                (float)(light_fields[3]) - self.origin_z])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.light_pub.header = header
        self.light_pub = pc2.create_cloud_xyz32(self.light_pub.header, self.light_info_list)
        light_op.close()

        ### lane center points #################################################################################
        lane_center_op = open(lane_center_path, "r")
        lane_center_lines = lane_center_op.readlines()
        lane_center_lines = lane_center_lines[1:]

        for line in lane_center_lines:
            fields = line.split(",")
            lane_center_fields = fields[1].split(" ")
            points_num = len(lane_center_fields)/3
            for i in range(int(points_num)):
                self.lane_center_info_list.append(  \
                [(float)(lane_center_fields[i*3 + 0]) - self.origin_x, \
                (float)(lane_center_fields[i*3 + 1]) - self.origin_y, \
                (float)(lane_center_fields[i*3 + 2]) - self.origin_z])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.lane_center_pub.header = header
        self.lane_center_pub = pc2.create_cloud_xyz32(self.lane_center_pub.header, self.lane_center_info_list)
        lane_center_op.close()
 
        ### goal points #######################################################################################
        goal_op = open(goal_path, "r")
        goal_lines = goal_op.readlines()
        goal_lines = goal_lines[1:]

        for line in goal_lines:
            fields = line.split(",")
            goal_fields = fields[1].split(" ")
            lat, lon, H = float(goal_fields[1]), float(goal_fields[2]), float(goal_fields[3])
            x, y = BLH2XYZ(lat, lon, H)
            self.goal_info_list.append( [x - self.origin_x, y - self.origin_y, 0])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.goal_pub.header = header
        self.goal_pub = pc2.create_cloud_xyz32(self.goal_pub.header, self.goal_info_list)
        goal_op.close()
 
        ##################################################

        #TD: init state_info
        self.state = self._get_state()


    def step(self, action):
        # using action/gps info to count car position

        action = np.clip(action, *self.action_bound)[0]
        #self.car_info[2] += action * np.pi/30
        self.car_info[2] += action / np.pi*30
        #self.car_info[2] = self.car_info[2]%np.pi
        rad = self.car_info[2]/180*np.pi
        self.car_info[:2] = self.car_info[:2] + \
                self.speed*self.dt*np.array([np.cos(self.car_info[2]), np.sin(self.car_info[2])])

        #state = self.car_info[:3]

        carx, cary, car_heading, carwidth, carlength = self.car_info

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
            #rotate_carxys += [x,y]
            rotate_carxys.append([x,y])


        #self.p1 = Point(rotate_carxys[0][0] - self.origin_x, rotate_carxys[0][1] - self.origin_y)
        #self.p2 = Point(rotate_carxys[1][0] - self.origin_x, rotate_carxys[1][1] - self.origin_y)
        #self.p3 = Point(rotate_carxys[2][0] - self.origin_x, rotate_carxys[2][1] - self.origin_y)
        #self.p4 = Point(rotate_carxys[3][0] - self.origin_x, rotate_carxys[3][1] - self.origin_y)
        self.p1 = Point(rotate_carxys[0][0], rotate_carxys[0][1])
        self.p2 = Point(rotate_carxys[1][0], rotate_carxys[1][1])
        self.p3 = Point(rotate_carxys[2][0], rotate_carxys[2][1])
        self.p4 = Point(rotate_carxys[3][0], rotate_carxys[3][1])


        corners_list = []
        #corners_list.append([rotate_carxys[0][0] - self.origin_x, rotate_carxys[0][1] - self.origin_y, 0])
        #corners_list.append([rotate_carxys[1][0] - self.origin_x, rotate_carxys[1][1] - self.origin_y, 0])
        #corners_list.append([rotate_carxys[2][0] - self.origin_x, rotate_carxys[2][1] - self.origin_y, 0])
        #corners_list.append([rotate_carxys[3][0] - self.origin_x, rotate_carxys[3][1] - self.origin_y, 0])
        #print(corners_list)
        corners_list.append([rotate_carxys[0][0] , rotate_carxys[0][1] , 0])
        corners_list.append([rotate_carxys[1][0] , rotate_carxys[1][1] , 0])
        corners_list.append([rotate_carxys[2][0] , rotate_carxys[2][1] , 0])
        corners_list.append([rotate_carxys[3][0] , rotate_carxys[3][1] , 0])

        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'
        self.corners_pub.header = header
        self.corners_pub = pc2.create_cloud_xyz32(self.corners_pub.header, corners_list)


        self.reward += self.get_goal()

        is_collision = self.collision()

        if is_collision:
            self.reward = -1
            self.terminator = True
        else:
            self.reward += 1


        ### update car marker info
        self.marker_car.pose.position.x = self.car_info[0] #- self.origin_x
        self.marker_car.pose.position.y = self.car_info[1] #- self.origin_y
        self.marker_car.pose.position.z = 0.0
        #q_x, q_y, q_z, q_w = rpy2q( 0.0, self.car_info[2]*180/np.pi, 0.0)
        #roll, pitch, yaw
        quaternion = quaternion_from_euler(self.car_info[2],0.0, 0.0, axes = "sxyz" )
        self.marker_car.pose.orientation.x = quaternion[0]
        self.marker_car.pose.orientation.y = quaternion[1]
        self.marker_car.pose.orientation.z = quaternion[2]
        self.marker_car.pose.orientation.w = quaternion[3]

        print(self.car_info)

        self.state = self._get_state()
        return self.state, self.reward, self.terminator, self.car_info



    def reset(self):
        #self.terminal = False
        #self.car_info[:3] = np.array([*self.start_point, self.])
        #self.car_info[:5] = self.origin_car_info[0:5]
        #print(self.origin_car_info)
        self.__init__()
        rospy.loginfo("Reset Now")
        return self._get_state()


    def _get_state(self):
        s = self.car_info[0:5].flatten()
        return s


    def sample_action(self):
        a = np.random.uniform(*self.action_bound, size = 1)
        return a

    def get_goal(self):
        #judge if get goal and set goal which has been get as invalid

        is_get_goal = False
        goal_info_list_general = self.goal_info_list[0:-1]
        for goal in goal_info_list_general:
            is_get_goal = pointInRectangle(self.p1, self.p2, self.p3, self.p4, Point(goal[0], goal[1]))
            if is_get_goal:
                break

        is_get_final_goal = False
        is_get_final_goal = pointInRectangle(self.p1, self.p2, self.p3, self.p4,  \
                            Point(self.goal_info_list[-1][0], self.goal_info_list[-1][1]))
        if is_get_final_goal:
            rospy.loginfo("Get Final Goal!")
            self.terminator = True
            return 1000

        if is_get_goal:
            rospy.loginfo("Get Goal!")
            return 100
        else:
            return 0


    def collision(self):
        #lane data in car box data(x-y)
        mark = False
        for lane_point in self.lane_info_list:
            mark = pointInRectangle(self.p1, self.p2, self.p3, self.p4, Point(lane_point[0], lane_point[1]))
            if mark:
                break

        if mark:
            rospy.loginfo("Collision!")
            return True
        else:
            rospy.loginfo("Not Collision!")
            return False

        for border_point in self.border_info_list:
            mark = pointInRectangle(self.p1, self.p2, self.p3, self.p4, Point(boder_point[0], boder_point[1]))
            if mark:
                break

        if mark:
            rospy.loginfo("Collision!")
            return True
        else:
            rospy.loginfo("Not Collision!")
            return False


if __name__ == '__main__':
    rospy.init_node('ruihu_rtk_navigation',anonymous=True)
    np.random.seed(1)
    env = CarEnv()

    lane_points_pub        = rospy.Publisher("lane_topic",        PointCloud2, queue_size = 10)
    border_points_pub      = rospy.Publisher("border_topic",      PointCloud2, queue_size = 10)
    light_points_pub       = rospy.Publisher("light_topic",       PointCloud2, queue_size = 10)
    lane_center_points_pub = rospy.Publisher("lane_center_topic", PointCloud2, queue_size = 10)
    goal_points_pub        = rospy.Publisher("goal_topic",        PointCloud2, queue_size = 10)
    corners_points_pub     = rospy.Publisher("car_corners_topic", PointCloud2, queue_size = 10)

    car_location_pub       = rospy.Publisher("car_position",      Marker, queue_size=10)

    #x, y = BLH2XYZ(39.86186573, 116.17898188, 71.37)
    #print(x,y)
    rate = rospy.Rate(10)
    #rospy.spin()
    for ep in range(20):
        s = env.reset()
        done = False
        #for t in range(100000):
        while 1:
            s, r, done, carinfo = env.step(env.sample_action())

            lane_points_pub.publish(env.lane_pub)
            border_points_pub.publish(env.border_pub)
            light_points_pub.publish(env.light_pub)
            lane_center_points_pub.publish(env.lane_center_pub)
            goal_points_pub.publish(env.goal_pub)
            car_location_pub.publish(env.marker_car)
            corners_points_pub.publish(env.corners_pub)
            rate.sleep()
            rospy.loginfo(done)

            if done:
                break
