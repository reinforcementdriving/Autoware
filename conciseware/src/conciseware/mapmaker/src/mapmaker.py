#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from   sensor_msgs.msg import PointCloud2, NavSatFix
import std_msgs.msg
import math



cur_gps_point = NavSatFix()
map_xyx_list_all = [] #after converted; map
xyz_list_in_everymsg = []
cloud = []
map_pc_list_msg = PointCloud2()
mapPoints = PointCloud2()
mark = False
origin = [0,0]
cur_xy = [0,0]
pcloud = PointCloud2()
pc_convert_in_everymsg = PointCloud2()


def BLH2XYZ(B,L,H):
    Lat,Lon = B,L

    N, E, h = 0,0,0
    L0 = (int((L - 1.5) / 3.0) + 1 ) * 3.0  	#
    
    a = 6378245.0            	                #
    F = 298.257223563        	                #
    iPI = 0.0174532925199433 	                #

    f = 1 / F
    b = a * (1 - f)
    ee = (a * a - b * b) / (a * a)
    e2 = (a * a - b * b) / (b * b)
    n = (a - b) / (a + b) 
    n2 = (n * n)
    n3 = (n2 * n)
    n4 = (n2 * n2)
    n5 = (n4 * n)
    al = (a + b) * (1 + n2 / 4 + n4 / 64) / 2
    bt = -3 * n / 2 + 9 * n3 / 16 - 3 * n5 / 32
    gm = 15 * n2 / 16 - 15 * n4 / 32
    dt = -35 * n3 / 48 + 105 * n5 / 256
    ep = 315 * n4 / 512

    B = B * iPI
    L = L * iPI
    L0 = L0 * iPI
    l = L - L0
    cl = (math.cos(B) * l) 
    cl2 = (cl * cl)
    cl3 = (cl2 * cl)
    cl4 = (cl2 * cl2)
    cl5 = (cl4 * cl)
    cl6 = (cl5 * cl)
    cl7 = (cl6 * cl)
    cl8 = (cl4 * cl4)

    lB = al * (B + bt * math.sin(2 * B) + gm * math.sin(4 * B) + dt * math.sin(6 * B) + ep * math.sin(8 * B))
    t = math.tan(B)
    t2 = (t * t) 
    t4 = (t2 * t2) 
    t6 = (t4 * t2)
    Nn = a / math.sqrt(1 - ee * math.sin(B) * math.sin(B))
    yt = e2 * math.cos(B) * math.cos(B)
    N = lB
    N += t * Nn * cl2 / 2
    N += t * Nn * cl4 * (5 - t2 + 9 * yt + 4 * yt * yt) / 24
    N += t * Nn * cl6 * (61 - 58 * t2 + t4 + 270 * yt - 330 * t2 * yt) / 720
    N += t * Nn * cl8 * (1385 - 3111 * t2 + 543 * t4 - t6) / 40320

    E = Nn * cl
    E += Nn * cl3 * (1 - t2 + yt) / 6
    E += Nn * cl5 * (5 - 18 * t2 + t4 + 14 * yt - 58 * t2 * yt) / 120
    E += Nn * cl7 * (61 - 479 * t2 + 179 * t4 - t6) / 5040

    E += 500000

    N = 0.9999 * N
    E = 0.9999 * (E - 500000.0) + 250000.0

    return E,N #x,y

#sub gps 
def callback_sub_gps(data):
    global cur_gps_point, mark, origin, cur_xy

    cur_gps_point = data 
    cur_x, cur_y = BLH2XYZ(cur_gps_point.latitude, cur_gps_point.longitude, cur_gps_point.altitude)
    cur_xy = [cur_x, cur_y]
    #rospy.loginfo("cur_xy:%f, %f", cur_x, cur_y)
    #rospy.loginfo("cur_xy:%f, %f", cur_xy[0], cur_xy[1])
    if not mark:
        origin = [cur_x, cur_y]
        mark = True
   
    

#sub pointcloud
def callback_sub_pointcloud(ros_cloud):
    global origin, cur_xy, xyz_list_in_everymsg
    #points_list = []
    #pc2.fromPCLPointCloud2(data, cloud_every_msg)
    #convert pc data to xyz
    #for i in cloud_every_msg.length:
    #    cloud.append([,,])
    ####xyz_list_in_everymsg = []
#rospy.loginfo("cur - origin:%f, %f", cur_xy[0] - origin[0], cur_xy[1] - origin[1])

    for data in pc2.read_points(ros_cloud, skip_nans=True):
         xyz_list_in_everymsg.append([cur_xy[1] + data[0] - origin[1], cur_xy[0] + data[1] - origin[0],  data[2]])
   #rospy.loginfo("length:%d", len(xyz_list_in_everymsg))

    pub_function()
    
    
    
def pub_function():
    global  xyz_list_in_everymsg, pcloud, map_puber, rate

    #while not rospy.is_shutdown():
    pc_convert_in_everymsg = PointCloud2()
    pcloud.header = std_msgs.msg.Header()
    pcloud.header.stamp = rospy.Time.now()
    pcloud.header.frame_id = "/map"

    #pc_convert_in_everymsg.data = []
    #pc_convert_in_everymsg = pc2.create_cloud_xyz32(pcloud.header, xyz_list_in_everymsg)
    pcloud = pc2.create_cloud_xyz32(pcloud.header, xyz_list_in_everymsg)
    #pcloud.data = pcloud.data + pc_convert_in_everymsg.data
    map_puber.publish(pcloud)
    #rospy.loginfo("publish map points")
    rospy.loginfo("length:%d", len(pcloud.data))
        
    rate.sleep()



#calculate coordinates of every points

#save and publish all points
   

#convert all points into pcd  and save it [optinal]


if __name__ == "__main__":
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)
    
    rospy.Subscriber("/ruihu_gps_locator_topic", NavSatFix, callback_sub_gps)
    rospy.Subscriber("/ns2/rslidar_points", PointCloud2, callback_sub_pointcloud)
    map_puber = rospy.Publisher('/map_point_topic', PointCloud2, queue_size=10)
    #pub_function()

    rospy.spin()

