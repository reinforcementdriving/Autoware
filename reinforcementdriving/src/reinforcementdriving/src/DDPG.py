#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
1.Environment is a 2D car.
2.Car has 5 sensors to obtain distance information.
3.Car collision => reward = -1, otherwise => reward = 0.
4.You can train this RL by using LOAD = False, after training, this model will be store in the a local folder.Using LOAD = True to reload the trained model for playing.
"""

import rospy
import tensorflow as tf
import numpy as np
import os
import shutil
from   car_env import CarEnv
import time
import matplotlib.pyplot as plt
import math as math
 
#from   ddpg.msg import pose
#from   nav_msgs.msg import Odometry
#from   geometry_msgs.msg import PoseWithCovarianceStamped
from   geometry_msgs.msg import Twist #steer and speed
from   sensor_msgs.msg import NavSatFix #GPS position
from   sensor_msgs.msg import Imu #imu

log_dir = "./log/ddpg/"
np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 2000
MAX_EP_STEPS = 1000
LR_A  = 1e-4  # learning rate for actor
LR_C  = 1e-4  # learning rate for critic
GAMMA = 0.9   # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 1500
BATCH_SIZE = 16
VAR_MIN = 1.0 #2.0  #0.1
RENDER = True
LOAD   = True 
#LOAD   = False
DISCRETE_ACTION = False

env = CarEnv( discrete_action = DISCRETE_ACTION)
STATE_DIM = env.state_dim
ACTION_DIM = env.action_dim
ACTION_BOUND = env.action_bound

def ou(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, shape=[None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')

if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
tf.gfile.MakeDirs(log_dir)

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.a_q = []
        self.a_running_q = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='actor_eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='actor_target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/actor_eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/actor_target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense( s, 100, activation = tf.nn.relu,
                                  kernel_initializer = init_w, bias_initializer = init_b, name = 'actor_layer_1',
                                  trainable = trainable)

            net = tf.layers.dense( net, 20, activation = tf.nn.relu,
                                  kernel_initializer = init_w, bias_initializer = init_b, name = 'actor_layer_2',
                                  trainable = trainable)
            
            #tf.summary.histogram('actor_init_w', init_w)
            #tf.summary.histogram('actor_init_b', init_b)
            tf.summary.histogram('actor_init_net', net)

            with tf.variable_scope('actions'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, name='actions', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_actions')  # Scale output to -action_bound to action_bound
                
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run( self.train_op, feed_dict={S: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run( [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)] )
        self.t_replace_counter += 1

    def choose_action(self, s): 
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict = {S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients( ys = self.a, xs = self.e_params, grad_ys = a_grads)

        with tf.variable_scope('actor_train'):
            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients( zip(self.policy_grads, self.e_params))

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
       
        self.c_cost = []
        self.c_target_q = []
        self.c_q = []
        self.c_q_ = []
        self.c_running_q = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'critic_eval_net', trainable = True)
            self.c_q.append(self.c_q)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'critic_target_net', trainable = False)    # target_q is based on a_ from Actor's target_net
            self.c_q_.append( self.q_)

            self.e_params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/critic_eval_net')
            self.t_params = tf.get_collection( tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/critic_target_net')

        with tf.variable_scope('critic_target_q'):
            self.target_q = R + self.gamma * self.q_
            #self.c_target_q.append(self.target_q)
            #tf.summary.scalar('t_q', self.target_q[0]) 
            #print("target_q"%self.target_q)
            tf.summary.histogram("critic_target_q", self.target_q )

        with tf.variable_scope('error'):# TD_error
            self.loss = tf.reduce_mean( tf.squared_difference( self.target_q, self.q))
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('critic_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize( self.loss)

        with tf.variable_scope('action_gradients'):
            self.a_grads = tf.gradients( self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope( scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer( 0.01 )

            with tf.variable_scope('critic_layer_1'):
                n_l1 = 100
                w1_s = tf.get_variable('weight1_state', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('weight1_action', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1  = tf.get_variable('bias1', [1, n_l1], initializer = init_b, trainable = trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
                net = tf.layers.dense(net, 20, activation = tf.nn.relu,
                                  kernel_initializer = init_w, bias_initializer = init_b, name = 'layer_2',
                                  trainable = trainable)

                tf.summary.histogram('critic_w1s', w1_s)
                tf.summary.histogram('critic_w1a', w1_a)
                tf.summary.histogram('critic_b1', b1)
                tf.summary.histogram('critic_net', net)

            with tf.variable_scope('q'):
                q = tf.layers.dense( net, 1, kernel_initializer = init_w, bias_initializer = init_b, trainable = trainable)   # Q(s,a)
                tf.summary.histogram('critic_q', q)

        return q

    def learn(self, s, a, r, s_, step):
        #self.sess.run(self.train_op, feed_dict = { S: s, self.a: a, R: r, S_: s_})
        result, _ = self.sess.run([merge_op, self.train_op], feed_dict = { S: s, self.a: a, R: r, S_: s_})
        writer.add_summary(result, step)

        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run( [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)] )
        self.t_replace_counter += 1

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[ indices, :]

sess = tf.Session()

# Create actor and critic.
actor  = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory( MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

writer = tf.summary.FileWriter( log_dir, graph=sess.graph)
merge_op = tf.summary.merge_all() 

saver = tf.train.Saver()
path = './discrete' if DISCRETE_ACTION else './continuous'

if LOAD:
    print( "now let's begin the replay")
    saver.restore( sess, tf.train.latest_checkpoint(path))
else:
    sess.run( tf.global_variables_initializer())

def train():
    rospy.init_node( "ddpg_node", anonymous = True)
    var = 2.  # control exploration
    max_step = 0
    max_step_episod = 0
    max_step_episod_mark = False

    for ep in range( MAX_EPISODES):
        if max_step == MAX_EP_STEPS:
            break

        s = env.reset()
        ep_step = 0

        for t in range( MAX_EP_STEPS):
            if max_step == MAX_EP_STEPS:
               break
            if RENDER:
                env.render()

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip( np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration
            s_, r, done, car_info = env.step(a)
            M.store_transition(s, a, r, s_)
            #if (max_step < MAX_EP_STEPS or M.pointer < MEMORY_CAPACITY):
            #    M.store_transition(s, a, r, s_)

            if M.pointer >= MEMORY_CAPACITY:
                var = max([ var * .9995, VAR_MIN])    # decay the action randomness
                b_M = M.sample( BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_, ep)
                actor.learn(b_s)
            
            s = s_
            ep_step += 1

            max_step = max( ep_step, max_step)
            if max_step == MAX_EP_STEPS and max_step_episod_mark == False:
                max_step_episod = ep
                max_step_episod_mark = True
            rospy.loginfo('Ep:%d, |t:%d, |Steps: %i, |Explore: %.2f, |step_reward:%.2f', ep, t, int(ep_step), var, r)

            if done or t == MAX_EP_STEPS - 1:
            #    rospy.loginfo('Ep:%d, | Steps: %i , | Explore: %.2f, | r:%.5f', ep, int(ep_step) , var, r)
                break

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph = False)
    print("\nSave Model %s\n" %save_path)
    print("\nMax steps: %d\n" %max_step)
    print("Max_step_episod:%d" %max_step_episod)

def BLH2XYZ(B,L,H):
    '''B: lat L: lon  H: height'''
    Lat,Lon = B,L

    N, E, h = 0,0,0
    L0 = (int((L - 1.5) / 3.0) + 1 ) * 3.0  	#根据经度求中央子午线经度
    
    a = 6378245.0            	                #地球半径  北京6378245
    F = 298.257223563        	                #地球扁率
    iPI = 0.0174532925199433 	                #2pi除以360，用于角度转换

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

def gps_sub_callback(gps_msg):
    # data.data(gps) --> pygame(x,y)
    # gps_msg.latitude, gps_msg.longitude, gps_msg.height

    x, y = BLH2XYZ(gps_msg.latitude, gps_msg.longitude, gps_msg.height)
    rospy.loginfo("lat,lon:%f, %f" %gps_msg.latitude % gps_msg.longitude )
    rospy.loginfo("x-y after transformed:%f,%f" % x %y)
    return x,y

def car_pos_sub_callback(data):
    # data.data(position) --> pyagme
    angle = data.data
    rospy.loginfo("car position Subscriber")

def eval():
    env.set_fps(30)
    sub_topic = "/gps"
    pos_topic = "/IMU"
    pub_topic = "/cmd_vel"

    rospy.init_node("ddpg_node", anonymous = False)
    rate = rospy.Rate(30)

    #record angle for draw:
    #-------------------------------------
    angle_list = []
    angle_rad_list = []
    x_list = []
    pos_list_x = []
    pos_list_y = []
    pos_list_a = []

    #fig1 = plt.figure(1)
    #fig2 = plt.figure(1)
    #fig3 = plt.figure(1)
    #fig4 = plt.figure(1)

    step = 0

    #ax1 = fig.add_subplot(221)
    #ax1.set_title('Action')

    #ax2 = fig.add_subplot(222)
    #ax2.set_title('action angle rad')
    
    #ax3 = fig.add_subplot(223)
    #ax3.set_title('Position in Simulator')
    
    #ax4 = fig.add_subplot(224)
    #ax4.set_title('Car Heading in Simulator')
    #------------------------------------

    # publisher for vel and steer
    twist_pub = rospy.Publisher(pub_topic, Twist, queue_size = 10)
    gps_sub   = rospy.Subscriber(sub_topic, NavSatFix, gps_sub_callback)
    car_pos_sub = rospy.Subscriber(pos_topic, Imu, car_pos_sub_callback)

    move_cmd = Twist()
    #rospy.spin()
    while not rospy.is_shutdown():
        s = env.reset()
        #sub a GPS and generate a state
        while not rospy.is_shutdown():
            begin = rospy.get_rostime()
            x_list.append(step)
            step += 1 
            env.render()
            ###s = State(position_in_pygame, angle)
            #position_in_pygame = 0
            #angle = 0
            a = actor.choose_action(s)
            # pub an action
            vel   = 1 #const value, ex. 10-->Xm/s
            steer = a #angle to steer
            move_cmd.linear.x = vel
            move_cmd.linear.y = 0
            move_cmd.linear.z = 0
            move_cmd.angular.x = 0
            move_cmd.angular.y = 0
            move_cmd.angular.z = steer
            #rospy.loginfo("Pub a move_cmd:%d", step)
            
            rospy.loginfo("Steps:%d", step)
            rospy.loginfo("Pub a move_cmd angle:[%f] = [%f]", a, a / math.pi * 180)
            twist_pub.publish( move_cmd )
            stop = rospy.get_rostime()
            
            rospy.loginfo("cost time:%f ms", (stop.nsecs-begin.nsecs)/1000000.0)

            angle_rad_list.append(steer)
            angle_list.append(steer/math.pi*180)

            s_, r, done, car_position = env.step(a)
            pos_list_x.append( car_position[0])
            pos_list_y.append( car_position[1])
            #pos_list_a.append( car_position[2])
            
            #if -np.pi<car_position[2]<np.pi:
            #    rr = car_position[2]/np.pi*180
            #elif np.pi <= car_position[2] <= 2*np.pi:
            #    rr = (car_position[2]- np.pi)/np.pi*180 - 180
            #elif 2*np.pi < car_position[2] < 3*np.pi:
            #    rr = (car_position[2]- 2*np.pi)/np.pi*180 
            #elif 3*np.pi <= car_position[2] < 4*np.pi:
            #    rr = (car_position[2]- 3*np.pi)/np.pi*180 - 180
            

            if 0 <= car_position[2] < 2*np.pi:
                rr = car_position[2]/np.pi*180
            elif 2*np.pi <= car_position[2] <= 4*np.pi:
                rr = (car_position[2] - 2*np.pi)/np.pi*180 
            elif 4*np.pi < car_position[2] < 6*np.pi:
                rr = (car_position[2] - 4*np.pi)/np.pi*180 
            elif 6*np.pi <= car_position[2]  < 8*np.pi:
                rr = (car_position[2]- 6*np.pi)/np.pi*180
            elif -2*np.pi <= car_position[2] <0:
                rr = (car_position[2] + 2*np.pi)/np.pi*180 
            elif -4*np.pi <= car_position[2] < -2*np.pi:
                rr = (car_position[2] + 4*np.pi)/np.pi*180 
            elif -6*np.pi <= car_position[2] < -4*np.pi:
                rr = (car_position[2] + 6*np.pi)/np.pi*180 

            pos_list_a.append( rr)

            s = s_

            if done:
                break
 
            if step == 900: #1000
                break
            rate.sleep()
        if step == 900: #1000
            break

    fonts = 14
    #agent steering
    ##---------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.plot( x_list, angle_list, "-",color="deeppink", markersize = 1, lw = 0.5, label='Steering Angles') #angle 
    plt.legend(loc='upper right', fontsize=fonts, scatterpoints = 1, numpoints =1)
    plt.title('Agent Steering Actions', fontsize = fonts)
    plt.xticks(fontsize = fonts)
    plt.yticks(fontsize = fonts)
    plt.xlabel('Steps', fontsize = fonts)
    plt.ylabel(u'Angle',fontsize = fonts)
    plt.show()

    plt.figure(figsize=(5, 4))
    plt.bar(x_list, angle_list, width = 3, label = "Angle", color = "darkviolet")
    plt.xticks(fontsize = fonts)
    plt.yticks(fontsize = fonts)
    plt.title('Agent Steering Action', fontsize = fonts)
    plt.xlabel('Steps', fontsize = fonts)
    plt.ylabel('Steering Angle',fontsize = fonts)
    plt.legend(loc='upper right', fontsize= fonts, scatterpoints = 1, numpoints = 1)
    plt.show()

    #agent position 
    ##---------------------------------------------------------
    plt.figure(figsize= ( 2.5, 4))
    plt.plot( pos_list_x, pos_list_y, "r-", markersize = 1, lw = 1.5, label = "Position")  #position in pyglet 
    plt.legend(loc='upper left', fontsize = fonts, scatterpoints = 1, numpoints = 1)
    plt.xticks(fontsize = fonts)
    plt.yticks(fontsize = fonts)
    plt.title('Agent Position', fontsize = fonts)
    plt.xlabel('X', fontsize = fonts)
    plt.ylabel("Y", fontsize = fonts)
    plt.show()

    #agent heading
    ##---------------------------------------------------------
    plt.figure(figsize= (5, 4))
    plt.plot( x_list, pos_list_a, "b-", markersize = 1, lw = 1, label = "Heading Angle") #
    plt.xticks(fontsize = fonts)
    plt.yticks(fontsize = fonts)
    plt.legend(loc='upper left', fontsize = fonts, scatterpoints = 1, numpoints = 1)
    plt.title('Agent Heading Angles in Simulator', fontsize = fonts)
    plt.xlabel('Steps', fontsize =fonts)
    plt.ylabel("Heading", fontsize =fonts)
    plt.show()
    #plt.show()

if __name__ == '__main__':
    begin_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    if LOAD:
        rospy.loginfo("Eval Model")
        eval()
    else:
        rospy.loginfo("Train Model")
        train()
    #env.close()
    print("Begin Time:%s" %begin_time)
    print("Stop Time:%s" %time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))


