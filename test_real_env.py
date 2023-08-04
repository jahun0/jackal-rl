#!/usr/bin/env python3

import rospy
import random
import math
import time
import csv
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from dynamic_reconfigure.client import Client
import tf as ros_tf
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np
from collections import deque
from rosgraph_msgs.msg import Clock

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt
import os
import actionlib
from actionlib_msgs.msg import GoalStatusArray
from std_srvs.srv import Empty

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from rospy.timer import Timer

# Check if GPU is available and set the device accordingly
if tf.config.list_physical_devices('GPU'):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first available GPU
    print("Using GPU for training.")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use the CPU
    print("Using CPU for training.")

from keras.models import Model
from keras.layers import Dense, Input, Add, Lambda
from keras.optimizers import Adam
from keras import backend as K

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class NavigationEnvROS:
    def __init__(self, node=None, goal=None, pose=None, start=None, priority=None):
        rospy.init_node("navigation_env", anonymous=True)
        self.initial_location = None

        if priority is not None:
            self.priority = priority
        else:
            self.priority = 2
        # Define state space
        self.observation_space = (8, )

        # Define action space (0: use small costmap, 1: use big costmap)
        self.action_space = 2

        # Initialize ROS subscribers and publishers
        rospy.Subscriber("/cmd_vel", Twist, self._callback)

        # -------- REAL ROBOT -------- 
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
        rospy.Subscriber("/move_base/current_goal", PoseStamped, self.current_goal_callback)

        self.move_base_local_costmap_client = Client("/move_base/local_costmap")
        self.move_base_global_costmap_client = Client("/move_base/global_costmap")
        self.clear_costmaps_client = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        

        # Initialize variables
        self.current_action = 0
        self.start_time = None
        self.collisions = 0
        self.total_reward = 0
        self.local_execute = 0
        self.cloud_execute = 0
        self.cloud_execute_time = None
        self.cloud_execute_state = []
        self.cloud_execute_time_list = []
        self.cloud_execute_laser_range = []
        self.cloud_execute_distance_from_start = []
        self.cloud_execute_last_cloud_distance = []
        self.cloud_execute_pose = []

        self.simulation_time = rospy.Time()
        self.local_range = 1.2
        self.cloud_range = 12
        self.temperature = 35
        self.network_capacity = 4
        self.previous_distance_to_goal = float('inf')
        self.nearest_scan_range = float('inf')
        self.front_nearest_scan_range = float('inf')
        self.current_goal = None
        self.goal_reached_threshold = 0.3
        self.laser_range = 2  # Initialize with a small costmap size
        self.collision_threshold = 0.2  # Meters
        self.start_location = None
        self.distance_to_new_goal = []
        self.time_taken_to_reach_goal = []

        self.last_large_costmap_position = None
        self.last_large_costmap_time = 0
        self.distance_from_last_big_costmap = 0
        self.last_known_cells = 0       

        self.no_movement_threshold = 0.1
        self.no_movement_duration = rospy.Duration(5)  # 5 seconds
        self.last_movement_time = rospy.Time.now()
        self.last_robot_pose = None
        self.tf_listener = ros_tf.TransformListener()
        self.start = False
        self.count = 0

        self.x = None
        self.y = None
        
        if self.priority == 1:
            self.cloud_action_penalty_weight = 0.5
        elif self.priority == 2:
            self.cloud_action_penalty_weight = 1
        elif self.priority == 3:
            self.cloud_action_penalty_weight = 2

        # Timer to check for no movement
        # self.no_movement_timer = Timer(self.no_movement_duration, self.check_no_movement, oneshot=False)

        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.modified_scan_pub = rospy.Publisher("/modified_scan", LaserScan, queue_size=10)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.clock_subscriber = rospy.Subscriber("/clock", Clock, self.clock_callback)

        # Wait for the action server to become available
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base_client.wait_for_server()

        self.episode_number = 0

        self.should_disregard_episode = False
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.move_base_status_callback)

    def clock_callback(self, data):
        # Update simulation time
        self.simulation_time = data.clock

    def move_backward(self):
        vel_msg = Twist()
        vel_msg.linear.x = -0.1
        vel_msg.angular.z = 0
        rate = rospy.Rate(10)
        t0 = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - t0 < 1.5:
            self.velocity_publisher.publish(vel_msg)
            rate.sleep()
        
        vel_msg.linear.x = 0
        self.velocity_publisher.publish(vel_msg)

    def _callback(self, msg):
        self.current_velocity = msg.linear.x

    def move_base_status_callback(self, status_msg):
        for status in status_msg.status_list:
            if status.status == status.ABORTED:
                if "a valid plan could not be found" in status.text:
                    rospy.loginfo("Aborting episode due to navigation failure.")
                    self.should_disregard_episode = True

    def send_goal_without_waiting(self, goal):
        move_base_goal = MoveBaseGoal()
        move_base_goal.target_pose.header.frame_id = goal.header.frame_id
        move_base_goal.target_pose.pose = goal.pose

        self.move_base_client.send_goal(move_base_goal)

    def current_goal_callback(self, msg):
        self.current_goal = msg

    def map_callback(self, msg):
        self.map = msg
        self.occupancy_grid = msg

    def scan_callback(self, msg):
        self.nearest_scan_range = min(msg.ranges)
        oracle_front_ranges = msg.ranges[120:240]
        self.oracle_front_nearest_scan_range = min(oracle_front_ranges)

        modified_scan = msg
        modified_scan.header = msg.header
        modified_scan.angle_min = msg.angle_min
        modified_scan.angle_max = msg.angle_max
        modified_scan.angle_increment = msg.angle_increment
        modified_scan.time_increment = msg.time_increment
        modified_scan.scan_time = msg.scan_time
        modified_scan.range_min = msg.range_min
        modified_scan.range_max = msg.range_max

        # Adjust the laser scan range based on the current_action
        if self.current_action == 0:  # Use short laser scan range
            self.laser_range = self.local_range
        elif self.current_action == 1:  # Use long laser scan range
            self.laser_range = self.cloud_range
        elif self.current_action == 2:  # Use Oracle
            if self.oracle_front_nearest_scan_range < 10:
                self.laser_range = self.oracle_front_nearest_scan_range + 2
            else:
                self.laser_range = 12
        else:
            rospy.loginfo("Action is invalid: {}".format(self.current_action))
            raise ValueError("Invalid action")

        modified_scan.ranges = [r if r <= self.laser_range else max(msg.ranges) for r in msg.ranges]

        self.modified_scan_pub.publish(modified_scan)
        front_ranges = modified_scan.ranges[120:240]
        self.front_nearest_scan_range = min(front_ranges)
        

    def odom_callback(self, msg):
        try:
            # Get the transform from "map" frame to "odom" frame
            self.tf_listener.waitForTransform("map", "odom", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "odom", rospy.Time(0))

            # Transform the robot pose from "odom" frame to "map" frame
            pose_in_odom = msg.pose.pose
            pose_in_map = PoseStamped()
            pose_in_map.header.frame_id = "odom"
            pose_in_map.pose = pose_in_odom
            robot_pose = self.tf_listener.transformPose("map", pose_in_map).pose

            # Store the initial location
            if self.start == True and self.initial_location is None:
                self.start = False
                self.initial_location = (robot_pose.position.x, robot_pose.position.y)

            self.robot_pose = robot_pose
            self.current_position = (robot_pose.position.x, robot_pose.position.y)
        except (ros_tf.LookupException, ros_tf.ConnectivityException, ros_tf.ExtrapolationException):
            pass

    def generate_goal(self, previous_side=None):
        if goal_x is None:
            x = 30
            y = 30
        else:
            x = goal_x
            y = goal_y
        side = True

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1.0

        return goal, side
    
    def check_no_movement(self, event):
        if self.last_robot_pose is not None:
            current_position = (self.robot_pose.position.x, self.robot_pose.position.y)
            last_position = (self.last_robot_pose.position.x, self.last_robot_pose.position.y)
            distance_moved = math.sqrt((current_position[0] - last_position[0])**2 +
                                       (current_position[1] - last_position[1])**2)
            if distance_moved < self.no_movement_threshold:
                if self.should_disregard_episode == True:
                    pass
                rospy.loginfo(f"Robot didn't move more {self.no_movement_threshold} meters for 5 seconds. Restarting episode.")
                # move backwards for 1 second by cmd_vel
                self.count += 1
                self.move_backward()
                if self.should_disregard_episode == False and self.count == 3:
                    self.should_disregard_episode = True
                    self.count = 0

            self.last_robot_pose = self.robot_pose
        else:
            self.last_robot_pose = self.robot_pose   
    def calculate_distance_to_goal(self):
        distance = math.sqrt((self.robot_pose.position.x - self.goal.pose.position.x)**2 +
                             (self.robot_pose.position.y - self.goal.pose.position.y)**2)
        # rospy.loginfo("Distance to goal: {}".format(distance))
        return distance

    def calculate_distance_to_start(self):
        # rospy.loginfo("Initial location: {}".format(self.initial_location))
        # rospy.loginfo("Robot pose: {}".format(self.robot_pose))
        distance = math.sqrt((self.robot_pose.position.x - self.initial_location[0])**2 +
                             (self.robot_pose.position.y - self.initial_location[1])**2)
        # rospy.loginfo("Distance to goal: {}".format(distance))
        return distance

    def check_goal_reached(self):
        goal_reached = self.calculate_distance_to_goal() < self.goal_reached_threshold
        if goal_reached:
            print("Goal reached!")
        return goal_reached

    def execute_action(self, action):
        if action == 0:  # Use small costmap
            self.current_action = action
            current_position = (self.robot_pose.position.x, self.robot_pose.position.y)
            if self.last_large_costmap_position is not None:
                self.distance_from_last_big_costmap = math.sqrt((current_position[0] - self.last_large_costmap_position[0])**2 +
                                                                (current_position[1] - self.last_large_costmap_position[1])**2)
            self.cloud_execute_time = rospy.Time.now().to_sec() - self.start_time - self.last_large_costmap_time
            self.local_execute += 1
        elif action == 1:  # Use big costmap
            rospy.logwarn("Using CLOUD EXECUTE")
            self.cloud_execute_laser_range.append(self.front_nearest_scan_range)
            self.cloud_execute_last_cloud_distance.append(self.distance_from_last_big_costmap)
                
            self.current_action = action
            self.last_large_costmap_position = (self.robot_pose.position.x, self.robot_pose.position.y)
            self.cloud_execute_pose.append(self.last_large_costmap_position)
            self.distance_from_last_big_costmap = 0
            self.cloud_execute_state.append(self.current_state)
            self.cloud_execute_distance_from_start.append(self.calculate_distance_to_start())
            self.last_large_costmap_time = rospy.Time.now().to_sec() - self.start_time
            self.cloud_execute_time = 0
            self.cloud_execute_time_list.append(self.last_large_costmap_time)
            self.cloud_execute += 1
            self.information_gain = self.calculate_occupancy_grid_gain()            

        elif action == 2:  # Use oracle
            self.current_action = action

        else:
            rospy.loginfo("Action is invalid: {}".format(action))   
            raise ValueError("Invalid action")

    def save_episode_data(self, csv_file_name):

        episode_time = rospy.Time.now().to_sec() - self.start_time
        self.time_taken_to_reach_goal.append(episode_time)

        with open(csv_file_name, 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([self.episode_number, str(self.goal.pose.position.x) + ", " +
                                 str(self.goal.pose.position.y), self.local_execute, self.cloud_execute, self.cloud_execute_time_list, self.cloud_execute_laser_range, self.cloud_execute_last_cloud_distance, self.cloud_execute_state ,episode_time,
                                 self.cloud_execute_distance_from_start, self.cloud_execute_pose, self.total_reward])

        self.episode_number += 1

    def reset_map(self):
        # Check if the occupancy grid has been received
        if self.occupancy_grid is not None:
            # Retrieve the dimensions of the grid
            width = self.occupancy_grid.info.width
            height = self.occupancy_grid.info.height

            # Convert occupancy_grid.data to a list for mutability
            data_list = list(self.occupancy_grid.data)

            # Reset all grid cells to empty (0)
            for i in range(width * height):
                data_list[i] = -1

            # Convert the list back to a tuple and update occupancy_grid.data
            self.occupancy_grid.data = tuple(data_list)
        else:
            rospy.logwarn("Occupancy grid has not been received yet. Cannot reset the map.")
    
    
    def calculate_occupancy_grid_gain(self):
        if self.occupancy_grid is not None:
            width = self.occupancy_grid.info.width
            height = self.occupancy_grid.info.height
            data_list = list(self.occupancy_grid.data)
            data_array = np.array(data_list)
            # unique_values, counts = np.unique(data_array, return_counts=True)
            # for value, count in zip(unique_values, counts):
            #     rospy.logwarn("Value: {} Count: {}".format(value, count))
            total_known_cells = np.count_nonzero(data_array == 100)
            if self.last_known_cells is not None:
                known_cells_gain = total_known_cells - self.last_known_cells
                rospy.logwarn("Known cells gain: {}".format(known_cells_gain))
                self.last_known_cells = total_known_cells
                if known_cells_gain < 0:
                    return 0
                elif known_cells_gain > 100:
                    return 100

            self.last_known_cells = total_known_cells
            # rospy.logwarn("Total known cells: {}".format(total_known_cells))
            return 0
        else:
            rospy.logwarn("Occupancy grid has not been received yet. Cannot calculate known cells gain.")
            return 0

    def step(self, action):

        self.execute_action(action)
        reward = self.calculate_reward()
        done = self.check_goal_reached()
        
        self.total_reward += reward
        # rospy.loginfo("Digregard: {}".format(self.should_disregard_episode))
        if self.should_disregard_episode:
            # Disregard the current episode and perform necessary actions
            self.reset(self.episode_number)
        rospy.loginfo("State: {}".format(self.current_state))
        return self.state, self.total_reward, done, {}

    def reset(self, episode):
        if not hasattr(self, 'previous_side'):
            if self.current_position is not None:
                rospy.logwarn("Current position in map frame: {}".format(self.current_position))
            self.previous_side = None
            self.start_location = PoseStamped()
            self.start_location.header.frame_id = "map"
            self.start_location.pose.position.x = self.current_position[0]
            self.start_location.pose.position.y = self.current_position[1]
            self.start_location.pose.orientation.w = 1.0
            self.goal = self.start_location
            self.current_action = 1
            self.send_goal_without_waiting(self.start_location)
            rospy.logwarn("initial poisition as the goal: x = {}, y = {}".format(self.start_location.pose.position.x, self.start_location.pose.position.y))
            initial_reached = False
            
            # Wait for the robot to return to the initial position
            while not initial_reached:
                distance_to_initial_position = self.calculate_distance_to_goal()
                if distance_to_initial_position < 0.7:
                    initial_reached = True
                    self.should_disregard_episode = False
                    # rospy.loginfo("Robot returned to initial position")
                    break
                rospy.sleep(0.1)  # Check every 100ms
            self.start=True
            rospy.loginfo("Robot returned to initial position")

            self.goal, self.previous_side = self.generate_goal(self.previous_side)
            rospy.logwarn("goal: x = {}, y = {}".format(self.goal.pose.position.x, self.goal.pose.position.y))
        elif self.should_disregard_episode:
            # rospy.logwarn("Robot returning to initial position")
            # Set the initial position as the goal

            self.current_action = 1
            self.goal = self.start_location
            self.send_goal_without_waiting(self.goal)
            rospy.logwarn("initial poisition as the goal: x = {}, y = {}".format(self.goal.pose.position.x, self.goal.pose.position.y))
            initial_reached = False
            
            # Wait for the robot to return to the initial position
            while not initial_reached:
                distance_to_initial_position = self.calculate_distance_to_goal()
                if distance_to_initial_position < 0.7:
                    initial_reached = True
                    self.should_disregard_episode = False
                    # rospy.loginfo("Robot returned to initial position")
                    break
                rospy.sleep(0.1)  # Check every 100ms
            self.start=True
            rospy.loginfo("Robot returned to initial position")
            self.goal, self.previous_side = self.generate_goal(self.previous_side)

        else:
            rospy.logwarn("End of episode {}, total reward: {}".format(episode, self.total_reward))
            # rospy.logwarn("Robot returning to initial position")
            # Set the initial position as the goal
            self.goal = self.start_location
            self.send_goal_without_waiting(self.goal)
            self.current_action = 1
            initial_reached = False
            
            # Wait for the robot to return to the initial position
            while not initial_reached:
                distance_to_initial_position = self.calculate_distance_to_goal()
                if distance_to_initial_position < 0.7:
                    initial_reached = True
                    self.should_disregard_episode = False
                    # rospy.loginfo("Robot returned to initial position")
                    break
                rospy.sleep(0.1)  # Check every 100ms

            self.goal, self.previous_side = self.generate_goal(self.previous_side)
            distance_to_goal = self.calculate_distance_to_goal()
            self.distance_to_new_goal.append(distance_to_goal)

        # Reset the occupancy grid map
        # Clear the costmaps
        self.current_action = 0
        self.reset_map()
        try:
            self.clear_costmaps_client()
            rospy.loginfo("Costmaps cleared")
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
        self.last_known_cells = 0
        self.reset_map()
        rospy.sleep(3)
        self.start_time = rospy.Time.now().to_sec()
        # rospy.logerr("Goal: x = {}, y = {}".format(self.goal.pose.position.x, self.goal.pose.position.y))
        self.send_goal_without_waiting(self.goal)  # Use the new function here

        self.collisions = 0
        self.total_reward = 0
        self.total_costmap_size = 0
        self.total_distance_from_last_big_costmap = 0
        self.total_large_costmap_time = 0
        self.local_execute = 0
        self.cloud_execute = 0
        self.cloud_execute_time = 4
        self.cloud_execute_time_list = []
        self.cloud_execute_laser_range = []
        self.cloud_execute_last_cloud_distance = []
        self.cloud_execute_pose = []
        self.previous_distance_to_goal = self.calculate_distance_to_goal()
        

        self.last_movement_time = rospy.Time.now().to_sec() - self.start_time
        self.last_robot_pose = None

        return self.state

    def calculate_angle_to_goal(self):
        angle_to_goal = math.atan2(self.goal.pose.position.y - self.robot_pose.position.y, self.goal.pose.position.x - self.robot_pose.position.x)
        angle_to_goal_degrees = math.degrees(angle_to_goal)
        return angle_to_goal_degrees


    def calculate_reward(self):
        reward = 0

        # Calculate distance to goal
        distance_to_goal = self.calculate_distance_to_goal()
        distance_to_start = self.calculate_distance_to_start() + 1
        # Check for collisions
        if self.nearest_scan_range < self.collision_threshold:
            self.collisions += 1
            rospy.logwarn("Collision detected, nearest scan range: {}".format(self.nearest_scan_range))
            reward -= 5000

        # Reward for getting closer to the goal
        if distance_to_goal < self.previous_distance_to_goal:
            reward += 5
        else:
            reward -= 5

        # Update the previous distance to goal
        self.previous_distance_to_goal = distance_to_goal

        # Penalize the size of the local costmap used * 10 per action taken
        reward -= 10

        if self.cloud_execute_time is not None and self.cloud_execute_time * self.current_velocity > self.cloud_range:
            reward -= 1
        # Penalize using action 1 (long laser scan range) further away from the start
        
        # calculate reward for information gain
        if self.current_action != 0:
            reward -= 5
            information_gain = self.information_gain / self.cloud_action_penalty_weight
            information_gain = information_gain / (distance_to_start)
            information_gain = round(information_gain, 1)
            rospy.loginfo("Distance to start: {}".format(distance_to_start))
            reward += information_gain
            rospy.loginfo("Reward for information gain: {}".format(information_gain))
            
        rospy.loginfo("Reward: {}".format(reward))
        return reward

    @property
    def state(self):
        max_laser_range = 12
        normalized_front_nearest_scan_range = round(self.front_nearest_scan_range / max_laser_range, 2)
        if normalized_front_nearest_scan_range == float('inf'):
            normalized_front_nearest_scan_range = 2
        max_velocity = 1.0
        velocity = round(self.current_velocity,1) / max_velocity
        temperature = self.temperature

        normalized_distance_to_goal = round(self.calculate_distance_to_goal()/max_velocity, 1)
        angle_to_goal = round(self.calculate_angle_to_goal(),1)
        normalized_distance_from_network = round(self.calculate_distance_to_start()/self.network_capacity, 1)
        total_information_gain = round(self.last_known_cells / max_laser_range ** 2, 1)
        if self.cloud_execute_time is None:
            cloud_execute_time = 5
        else:
            cloud_execute_time = round(self.cloud_execute_time, 0)
        # state space 8
        normalized_state = [normalized_front_nearest_scan_range, velocity, temperature,
                            normalized_distance_to_goal, angle_to_goal, normalized_distance_from_network,
                            total_information_gain, cloud_execute_time]
        self.current_state = normalized_state

        state_message = Float32MultiArray()
        state_message.data = self.current_state
        state_publisher.publish(state_message)

        return normalized_state
    
from keras.models import Model
from keras.layers import Dense, Input, Add, Lambda
from keras.optimizers import Adam
from keras import backend as K

def create_dqn_model(observation_space, action_space):
    input = Input(shape=(observation_space,))
    x = Dense(24, activation='relu')(input)
    x = Dense(48, activation='relu')(x)

    # Dueling DQN
    state_value = Dense(1, kernel_initializer='he_uniform')(x)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

    action_advantage = Dense(action_space, kernel_initializer='he_uniform')(x)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

    q_values = Add()([state_value, action_advantage])

    model = Model(inputs=input, outputs=q_values)
    model.compile(loss='mse', optimizer=Adam())

    return model


class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)



class OverlayState:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/camera/color/image_raw/overlay", Image, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.state_sub = rospy.Subscriber('state', Float32MultiArray, self.state_callback)
        self.state = None  # Initialize state to None

    def state_callback(self, data):
        self.state = np.array(data.data)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        state = self.state
        if state is None:
            pass
        else:
            if state[0] == 2:
                state_0 = "Out"
            else:
                state_0 = str(round(float(state[0])*100,2))+'%'
            state_1 = round(float(state[1])*100,2)
            state_3 = round(float(state[3]),2)
            state_4 = round(float(state[4]),2)
            state_info = [
                f'Front Nearest Scan Range: {state_0} of Max Range',
                f'Velocity: {state_1}% of Max Velocity',
                f'Temperature: {state[2]}',
                f'Goal Reach Time Estimate: {state_3} seconds on current velocity',
                f'Angle to Goal: {state_4} degrees',
                f'Network Latency Estimate: {round(1/state[5]*2,2)}',
                f'Situation Aware Value : {state[6]}',
                f'Last Cloud Execute Time(sec): {state[7]}'
            ]

            y = 50  # initial y-coordinate for the text
            for info in state_info:
                # Draw the outline by drawing the text in black with a thicker line width
                cv2.putText(cv_image, info, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                # Draw the text in white with a thinner line width on top of the outline
                cv2.putText(cv_image, info, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y += 20  # increment the y-coordinate for the next line

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == "__main__":
    
        file_name = input("Enter the file name: ")
        csv_file_name = file_name + ".csv"
        model_name = input("Enter the model name: ")
        model_file = model_name + '.h5'
        
        priority = float(input("Enter priority: 1: Awareness, 2: Balance:, 3:Cloud Efficiency:"))


        # Create CSV file
        if not os.path.exists(csv_file_name):
            with open(csv_file_name, 'w') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Episode", "Random location", "Local execution", "Cloud execution", "Cloud execution time list", "Cloud execute laser range", "Cloud execute last cloud distance", "Cloud Execute State", "Time taken", "Distance from Initial location", "Cloud Execute Position","Total reward"])
                
        env = NavigationEnvROS(priority = priority)
        state_publisher = rospy.Publisher('state', Float32MultiArray, queue_size=10)

        observation_space = env.observation_space[0]
        action_space = env.action_space
        overlay_state = OverlayState()
        episodes = 2

        # Ask user for goal position
        global goal_x
        goal_x = float(input("Enter the goal x coordinate: "))
        global goal_y
        goal_y = float(input("Enter the goal y coordinate: "))

        accumulated_rewards = []
        
        all_trajectory = []
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            model = create_dqn_model(observation_space, action_space)

        for episode in range(episodes):
            episode = 3
            trajectory = []
            rospy.loginfo("starting episode #{}".format(episode+1))
            state = env.reset(episode)
            state = np.array(state, dtype=np.float32)

            done = False
            episode_steps = 0
            while not done:
                try:
                    if episode == 0:
                        action = 0
                    elif episode == 1:
                        action = 1
                    elif episode == 2:
                        action = 2
                    else:
                        if episode < 5:
                            if episode_steps < 5 and state[7] >= 4 and state[0]==2:
                                rospy.loginfo("Action: 1 from start if")
                                action = 1
                            elif state[7] < 4:
                                rospy.loginfo("Action: 0 for recent execution")
                                action = 0
                            elif state[7] > 10:
                                rospy.loginfo("Action: random from network too stale")
                                action = 1
                            elif state[5] > 3:
                                rospy.loginfo("Action: 0 from network too far")
                                action = 0
                            else:
                                rospy.loginfo("Action: random from else")
                                action_probs = model.predict(state[np.newaxis, ...])[0]
                                q_values = model.predict(state[np.newaxis, ...])[0]
                                action = np.argmax(q_values)
                                rospy.logerr("q_values info : {}".format(q_values))

                    new_state, reward, done, _ = env.step(action)
                    rospy.sleep(1)
                    trajectory.append(env.current_position)
                    new_state = np.array(new_state, dtype=np.float32)
                    # rospy.loginfo("state info : {}".format(new_state))

                    state = new_state
                    if done:
                        env.save_episode_data(csv_file_name=csv_file_name)
                        all_trajectory.append(trajectory)
                        accumulated_rewards.append(env.total_reward)

                except rospy.ROSInterruptException:
                    pass    


        labels = ["small costmap", "big costmap", "dynamic costmap", "RL Model"]

        # Plot and save the combined trajectories
        for i, trajectory in enumerate(all_trajectory):
            plt.plot(*zip(*trajectory), marker="o", label=labels[i])

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("All Trajectories to goal")
        plt.legend()
        plt.savefig("all_trajectories.png")
        plt.show()
        print("Combined trajectories saved as 'all_trajectories.png'")

        import datetime

        # Plot the accumulated rewards
        plt.plot(accumulated_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Reward")
        plt.title("Accumulated Rewards per Episode")
        if not os.path.exists("accumulated_rewards_plot.png"):
            plt.savefig("accumulated_rewards_plot.png")
        else:
            plt.savefig("accumulated_rewards_plot_{}.png".format(datetime.datetime.now()))
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Distance to New Goal", color='tab:blue')
        ax1.plot(env.distance_to_new_goal, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel("Time Taken to Reach Goal", color='tab:red')
        ax2.plot(env.time_taken_to_reach_goal, color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.tight_layout()
        plt.title("Distance to New Goal and Time Taken to Reach Goal per Episode")
        
        if not os.path.exists("distance_time_plot.png"):
            plt.savefig("distance_time_plot.png")
        else: # save with datetime
            plt.savefig("distance_time_plot_{}.png".format(datetime.datetime.now()))
        plt.show()


