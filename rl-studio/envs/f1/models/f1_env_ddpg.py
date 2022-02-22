import math
import cv2
from cv2 import CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from sensor_msgs.msg import Image
from cprint import cprint
from icecream import ic
from datetime import datetime
import time
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from PIL import Image as im
from agents.utils import print_messages

#from agents.f1.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
#from settings import x_row, center_image, width, height, telemetry_mask, max_distance
from rl_studio.envs.f1.image_f1 import ImageF1
from rl_studio.envs.f1.models.f1_env import F1Env
from rl_studio.envs.gazebo_utils import set_new_pose

ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')

class F1DDPGCameraEnv(F1Env):

    def __init__(self, **config):

        F1Env.__init__(self, **config)
        self.image = ImageF1()
        #self.cv_image_pub = None
        self.image_raw_from_topic = None
        self.f1_image_camera = None

        self.sensor = config['sensor']

        # Image
        self.image_resizing = config['image_resizing'] / 100
        self.new_image_size = config['new_image_size']
        self.raw_image = config['raw_image']
        self.height = int(config['height_image'] * self.image_resizing)
        #self.height = int(config['height_image'])
        self.width = int(config['width_image'] * self.image_resizing)
        #self.width = int(config['width_image'])
        self.center_image = int(config['center_image'] * self.image_resizing)
        #self.center_image = int(config['center_image'])
        self.num_regions = config['num_regions']
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config['telemetry_mask']
        self.poi = config['x_row'][0]

        # States
        self.state_space = config['state_space']
        if self.state_space == 'spn':
            self.x_row = [i for i in range(1, int(self.height/2)-1)]
            #print(f"[spn] self.x_row: {self.x_row}")
        else:
            self.x_row = config['x_row']

        # Actions
        #self.beta_1 = -(config["actions"]['w_left'] / (config["actions"]['v_max'] - config["actions"]['v_min']))
        #self.beta_0 = -(self.beta_1 * config["actions"]['v_max'])
        self.action_space = config['action_space']
        self.actions = config["actions"]


        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config['min_reward']
        self.beta_1 = config['beta_1']
        self.beta_0 = config['beta_0']

        # 
        self.telemetry = config["telemetry"]

    def render(self, mode='human'):
        pass

    @staticmethod
    def all_same(items):
        return all(x == items[0] for x in items)


    def image_msg_to_image(self, img, cv_image):
        #ic("F1DQNCameraEnv.image_msg_to_image()")
        #print(f"\n F1QlearnCameraEnv.image_msg_to_image()\n")

        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image 

    @staticmethod
    def get_center(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            #print(f"No lines detected in the image")
            return 0

    def calculate_reward(self, error: float) -> float:

        d = np.true_divide(error, self.center_image)
        reward = np.round(np.exp(-d), 4)

        return reward


    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        #line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        #_, mask = cv2.threshold(line_pre_proc, 48, 63, cv2.THRESH_BINARY) #(240 -> 255)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY) #(240 -> 255)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        centrals = list(map(self.get_center, lines))

        # if centrals[-1] == 9:
        #     centrals[-1] = center_image

        if self.telemetry_mask:
            mask_points = np.zeros((self.height, self.width), dtype=np.uint8)
            for idx, point in enumerate(centrals):
                # mask_points[x_row[idx], centrals[idx]] = 255
                cv2.line(mask_points, (point, self.x_row[idx]), (point, self.x_row[idx]), (255, 255, 255), thickness=3)

            cv2.imshow("MASK", mask_points[image_middle_line:])
            cv2.waitKey(3)

        return centrals


    def calculate_observation(self, state: list) -> list:
        '''
        This is original Nacho's version. 
        I have other version. See f1_env_ddpg_camera.py
        '''
        #normalize = 40

        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.center_image - x) / self.pixel_region) + 1)
            #final_state.append(int(x / self.pixel_region) + 1)

        return final_state

    def rewards_discrete(self, center):
        if 0 <= center <= 0.2:
            reward = self.rewards['from_0_to_02']
        elif 0.2 < center <= 0.4:
            reward = self.rewards['from_02_to_04']
        else:
            reward = self.rewards['from_others']
        
        return reward

    def reward_v_w_center_linear(self, vel_cmd, center):
        '''
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward + center))) where Max value = 1

        Args: 
            linear and angular velocity
            center

        Returns: reward
        '''

        w_target = self.beta_0 + (self.beta_1 * abs(vel_cmd.linear.x))
        error = abs(w_target - abs(vel_cmd.angular.z))
        reward = 1/math.exp(error + center)
        
        return reward

    def reward_v_w_center_linear_first_formula(self, vel_cmd, center):
        '''
        Applies a linear regression between v and w
        Supposing there is a lineal relationship V and W. So, formula w = B_0 + x*v.

        Data for Formula1:
        Max W = 5 r/s we take max abs value. Correctly it is w left or right
        Max V = 100 m/s
        Min V = 20 m/s
        B_0 = -B_1 * Max V
        B_1 = -(W Max / (V Max - V Min))

        w target = B_0 + B_1 * v
        error = w_actual - w_target
        reward = 1/exp(reward)/sqrt(center^2+0.001)

        Args: 
            linear and angular velocity
            center

        Returns: reward
        '''

        num = 0.001
        w_target = self.beta_0 + (self.beta_1 * abs(vel_cmd.linear.x))
        error = abs(w_target - abs(vel_cmd.angular.z))
        reward = 1/math.exp(error)
        reward = (reward / math.sqrt(pow(center,2) + num)) #Maximize near center and avoid zero in denominator
        
        return reward


    def image_preprocessing_black_white_original_size(self, img):
        #print(f"entro image_preprocessing_black_white_original_size")
        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY) #(240 -> 255)
        maskBlackWhite3D = np.expand_dims(mask, axis=2)

        # now resizing is always 32x32 or 64x64
        #maskGray3D = cv2.resize(maskGray3D, (self.new_image_size, self.new_image_size), cv2.INTER_AREA)
        return maskBlackWhite3D

    def image_preprocessing_black_white_32x32(self, img):
        #print(f"entro image_preprocessing_black_white_32x32")
        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY) #(240 -> 255)

        maskBlackWhite3D = cv2.resize(mask, (32,32), cv2.INTER_AREA)
        maskBlackWhite3D = np.expand_dims(maskBlackWhite3D, axis=2)

        # now resizing is always 32x32 or 64x64
        #maskGray3D = cv2.resize(maskGray3D, (self.new_image_size, self.new_image_size), cv2.INTER_AREA)
        return maskBlackWhite3D

    def image_preprocessing_gray_32x32(self, img):
        #print(f"entro image_preprocessing_black_white_32x32")
        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        #line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        #_, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY) #(240 -> 255)

        imgGray3D = cv2.resize(img_proc, (32,32), cv2.INTER_AREA)
        imgGray3D = np.expand_dims(imgGray3D, axis=2)

        # now resizing is always 32x32 or 64x64
        #maskGray3D = cv2.resize(maskGray3D, (self.new_image_size, self.new_image_size), cv2.INTER_AREA)
        return imgGray3D


    def image_preprocessing_raw_original_size(self, img):
        print(f"entro image_preprocessing_raw_original_size")

        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]

        # now resizing is always 32x32 or 64x64
        #img_sliced = cv2.resize(img_sliced, (self.new_image_size, self.new_image_size), cv2.INTER_AREA)
        return img_sliced

    def image_preprocessing_color_quantization_original_size(self, img):
        #print(f"entro image_preprocessing_color_quantization_original_size")
        
        n_colors = 3
        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        #assert d == 3
        image_array = np.reshape(img_sliced, (w * h, d))

        image_array_sample = shuffle(image_array, random_state=0, n_samples=50)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        labels = kmeans.predict(image_array)

        return kmeans.cluster_centers_[labels].reshape(w, h, -1)


    def image_preprocessing_color_quantization_32x32x1(self, img):
        #print(f"entro image_preprocessing_color_quantization_32x32x1")
        
        n_colors = 3
        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        #assert d == 3
        image_array = np.reshape(img_sliced, (w * h, d))

        image_array_sample = shuffle(image_array, random_state=0, n_samples=500)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

        labels = kmeans.predict(image_array)
        im_reshape = kmeans.cluster_centers_[labels].reshape(w, h, -1)
        # convert to np array 32x32x1
        im_resize32x32x1 = np.expand_dims(np.resize(im_reshape, (32, 32)),axis=2)

        return im_resize32x32x1

    def image_preprocessing_reducing_color_PIL_original_size(self, img):
        #print(f"entro image_preprocessing_reducing_color_PIL_original_size")

        num_colors = 4
        image_middle_line = self.height // 2            
        img_sliced = img[image_middle_line:]

        array2pil = im.fromarray(img_sliced)
        array2pil_reduced = array2pil.convert('P', palette=im.ADAPTIVE, colors=num_colors)
        pil2array = np.expand_dims(np.array(array2pil_reduced), 2)
        #print_messages('image_preprocessing_reducing_color_PIL_original_size', img_shape=img.shape,num_colors=num_colors, image_middle_line=image_middle_line, img_sliced_shape = img_sliced.shape,pil2array_shape = pil2array.shape)
        return pil2array


    '''
    def image_segmentation(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        #line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        #_, mask = cv2.threshold(line_pre_proc, 48, 63, cv2.THRESH_BINARY) #(240 -> 255)
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY) #(240 -> 255)
        maskGray3D = np.expand_dims(mask, axis=2)

        return maskGray3D
    '''

    def image_callback(self, image_data):
        #global cv_image
        self.image_raw_from_topic = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        #self.f1_image_camera.width = image_data.width
        #self.f1_image_camera.height = image_data.height
        #self.f1_image_camera.format = "RGB8"
        #self.f1_image_camera.timeStamp = image_data.header.stamp.secs + (image_data.header.stamp.nsecs * 1e-9)
        #self.f1_image_camera.data = self.cv_image

        #self.cv_image_pub.publish(cv_image)


    def reset(self):
        '''
        Main reset. Depending of:
        - sensor
        - states: images or simplified perception (sp)

        '''
        #ic(sensor)
        if self.sensor == 'camera':
            return self.reset_camera()  

    def reset_camera_SUBScriber(self):
        '''
        ropsy.Subscriber()
        
        '''
        if self.alternate_pose:
            print(f"\n[INFO] ===> Necesary implementing self._gazebo_set_new_pose()...class F1DDPGCameraEnv(F1Env) -> def reset_camera() \n")
            #self._gazebo_set_new_pose() # Mine, it works fine!!!
            #pos_number = set_new_pose(self.circuit_positions_set) #not using. Just for V1.1
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        success = False
        #image_data = None        
        while self.cv_image is None or success is False:
            # SACADO DE https://stackoverflow.com/questions/57271100/how-to-feed-the-data-obtained-from-rospy-subscriber-data-into-a-variable
            rospy.Subscriber('/F1ROS/cameraL/image_raw', Image, self.image_callback, queue_size = 1)
            #image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=1)
            #cv_image_wait_for_message = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            #if (cv_image_wait_for_message == self.cv_image).all():
            #    print_messages('son iguales cv_image_wait_for_message == self.cv_image')


            if np.any(self.cv_image):
                success = True

        f1_image_camera = cv2.resize(self.cv_image, (int(self.cv_image.shape[1] * self.image_resizing), int(self.cv_image.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
        #print_messages('cv_image.shape', cv_image_shape = self.cv_image.shape, f1_image_camera_shape=f1_image_camera.shape)

        points = self.processed_image(self.cv_image)
        self._gazebo_pause()

        if self.state_space == 'image':
            if self.raw_image:
                state = np.array(self.image_preprocessing_reducing_color_PIL_original_size(f1_image_camera))
            else:    
                state = np.array(self.image_preprocessing_black_white_32x32(f1_image_camera))

            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size

    def reset_camera(self):
        '''
        goal: resetting environment to default in every new epoch
        rospy.wait_for_message()

        it can work with raw images, resizing images, 32x32 images, segmented images in black and white, gray images, quantization images, PIL color reducing images,
        and simplified perception in 1, 3 or n points
        return: 
        - state 
        - state_size: this field is just for debugging purposes
        '''
        if self.alternate_pose:
            #print(f"\n[INFO] ===> Necesary implement self._gazebo_set_new_pose()...class F1DDPGCameraEnv(F1Env) -> def reset_camera() \n")
            self._gazebo_set_new_pose() # Mine, it works fine!!!
            #pos_number = set_new_pose(self.circuit_positions_set) #not using. Just for V1.1
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

            # now resizing the image
            cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * self.image_resizing), int(cv_image.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            
            #f1_image_camera = cv_image
            if np.any(cv_image):
                success = True

        points = self.processed_image(f1_image_camera.data)
        self._gazebo_pause()

        if self.state_space == 'image':
            # Segment image in two colors, black and white to reducing image size
            #image_segmentated = self.image_segmentation(f1_image_camera.data)
            #state = np.array(image_segmentated)
            if self.raw_image:
                state = np.array(self.image_preprocessing_reducing_color_PIL_original_size(f1_image_camera.data))
            else:    
                state = np.array(self.image_preprocessing_black_white_original_size(f1_image_camera.data))
                #state = np.array(self.image_preprocessing_black_white_32x32(f1_image_camera.data))

            # Square image to fit in Conv2D neural net actor-critic
            #state = cv2.resize(state, (self.new_image_size, self.new_image_size), cv2.INTER_AREA)
            #state = np.array(cv_image)
            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size 

    def reset_camera_wait_for(self):
        '''
        goal: resetting environment to default in every new epoch
        rospy.wait_for_message()

        it can work with raw images, resizing images, 32x32 images, segmented images in black and white, gray images, quantization images, PIL color reducing images,
        and simplified perception in 1, 3 or n points
        return: 
        - state 
        - state_size: this field is just for debugging purposes
        '''
        if self.alternate_pose:
            print(f"\n[INFO] ===> Necesary implement self._gazebo_set_new_pose()...class F1DDPGCameraEnv(F1Env) -> def reset_camera() \n")
            #self._gazebo_set_new_pose() # Mine, it works fine!!!
            #pos_number = set_new_pose(self.circuit_positions_set) #not using. Just for V1.1
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

            # now resizing the image
            cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * self.image_resizing), int(cv_image.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            
            #f1_image_camera = cv_image
            if np.any(cv_image):
                success = True

        points = self.processed_image(f1_image_camera.data)
        self._gazebo_pause()

        if self.state_space == 'image':
            # Segment image in two colors, black and white to reducing image size
            #image_segmentated = self.image_segmentation(f1_image_camera.data)
            #state = np.array(image_segmentated)
            if self.raw_image:
                state = np.array(self.image_preprocessing_reducing_color_PIL_original_size(f1_image_camera.data))
            else:    
                state = np.array(self.image_preprocessing_black_white_original_size(f1_image_camera.data))
                #state = np.array(self.image_preprocessing_black_white_32x32(f1_image_camera.data))

            # Square image to fit in Conv2D neural net actor-critic
            #state = cv2.resize(state, (self.new_image_size, self.new_image_size), cv2.INTER_AREA)
            #state = np.array(cv_image)
            state_size = state.shape
            return state, state_size
        else:
            state = self.calculate_observation(points)
            state_size = len(state)
            return state, state_size 


    #####################################################################################################

    def step_NOTWORKING(self, action):   
        '''
        probando rospy.Subscriber
        '''

        self._gazebo_unpause()
        vel_cmd = Twist()

        if self.action_space == 'continuous':
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]
            
        self.vel_pub.publish(vel_cmd)

        print_messages('hola')
        success = False
        while self.image_raw_from_topic is None or success is False:
            rospy.Subscriber('/F1ROS/cameraL/image_raw', Image, self.image_callback, queue_size = 1)
        #rospy.Subscriber('/F1ROS/cameraL/image_raw', Image, self.image_callback, queue_size = 1)
            if np.any(self.image_raw_from_topic):
                success = True
        #print_messages('tiene que coincidir con el sgieuinte mensaje')
        #rospy.Subscriber('/F1ROS/cameraL/image_raw', Image, self.image_callback, queue_size = 1)

        #while self.cv_image is None or success is False:
            # SACADO DE https://stackoverflow.com/questions/57271100/how-to-feed-the-data-obtained-from-rospy-subscriber-data-into-a-variable
        #    image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=1)
        #    cv_image_wait_for_message = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        #    if (cv_image_wait_for_message == self.cv_image).all():
        #        print_messages('====> in STEP son iguales cv_image_wait_for_message == self.cv_image')

        #    if np.any(self.cv_image):
        #        success = True
        print_messages('hola2')
        f1_image_camera = cv2.resize(self.image_raw_from_topic, (int(self.image_raw_from_topic.shape[1] * self.image_resizing), int(self.image_raw_from_topic.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
        #f1_image_camera_wait_for_message = cv2.resize(cv_image_wait_for_message, (int(cv_image_wait_for_message.shape[1] * self.image_resizing), int(cv_image_wait_for_message.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
        points = self.processed_image(f1_image_camera)
        #points_wait_for_message = self.processed_image(f1_image_camera_wait_for_message)
        #print_messages('--step', f1_image_camera_shape=f1_image_camera.shape, f1_image_camera_wait_for_message=f1_image_camera_wait_for_message.shape,points=points, points_wait_for_message=points_wait_for_message)

        self._gazebo_pause()

        #points = self.processed_image(f1_image_camera.data)
        if self.state_space == 'spn':
            self.point = points[self.poi]
        else:
            self.point = points[0]    

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        if self.state_space == 'image':
            if self.raw_image:
                state = np.array(self.image_preprocessing_reducing_color_PIL_original_size(f1_image_camera))
            else:    
                state = np.array(self.image_preprocessing_black_white_32x32(f1_image_camera))

        else:
            state = self.calculate_observation(points)

        done = False
        # calculate reward
        if center > 0.9:
            done = True
            reward = self.rewards['from_done']
        else:
            reward = self.rewards_discrete(center)

        #print_messages('end step', reward=reward, state_shape=state.shape, done=done, center=center)
        return state, reward, done, {}

########################################################################################

    def step(self, action):   
        '''
        exec every step with action given. it is working!!!

        Args:
            action

        Returns: 
            state: 
            reward: 
            done: 
            info: is just used to follow OpenAI conventions)
        '''
        self._gazebo_unpause()
        vel_cmd = Twist()

        if self.action_space == 'continuous':
            vel_cmd.linear.x = action[0][0]
            vel_cmd.angular.z = action[0][1]
        else:
            vel_cmd.linear.x = self.actions[action][0]
            vel_cmd.angular.z = self.actions[action][1]
            
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            # now resizing the image
            cv_image = cv2.resize(cv_image, (int(cv_image.shape[1] * self.image_resizing), int(cv_image.shape[0] * self.image_resizing)), cv2.INTER_AREA)             
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            

        self._gazebo_pause()
        points = self.processed_image(f1_image_camera.data)
        if self.state_space == 'spn':
            self.point = points[self.poi]
        else:
            self.point = points[0]    

        center = abs(float(self.center_image - self.point) / (float(self.width) // 2))

        if self.state_space == 'image':
            if self.raw_image:
                state = np.array(self.image_preprocessing_reducing_color_PIL_original_size(f1_image_camera.data))
            else:    
                #state = np.array(self.image_preprocessing_black_white_32x32(f1_image_camera.data))
                state = np.array(self.image_preprocessing_black_white_original_size(f1_image_camera.data))
        else:
            #state = self.calculate_observation(points)
            state = self.calculate_observation(points)

        done = False
        # calculate reward
        if center > 0.9:
            done = True
            reward = self.rewards['penal']
        else:
            if self.reward_function == 'linear':
                reward = self.reward_v_w_center_linear(vel_cmd, center)
            else:
                reward = self.rewards_discrete(center)

        return state, reward, done, {}
