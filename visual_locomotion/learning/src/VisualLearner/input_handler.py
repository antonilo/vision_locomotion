#!/usr/bin/env python3
import time

from visualoco_msgs.msg import Proprioception
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from collections import deque
import cv2
from Architecture.models.learner import Learner

import numpy as np
import rospy


class InputHandler():
    def __init__(self, config):
        self.config = config
        self.learner = Learner(config=config, mode="deploy")
        self.learner.initialize_network()  # build the computation graph
        self.initiate_subscribers()

    def initiate_subscribers(self):
        self.updating_prop = False
        self.updating_img = False
        self.sending_inputs = False
        self.img_fts = None
        self.img_timestamp = 0
        self.prop = np.ones((40,))
        self.prop_timestamp = 0.0
        self.img_fallback_time = 0.5 #s
        self.propioception = Proprioception()
        self.img_list = deque(maxlen=3*self.config.frame_skip)
        self.prop_list = deque(maxlen=self.config.history_len)
        for _ in range(3*self.config.frame_skip):
            if self.config.input_use_depth:
                self.img_list.append(np.zeros((270,480), dtype=np.float32))
            elif self.config.input_use_imgs:
                self.img_list.append(np.zeros((240,424), dtype=np.float32))
        for i in range(self.config.history_len):
            self.prop_list.append(np.zeros((40,), dtype=np.float32))

        self.prop_sub = rospy.Subscriber("/agile_locomotion/proprioception",
                                         Proprioception,
                                         self.callback_prop, queue_size=1, tcp_nodelay=True)

        if self.config.input_use_imgs:
            self.bridge = CvBridge()
            if self.config.input_use_depth:
                self.img_sub = rospy.Subscriber("/front/depth/image_rect_raw",
                                                Image,
                                                self.callback_depth,
                                                queue_size=1,
                                                tcp_nodelay=True)
            else:
                self.img_sub = rospy.Subscriber("/front/color/image_raw",
                                                Image,
                                                self.callback_img,
                                                queue_size=1,
                                                tcp_nodelay=True)


    def callback_prop(self, data):
        while self.sending_inputs:
            time.sleep(0.0001)
        self.updating_prop = True
        self.process_prop(data)
        self.updating_prop = False

    def callback_img(self, data):
        while self.sending_inputs:
            time.sleep(0.0001)
        self.updating_img = True
        self.img_timestamp = data.header.stamp.to_sec()
        try:
            image = self.bridge.imgmsg_to_cv2(data)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except CvBridgeError as e:
            print(e)
            return
        self.img_list.append(image)
        self.img_fts = self.learner.getImageFts(self.img_list)
        self.updating_img = False

    def callback_depth(self, data):
        while self.sending_inputs:
            time.sleep(0.0001)
        self.updating_img = True
        self.img_timestamp = data.header.stamp.to_sec()
        try:
            depth = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            print(e)
            return
        self.img_list.append(np.asarray(depth, np.float32))
        self.img_fts = self.learner.getImageFts(self.img_list)
        self.updating_img = False

    def process_prop(self, prop_msg):
        self.prop_msg = prop_msg
        self.prop_timestamp = prop_msg.header.stamp.to_sec()
        observation = []
        observation.append(prop_msg.rpy_0)
        observation.append(prop_msg.rpy_1)
        observation.extend(prop_msg.joint_angles)
        observation.extend(prop_msg.joint_vel)
        observation.extend(prop_msg.last_action)
        observation.extend(prop_msg.command[:2])
        prop_array = np.array(observation, dtype=np.float32)
        self.prop_list.append(prop_array)
        return

    def prepare_net_inputs(self):
        while self.updating_prop or self.updating_img:
            time.sleep(0.0001)
        self.sending_inputs = True
        prop_array = np.vstack(self.prop_list)
        # get history of frame
        input_dict = {'img_fts': self.img_fts,
                      'prop': np.expand_dims(prop_array, axis=0),
                      'img_ts': self.img_timestamp,
                      'prop_ts': self.prop_timestamp}
        self.sending_inputs = False
        return input_dict
