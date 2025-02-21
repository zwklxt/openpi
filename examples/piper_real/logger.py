#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
import csv
import os
from datetime import datetime
import cv2
import pandas as pd
from cv_bridge import CvBridge
from threading import Thread
import numpy as np
from PIL import Image as PImage




log_dir = os.path.join(os.path.dirname(__file__), 'logs') + '/' + datetime.now().strftime('%m%d%H%M') + '/'
# 创建日志文件夹

# 创建CvBridge对象
bridge = CvBridge()

class InputJointStateLogger:
    def __init__(self):
        rospy.init_node('input_joint_state_logger', anonymous=True)
        # 订阅/puppet/joint_left话题
        self.subscription_left = rospy.Subscriber('/puppet/joint_left', JointState, self.joint_state_callback_left)
        # 订阅/puppet/joint_right话题
        self.subscription_right = rospy.Subscriber('/puppet/joint_right', JointState, self.joint_state_callback_right)

        # 设置CSV文件路径
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file_path_left = log_dir + 'joint_states_left.csv'
        self.csv_file_path_right = log_dir + 'joint_states_right.csv'
        # 写入CSV文件头
        with open(self.csv_file_path_left, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'frame_id', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper'])
        # 写入CSV文件头
        with open(self.csv_file_path_right, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'frame_id', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper'])
            
        #spin thread
        self.spin_thread = Thread(target=self.spin)
        self.spin_thread.start()
    
    def spin(self):
        try:
            # 保持节点运行，直到有外部中断信号（如Ctrl+C）
            rospy.spin()
        except KeyboardInterrupt:
            # 捕获Ctrl+C中断
            print(" shutting down")
        finally:
            # 不管是否发生异常，都执行清理操作
            rospy.signal_shutdown("User requested shutdown")


    def joint_state_callback_left(self, msg):
        # 提取时间戳和位置信息
        timestamp = msg.header.stamp.to_sec()
        frame_id = msg.header.frame_id
        positions = ','.join(map(str, msg.position)).split(',')

        # 将数据写入CSV文件
        with open(self.csv_file_path_left, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_id, positions])
        rospy.loginfo(f"Logged left timestamp: {timestamp},frame_id:{frame_id}, positions: {positions}")

    def joint_state_callback_right(self, msg):
        # 提取时间戳和位置信息
        timestamp = msg.header.stamp.to_sec()
        frame_id = msg.header.frame_id
        positions = ','.join(map(str, msg.position)).split(',')

        # 将数据写入CSV文件
        with open(self.csv_file_path_right, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_id, positions])
        rospy.loginfo(f"Logged right timestamp: {timestamp},frame_id:{frame_id}, positions: {positions}")


class InputImgLogger:
    def __init__(self):
        rospy.init_node('input_img_logger', anonymous=True)
        # 订阅/camera_f/color/image_raw话题
        self.subscription_front = rospy.Subscriber('/camera_f/color/image_raw', Image, self.image_callback_front)
        # 订阅/camera_l/color/image_raw话题
        self.subscription_left = rospy.Subscriber('/camera_l/color/image_raw', Image, self.image_callback_left)
        # 订阅/camera_r/color/image_raw话题
        self.subscription_right = rospy.Subscriber('/camera_r/color/image_raw', Image, self.image_callback_right)

        # 设置图片保存路径
        os.makedirs(log_dir, exist_ok=True)
        self.image_dir = log_dir + 'images/'
        os.makedirs(self.image_dir, exist_ok=True)

        # 初始化图片计数器
        self.image_count_front = 0
        self.image_count_left = 0
        self.image_count_right = 0

        #spin thread
        self.spin_thread = Thread(target=self.spin)
        self.spin_thread.start()
    
    def spin(self):
        try:
            # 保持节点运行，直到有外部中断信号（如Ctrl+C）
            rospy.spin()
        except KeyboardInterrupt:
            # 捕获Ctrl+C中断
            print(" shutting down")
        finally:
            # 不管是否发生异常，都执行清理操作
            rospy.signal_shutdown("User requested shutdown")
            
    def image_callback_front(self, msg):
        # 将图片保存为JPG文件
        image_path = os.path.join(self.image_dir, f'image_{self.image_count_front}_front.jpg')
        cv2.imwrite(image_path, bridge.imgmsg_to_cv2(msg, "bgr8"))
        rospy.loginfo(f"Saved front image: {image_path}")
        self.image_count_front += 1

    def image_callback_left(self, msg):
        # 将图片保存为JPG文件
        image_path = os.path.join(self.image_dir, f'image_{self.image_count_left}_left.jpg')
        cv2.imwrite(image_path, bridge.imgmsg_to_cv2(msg, "bgr8"))
        rospy.loginfo(f"Saved left image: {image_path}")
        self.image_count_left += 1

    def image_callback_right(self, msg):
        # 将图片保存为JPG文件
        image_path = os.path.join(self.image_dir, f'image_{self.image_count_right}_right.jpg')
        cv2.imwrite(image_path, bridge.imgmsg_to_cv2(msg, "bgr8"))
        rospy.loginfo(f"Saved right image: {image_path}")
        self.image_count_right += 1


class OutputJointStateLogger:
    def __init__(self):
        rospy.init_node('output_joint_state_logger', anonymous=True)
        # 订阅/master/joint_left话题
        self.subscription_left = rospy.Subscriber('/master/joint_left', JointState, self.joint_state_callback_left)
        # 订阅/master/joint_right话题
        self.subscription_right = rospy.Subscriber('/master/joint_right', JointState, self.joint_state_callback_right)

        # 设置CSV文件路径
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file_path_left = log_dir + 'joint_states_left.csv'
        self.csv_file_path_right = log_dir + 'joint_states_right.csv'
        # 写入CSV文件头
        with open(self.csv_file_path_left, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'frame_id', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper'])
        # 写入CSV文件头
        with open(self.csv_file_path_right, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'frame_id', 'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper'])

        #spin thread
        self.spin_thread = Thread(target=self.spin)
        self.spin_thread.start()
    
    def spin(self):
        try:
            # 保持节点运行，直到有外部中断信号（如Ctrl+C）
            rospy.spin()
        except KeyboardInterrupt:
            # 捕获Ctrl+C中断
            print(" shutting down")
        finally:
            # 不管是否发生异常，都执行清理操作
            rospy.signal_shutdown("User requested shutdown")
            
    def joint_state_callback_left(self, msg):
        # 提取时间戳和位置信息
        timestamp = msg.header.stamp.to_sec()
        frame_id = msg.header.frame_id
        positions = ','.join(map(str, msg.position)).split(',')  # 用split(',')分割position string
        

        # 将数据写入CSV文件
        with open(self.csv_file_path_left, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_id, positions])
        rospy.loginfo(f"Logged left timestamp: {timestamp},frame_id:{frame_id}, positions: {positions}")

    def joint_state_callback_right(self, msg):
        # 提取时间戳和位置信息
        timestamp = msg.header.stamp.to_sec()
        frame_id = msg.header.frame_id
        positions = ','.join(map(str, msg.position)).split(',')

        # 将数据写入CSV文件
        with open(self.csv_file_path_right, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, frame_id, positions])
        rospy.loginfo(f"Logged right timestamp: {timestamp},frame_id:{frame_id}, positions: {positions}")



class ModelInputObservationSaver:
    def __init__(self, save_dir='model_input_observation'):
        self.save_dir = log_dir + save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def save_input_state_to_csv(self, state, filename='state.csv'):
        # 将状态（state）保存为CSV文件
        filepath = os.path.join(self.save_dir, filename)
        header=['left_joint0', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_gripper',
                                                  'right_joint0', 'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_gripper']
        
        df = pd.DataFrame([state], columns=header)
        
        if os.path.exists(filepath):
            # 如果文件已存在，则以追加模式添加新行而不添加header
            df.to_csv(filepath, mode='a', index=False, header=False)
        else:
            # 如果文件不存在，则创建新文件并写入header和新行
            df.to_csv(filepath, index=False, header=True)
    
    def save_output_action_to_csv(self, action, filename='action.csv'):
        # 将动作（action）保存为CSV文件
        filepath = os.path.join(self.save_dir, filename)
        header=['left_joint0', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_gripper',
                                                  'right_joint0', 'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_gripper']
        
        df = pd.DataFrame([action], columns=header)
        
        if os.path.exists(filepath):
            # 如果文件已存在，则以追加模式添加新行而不添加header
            df.to_csv(filepath, mode='a', index=False, header=False)
        else:
            # 如果文件不存在，则创建新文件并写入header和新行
            df.to_csv(filepath, index=False, header=True)
    
    def save_images_to_folder(self, images, frame_id,folder_name='images'):
        # 创建图像存储目录
        img_save_dir = os.path.join(self.save_dir, folder_name)
        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)
        
        for img_name, img_data in images.items():
            # 假定img_data形状为(c, h, w)，需要转换为(h, w, c)以供PIL使用
            img_array = np.transpose(img_data, (1, 2, 0))
            img = PImage.fromarray(img_array.astype(np.uint8))
            
            #update image name by frame_id
            img_name = img_name + '_' + str(frame_id)
            # 保存图片
            img_path = os.path.join(img_save_dir, f"{img_name}.png")
            img.save(img_path)




# main:设置参数，根据参数选择需要记录的logger,并启动logger,参数包括 input,output,img,all
if __name__ == '__main__':
    rospy.loginfo("Start logger node")
    rospy.loginfo("Please input the logger type: input/output/img/all")
    logger_type = input()
    if logger_type == 'input':
        input_joint_state_logger = InputJointStateLogger()
        input_img_logger = InputImgLogger()
    elif logger_type == 'output':
        output_joint_state_logger = OutputJointStateLogger()
    elif logger_type == 'img':
        input_img_logger = InputImgLogger()
    elif logger_type == 'all':
        input_joint_state_logger = InputJointStateLogger()
        output_joint_state_logger = OutputJointStateLogger()
        input_img_logger = InputImgLogger()
    else:
        rospy.loginfo("Invalid logger type")
    rospy.spin()