

from typing import Optional, List
import dm_env
import numpy as np

#this is a ROS package
import rospy
import cv2
import torch

from threading import Thread
from  examples.piper_real import ros_oper as _ros_oper

#this is  a camera name list for config
CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

ros_config = {
    "img_front_topic": "/camera_f/color/image_raw",
    "img_left_topic": "/camera_l/color/image_raw",
    "img_right_topic": "/camera_r/color/image_raw",

    "img_front_depth_topic": "/camera_f/depth/image_raw",
    "img_left_depth_topic": "/camera_l/depth/image_raw",
    "img_right_depth_topic": "/camera_r/depth/image_raw",

    "puppet_arm_left_topic": "/puppet/joint_left",
    "puppet_arm_right_topic": "/puppet/joint_right",

    "puppet_arm_left_cmd_topic": "/master/joint_left",
    "puppet_arm_right_cmd_topic": "/master/joint_right",

    "robot_base_topic": "/odom_raw",
    "robot_base_cmd_topic": "/cmd_vel",
    "use_robot_base": False,

    "publish_rate": 30,
    "ctrl_freq": 25,
    "state_dim": 14,
    "chunk_size": 64,
    "arm_steps_length": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2],

    "use_actions_interpolation": True,
    "use_depth_image": False,

    "disable_puppet_arm": False,
    "disable_robot_base": False,
}

# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args["arm_steps_length"]), np.array(args["arm_steps_length"])), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

class PiperRealEnv:
    """
    Environment for real robot bi-manual manipulation
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(self, init_node, *, reset_pos:Optional[List[float]] = None, setup_robots: bool = False):
        if init_node:
            rospy.init_node('joint_state_publisher_pi0_debug', anonymous=True)
            self.spin_thread = Thread(target=self.spin)
            self.spin_thread.start()
        self._reset_pos = reset_pos
        self.ros_operator = _ros_oper.RosOperator(ros_config)
        self.rate = rospy.Rate(ros_config["publish_rate"])
        # self.action = None
        self.pre_action = np.zeros(ros_config['state_dim'])

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

    def setup_robots(self):
        pass

    def reset(self,*, fake=False):
        if not fake:
            left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                     -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
            right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656,
                      -0.00476837158203125, -0.00209808349609375, 3.557830810546875]
            left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                     -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
            right1 = [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156,
                      -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
            self.ros_operator.puppet_arm_publish_continuous(left0, right0)
            input("Press enter to continue")
            self.ros_operator.puppet_arm_publish_continuous(left1, right1)

            # Initialize the previous action to be the initial robot state

            self.pre_action[:14] = np.array(
                [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875,
                 0.00095367431640625, -0.3393220901489258] +
                [-0.00133514404296875, 0.00247955322265625, 0.01583099365234375, -0.032616615295410156,
                 -0.00286102294921875,
                 0.00095367431640625, -0.3397035598754883]
            )

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )

    def get_reward(self):
        return 0

    def get_observation(self):
        def jpeg_mapping(img):
            img = cv2.imencode('.jpg', img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img


        print_flag = True


        while True and not rospy.is_shutdown():
            result = self.ros_operator.get_frame()
            
            if not result:
                if print_flag:
                    print("syn fail when get_ros_observation")
                    print_flag = False
                self.rate.sleep()
                continue
            print_flag = True
            print(f"get_ros_observation success") 
            
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
             puppet_arm_left, puppet_arm_right, robot_base) = result
            # print(f"sync success when get_ros_observation")
            
            img_front = jpeg_mapping(img_front)
            img_left = jpeg_mapping(img_left)
            img_right = jpeg_mapping(img_right)

            qpos = np.concatenate(
                (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            # qpos = torch.from_numpy(qpos).float().cuda()
            # qpos = qpos.unsqueeze(0)

            obs = {
                    'qpos': qpos,
                    'images':
                        {
                            "cam_high": img_front,
                            "cam_right_wrist": img_right,
                            "cam_left_wrist": img_left,
                        },
                }
            
            return obs


    def step(self, action):
        interp_actions = None

        if ros_config["use_actions_interpolation"]:
            # print(f"Time {t}, pre {pre_action}, act {action}")
            interp_actions = interpolate_action(ros_config, self.pre_action, action)
        else:
            interp_actions = action[np.newaxis, :]

        # Execute the interpolated actions one by one
        for act in interp_actions:
            state_len = int(len(act) / 2)
            left_action = act[:state_len]
            right_action = act[state_len:]

            if not ros_config["disable_puppet_arm"]:
                self.ros_operator.puppet_arm_publish(left_action, right_action)  # puppet_arm_publish_continuous_thread

            if ros_config["use_robot_base"]:
                vel_action = act[14:16]
                self.ros_operator.robot_base_publish(vel_action)
            self.rate.sleep()

        self.pre_action = action.copy()

        # get next frame obs
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
        )




def make_real_env(init_node, *, reset_position: Optional[List[float]] = None, setup_robots: bool = True) -> PiperRealEnv:
    return PiperRealEnv(init_node, reset_pos=reset_position, setup_robots=setup_robots)
