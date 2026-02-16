#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fullbody controller for G1 humanoid robot with Inspire hands (29 DoF body + 24 hand DoF = 53 joints).

Adapted from h1_fullbody_controller.py for the Unitree G1 robot trained with
Isaac Lab using the Isaac-Velocity-Flat-G1-Inspire-v0 task.

Key differences from H1:
- 53 joints total (vs 19 for H1) including finger joints
- Different joint ordering following the USD kinematic tree
- Different default standing pose (init height 0.74m vs 1.05m)
- Observation vector is 171-dimensional (vs 69 for H1)
- Policy trained with Isaac Lab (isaaclab.*) not omni.isaac.lab
"""

import rclpy
import torch
import numpy as np
import io
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer


class G1FullbodyController(Node):
    """Fullbody controller for G1 humanoid robot with Inspire hands.

    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a neural network policy, and publishes
    joint commands for controlling the G1 robot's movements.

    The policy controls all 53 joints (legs, arms, waist, and hands) using
    position control with PD actuators.
    """

    # -------------------------------------------------------------------------
    # Joint configuration for G1 with Inspire hands (53 joints)
    # Order follows the USD kinematic tree traversal in Isaac Lab.
    # -------------------------------------------------------------------------
    JOINT_NAMES = [
        "waist_pitch_joint",           # [0]  Waist
        "left_shoulder_pitch_joint",   # [1]  Left Arm
        "right_shoulder_pitch_joint",  # [2]  Right Arm
        "waist_roll_joint",            # [3]  Waist
        "left_shoulder_roll_joint",    # [4]  Left Arm
        "right_shoulder_roll_joint",   # [5]  Right Arm
        "waist_yaw_joint",             # [6]  Waist
        "left_shoulder_yaw_joint",     # [7]  Left Arm
        "right_shoulder_yaw_joint",    # [8]  Right Arm
        "left_hip_pitch_joint",        # [9]  Left Leg
        "right_hip_pitch_joint",       # [10] Right Leg
        "left_elbow_joint",            # [11] Left Arm
        "right_elbow_joint",           # [12] Right Arm
        "left_hip_roll_joint",         # [13] Left Leg
        "right_hip_roll_joint",        # [14] Right Leg
        "left_wrist_roll_joint",       # [15] Left Arm
        "right_wrist_roll_joint",      # [16] Right Arm
        "left_hip_yaw_joint",          # [17] Left Leg
        "right_hip_yaw_joint",         # [18] Right Leg
        "left_wrist_pitch_joint",      # [19] Left Arm
        "right_wrist_pitch_joint",     # [20] Right Arm
        "left_knee_joint",             # [21] Left Leg
        "right_knee_joint",            # [22] Right Leg
        "left_wrist_yaw_joint",        # [23] Left Arm
        "right_wrist_yaw_joint",       # [24] Right Arm
        "left_ankle_pitch_joint",      # [25] Left Leg
        "right_ankle_pitch_joint",     # [26] Right Leg
        "L_index_proximal_joint",      # [27] Left Hand
        "L_middle_proximal_joint",     # [28] Left Hand
        "L_pinky_proximal_joint",      # [29] Left Hand
        "L_ring_proximal_joint",       # [30] Left Hand
        "L_thumb_proximal_yaw_joint",  # [31] Left Hand
        "R_index_proximal_joint",      # [32] Right Hand
        "R_middle_proximal_joint",     # [33] Right Hand
        "R_pinky_proximal_joint",      # [34] Right Hand
        "R_ring_proximal_joint",       # [35] Right Hand
        "R_thumb_proximal_yaw_joint",  # [36] Right Hand
        "left_ankle_roll_joint",       # [37] Left Leg
        "right_ankle_roll_joint",      # [38] Right Leg
        "L_index_intermediate_joint",  # [39] Left Hand
        "L_middle_intermediate_joint", # [40] Left Hand
        "L_pinky_intermediate_joint",  # [41] Left Hand
        "L_ring_intermediate_joint",   # [42] Left Hand
        "L_thumb_proximal_pitch_joint",# [43] Left Hand
        "R_index_intermediate_joint",  # [44] Right Hand
        "R_middle_intermediate_joint", # [45] Right Hand
        "R_pinky_intermediate_joint",  # [46] Right Hand
        "R_ring_intermediate_joint",   # [47] Right Hand
        "R_thumb_proximal_pitch_joint",# [48] Right Hand
        "L_thumb_intermediate_joint",  # [49] Left Hand
        "R_thumb_intermediate_joint",  # [50] Right Hand
        "L_thumb_distal_joint",        # [51] Left Hand
        "R_thumb_distal_joint",        # [52] Right Hand
    ]

    NUM_JOINTS = 53

    # Default standing pose (non-zero values only)
    # From G1_INSPIRE_LOCOMOTION_CFG init_state in Isaac Lab
    DEFAULT_JOINT_POS_DICT = {
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_pitch_joint": 0.35,
        "left_shoulder_roll_joint": 0.16,
        "right_shoulder_roll_joint": -0.16,
        "left_hip_pitch_joint": -0.20,
        "right_hip_pitch_joint": -0.20,
        "left_elbow_joint": 0.87,
        "right_elbow_joint": 0.87,
        "left_knee_joint": 0.42,
        "right_knee_joint": 0.42,
        "left_ankle_pitch_joint": -0.23,
        "right_ankle_pitch_joint": -0.23,
    }

    # Observation vector layout (171 dimensions total):
    #   [0:3]     base_lin_vel      (3)
    #   [3:6]     base_ang_vel      (3)
    #   [6:9]     projected_gravity (3)
    #   [9:12]    velocity_commands  (3)
    #   [12:65]   joint_pos_rel     (53)
    #   [65:118]  joint_vel_rel     (53)
    #   [118:171] last_action       (53)
    OBS_DIM = 171

    def __init__(self):
        """Initialize the G1 fullbody controller node."""
        super().__init__('g1_fullbody_controller')

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', 'policy/g1_policy.pt')
        self.set_parameters(
            [rclpy.parameter.Parameter(
                'use_sim_time',
                rclpy.Parameter.Type.BOOL,
                True
            )]
        )

        self._logger = self.get_logger()

        # Configure QoS profile for simulation
        sim_qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            history=rclpy.qos.HistoryPolicy.KEEP_ALL,
        )

        # Create subscription for velocity commands
        self._cmd_vel_subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self._cmd_vel_callback,
            qos_profile=10)

        # Create publisher for joint commands
        self._joint_publisher = self.create_publisher(
            JointState,
            'joint_command',
            qos_profile=sim_qos_profile)

        # Setup synchronized subscribers for IMU and joint state data
        self._imu_sub_filter = Subscriber(
            self,
            Imu,
            'imu',
            qos_profile=sim_qos_profile,
        )
        self._joint_states_sub_filter = Subscriber(
            self,
            JointState,
            'joint_states',
            qos_profile=sim_qos_profile,
        )
        queue_size = 10
        subscribers = [self._joint_states_sub_filter, self._imu_sub_filter]

        # Time synchronizer to ensure joint state and IMU data are processed
        # together
        self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync.registerCallback(self._tick)

        # Load neural network policy
        self.policy_path = self.get_parameter('policy_path').value
        self.load_policy()

        # Initialize state variables
        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._action_scale = 0.5  # Scale factor for policy output (from env.yaml)
        self._previous_action = np.zeros(self.NUM_JOINTS)
        self._policy_counter = 0
        self._decimation = 4  # Run policy every 4 ticks (from env.yaml decimation=4)
        self._last_tick_time = self.get_clock().now().nanoseconds * 1e-9
        self._lin_vel_b = np.zeros(3)  # Linear velocity in body frame
        self._dt = 0.0  # Time delta between ticks

        # Build default position array from dictionary
        self.default_pos = np.zeros(self.NUM_JOINTS)
        for i, name in enumerate(self.JOINT_NAMES):
            if name in self.DEFAULT_JOINT_POS_DICT:
                self.default_pos[i] = self.DEFAULT_JOINT_POS_DICT[name]

        self._logger.info("Initializing G1FullbodyController")
        self._logger.info(f"  Number of joints: {self.NUM_JOINTS}")
        self._logger.info(f"  Observation dim: {self.OBS_DIM}")
        self._logger.info(f"  Action scale: {self._action_scale}")
        self._logger.info(f"  Decimation: {self._decimation}")

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _tick(self, joint_state: JointState, imu: Imu):
        """Process synchronized joint state and IMU data to generate robot commands.

        This method is called whenever new joint state and IMU data are available.
        It computes the policy's action and publishes the resulting joint commands.

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data (orientation, angular velocity, acceleration)
        """
        # Reset if time jumped backwards (most likely due to sim time reset)
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self._last_tick_time:
            self._logger.error(
                f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.'
            )

        # Calculate time delta since last tick
        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now

        # Run the control policy
        self.forward(joint_state, imu)

        # Prepare and publish the joint command message
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = list(self.JOINT_NAMES)

        # Compute final joint positions by adding scaled actions to default positions
        action_pos = self.default_pos + self.action * self._action_scale
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(self.NUM_JOINTS).tolist()
        self._joint_command.effort = np.zeros(self.NUM_JOINTS).tolist()
        self._joint_publisher.publish(self._joint_command)

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        """Compute the policy observation vector from robot state.

        Constructs a 171-dimensional observation vector from robot sensor data:
        - Linear velocity (body frame)               [0:3]    (3)
        - Angular velocity (body frame)               [3:6]    (3)
        - Gravity direction (body frame)              [6:9]    (3)
        - Command velocity                            [9:12]   (3)
        - Joint positions relative to default         [12:65]  (53)
        - Joint velocities                            [65:118] (53)
        - Previous action                             [118:171](53)

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data

        Returns:
            np.ndarray: 171-dimensional observation vector for the policy
        """
        # Extract quaternion orientation from IMU
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix
        # (transpose for body to inertial frame)
        R_BI = self.quat_to_rot_matrix(quat_array).T

        # Extract linear acceleration and integrate to estimate velocity
        lin_acc_b = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z
        ])

        # Simple integration to estimate velocity
        self._lin_vel_b = lin_acc_b * self._dt + self._lin_vel_b

        # Extract angular velocity
        ang_vel_b = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z
        ])

        # Calculate gravity direction in body frame
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # Initialize observation vector
        obs = np.zeros(self.OBS_DIM)

        # Fill observation vector components:
        # Base linear velocity (3)
        obs[0:3] = self._lin_vel_b

        # Base angular velocity (3)
        obs[3:6] = ang_vel_b

        # Gravity direction (3)
        obs[6:9] = gravity_b

        # Velocity commands (3)
        cmd_vel = [
            self._cmd_vel.linear.x,
            self._cmd_vel.linear.y,
            self._cmd_vel.angular.z
        ]
        obs[9:12] = np.array(cmd_vel)

        # Joint states (53 positions + 53 velocities)
        current_joint_pos = np.zeros(self.NUM_JOINTS)
        current_joint_vel = np.zeros(self.NUM_JOINTS)

        # Map joint states from message to our ordered arrays
        for i, name in enumerate(self.JOINT_NAMES):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        # Store joint positions relative to default pose
        obs[12:65] = current_joint_pos - self.default_pos

        # Store joint velocities
        obs[65:118] = current_joint_vel

        # Store previous actions
        obs[118:171] = self._previous_action

        return obs

    def _compute_action(self, obs):
        """Run the neural network policy to compute an action from the observation.

        Args:
            obs: 171-dim observation vector containing robot state information

        Returns:
            np.ndarray: 53-dim action vector containing joint position adjustments
        """
        # Run inference with the PyTorch policy
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Process sensor data and compute control actions.

        This combines observation computation and policy evaluation.
        The policy is run at a reduced rate (decimation) to save computation.

        Args:
            joint_state: Current joint positions and velocities
            imu: Current IMU data
        """
        # Compute observation from current state
        obs = self._compute_observation(joint_state, imu)

        # Run policy at reduced frequency (every _decimation ticks)
        if self._policy_counter % self._decimation == 0:
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()
        self._policy_counter += 1

    def quat_to_rot_matrix(self, quat: np.ndarray) -> np.ndarray:
        """Convert input quaternion to rotation matrix.

        Args:
            quat (np.ndarray): Input quaternion (w, x, y, z).

        Returns:
            np.ndarray: A 3x3 rotation matrix.
        """
        q = np.array(quat, dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < 1e-10:
            return np.identity(3)
        q *= np.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array(
            (
                (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]),
                (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]),
                (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]),
            ),
            dtype=np.float64,
        )

    def load_policy(self):
        """Load the neural network policy from the specified path."""
        self._logger.info(f"Loading policy from: {self.policy_path}")
        # Load policy from file to io.BytesIO object
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        # Load TorchScript model from buffer
        self.policy = torch.jit.load(buffer)
        self._logger.info("Policy loaded successfully")

    def _get_stamp_prefix(self) -> str:
        """Create a timestamp prefix for logging with both system and ROS time.

        Returns:
            str: Formatted timestamp string with system and ROS time
        """
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'

    def header_time_in_seconds(self, header) -> float:
        """Convert a ROS message header timestamp to seconds.

        Args:
            header: ROS message header containing timestamp

        Returns:
            float: Time in seconds
        """
        return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(args=None):
    """Main function to initialize and run the G1 fullbody controller node."""
    rclpy.init(args=args)
    node = G1FullbodyController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
