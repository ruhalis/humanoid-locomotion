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

import rclpy
import torch
import numpy as np
import io
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from message_filters import Subscriber, TimeSynchronizer


class Go2Controller(Node):
    """Controller for Unitree Go2 quadruped robot.

    This ROS 2 node subscribes to velocity commands and synchronized joint/IMU
    data, processes the data through a neural network policy trained in IsaacLab,
    and publishes joint commands for controlling the Go2 robot's movements.

    The policy was trained using IsaacLab with the Isaac-Velocity-Flat-Unitree-Go2-v0
    environment configuration.
    """

    NUM_JOINTS = 12
    # Observation: 3 lin_vel + 3 ang_vel + 3 gravity + 3 cmd_vel + 12 joint_pos + 12 joint_vel + 12 actions
    OBS_DIM = 48

    def __init__(self):
        """Initialize the Go2 controller node."""
        super().__init__('go2_controller')

        # Declare and set parameters
        self.declare_parameter('publish_period_ms', 5)
        self.declare_parameter('policy_path', 'policy/go2_policy.pt')
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

        # Time synchronizer to ensure joint state and IMU data are processed together
        self.sync = TimeSynchronizer(subscribers, queue_size)
        self.sync.registerCallback(self._tick)

        # Load neural network policy
        self.declare_parameter('debug_default_only', False)
        self._debug_default_only = self.get_parameter('debug_default_only').value
        self.policy_path = self.get_parameter('policy_path').value
        if not self._debug_default_only:
            self.load_policy()

        # Initialize state variables
        self._joint_state = JointState()
        self._joint_command = JointState()
        self._cmd_vel = Twist()
        self._imu = Imu()
        self._action_scale = 0.25  # Scale factor for policy output (matches IsaacLab config)
        self._previous_action = np.zeros(self.NUM_JOINTS)
        self._policy_counter = 0
        self._decimation = 4  # Run policy every 4 ticks (matches IsaacLab config)
        self._last_tick_time = None  # Will be set on first tick
        self._lin_vel_b = np.zeros(3)  # Linear velocity in body frame
        self._dt = 0.0  # Time delta between ticks

        # Joint names in the order from Isaac Sim articulation (must match training order)
        self.joint_names = [
            'FL_hip_joint',
            'FR_hip_joint',
            'RL_hip_joint',
            'RR_hip_joint',
            'FL_thigh_joint',
            'FR_thigh_joint',
            'RL_thigh_joint',
            'RR_thigh_joint',
            'FL_calf_joint',
            'FR_calf_joint',
            'RL_calf_joint',
            'RR_calf_joint',
        ]

        # Default joint positions representing the nominal stance
        # These values match the init_state from UNITREE_GO2_CFG
        self.default_pos = np.array([
            0.1,    # FL_hip_joint
            -0.1,   # FR_hip_joint
            0.1,    # RL_hip_joint
            -0.1,   # RR_hip_joint
            0.8,    # FL_thigh_joint
            0.8,    # FR_thigh_joint
            1.0,    # RL_thigh_joint
            1.0,    # RR_thigh_joint
            -1.5,   # FL_calf_joint
            -1.5,   # FR_calf_joint
            -1.5,   # RL_calf_joint
            -1.5,   # RR_calf_joint
        ])

        self._logger.info("Initializing Go2Controller")
        self._logger.info(f"Number of joints: {self.NUM_JOINTS}")
        self._logger.info(f"Observation dimension: {self.OBS_DIM}")

    def _cmd_vel_callback(self, msg):
        """Store the latest velocity command."""
        self._cmd_vel = msg

    def _tick(self, joint_state: JointState, imu: Imu):
        """Process synchronized joint state and IMU data to generate robot commands."""
        now = self.get_clock().now().nanoseconds * 1e-9

        # Skip first tick to establish timing baseline
        if self._last_tick_time is None:
            self._last_tick_time = now
            self._dt = 0.0
            self._logger.info("First tick received, establishing timing baseline.")
            # Send default pose on first tick
            self._joint_command.header.stamp = self.get_clock().now().to_msg()
            self._joint_command.name = self.joint_names
            self._joint_command.position = self.default_pos.tolist()
            self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
            self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
            self._joint_publisher.publish(self._joint_command)
            self.action = np.zeros(self.NUM_JOINTS)
            return

        # Reset if time jumped backwards (most likely due to sim time reset)
        if now < self._last_tick_time:
            self._logger.error(
                f'{self._get_stamp_prefix()} Time jumped backwards. Resetting.'
            )
            self._lin_vel_b = np.zeros(3)

        # Calculate time delta since last tick
        self._dt = (now - self._last_tick_time)
        self._last_tick_time = now

        # Run the control policy
        if self._debug_default_only:
            self.action = np.zeros(self.NUM_JOINTS)
        else:
            self.forward(joint_state, imu)

        # Prepare and publish the joint command message
        self._joint_command.header.stamp = self.get_clock().now().to_msg()
        self._joint_command.name = self.joint_names

        # Compute final joint positions by adding scaled actions to default positions
        action_pos = self.default_pos + self.action * self._action_scale
        self._joint_command.position = action_pos.tolist()
        self._joint_command.velocity = np.zeros(len(self.joint_names)).tolist()
        self._joint_command.effort = np.zeros(len(self.joint_names)).tolist()
        self._joint_publisher.publish(self._joint_command)

        # Log first few ticks for debugging
        if self._policy_counter <= 5:
            self._logger.info(f"Tick {self._policy_counter}: dt={self._dt:.4f}")
            self._logger.info(f"  lin_vel_b={np.round(self._lin_vel_b, 3)}")
            self._logger.info(f"  action={np.round(self.action, 3)}")
            self._logger.info(f"  action_pos={np.round(action_pos, 3)}")

    def _compute_observation(self, joint_state: JointState, imu: Imu):
        """Compute the policy observation vector from robot state.

        Constructs a 48-dimensional observation vector from robot sensor data:
        - Linear velocity (body frame): 3
        - Angular velocity (body frame): 3
        - Gravity direction (body frame): 3
        - Command velocity: 3
        - Joint positions (relative to default): 12
        - Joint velocities: 12
        - Previous action: 12
        """
        # Extract quaternion orientation from IMU
        quat_I = imu.orientation
        quat_array = np.array([quat_I.w, quat_I.x, quat_I.y, quat_I.z])

        # Convert quaternion to rotation matrix (transpose for body to inertial frame)
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

        # Base linear velocity (3)
        obs[:3] = self._lin_vel_b

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

        # Joint states (12 positions + 12 velocities)
        current_joint_pos = np.zeros(self.NUM_JOINTS)
        current_joint_vel = np.zeros(self.NUM_JOINTS)

        # Map joint states from message to our ordered arrays
        for i, name in enumerate(self.joint_names):
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                current_joint_pos[i] = joint_state.position[idx]
                current_joint_vel[i] = joint_state.velocity[idx]

        # Store joint positions relative to default pose
        obs[12:24] = current_joint_pos - self.default_pos

        # Store joint velocities
        obs[24:36] = current_joint_vel

        # Store previous actions
        obs[36:48] = self._previous_action

        return obs

    def _compute_action(self, obs):
        """Run the neural network policy to compute an action from the observation."""
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def forward(self, joint_state: JointState, imu: Imu):
        """Process sensor data and compute control actions."""
        obs = self._compute_observation(joint_state, imu)

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
        with open(self.policy_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        self.policy = torch.jit.load(buffer)
        self._logger.info(f"Loaded policy from: {self.policy_path}")

    def _get_stamp_prefix(self) -> str:
        """Create a timestamp prefix for logging."""
        now = time.time()
        now_ros = self.get_clock().now().nanoseconds / 1e9
        return f'[{now}][{now_ros}]'

    def header_time_in_seconds(self, header) -> float:
        """Convert a ROS message header timestamp to seconds."""
        return header.stamp.sec + header.stamp.nanosec * 1e-9


def main(args=None):
    """Main function to initialize and run the Go2 controller node."""
    rclpy.init(args=args)
    node = Go2Controller()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
