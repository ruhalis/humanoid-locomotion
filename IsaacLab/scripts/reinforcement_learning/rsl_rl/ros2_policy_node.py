#!/usr/bin/env python3
"""ROS2 node for G1 humanoid locomotion policy inference.

This node runs the trained locomotion policy and publishes joint position targets.

Subscriptions:
    - /imu/data (sensor_msgs/Imu): IMU data for base velocity and orientation
    - /joint_states (sensor_msgs/JointState): Current joint positions and velocities
    - /cmd_vel (geometry_msgs/Twist): Velocity commands

Publications:
    - /joint_position_targets (sensor_msgs/JointState): Target joint positions

Usage:
    1. Copy policy_inference.py and this file to your ROS2 package
    2. Copy the exported policy.pt file
    3. Run: ros2 run your_package ros2_policy_node --ros-args -p policy_path:=/path/to/policy.pt

Requirements:
    - torch
    - numpy
    - rclpy
    - sensor_msgs
    - geometry_msgs
"""

import numpy as np
import torch
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist

# Import the PolicyInference class (copy policy_inference.py to your package)
from policy_inference import PolicyInference, G1InspireJointConfig


class G1PolicyNode(Node):
    """ROS2 node for G1 locomotion policy inference."""

    def __init__(self):
        super().__init__('g1_policy_node')

        # Declare parameters
        self.declare_parameter('policy_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('action_scale', 0.5)
        self.declare_parameter('control_frequency', 50.0)  # Hz
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('joint_targets_topic', '/joint_position_targets')

        # Get parameters
        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        device = self.get_parameter('device').get_parameter_value().string_value
        action_scale = self.get_parameter('action_scale').get_parameter_value().double_value
        control_freq = self.get_parameter('control_frequency').get_parameter_value().double_value

        if not policy_path:
            self.get_logger().error('policy_path parameter is required!')
            raise ValueError('policy_path parameter is required')

        # Initialize policy
        self.get_logger().info(f'Loading policy from: {policy_path}')
        self.policy = PolicyInference(
            policy_path=policy_path,
            device=device,
            action_scale=action_scale,
        )

        # State buffers
        self.base_lin_vel = np.zeros(3)
        self.base_ang_vel = np.zeros(3)
        self.base_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.projected_gravity = np.array([0.0, 0.0, -1.0])
        self.velocity_cmd = np.zeros(3)
        self.joint_pos = self.policy.get_default_joint_positions()
        self.joint_vel = np.zeros(G1InspireJointConfig.NUM_JOINTS)

        # Joint name to index mapping for incoming joint states
        self.joint_name_to_policy_idx = {
            name: i for i, name in enumerate(G1InspireJointConfig.JOINT_NAMES)
        }

        # Data received flags
        self.imu_received = False
        self.joints_received = False

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        joint_states_topic = self.get_parameter('joint_states_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback, sensor_qos
        )
        self.joint_states_sub = self.create_subscription(
            JointState, joint_states_topic, self.joint_states_callback, sensor_qos
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_vel_callback, 10
        )

        # Publisher
        joint_targets_topic = self.get_parameter('joint_targets_topic').get_parameter_value().string_value
        self.joint_targets_pub = self.create_publisher(JointState, joint_targets_topic, 10)

        # Control timer
        control_period = 1.0 / control_freq
        self.control_timer = self.create_timer(control_period, self.control_callback)

        self.get_logger().info(f'G1 Policy Node initialized')
        self.get_logger().info(f'  Control frequency: {control_freq} Hz')
        self.get_logger().info(f'  IMU topic: {imu_topic}')
        self.get_logger().info(f'  Joint states topic: {joint_states_topic}')
        self.get_logger().info(f'  Cmd vel topic: {cmd_vel_topic}')
        self.get_logger().info(f'  Joint targets topic: {joint_targets_topic}')

    def imu_callback(self, msg: Imu):
        """Process IMU data."""
        # Extract angular velocity (in base frame)
        self.base_ang_vel[0] = msg.angular_velocity.x
        self.base_ang_vel[1] = msg.angular_velocity.y
        self.base_ang_vel[2] = msg.angular_velocity.z

        # Extract linear acceleration and estimate velocity
        # Note: For proper velocity estimation, you may need sensor fusion
        # This is a simplified version - real implementation should use
        # state estimation (e.g., EKF with leg odometry)
        # For now, we use linear acceleration as a proxy (not accurate!)
        # You should replace this with proper velocity estimation
        self.base_lin_vel[0] = msg.linear_acceleration.x * 0.02  # Simple integration approximation
        self.base_lin_vel[1] = msg.linear_acceleration.y * 0.02
        self.base_lin_vel[2] = msg.linear_acceleration.z * 0.02

        # Extract orientation (quaternion: w, x, y, z)
        self.base_orientation[0] = msg.orientation.w
        self.base_orientation[1] = msg.orientation.x
        self.base_orientation[2] = msg.orientation.y
        self.base_orientation[3] = msg.orientation.z

        # Compute projected gravity
        self.projected_gravity = PolicyInference.compute_projected_gravity(self.base_orientation)

        self.imu_received = True

    def joint_states_callback(self, msg: JointState):
        """Process joint states."""
        for i, name in enumerate(msg.name):
            if name in self.joint_name_to_policy_idx:
                idx = self.joint_name_to_policy_idx[name]
                if i < len(msg.position):
                    self.joint_pos[idx] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_vel[idx] = msg.velocity[i]

        self.joints_received = True

    def cmd_vel_callback(self, msg: Twist):
        """Process velocity commands."""
        self.velocity_cmd[0] = msg.linear.x   # Forward velocity
        self.velocity_cmd[1] = msg.linear.y   # Lateral velocity
        self.velocity_cmd[2] = msg.angular.z  # Angular velocity (yaw rate)

    def control_callback(self):
        """Run policy inference and publish joint targets."""
        # Check if we have received sensor data
        if not self.imu_received:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000,  # Warn every 5 seconds
                'Waiting for IMU data...'
            )
            return

        if not self.joints_received:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000,
                'Waiting for joint states...'
            )
            return

        # Run policy inference
        try:
            joint_targets = self.policy.infer(
                base_lin_vel=self.base_lin_vel,
                base_ang_vel=self.base_ang_vel,
                projected_gravity=self.projected_gravity,
                velocity_cmd=self.velocity_cmd,
                joint_pos=self.joint_pos,
                joint_vel=self.joint_vel,
            )
        except Exception as e:
            self.get_logger().error(f'Policy inference failed: {e}')
            return

        # Publish joint targets
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = G1InspireJointConfig.JOINT_NAMES
        msg.position = joint_targets.tolist()

        self.joint_targets_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = G1PolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
