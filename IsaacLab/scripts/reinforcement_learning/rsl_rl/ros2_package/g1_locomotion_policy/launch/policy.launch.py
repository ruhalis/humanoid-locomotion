"""Launch file for G1 locomotion policy node."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package share directory
    pkg_share = get_package_share_directory('g1_locomotion_policy')

    # Declare launch arguments
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        default_value=os.path.join(pkg_share, 'models', 'policy.pt'),
        description='Path to the policy .pt file'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Device for inference (cpu or cuda:0)'
    )

    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='50.0',
        description='Control loop frequency in Hz'
    )

    # Policy node
    policy_node = Node(
        package='g1_locomotion_policy',
        executable='policy_node',
        name='g1_policy_node',
        output='screen',
        parameters=[{
            'policy_path': LaunchConfiguration('policy_path'),
            'device': LaunchConfiguration('device'),
            'control_frequency': LaunchConfiguration('control_frequency'),
            'action_scale': 0.5,
            'imu_topic': '/imu/data',
            'joint_states_topic': '/joint_states',
            'cmd_vel_topic': '/cmd_vel',
            'joint_targets_topic': '/joint_position_targets',
        }],
    )

    return LaunchDescription([
        policy_path_arg,
        device_arg,
        control_frequency_arg,
        policy_node,
    ])
