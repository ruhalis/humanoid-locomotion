# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .rough_env_cfg import G1RoughEnvCfg

##
# Pre-defined configs
##


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None


##
# G1 Inspire Flat Environment Configuration
# Incorporates recommendations from literature review:
# - Domain randomization (mass, friction, pushes, PD gains)
# - Tighter velocity tracking (std=0.25)
# - Single-foot-contact reward for proper bipedal gait
# - Stronger vertical velocity penalty
# - Base height tracking reward
# - Reduced action scale (0.25)
##


@configclass
class G1InspireRewards(RewardsCfg):
    """Reward terms for G1 Inspire locomotion, tuned per literature best practices."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- Task rewards: strong tracking signal to overcome penalty local minimum --
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # -- Gait shaping: the essential terms for bipedal walking --
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # -- Balance and posture --
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": 0.74},
    )

    # -- Smoothness penalties --
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )

    # -- Joint limit and deviation penalties --
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


@configclass
class G1InspireFlatEnvCfg(G1RoughEnvCfg):
    """G1 Inspire flat-ground locomotion environment with literature-recommended settings.

    Key improvements over the base G1FlatEnvCfg:
    - Uses G1_INSPIRE_LOCOMOTION_CFG (53 joints: 29 body + 24 hand)
    - Domain randomization enabled (mass, friction, pushes)
    - Tighter velocity tracking (std=0.25 vs 0.5)
    - Stronger vertical velocity penalty (-2.0 vs -0.2)
    - Base height tracking reward (target 0.74m)
    - Reduced action scale (0.25 vs 0.5)
    """

    rewards: G1InspireRewards = G1InspireRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -- Flat terrain (robot is G1_MINIMAL_CFG from parent G1RoughEnvCfg) --
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # -- Action scale: 0.25 rad (literature standard, down from 0.5) --
        self.actions.joint_pos.scale = 0.25

        # -- Re-apply reward weights that parent __post_init__ may have overridden --
        # Parent G1RoughEnvCfg sets lin_vel_z_l2=0.0, action_rate_l2=-0.005, etc.
        # We want our G1InspireRewards class-level values instead.
        self.rewards.lin_vel_z_l2.weight = -0.5
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # -- Domain randomization: re-enable everything --
        # Mass randomization: ±20% of torso mass
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-3.0, 3.0),
                "operation": "add",
            },
        )
        # Friction randomization: 0.3-1.5
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.5)
        # Random pushes during training: 50-200N equivalent via velocity impulses
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
        # External force/torque on torso
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.base_external_force_torque.params["force_range"] = (-50.0, 50.0)
        self.events.base_external_force_torque.params["torque_range"] = (-10.0, 10.0)
        # Joint reset with slight randomization
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        # Initial velocity randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        }

        # -- Commands: minimum forward vel > 0 so robot must always walk --
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # -- Terminations --
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"


@configclass
class G1InspireFlatEnvCfg_PLAY(G1InspireFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
