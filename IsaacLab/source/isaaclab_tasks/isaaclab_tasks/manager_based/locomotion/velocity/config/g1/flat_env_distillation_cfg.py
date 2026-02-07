# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Inspire distillation environment configuration.

This module defines environment configurations for teacher-student distillation,
where the teacher uses privileged observations (including base_lin_vel) and
the student uses only sensor-available observations (no base_lin_vel).
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import G1InspireFlatEnvCfg


@configclass
class StudentPolicyCfg(ObsGroup):
    """Student observations without privileged terms (no base_lin_vel).
    
    This observation group contains only sensor-available observations:
    - Angular velocity (from IMU)
    - Projected gravity (from IMU orientation)
    - Velocity commands (user input)
    - Joint positions (from encoders)
    - Joint velocities (from encoders)
    - Previous actions (known)
    
    Note: base_lin_vel is removed as it cannot be directly measured by real sensors.
    """

    # observation terms (order preserved) - NO base_lin_vel
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class TeacherPolicyCfg(ObsGroup):
    """Teacher observations with privileged terms (includes base_lin_vel).
    
    This observation group contains all observations including privileged ones:
    - Linear velocity (ground truth from simulation - PRIVILEGED)
    - Angular velocity (from IMU)
    - Projected gravity (from IMU orientation)
    - Velocity commands (user input)
    - Joint positions (from encoders)
    - Joint velocities (from encoders)
    - Previous actions (known)
    """

    # observation terms (order preserved) - WITH base_lin_vel
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class G1InspireDistillationObservationsCfg:
    """Observation configuration for distillation with both teacher and student groups."""

    # The 'policy' group is used by the RL env for the default policy input
    # For distillation, we use student_policy for the student and teacher for the teacher
    policy: StudentPolicyCfg = StudentPolicyCfg()  # Required by parent class
    student_policy: StudentPolicyCfg = StudentPolicyCfg()
    teacher: TeacherPolicyCfg = TeacherPolicyCfg()


@configclass
class G1InspireDistillationEnvCfg(G1InspireFlatEnvCfg):
    """Distillation environment configuration for G1 with Inspire hand.
    
    This environment provides both teacher and student observation groups
    for teacher-student distillation. The teacher uses privileged observations
    (base_lin_vel) while the student uses only sensor-available observations.
    """

    observations: G1InspireDistillationObservationsCfg = G1InspireDistillationObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Ensure all observation groups have corruption enabled for training robustness
        self.observations.policy.enable_corruption = True
        self.observations.student_policy.enable_corruption = True
        self.observations.teacher.enable_corruption = True
