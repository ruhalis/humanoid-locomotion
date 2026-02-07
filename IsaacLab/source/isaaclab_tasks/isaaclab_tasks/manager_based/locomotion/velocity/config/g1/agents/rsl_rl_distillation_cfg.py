# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL distillation configuration for G1 Inspire.

This module provides the distillation runner configuration for training
a student policy that removes privileged observations (base_lin_vel).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class G1InspireDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner config for G1 Inspire.
    
    The student learns to match the teacher's actions using only
    sensor-available observations (student_policy), while the teacher
    uses all observations including privileged ones (teacher).
    
    Teacher observation dim: 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (cmd) + 53 (joint_pos) + 53 (joint_vel) + 53 (actions) = 171
    Student observation dim: 3 (ang_vel) + 3 (gravity) + 3 (cmd) + 53 (joint_pos) + 53 (joint_vel) + 53 (actions) = 168
    """

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    # Use same experiment_name as teacher so --load_run finds the teacher checkpoint
    experiment_name = "g1_flat"
    
    # Map observation groups: student uses student_policy, teacher uses teacher
    obs_groups = {"policy": ["student_policy"], "teacher": ["teacher"]}
    
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        # Match the network architecture from PPO training
        student_hidden_dims=[256, 128, 128],
        teacher_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
    )
