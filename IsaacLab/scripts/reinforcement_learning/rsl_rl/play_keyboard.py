# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with keyboard control for velocity commands."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with keyboard control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Keyboard control parameters
parser.add_argument("--v_x_sensitivity", type=float, default=0.8, help="Linear velocity X sensitivity.")
parser.add_argument("--v_y_sensitivity", type=float, default=0.4, help="Linear velocity Y sensitivity.")
parser.add_argument("--omega_z_sensitivity", type=float, default=1.0, help="Angular velocity Z sensitivity.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

import carb
import omni

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class KeyboardController:
    """Keyboard controller for SE(2) velocity commands.

    Key bindings:
        - Arrow Up / W: Move forward (+x)
        - Arrow Down / S: Move backward (-x)
        - Arrow Left / A: Move left (+y)
        - Arrow Right / D: Move right (-y)
        - Z / Q: Rotate left (+omega_z)
        - X / E: Rotate right (-omega_z)
        - L: Reset commands to zero
        - ESC: Exit
    """

    def __init__(self, num_envs: int, device: str, v_x: float = 0.8, v_y: float = 0.4, omega_z: float = 1.0):
        """Initialize the keyboard controller.

        Args:
            num_envs: Number of environments.
            device: Device to create tensors on.
            v_x: Linear velocity X sensitivity.
            v_y: Linear velocity Y sensitivity.
            omega_z: Angular velocity Z sensitivity.
        """
        self.num_envs = num_envs
        self.device = device
        self.v_x_sensitivity = v_x
        self.v_y_sensitivity = v_y
        self.omega_z_sensitivity = omega_z

        # Command buffer: [vx, vy, omega_z]
        self._base_command = np.zeros(3)

        # Setup keyboard interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        # Key bindings
        self._create_key_bindings()

        print(self)

    def __del__(self):
        """Cleanup keyboard subscription."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)

    def __str__(self) -> str:
        """Return string representation with key bindings."""
        msg = "\n" + "=" * 60 + "\n"
        msg += "Keyboard Controller for Velocity Commands\n"
        msg += "=" * 60 + "\n"
        msg += f"  Sensitivities: vx={self.v_x_sensitivity}, vy={self.v_y_sensitivity}, wz={self.omega_z_sensitivity}\n"
        msg += "-" * 60 + "\n"
        msg += "  Move forward   (+x): Arrow Up    / W\n"
        msg += "  Move backward  (-x): Arrow Down  / S\n"
        msg += "  Move left      (+y): Arrow Left  / A\n"
        msg += "  Move right     (-y): Arrow Right / D\n"
        msg += "  Rotate left   (+wz): Z / Q\n"
        msg += "  Rotate right  (-wz): X / E\n"
        msg += "  Reset commands     : L\n"
        msg += "=" * 60 + "\n"
        return msg

    def _create_key_bindings(self):
        """Create key-to-command mappings."""
        self._key_mapping = {
            # Forward/backward
            "UP": np.array([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "W": np.array([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "DOWN": np.array([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.array([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # Left/right strafe
            "LEFT": np.array([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "A": np.array([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "RIGHT": np.array([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "D": np.array([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            # Rotation
            "Z": np.array([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "Q": np.array([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "X": np.array([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
            "E": np.array([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name in self._key_mapping:
                self._base_command += self._key_mapping[event.input.name]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._key_mapping:
                self._base_command -= self._key_mapping[event.input.name]
        return True

    def reset(self):
        """Reset commands to zero."""
        self._base_command.fill(0.0)
        print("[Keyboard] Commands reset to zero")

    def get_command(self) -> torch.Tensor:
        """Get current velocity command as tensor.

        Returns:
            Tensor of shape (num_envs, 3) with [vx, vy, omega_z].
        """
        cmd = torch.tensor(self._base_command, dtype=torch.float32, device=self.device)
        # Broadcast to all environments
        return cmd.unsqueeze(0).expand(self.num_envs, -1)


def get_velocity_command_indices(env) -> tuple:
    """Determine the observation indices for velocity commands.

    For standard locomotion envs with policy observations:
    - base_lin_vel: 3
    - base_ang_vel: 3
    - projected_gravity: 3
    - velocity_commands: 3  <-- indices 9:12

    Returns:
        Tuple of (start_idx, end_idx) for velocity command slice.
    """
    # Default for standard locomotion velocity environments
    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) = 9
    return 9, 12


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using keyboard control."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Initialize keyboard controller
    keyboard = KeyboardController(
        num_envs=env_cfg.scene.num_envs,
        device=env.unwrapped.device,
        v_x=args_cli.v_x_sensitivity,
        v_y=args_cli.v_y_sensitivity,
        omega_z=args_cli.omega_z_sensitivity,
    )

    # Get velocity command indices in observation
    cmd_start, cmd_end = get_velocity_command_indices(env)
    print(f"[INFO] Velocity command indices in observation: [{cmd_start}:{cmd_end}]")

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # Get keyboard command
        keyboard_cmd = keyboard.get_command()

        # run everything in inference mode
        with torch.inference_mode():
            # Clone obs and override velocity commands with keyboard input
            obs_modified = obs.clone()
            obs_modified[:, cmd_start:cmd_end] = keyboard_cmd

            # agent stepping with modified observations
            actions = policy(obs_modified)

            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
