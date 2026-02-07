"""Standalone policy inference module for G1 humanoid locomotion.

This module provides:
1. PolicyInference class - for ROS2 integration (no Isaac Sim required)
2. Simulation mode - spawns robot in Isaac Sim and runs policy

Usage (standalone inference for ROS2):
    from policy_inference import PolicyInference

    policy = PolicyInference(policy_path="path/to/policy.pt", device="cpu")
    joint_targets = policy.infer(
        base_lin_vel=[0.0, 0.0, 0.0],
        base_ang_vel=[0.0, 0.0, 0.0],
        projected_gravity=[0.0, 0.0, -1.0],
        velocity_cmd=[0.5, 0.0, 0.0],
        joint_pos=current_joint_positions,
        joint_vel=current_joint_velocities,
    )

Usage (Isaac Sim simulation):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/policy_inference.py \
        --task Isaac-Velocity-Flat-G1-Inspire-v0
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

# Check if running in Isaac Sim environment
_ISAAC_SIM_AVAILABLE = False


@dataclass
class G1InspireJointConfig:
    """Joint configuration for G1 robot with Inspire hands.

    This defines the 53 joints in the order expected by the policy.
    Joint order follows the kinematic tree traversal in Isaac Lab.

    IMPORTANT: This order was obtained by running check_joint_order.py in Isaac Lab.
    If your robot has a different joint order, run check_joint_order.py to get
    the correct order for your configuration.
    """

    # Joint names in the order expected by the policy
    # This order is determined by the USD kinematic tree in Isaac Lab
    JOINT_NAMES = [
        "waist_pitch_joint",  # [0]
        "left_shoulder_pitch_joint",  # [1]
        "right_shoulder_pitch_joint",  # [2]
        "waist_roll_joint",  # [3]
        "left_shoulder_roll_joint",  # [4]
        "right_shoulder_roll_joint",  # [5]
        "waist_yaw_joint",  # [6]
        "left_shoulder_yaw_joint",  # [7]
        "right_shoulder_yaw_joint",  # [8]
        "left_hip_pitch_joint",  # [9]
        "right_hip_pitch_joint",  # [10]
        "left_elbow_joint",  # [11]
        "right_elbow_joint",  # [12]
        "left_hip_roll_joint",  # [13]
        "right_hip_roll_joint",  # [14]
        "left_wrist_roll_joint",  # [15]
        "right_wrist_roll_joint",  # [16]
        "left_hip_yaw_joint",  # [17]
        "right_hip_yaw_joint",  # [18]
        "left_wrist_pitch_joint",  # [19]
        "right_wrist_pitch_joint",  # [20]
        "left_knee_joint",  # [21]
        "right_knee_joint",  # [22]
        "left_wrist_yaw_joint",  # [23]
        "right_wrist_yaw_joint",  # [24]
        "left_ankle_pitch_joint",  # [25]
        "right_ankle_pitch_joint",  # [26]
        "L_index_proximal_joint",  # [27]
        "L_middle_proximal_joint",  # [28]
        "L_pinky_proximal_joint",  # [29]
        "L_ring_proximal_joint",  # [30]
        "L_thumb_proximal_yaw_joint",  # [31]
        "R_index_proximal_joint",  # [32]
        "R_middle_proximal_joint",  # [33]
        "R_pinky_proximal_joint",  # [34]
        "R_ring_proximal_joint",  # [35]
        "R_thumb_proximal_yaw_joint",  # [36]
        "left_ankle_roll_joint",  # [37]
        "right_ankle_roll_joint",  # [38]
        "L_index_intermediate_joint",  # [39]
        "L_middle_intermediate_joint",  # [40]
        "L_pinky_intermediate_joint",  # [41]
        "L_ring_intermediate_joint",  # [42]
        "L_thumb_proximal_pitch_joint",  # [43]
        "R_index_intermediate_joint",  # [44]
        "R_middle_intermediate_joint",  # [45]
        "R_pinky_intermediate_joint",  # [46]
        "R_ring_intermediate_joint",  # [47]
        "R_thumb_proximal_pitch_joint",  # [48]
        "L_thumb_intermediate_joint",  # [49]
        "R_thumb_intermediate_joint",  # [50]
        "L_thumb_distal_joint",  # [51]
        "R_thumb_distal_joint",  # [52]
    ]

    # Default joint positions (standing pose) from env.yaml
    # Only non-zero values are listed
    DEFAULT_JOINT_POS = {
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

    NUM_JOINTS = 53

    @classmethod
    def get_default_positions(cls) -> np.ndarray:
        """Get default joint positions as numpy array."""
        defaults = np.zeros(cls.NUM_JOINTS)
        for i, name in enumerate(cls.JOINT_NAMES):
            if name in cls.DEFAULT_JOINT_POS:
                defaults[i] = cls.DEFAULT_JOINT_POS[name]
        return defaults

    @classmethod
    def get_joint_index(cls, name: str) -> int:
        """Get index of joint by name."""
        return cls.JOINT_NAMES.index(name)


class PolicyInference:
    """Standalone policy inference for G1 humanoid locomotion.

    This class loads a trained policy and provides methods to run inference
    without requiring Isaac Sim or the full RL environment.
    """

    def __init__(
        self,
        policy_path: str,
        device: str = "cpu",
        action_scale: float = 0.5,
        env_yaml_path: Optional[str] = None,
    ):
        """Initialize the policy inference module.

        Args:
            policy_path: Path to the exported policy (.pt file).
            device: Device to run inference on ("cpu" or "cuda:0").
            action_scale: Action scaling factor (default 0.5 from env config).
            env_yaml_path: Optional path to env.yaml for loading default positions.
        """
        self.device = torch.device(device)
        self.action_scale = action_scale

        # Load policy
        print(f"[PolicyInference] Loading policy from: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        # Load default joint positions
        if env_yaml_path:
            self._load_defaults_from_yaml(env_yaml_path)
        else:
            self.default_joint_pos = G1InspireJointConfig.get_default_positions()

        self.default_joint_pos = torch.tensor(
            self.default_joint_pos, dtype=torch.float32, device=self.device
        )

        # Initialize last action buffer
        self.last_action = torch.zeros(G1InspireJointConfig.NUM_JOINTS, device=self.device)

        # Observation indices
        self.obs_dim = 171
        self.action_dim = G1InspireJointConfig.NUM_JOINTS

        print(f"[PolicyInference] Policy loaded successfully")
        print(f"[PolicyInference] Observation dim: {self.obs_dim}")
        print(f"[PolicyInference] Action dim: {self.action_dim}")
        print(f"[PolicyInference] Device: {self.device}")

    def _load_defaults_from_yaml(self, yaml_path: str):
        """Load default joint positions from env.yaml."""
        import yaml
        try:
            with open(yaml_path, 'r') as f:
                # Use unsafe_load for Python-specific YAML tags
                env_cfg = yaml.unsafe_load(f)

            joint_pos_cfg = env_cfg.get('scene', {}).get('robot', {}).get('init_state', {}).get('joint_pos', {})
            self.default_joint_pos = G1InspireJointConfig.get_default_positions()

            # Apply regex patterns from config (simplified matching)
            for pattern, value in joint_pos_cfg.items():
                for i, name in enumerate(G1InspireJointConfig.JOINT_NAMES):
                    # Simple pattern matching (handles .* prefix)
                    pattern_clean = pattern.replace(".*", "")
                    if pattern_clean in name:
                        self.default_joint_pos[i] = value
        except Exception as e:
            print(f"[PolicyInference] Warning: Could not load env.yaml: {e}")
            print("[PolicyInference] Using hardcoded default joint positions")
            self.default_joint_pos = G1InspireJointConfig.get_default_positions()

    def reset(self):
        """Reset the policy state (last action buffer)."""
        self.last_action.zero_()
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def build_observation(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        velocity_cmd: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> torch.Tensor:
        """Build observation tensor from sensor data.

        Args:
            base_lin_vel: Linear velocity in base frame [vx, vy, vz] (3,)
            base_ang_vel: Angular velocity in base frame [wx, wy, wz] (3,)
            projected_gravity: Gravity vector in base frame (normalized) (3,)
            velocity_cmd: Velocity command [vx_cmd, vy_cmd, wz_cmd] (3,)
            joint_pos: Current joint positions (53,)
            joint_vel: Current joint velocities (53,)

        Returns:
            Observation tensor of shape (1, 171)
        """
        # Convert to tensors
        base_lin_vel = torch.tensor(base_lin_vel, dtype=torch.float32, device=self.device)
        base_ang_vel = torch.tensor(base_ang_vel, dtype=torch.float32, device=self.device)
        projected_gravity = torch.tensor(projected_gravity, dtype=torch.float32, device=self.device)
        velocity_cmd = torch.tensor(velocity_cmd, dtype=torch.float32, device=self.device)
        joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self.device)
        joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=self.device)

        # Compute relative joint positions (relative to default)
        joint_pos_rel = joint_pos - self.default_joint_pos

        # Joint velocities are relative to 0 (no scaling needed)
        joint_vel_rel = joint_vel

        # Concatenate observation
        obs = torch.cat([
            base_lin_vel,      # 3
            base_ang_vel,      # 3
            projected_gravity, # 3
            velocity_cmd,      # 3
            joint_pos_rel,     # 53
            joint_vel_rel,     # 53
            self.last_action,  # 53
        ])

        return obs.unsqueeze(0)  # Add batch dimension

    @torch.no_grad()
    def infer(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        velocity_cmd: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> np.ndarray:
        """Run policy inference and return joint position targets.

        Args:
            base_lin_vel: Linear velocity in base frame [vx, vy, vz] (3,)
            base_ang_vel: Angular velocity in base frame [wx, wy, wz] (3,)
            projected_gravity: Gravity vector in base frame (normalized) (3,)
            velocity_cmd: Velocity command [vx_cmd, vy_cmd, wz_cmd] (3,)
            joint_pos: Current joint positions (53,)
            joint_vel: Current joint velocities (53,)

        Returns:
            Joint position targets as numpy array (53,)
        """
        # Build observation
        obs = self.build_observation(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            velocity_cmd=velocity_cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

        # Run inference
        action = self.policy(obs).squeeze(0)

        # Store for next iteration
        self.last_action = action.clone()

        # Scale action and add to default position
        # action_target = default_pos + action * scale
        joint_targets = self.default_joint_pos + action * self.action_scale

        return joint_targets.cpu().numpy()

    def get_default_joint_positions(self) -> np.ndarray:
        """Get default (standing) joint positions."""
        return self.default_joint_pos.cpu().numpy()

    @staticmethod
    def joint_dict_to_array(joint_dict: dict) -> np.ndarray:
        """Convert a dictionary of joint values to a numpy array in policy order.

        Args:
            joint_dict: Dictionary mapping joint names to values

        Returns:
            Numpy array of shape (53,) with values in policy joint order
        """
        arr = np.zeros(G1InspireJointConfig.NUM_JOINTS)
        for name, value in joint_dict.items():
            if name in G1InspireJointConfig.JOINT_NAMES:
                idx = G1InspireJointConfig.JOINT_NAMES.index(name)
                arr[idx] = value
        return arr

    @staticmethod
    def array_to_joint_dict(arr: np.ndarray) -> dict:
        """Convert a numpy array to a dictionary of joint values.

        Args:
            arr: Numpy array of shape (53,) with values in policy joint order

        Returns:
            Dictionary mapping joint names to values
        """
        return {name: arr[i] for i, name in enumerate(G1InspireJointConfig.JOINT_NAMES)}

    @staticmethod
    def compute_projected_gravity(quaternion: np.ndarray) -> np.ndarray:
        """Compute projected gravity from base orientation quaternion.

        Args:
            quaternion: Base orientation as [w, x, y, z] quaternion

        Returns:
            Gravity vector projected into base frame (normalized)
        """
        w, x, y, z = quaternion

        # Gravity in world frame (pointing down)
        gravity_world = np.array([0.0, 0.0, -1.0])

        # Rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

        # Project gravity into base frame (transpose for world->base)
        gravity_base = R.T @ gravity_world

        return gravity_base


# =============================================================================
# Isaac Sim Simulation Mode
# =============================================================================

def run_isaac_sim():
    """Run policy inference in Isaac Sim with robot visualization."""
    import argparse
    import sys

    from isaaclab.app import AppLauncher

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run policy inference in Isaac Sim.")
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-Inspire-v0", help="Task name")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
    parser.add_argument(
        "--velocity_cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0],
        help="Velocity command [vx, vy, wz] (default: standing still)"
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Clear sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    # Launch Isaac Sim
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Now import Isaac Lab modules
    import gymnasium as gym
    import os
    import time
    import torch

    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
        """Run policy in Isaac Sim environment."""
        # Get task name
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # Configure environment
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Disable observation corruption for inference
        env_cfg.observations.policy.enable_corruption = False

        # Set up log directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        # Find checkpoint
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # Create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # Handle multi-agent envs
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Wrap for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")

        # Load runner and policy
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)

        # Get inference policy
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Get step dt for real-time mode
        dt = env.unwrapped.step_dt

        # Velocity command (constant)
        velocity_cmd = torch.tensor(
            args_cli.velocity_cmd, dtype=torch.float32, device=env.unwrapped.device
        ).unsqueeze(0).expand(args_cli.num_envs, -1)

        # Velocity command indices in observation (base_lin_vel:3, base_ang_vel:3, proj_grav:3, then vel_cmd:3)
        cmd_start, cmd_end = 9, 12

        print("\n" + "=" * 60)
        print("Policy Inference in Isaac Sim")
        print("=" * 60)
        print(f"  Task: {args_cli.task}")
        print(f"  Checkpoint: {resume_path}")
        print(f"  Velocity command: vx={args_cli.velocity_cmd[0]:.2f}, vy={args_cli.velocity_cmd[1]:.2f}, wz={args_cli.velocity_cmd[2]:.2f}")
        print(f"  Real-time: {args_cli.real_time}")
        print(f"  Num envs: {args_cli.num_envs}")
        print("=" * 60)
        print("\nPress Ctrl+C to exit\n")

        # Reset environment
        obs = env.get_observations()

        # Main loop
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # Override velocity commands in observation
                obs_modified = obs.clone()
                obs_modified[:, cmd_start:cmd_end] = velocity_cmd

                # Run policy
                actions = policy(obs_modified)

                # Step environment
                obs, _, _, _ = env.step(actions)

            # Real-time delay
            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()

    # Run main
    main()
    simulation_app.close()


def main_standalone():
    """Example usage of PolicyInference (standalone, no Isaac Sim)."""
    import os

    # Find the policy file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    policy_path = os.path.join(
        base_dir, "logs/rsl_rl/g1_flat/2026-02-02_17-19-26/exported/policy.pt"
    )
    env_yaml_path = os.path.join(
        base_dir, "logs/rsl_rl/g1_flat/2026-02-02_17-19-26/params/env.yaml"
    )

    if not os.path.exists(policy_path):
        print(f"Policy not found at: {policy_path}")
        print("Please provide a valid policy path.")
        return

    # Initialize policy
    policy = PolicyInference(
        policy_path=policy_path,
        device="cpu",
        action_scale=0.5,
        env_yaml_path=env_yaml_path if os.path.exists(env_yaml_path) else None,
    )

    # Get default positions
    default_pos = policy.get_default_joint_positions()
    print(f"\nDefault joint positions:\n{default_pos}")

    # Example inference
    print("\n--- Running example inference ---")

    # Simulated sensor data (robot standing still, upright)
    base_lin_vel = np.array([0.0, 0.0, 0.0])
    base_ang_vel = np.array([0.0, 0.0, 0.0])
    projected_gravity = np.array([0.0, 0.0, -1.0])  # Upright
    velocity_cmd = np.array([0.5, 0.0, 0.0])  # Walk forward
    joint_pos = default_pos.copy()
    joint_vel = np.zeros(53)

    # Run inference
    for step in range(5):
        joint_targets = policy.infer(
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            velocity_cmd=velocity_cmd,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

        print(f"\nStep {step + 1}:")
        print(f"  Velocity command: vx={velocity_cmd[0]:.2f}, vy={velocity_cmd[1]:.2f}, wz={velocity_cmd[2]:.2f}")
        print(f"  Joint targets (first 12):")
        for i in range(12):
            print(f"    {G1InspireJointConfig.JOINT_NAMES[i]}: {joint_targets[i]:.4f}")

        # Update joint_pos for next iteration
        joint_pos = joint_targets.copy()

    print("\n--- Joint names and indices ---")
    for i, name in enumerate(G1InspireJointConfig.JOINT_NAMES):
        print(f"  [{i:2d}] {name}")


if __name__ == "__main__":
    import sys

    # Check if running with Isaac Sim (has --task argument or isaaclab.sh launcher)
    if any(arg.startswith("--task") for arg in sys.argv) or "isaacsim" in sys.executable.lower():
        # Isaac Sim mode
        run_isaac_sim()
    else:
        # Standalone mode (no Isaac Sim)
        main_standalone()
