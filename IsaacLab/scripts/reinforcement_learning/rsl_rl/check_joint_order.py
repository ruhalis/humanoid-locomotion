"""Utility script to check the joint order in Isaac Lab.

Run this script in your Isaac Lab environment to get the actual joint order
used by the trained policy.

Usage:
    cd /path/to/IsaacLab
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/check_joint_order.py \
        --task Isaac-Velocity-Flat-G1-Inspire-v0
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check joint order for G1 robot.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-Inspire-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    """Print the joint order from the environment."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Get the robot asset
    robot = env.unwrapped.scene["robot"]

    print("\n" + "=" * 70)
    print("JOINT ORDER FOR POLICY INFERENCE")
    print("=" * 70)

    joint_names = robot.joint_names
    print(f"\nTotal joints: {len(joint_names)}")
    print("\nJoint names in order (copy this to policy_inference.py):\n")

    print("JOINT_NAMES = [")
    for i, name in enumerate(joint_names):
        print(f'    "{name}",  # [{i}]')
    print("]")

    print("\n" + "-" * 70)
    print("Default joint positions:")
    print("-" * 70)

    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    print("\nDEFAULT_JOINT_POS = {")
    for i, name in enumerate(joint_names):
        if abs(default_pos[i]) > 1e-6:
            print(f'    "{name}": {default_pos[i]:.4f},')
    print("}")

    print("\n" + "-" * 70)
    print("Python dict for all defaults:")
    print("-" * 70)
    print("\ndefault_positions = {")
    for i, name in enumerate(joint_names):
        print(f'    "{name}": {default_pos[i]:.6f},')
    print("}")

    # Also show observation space info
    print("\n" + "=" * 70)
    print("OBSERVATION SPACE INFO")
    print("=" * 70)
    print(f"\nObservation shape: {env.observation_space.shape}")
    print(f"Action shape: {env.action_space.shape}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
