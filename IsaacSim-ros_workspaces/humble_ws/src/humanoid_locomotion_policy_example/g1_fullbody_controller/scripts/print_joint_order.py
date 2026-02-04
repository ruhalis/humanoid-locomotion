#!/usr/bin/env python3
"""
Script to print the actual joint order from IsaacLab's G1 robot configuration.
Run this inside the IsaacLab environment to get the correct joint ordering.

Usage (from IsaacLab directory):
    python scripts/print_joint_order.py

Or with full path:
    cd /path/to/IsaacLab
    ./isaaclab.sh -p /path/to/this/print_joint_order.py
"""

def main():
    try:
        # Try to import IsaacLab components
        from isaaclab.app import AppLauncher

        # Launch minimal app
        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app

        import torch
        from isaaclab.assets import Articulation
        from isaaclab_assets.robots.unitree import G1_INSPIRE_LOCOMOTION_CFG
        import isaaclab.sim as sim_utils

        # Create a minimal scene
        sim_cfg = sim_utils.SimulationCfg(device="cpu")
        sim = sim_utils.SimulationContext(sim_cfg)

        # Set up ground plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)

        # Spawn the robot
        robot_cfg = G1_INSPIRE_LOCOMOTION_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        robot = Articulation(robot_cfg)

        # Play the sim to initialize
        sim.reset()
        robot.reset()

        # Print joint names
        print("\n" + "=" * 70)
        print("G1 INSPIRE LOCOMOTION - JOINT ORDER FROM ISAACLAB")
        print("=" * 70)
        print(f"\nTotal joints: {robot.num_joints}")
        print("\nJoint names in IsaacLab order:")
        for i, name in enumerate(robot.joint_names):
            print(f"    '{name}',")

        print("\n" + "=" * 70)
        print("Copy the above list to your controller's joint_names array")
        print("=" * 70 + "\n")

        # Also print as Python list for easy copy-paste
        print("\nAs Python list:")
        print("joint_names = [")
        for name in robot.joint_names:
            print(f"    '{name}',")
        print("]")

        simulation_app.close()

    except ImportError as e:
        print(f"Error: Cannot import IsaacLab components: {e}")
        print("\nThis script must be run inside the IsaacLab environment.")
        print("Usage: cd /path/to/IsaacLab && ./isaaclab.sh -p /path/to/print_joint_order.py")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
