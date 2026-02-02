# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert G1 Inspire hand USD from fixed base to floating base for locomotion.

This script:
1. Flattens the USD (merges all references into a single file)
2. Removes the fixed root_joint
3. Applies ArticulationRootAPI to the pelvis link

Usage:
    python scripts/tools/convert_g1_inspire_to_floating.py
"""

import argparse
import os

# Create argument parser
parser = argparse.ArgumentParser(description="Convert G1 Inspire USD to floating base")
parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")

# Parse known args to handle IsaacLab launcher args
args, _ = parser.parse_known_args()

# Launch Isaac Sim App (must be done before importing pxr)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now we can import pxr modules
from pxr import Usd, UsdPhysics, Sdf, PhysxSchema
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


def flatten_and_convert(input_usd_path: str, output_usd_path: str):
    """Flatten USD and convert to floating base for locomotion.

    Args:
        input_usd_path: Path to the input USD file.
        output_usd_path: Path to save the modified USD file.
    """
    print(f"Loading USD from: {input_usd_path}")

    # Open the USD stage
    stage = Usd.Stage.Open(input_usd_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {input_usd_path}")

    # Flatten the stage (resolve all references and payloads)
    print("\nFlattening USD (merging all references)...")
    flattened_layer = stage.Flatten()

    # Create output directory and save flattened USD
    os.makedirs(os.path.dirname(output_usd_path), exist_ok=True)
    flattened_layer.Export(output_usd_path)
    print(f"Flattened USD saved to: {output_usd_path}")

    # Now open the flattened USD and modify it
    print("\nOpening flattened USD for modification...")
    flat_stage = Usd.Stage.Open(output_usd_path)

    # Find the root prim and pelvis
    root_prim_path = None
    pelvis_prim_path = None
    root_joint_path = None

    # First pass: find important prims
    print("\nSearching for root structure...")
    for prim in flat_stage.Traverse():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        prim_type = prim.GetTypeName()

        # Find the robot root (usually named like g1_29dof_rev_1_0)
        if "g1_29dof" in prim_name.lower() and prim.GetParent().GetPath() == Sdf.Path("/"):
            root_prim_path = prim_path
            print(f"  Found robot root: {prim_path}")

        # Find pelvis
        if prim_name == "pelvis" or "pelvis" in prim_name.lower():
            pelvis_prim_path = prim_path
            print(f"  Found pelvis: {prim_path}")

        # Find root_joint
        if prim_name == "root_joint":
            root_joint_path = prim_path
            print(f"  Found root_joint: {prim_path} (type: {prim_type})")

    # If no pelvis found, try to find the base link
    if pelvis_prim_path is None:
        for prim in flat_stage.Traverse():
            prim_name = prim.GetName()
            if "base" in prim_name.lower() or "torso" in prim_name.lower():
                pelvis_prim_path = str(prim.GetPath())
                print(f"  Using as base link: {pelvis_prim_path}")
                break

    if root_joint_path is None:
        print("\nWARNING: No root_joint found!")
        flat_stage.GetRootLayer().Save()
        return []

    # Get the root_joint prim to check its APIs before removing
    root_joint_prim = flat_stage.GetPrimAtPath(root_joint_path)

    # Check if ArticulationRootAPI is on the root_joint
    has_artic_api = root_joint_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    print(f"\n  root_joint has ArticulationRootAPI: {has_artic_api}")

    # Find where to apply ArticulationRootAPI (the robot root or pelvis)
    target_for_artic_api = root_prim_path if root_prim_path else pelvis_prim_path

    if target_for_artic_api:
        print(f"\nApplying ArticulationRootAPI to: {target_for_artic_api}")
        target_prim = flat_stage.GetPrimAtPath(target_for_artic_api)

        # Apply ArticulationRootAPI
        if not target_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI.Apply(target_prim)
            print("  Applied UsdPhysics.ArticulationRootAPI")

        # Also apply PhysxArticulationAPI for PhysX-specific settings
        if not target_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(target_prim)
            print("  Applied PhysxSchema.PhysxArticulationAPI")

        # Set articulation properties for floating base
        artic_api = PhysxSchema.PhysxArticulationAPI(target_prim)
        artic_api.CreateEnabledSelfCollisionsAttr(False)
        artic_api.CreateSolverPositionIterationCountAttr(8)
        artic_api.CreateSolverVelocityIterationCountAttr(4)
        print("  Set articulation solver properties")

    # Now remove the root_joint
    print(f"\nRemoving root_joint: {root_joint_path}")
    flat_stage.RemovePrim(root_joint_path)

    # Save the modified stage
    flat_stage.GetRootLayer().Save()
    print(f"\nSaved modified USD to: {output_usd_path}")

    return [root_joint_path]


def inspect_usd(usd_path: str):
    """Inspect a USD file to show its structure."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {usd_path}")
    print('='*60)

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"ERROR: Failed to open USD file")
        return

    # Check for ArticulationRootAPI
    print("\nPrims with ArticulationRootAPI:")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  {prim.GetPath()} (type: {prim.GetTypeName()})")

    # Show first few joints
    print("\nFirst 5 joints:")
    joint_count = 0
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        if "Joint" in prim_type:
            joint_count += 1
            if joint_count <= 5:
                print(f"  {prim.GetPath()} (type: {prim_type})")

    print(f"\nTotal joints: {joint_count}")


if __name__ == "__main__":
    # Input path from Nucleus
    input_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd"

    # Output path
    output_dir = "/home/nurtay/humanoid-locomotion/IsaacLab/source/isaaclab_assets/data/Robots/Unitree/G1"
    output_path = f"{output_dir}/g1_29dof_inspire_hand_floating.usd"

    print(f"Input USD: {input_path}")
    print(f"Output USD: {output_path}")

    # First inspect the input
    inspect_usd(input_path)

    # Flatten and convert
    print("\n" + "="*60)
    print("Flattening and converting to floating base...")
    print("="*60)
    removed = flatten_and_convert(input_path, output_path)

    # Verify the output
    print("\n" + "="*60)
    print("Verifying output USD...")
    print("="*60)
    inspect_usd(output_path)

    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    print(f"\nNew USD saved to:\n{output_path}")
    print(f"\nRemoved joints: {removed}")
    print("\nTry training with: --num_envs 1024")

    # Close the simulation app
    simulation_app.close()
