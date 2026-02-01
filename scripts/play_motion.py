"""
Motion Playback Script - Interactive MuJoCo Viewer

Plays AMASS motion data on the SMPL-H MuJoCo model interactively.
Must be run with mjpython on macOS.

Usage:
    mjpython scripts/play_motion.py /path/to/motion.npz
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.motion.amass_loader import load_amass_clip


def play_motion_interactive(motion_path: str, model_path: str = None, speed: float = 1.0):
    """
    Play AMASS motion interactively in MuJoCo viewer.
    
    Args:
        motion_path: Path to AMASS .npz file
        model_path: Path to SMPL-H MJCF (default: built-in asset)
        speed: Playback speed multiplier
    """
    # Load motion clip
    print(f"Loading motion: {motion_path}")
    clip = load_amass_clip(motion_path, include_hands=False)
    print(f"  Duration: {clip.duration:.2f}s, Frames: {clip.num_frames}, FPS: {clip.fps}")
    
    # Load MuJoCo model
    if model_path is None:
        model_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
    
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Disable gravity for kinematic playback
    model.opt.gravity[:] = 0
    
    # Get joint name to qpos index mapping
    joint_name_to_idx = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "root":
            qpos_idx = model.jnt_qposadr[i]
            joint_name_to_idx[name] = qpos_idx
    
    # Joint names from SMPL-H (skipping Pelvis which uses freejoint)
    smplh_joint_names = [
        "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee", "Spine",
        "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe", "Neck",
        "L_Thorax", "R_Thorax", "Head", "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    ]
    
    # Build mapping from motion data joint index to qpos indices
    motion_to_qpos = []
    for joint_idx, joint_name in enumerate(smplh_joint_names):
        indices = []
        for axis in ["x", "y", "z"]:
            full_name = f"{joint_name}_{axis}"
            if full_name in joint_name_to_idx:
                indices.append(joint_name_to_idx[full_name])
            else:
                indices.append(None)
        motion_to_qpos.append(indices)
    
    # Playback state
    frame_idx = 0
    last_update = time.time()
    paused = False
    
    def set_pose(frame_idx: int):
        """Set model pose from motion frame."""
        joint_angles, root_trans, root_quat = clip.get_frame(frame_idx)
        
        # Set root position and orientation
        data.qpos[0:3] = root_trans
        data.qpos[3:7] = root_quat
        
        # Set joint angles (skip index 0 which is Pelvis/root)
        for joint_idx, (angles, qpos_indices) in enumerate(zip(joint_angles[1:], motion_to_qpos)):
            for axis_idx, qpos_idx in enumerate(qpos_indices):
                if qpos_idx is not None:
                    data.qpos[qpos_idx] = angles[axis_idx]
        
        mujoco.mj_kinematics(model, data)
    
    def key_callback(key):
        """Handle keyboard input."""
        nonlocal paused
        if key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")
    
    print("\nLaunching interactive viewer...")
    print("  SPACE: pause/resume")
    print("  Close window to exit")
    
    # Set initial pose
    set_pose(0)
    
    # Launch passive viewer (works with mjpython on macOS)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            current_time = time.time()
            
            if not paused:
                # Advance frame based on elapsed time
                dt = current_time - last_update
                frames_to_advance = int(dt * clip.fps * speed)
                
                if frames_to_advance > 0:
                    frame_idx = (frame_idx + frames_to_advance) % clip.num_frames
                    last_update = current_time
                    set_pose(frame_idx)
            else:
                last_update = current_time
            
            viewer.sync()
            time.sleep(0.008)
    
    print("Viewer closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play AMASS motion interactively")
    parser.add_argument("motion_path", type=str, help="Path to AMASS .npz file")
    parser.add_argument("--model", type=str, default=None, help="Path to MJCF model")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    
    args = parser.parse_args()
    
    play_motion_interactive(args.motion_path, args.model, args.speed)
