"""
Test script for PD-controlled human motion.

Demonstrates the HumanController with AMASS motion playback.
Must be run with mjpython on macOS.

Usage:
    mjpython scripts/test_pd_controller.py /path/to/motion.npz
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

from safety_bigym.human import HumanController, ScenarioParams


def test_pd_controller(motion_path: str, model_path: str = None):
    """
    Test PD controller with AMASS motion playback.
    
    Args:
        motion_path: Path to AMASS .npz file
        model_path: Path to SMPL-H MJCF
    """
    # Load model
    if model_path is None:
        model_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
    
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    
    # Create human controller
    print(f"Loading motion: {motion_path}")
    controller = HumanController(model, data)
    
    # Set up scenario (no IK trigger for this test - pure AMASS playback)
    scenario = ScenarioParams(
        clip_path=motion_path,
        trigger_time=1000.0,  # Never trigger IK in this test
        speed_multiplier=1.0,
    )
    controller.set_scenario(scenario)
    controller.reset()
    
    print(f"  Clip duration: {controller.clip.duration:.2f}s")
    print(f"  Clip FPS: {controller.clip.fps}")
    
    # Playback state
    paused = False
    
    def key_callback(key):
        nonlocal paused
        if key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'} (Phase: {controller.current_phase})")
    
    print("\nLaunching PD controller test...")
    print("  SPACE: pause/resume")
    print("  Close window to exit")
    
    # Physics timestep
    dt = model.opt.timestep
    
    last_time = time.time()
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            current_time = time.time()
            
            if not paused:
                # Step the controller (updates targets)
                controller.step(dt)
                
                # Step physics (applies controls)
                mujoco.mj_step(model, data)
            
            viewer.sync()
            
            # Regulate to roughly real-time
            elapsed = time.time() - current_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("Test complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PD controller with AMASS motion")
    parser.add_argument("motion_path", type=str, help="Path to AMASS .npz file")
    parser.add_argument("--model", type=str, default=None, help="Path to MJCF model")
    
    args = parser.parse_args()
    
    test_pd_controller(args.motion_path, args.model)
