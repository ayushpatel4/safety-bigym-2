"""
Contact Sanity Check

Tests that the PD-controlled human responds naturally to collisions.
Adds a simple wall obstacle in the motion path.

Usage:
    mjpython scripts/test_contact.py /path/to/motion.npz
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import sys
import tempfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human import HumanController, ScenarioParams


# MJCF with obstacle added
SCENE_WITH_OBSTACLE = """
<mujoco model="contact_test">
  <include file="{smplh_path}"/>
  
  <worldbody>
    <!-- Ground plane (unique name to avoid conflict with model) -->
    <geom name="ground_plane" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1" 
          pos="0 0 0" contype="1" conaffinity="1"/>
    
    <!-- Wall obstacle in front of human -->
    <body name="wall" pos="0 2 0.75">
      <geom name="wall_geom" type="box" size="1.5 0.1 0.75" 
            rgba="0.9 0.3 0.3 0.8" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Floating box to interact with -->
    <body name="box" pos="0.5 0.5 1.0">
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" 
            rgba="0.3 0.3 0.9 1" mass="0.5" contype="1" conaffinity="1"/>
      <freejoint/>
    </body>
  </worldbody>
</mujoco>
"""


def test_contact(motion_path: str):
    """
    Test contact response with obstacle.
    
    Args:
        motion_path: Path to AMASS .npz file
    """
    # Get SMPL-H model path
    smplh_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
    
    # Create temporary scene file with obstacle
    scene_xml = SCENE_WITH_OBSTACLE.format(smplh_path=str(smplh_path))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(scene_xml)
        scene_path = f.name
    
    print(f"Loading scene with obstacle...")
    try:
        model = mujoco.MjModel.from_xml_path(scene_path)
    except Exception as e:
        print(f"Error loading scene: {e}")
        # Fall back to just the human model
        print("Falling back to human model only...")
        model = mujoco.MjModel.from_xml_path(str(smplh_path))
    
    data = mujoco.MjData(model)
    
    # Create human controller
    print(f"Loading motion: {motion_path}")
    controller = HumanController(model, data)
    
    scenario = ScenarioParams(
        clip_path=motion_path,
        trigger_time=1000.0,  # No IK trigger
        speed_multiplier=1.0,
    )
    controller.set_scenario(scenario)
    controller.reset()
    
    print(f"  Clip duration: {controller.clip.duration:.2f}s")
    
    # Tracking variables
    max_contact_force = 0.0
    contact_count = 0
    
    paused = False
    
    def key_callback(key):
        nonlocal paused
        if key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")
    
    print("\nLaunching contact test...")
    print("  SPACE: pause/resume")
    print("  Watch for contact forces in console")
    print("  Close window to exit")
    
    dt = model.opt.timestep
    step_count = 0
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused:
                # Step controller
                controller.step(dt)
                
                # Step physics
                mujoco.mj_step(model, data)
                
                # Check for contacts
                for i in range(data.ncon):
                    contact = data.contact[i]
                    
                    # Get contact force
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    force_mag = np.linalg.norm(force[:3])
                    
                    if force_mag > 1.0:  # Threshold to filter noise
                        contact_count += 1
                        if force_mag > max_contact_force:
                            max_contact_force = force_mag
                            geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or f"geom{contact.geom1}"
                            geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or f"geom{contact.geom2}"
                            print(f"  Contact: {geom1} <-> {geom2}, Force: {force_mag:.1f}N")
                
                step_count += 1
            
            viewer.sync()
            time.sleep(0.001)
    
    print(f"\nTest complete:")
    print(f"  Total steps: {step_count}")
    print(f"  Contact events: {contact_count}")
    print(f"  Max contact force: {max_contact_force:.1f}N")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test contact response")
    parser.add_argument("motion_path", type=str, help="Path to AMASS .npz file")
    
    args = parser.parse_args()
    
    test_contact(args.motion_path)
