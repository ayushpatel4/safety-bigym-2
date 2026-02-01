"""
Visual test for Human IK solver.

Displays the human reaching for a visible moving target sphere.
Uses kinematic mode (no physics) to test IK in isolation.

Run with: mjpython scripts/test_ik.py
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.human import HumanIK


# Scene with movable target marker (mocap body)
SCENE_XML = """
<mujoco model="ik_test">
  <include file="{smplh_path}"/>
  
  <worldbody>
    <!-- Movable target sphere (mocap body can be positioned freely) -->
    <body name="target_marker" mocap="true" pos="0.3 0.5 1.0">
      <geom name="target_sphere" type="sphere" size="0.05" 
            rgba="1 0.2 0.2 0.9" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_ik_visual():
    """Visual test of IK solver using kinematic mode."""
    # Create scene with target marker
    smplh_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
    scene_xml = SCENE_XML.format(smplh_path=str(smplh_path))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(scene_xml)
        scene_path = f.name
    
    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # Get mocap body ID for target
    target_mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
    
    # Initialize
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Create IK solver
    ik = HumanIK(model, data)
    
    # Save initial pose
    initial_qpos = data.qpos.copy()
    
    print("Launching IK test with visible target...")
    print("  RED SPHERE = Target position (moves in a circle)")
    print("  Watch the arm reach toward the sphere")
    print("  Press SPACE to pause/resume")
    print("  Close window to exit")
    
    paused = False
    time_elapsed = 0.0
    
    def key_callback(key):
        nonlocal paused
        if key == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")
    
    dt = 0.01
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused:
                time_elapsed += dt
                
                # Move target in a circle (within arm reach)
                target_x = 0.3 + 0.15 * np.sin(time_elapsed * 0.5)
                target_y = 0.3 + 0.15 * np.cos(time_elapsed * 0.5)
                target_z = 0.9 + 0.1 * np.sin(time_elapsed * 0.3)
                target = np.array([target_x, target_y, target_z])
                
                # Update mocap body position (visible target sphere)
                data.mocap_pos[0] = target
                
                # Reset human pose (except arm will be set by IK)
                data.qpos[:] = initial_qpos
                
                # Solve IK
                chain_name, angles = ik.solve_with_selection(target, max_iterations=50)
                
                # Apply solution
                qpos_indices = ik._chain_cache[chain_name]["qpos_indices"]
                for i, idx in enumerate(qpos_indices):
                    data.qpos[idx] = angles[i]
                
                # Kinematics update
                mujoco.mj_kinematics(model, data)
                
                # Get EE position
                ee_pos = ik.get_end_effector_pos(chain_name)
                error = np.linalg.norm(target - ee_pos)
                
                # Print occasionally
                if int(time_elapsed * 10) % 30 == 0:
                    print(f"  Error: {error:.3f}m | Arm: {chain_name}")
            
            viewer.sync()
            time.sleep(0.01)
    
    print("Test complete.")


if __name__ == "__main__":
    test_ik_visual()
