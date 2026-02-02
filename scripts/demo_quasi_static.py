"""
Visual Demo: Quasi-Static Clamping

Shows quasi-static contact classification when robot pins
human arm against a wall.

Usage:
    mjpython scripts/demo_quasi_static.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.safety import ISO15066Wrapper, SSMConfig, PFL_LIMITS


# Simple scene with stable arm attached to body, wall, and robot
SCENE = """
<mujoco model="quasi_static_demo">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>
    
    <!-- Wall - the fixed surface -->
    <geom name="wall" type="box" size="0.3 0.02 0.3" pos="0 0.4 1" rgba="0.4 0.4 0.5 1"/>
    
    <!-- Human arm segment - positioned right in front of wall -->
    <!-- The arm is a STATIC body between wall and robot - no joints, just geom -->
    <body name="human_arm" pos="0 0.3 1">
      <freejoint name="arm_free"/>
      <geom name="L_Wrist_col" type="sphere" size="0.06" mass="0.5" rgba="1 0.8 0.6 1"/>
    </body>
    
    <!-- Robot - pushes toward wall (in +Y direction) -->
    <body name="robot" pos="0 -0.2 1">
      <joint name="robot_slide" type="slide" axis="0 1 0" range="0 0.5" damping="30"/>
      <geom name="robot_geom" type="box" size="0.1 0.08 0.1" rgba="0.3 0.5 0.8 1"/>
    </body>
  </worldbody>
  
  <actuator>
    <position name="robot_push" joint="robot_slide" kp="1000" ctrlrange="0 0.5"/>
  </actuator>
</mujoco>
"""


def run_demo():
    """Demonstrate quasi-static vs transient contact."""
    
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Create safety wrapper
    wrapper = ISO15066Wrapper(model, data)
    wrapper.add_robot_geom("robot_geom")
    wrapper.add_fixture_geom("wall")
    
    # Get body IDs
    arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_arm")
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    
    print("\n" + "="*70)
    print("QUASI-STATIC CLAMPING DEMO")
    print("="*70)
    print("\nRobot pushes hand/wrist against wall")
    print("- Transient limit for wrist: 280N")
    print("- QUASI-STATIC limit: 140N (when clamped against wall)")
    print("="*70 + "\n")
    
    # Animation
    push_target = 0.0
    phase = "approach"
    phase_start = 0.0
    
    transient_seen = False
    quasi_static_seen = False
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0, 0.2, 1]
        
        while viewer.is_running():
            t = data.time
            
            # Animation phases
            if phase == "approach":
                push_target = min(0.45, push_target + 0.002)
                data.ctrl[0] = push_target
                
                if push_target > 0.4:
                    phase = "clamping"
                    phase_start = t
                    
            elif phase == "clamping":
                data.ctrl[0] = 0.45
                if t - phase_start > 4:
                    phase = "release"
                    
            elif phase == "release":
                push_target = max(0.0, push_target - 0.003)
                data.ctrl[0] = push_target
                if push_target < 0.05:
                    phase = "approach"
            
            # Get positions
            human_pos = data.xpos[arm_id].copy()
            robot_pos = data.xpos[robot_id].copy()
            
            # Check safety
            safety_info = wrapper.check_safety_no_step(
                robot_pos=robot_pos,
                robot_vel=0.0,
                human_pos=human_pos,
                human_vel=0.0,
            )
            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Print status every 0.3s
            if int(t * 3) != int((t - model.opt.timestep) * 3):
                # Debug: show what human parts are contacting
                if wrapper._human_contacts_this_step:
                    contacts = {k: list(v) for k, v in wrapper._human_contacts_this_step.items()}
                    print(f"    [contacts: {contacts}]")
                
                if safety_info.max_contact_force > 5:
                    region = safety_info.contact_region or "unknown"
                    c_type = safety_info.contact_type or "unknown"
                    force = safety_info.max_contact_force
                    
                    limits = PFL_LIMITS.get(region)
                    if limits:
                        if c_type == "quasi_static":
                            limit = limits.quasi_static_force
                            quasi_static_seen = True
                        else:
                            limit = limits.transient_force
                            transient_seen = True
                    else:
                        limit = 999
                    
                    status = "🔴 VIOLATION" if safety_info.pfl_violation else "🟢 OK"
                    
                    if c_type == "quasi_static":
                        print(f"t={t:5.1f}s | ⚠️  QUASI-STATIC | {region} | {force:.0f}N / {limit}N | {status}")
                    else:
                        print(f"t={t:5.1f}s | Transient | {region} | {force:.0f}N / {limit}N | {status}")
                else:
                    print(f"t={t:5.1f}s | No contact")
            
            viewer.sync()
            time.sleep(0.001)
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    print(f"  Transient contact seen:    {'✅ YES' if transient_seen else '❌ NO'}")
    print(f"  Quasi-static detected:     {'✅ YES' if quasi_static_seen else '❌ NO'}")
    if quasi_static_seen:
        print("\n  ✅ Quasi-static classification working!")
        print("     Arm was clamped between robot and wall.")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_demo()
