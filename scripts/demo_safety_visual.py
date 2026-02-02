"""
Visual Demo: ISO 15066 Safety Wrapper

Shows the safety wrapper detecting violations:
1. Robot approaches human → SSM violation
2. Robot collides with human → PFL contact detected
3. Status printed in real-time with colors

Usage:
    mjpython scripts/demo_safety_visual.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import tempfile
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.safety import ISO15066Wrapper, SSMConfig, PFL_LIMITS


# Scene with actuated robot that can reliably move toward human
VISUAL_SCENE = """
<mujoco model="safety_visual_demo">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="3 3 0.1" rgba="0.9 0.9 0.9 1"/>
    
    <!-- Human - static body -->
    <body name="human" pos="0 0 1">
      <geom name="Pelvis_col" type="sphere" size="0.12" rgba="0.9 0.7 0.6 1"/>
      <geom name="Chest_col" type="capsule" size="0.1" fromto="0 0 0 0 0 0.35" rgba="0.9 0.7 0.6 1"/>
      <geom name="Head_col" type="sphere" size="0.09" pos="0 0 0.45" rgba="0.9 0.7 0.6 1"/>
      <geom name="R_Elbow_col" type="capsule" size="0.04" fromto="0.15 0 0.25 0.4 0 0.25" rgba="0.9 0.7 0.6 1"/>
      <geom name="L_Elbow_col" type="capsule" size="0.04" fromto="-0.15 0 0.25 -0.4 0 0.25" rgba="0.9 0.7 0.6 1"/>
    </body>
    
    <!-- Robot - slides on track toward human -->
    <body name="robot_base" pos="0 2 1">
      <joint name="robot_slide" type="slide" axis="0 -1 0" range="0 2" damping="50"/>
      <geom name="robot_base_geom" type="box" size="0.08 0.08 0.15" rgba="0.3 0.3 0.3 1"/>
      
      <body name="robot_arm" pos="0 -0.12 0">
        <joint name="robot_arm_joint" type="hinge" axis="1 0 0" range="-0.5 0.5" damping="10"/>
        <geom name="robot_arm_geom" type="capsule" size="0.035" fromto="0 0 0 0 -0.3 0" rgba="0.4 0.5 0.7 1"/>
        
        <body name="robot_ee" pos="0 -0.35 0">
          <geom name="robot_ee_geom" type="sphere" size="0.07" rgba="1 0.3 0.3 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <position name="slide_motor" joint="robot_slide" kp="500" ctrlrange="0 2"/>
    <position name="arm_motor" joint="robot_arm_joint" kp="100" ctrlrange="-1.5 1.5"/>
  </actuator>
</mujoco>
"""


def run_visual_demo():
    """Visual demo showing safety wrapper in action."""
    
    # Create scene
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(VISUAL_SCENE)
        scene_path = f.name
    
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Create safety wrapper
    ssm_config = SSMConfig(T_r=0.1, T_s=0.05, a_max=5.0, C=0.1, v_h_max=0.0)
    wrapper = ISO15066Wrapper(model, data, ssm_config=ssm_config)
    wrapper.add_robot_geom("robot_ee_geom")
    wrapper.add_robot_geom("robot_arm_geom")
    wrapper.add_robot_geom("robot_base_geom")
    
    # Get body IDs
    human_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human")
    robot_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_ee")
    
    print("\n" + "="*70)
    print("ISO 15066 SAFETY VISUAL DEMO")
    print("="*70)
    print("Watch the MuJoCo viewer - robot will approach human automatically")
    print("Console shows real-time safety status")
    print("="*70 + "\n")
    
    # Animation state
    approach_target = 0.0  # Start position
    approach_speed = 0.3   # m/s target movement
    phase = "approach"
    phase_start_time = 0.0
    
    ssm_violated = False
    pfl_violated = False
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        viewer.cam.distance = 4
        viewer.cam.lookat[:] = [0, 0.5, 1]
        
        while viewer.is_running():
            current_time = data.time
            
            # Get positions
            human_pos = data.xpos[human_body_id].copy()
            robot_pos = data.xpos[robot_ee_id].copy()
            robot_vel = np.linalg.norm(data.cvel[robot_ee_id, 3:6])
            
            # Animation control
            if phase == "approach":
                # Move robot toward human - faster approach
                approach_target = min(1.9, approach_target + 0.002)
                data.ctrl[0] = approach_target  # Slide toward human
                data.ctrl[1] = 0.0  # Keep arm level
                
                # Check if we should stop (collision or close enough)
                distance = np.linalg.norm(robot_pos - human_pos)
                if distance < 0.15 or approach_target > 1.85:
                    phase = "contact"
                    phase_start_time = current_time
                    
            elif phase == "contact":
                # Hold position for contact
                if current_time - phase_start_time > 3:
                    phase = "retreat"
                    
            elif phase == "retreat":
                approach_target = max(0.0, approach_target - 0.002)
                data.ctrl[0] = approach_target
                if approach_target < 0.1:
                    phase = "approach"
            
            # Check safety
            safety_info = wrapper.check_safety_no_step(
                robot_pos=robot_pos,
                robot_vel=robot_vel,
                human_pos=human_pos,
                human_vel=0.0,
            )
            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Print status every 0.2s
            if int(current_time * 5) != int((current_time - model.opt.timestep) * 5):
                distance = np.linalg.norm(robot_pos - human_pos)
                
                # SSM status
                if safety_info.ssm_violation:
                    ssm_status = "🔴 SSM VIOLATION"
                    ssm_violated = True
                else:
                    ssm_status = "🟢 SSM OK"
                
                # PFL status
                if safety_info.max_contact_force > 5:
                    if safety_info.pfl_violation:
                        pfl_status = f"🔴 PFL VIOLATION {safety_info.max_contact_force:.0f}N"
                        pfl_violated = True
                    else:
                        pfl_status = f"🟡 PFL Contact {safety_info.max_contact_force:.0f}N ({safety_info.contact_region})"
                else:
                    pfl_status = "⚪ No contact"
                
                print(f"t={current_time:5.1f}s | Dist={distance:.2f}m | Margin={safety_info.ssm_margin:+.2f}m | {ssm_status} | {pfl_status}")
            
            viewer.sync()
            time.sleep(0.001)
    
    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)
    print(f"  SSM Violation Detected: {'✅ YES' if ssm_violated else '❌ NO'}")
    print(f"  PFL Contact Detected:   {'✅ YES' if pfl_violated else '❌ NO'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_visual_demo()
