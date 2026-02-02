"""
Visual Demo: ISO 15066 Safety Monitoring

Demonstrates the safety wrapper in action:
1. SSM - Shows separation distance and required buffer
2. PFL - Shows contact forces and body region limits
3. Contact classification - quasi-static vs transient

Usage:
    mjpython scripts/demo_safety.py
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


# Scene with human, robot, wall, and falling object
DEMO_SCENE_XML = """
<mujoco model="safety_demo">
  <option timestep="0.002"/>
  
  <default>
    <geom contype="1" conaffinity="1"/>
  </default>
  
  <asset>
    <material name="human_skin" rgba="0.9 0.7 0.6 1"/>
    <material name="robot_metal" rgba="0.3 0.5 0.8 1"/>
    <material name="wall_mat" rgba="0.6 0.6 0.6 1"/>
    <material name="safe_zone" rgba="0.2 0.8 0.2 0.3"/>
    <material name="warning_zone" rgba="0.8 0.8 0.2 0.3"/>
    <material name="danger_zone" rgba="0.8 0.2 0.2 0.3"/>
  </asset>
  
  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
    
    <!-- Wall (fixture for quasi-static test) -->
    <geom name="wall" type="box" size="0.05 1 1" pos="-0.8 0 0.5" material="wall_mat"/>
    
    <!-- Human body (simplified SMPL-H style) -->
    <body name="human_root" pos="0 0 0.95">
      <freejoint name="human_free"/>
      
      <!-- Pelvis -->
      <geom name="Pelvis_col" type="capsule" size="0.12" fromto="0 -0.15 0 0 0.15 0" 
            rgba="0.9 0.7 0.6 1"/>
      
      <!-- Torso -->
      <body name="torso" pos="0 0 0.25">
        <geom name="Torso_col" type="capsule" size="0.13" fromto="0 -0.12 0 0 0.12 0"
              rgba="0.9 0.7 0.6 1"/>
        
        <!-- Chest -->
        <body name="chest" pos="0 0 0.25">
          <geom name="Chest_col" type="capsule" size="0.14" fromto="0 -0.14 0 0 0.14 0"
                rgba="0.9 0.7 0.6 1"/>
          
          <!-- Head -->
          <body name="head" pos="0 0 0.25">
            <geom name="Head_col" type="sphere" size="0.1" rgba="0.9 0.7 0.6 1"/>
          </body>
          
          <!-- Right arm -->
          <body name="r_shoulder" pos="0.2 0 0">
            <joint name="r_shoulder_x" type="hinge" axis="1 0 0" range="-2 2"/>
            <joint name="r_shoulder_y" type="hinge" axis="0 1 0" range="-2 2"/>
            <geom name="R_Shoulder_col" type="capsule" size="0.045" fromto="0 0 0 0.25 0 0"
                  rgba="0.9 0.7 0.6 1"/>
            
            <body name="r_elbow" pos="0.3 0 0">
              <joint name="r_elbow_y" type="hinge" axis="0 1 0" range="0 2.5"/>
              <geom name="R_Elbow_col" type="capsule" size="0.04" fromto="0 0 0 0.2 0 0"
                    rgba="0.9 0.7 0.6 1"/>
              
              <body name="r_wrist" pos="0.25 0 0">
                <joint name="r_wrist_y" type="hinge" axis="0 1 0" range="-1 1"/>
                <geom name="R_Wrist_col" type="capsule" size="0.03" fromto="0 0 0 0.08 0 0"
                      rgba="0.9 0.7 0.6 1"/>
              </body>
            </body>
          </body>
          
          <!-- Left arm -->
          <body name="l_shoulder" pos="-0.2 0 0">
            <joint name="l_shoulder_x" type="hinge" axis="1 0 0" range="-2 2"/>
            <joint name="l_shoulder_y" type="hinge" axis="0 1 0" range="-2 2"/>
            <geom name="L_Shoulder_col" type="capsule" size="0.045" fromto="0 0 0 -0.25 0 0"
                  rgba="0.9 0.7 0.6 1"/>
            
            <body name="l_elbow" pos="-0.3 0 0">
              <joint name="l_elbow_y" type="hinge" axis="0 1 0" range="-2.5 0"/>
              <geom name="L_Elbow_col" type="capsule" size="0.04" fromto="0 0 0 -0.2 0 0"
                    rgba="0.9 0.7 0.6 1"/>
              
              <body name="l_wrist" pos="-0.25 0 0">
                <joint name="l_wrist_y" type="hinge" axis="0 1 0" range="-1 1"/>
                <geom name="L_Wrist_col" type="capsule" size="0.03" fromto="0 0 0 -0.08 0 0"
                      rgba="0.9 0.7 0.6 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Robot arm (simplified) -->
    <body name="robot_base" pos="0 1.5 0.5">
      <geom name="robot_base_geom" type="box" size="0.15 0.15 0.25" material="robot_metal"/>
      
      <body name="robot_link1" pos="0 -0.2 0.25">
        <joint name="robot_j1" type="hinge" axis="1 0 0" range="-2 2" damping="10"/>
        <geom name="robot_link1_geom" type="capsule" size="0.05" fromto="0 0 0 0 -0.3 0"
              material="robot_metal"/>
        
        <body name="robot_link2" pos="0 -0.35 0">
          <joint name="robot_j2" type="hinge" axis="1 0 0" range="-2 2" damping="10"/>
          <geom name="robot_link2_geom" type="capsule" size="0.04" fromto="0 0 0 0 -0.25 0"
                material="robot_metal"/>
          
          <body name="robot_ee" pos="0 -0.3 0">
            <geom name="robot_ee_geom" type="sphere" size="0.06" rgba="1 0.3 0.3 1"/>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Falling impact object -->
    <body name="impact_ball" pos="0.6 0 2.5">
      <freejoint name="ball_joint"/>
      <geom name="impact_sphere" type="sphere" size="0.08" mass="2" rgba="1 0.5 0 1"/>
    </body>
    
    <!-- SSM visualization markers -->
    <body name="ssm_marker" mocap="true" pos="0 0 0">
      <geom name="ssm_zone" type="cylinder" size="0.5 0.01" rgba="0.2 0.8 0.2 0.3" 
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="robot_m1" joint="robot_j1" gear="100"/>
    <motor name="robot_m2" joint="robot_j2" gear="100"/>
    <motor name="arm_reach" joint="r_shoulder_y" gear="50"/>
  </actuator>
</mujoco>
"""


def demo_safety():
    """Interactive safety monitoring demo."""
    
    # Create scene
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(DEMO_SCENE_XML)
        scene_path = f.name
    
    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Create safety wrapper
    ssm_config = SSMConfig(
        T_r=0.1,
        T_s=0.05,
        a_max=5.0,
        C=0.1,
        v_h_max=1.6,
    )
    
    wrapper = ISO15066Wrapper(model, data, ssm_config=ssm_config)
    
    # Register robot and fixture geoms
    wrapper.add_robot_geom("robot_ee_geom")
    wrapper.add_robot_geom("robot_link1_geom")
    wrapper.add_robot_geom("robot_link2_geom")
    wrapper.add_robot_geom("impact_sphere")
    wrapper.add_fixture_geom("wall")
    
    # Get body/geom IDs
    robot_ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_ee")
    human_pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_root")
    ssm_mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ssm_marker")
    
    print("\n" + "="*60)
    print("ISO 15066 SAFETY MONITORING DEMO")
    print("="*60)
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  R - Reset scene")
    print("  M - Move robot toward human")
    print("  D - Drop impact ball")
    print("  A - Reach arm toward wall (quasi-static test)")
    print("  Close window to exit")
    print("="*60)
    print("\nLegend:")
    print("  GREEN zone = Safe separation")
    print("  YELLOW = Warning (approaching limit)")
    print("  RED = Violation")
    print("="*60 + "\n")
    
    # State
    paused = False
    robot_target = 0.0
    should_drop_ball = False
    should_reach = False
    
    def key_callback(key):
        nonlocal paused, robot_target, should_drop_ball, should_reach
        # Convert to lowercase for comparison
        key_char = chr(key).lower() if 32 <= key <= 126 else ''
        
        if key == 32:  # SPACE
            paused = not paused
            print(f"[KEY] {'Paused' if paused else 'Resumed'}")
        elif key_char == 'r':
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            robot_target = 0.0
            print("[KEY] Reset scene")
        elif key_char == 'm':
            robot_target += 0.5
            print(f"[KEY] Move robot (target={robot_target:.1f})")
        elif key_char == 'd':
            should_drop_ball = True
            print("[KEY] Drop ball!")
        elif key_char == 'a':
            should_reach = True
            print("[KEY] Reach arm!")
    
    dt = model.opt.timestep
    last_print = 0.0
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if not paused:
                current_time = data.time
                
                # Apply robot control
                data.ctrl[0] = robot_target
                data.ctrl[1] = robot_target * 0.5
                
                # Drop ball command
                if should_drop_ball:
                    ball_qpos_start = 7  # After human's freejoint
                    data.qpos[ball_qpos_start:ball_qpos_start+3] = [0.6, 0, 2.0]
                    data.qvel[ball_qpos_start+3:ball_qpos_start+6] = [0, 0, -3]
                    should_drop_ball = False
                
                # Reach arm toward wall
                if should_reach:
                    data.ctrl[2] = -1.5  # Extend arm to left
                    should_reach = False
                
                # Get positions
                robot_pos = data.xpos[robot_ee_id].copy()
                human_pos = data.xpos[human_pelvis_id].copy()
                
                # Compute robot velocity (approximate)
                robot_vel = np.linalg.norm(data.cvel[robot_ee_id, 3:6])
                
                # Check safety (without stepping - BiGym integration mode)
                safety_info = wrapper.check_safety_no_step(
                    robot_pos=robot_pos,
                    robot_vel=robot_vel,
                    human_pos=human_pos,
                    human_vel=0.0,  # Human stationary in demo
                )
                
                # Update SSM visualization marker
                S_p = ssm_config.compute_separation_distance(robot_vel, 0.0)
                data.mocap_pos[0] = human_pos
                data.mocap_pos[0, 2] = 0.01  # Flat on ground
                
                # Step physics
                mujoco.mj_step(model, data)
                
                # Print status periodically
                if current_time - last_print > 0.5:
                    last_print = current_time
                    
                    # SSM status
                    ssm_status = "🔴 VIOLATION" if safety_info.ssm_violation else "🟢 OK"
                    print(f"t={current_time:.1f}s | SSM: {ssm_status} margin={safety_info.ssm_margin:.2f}m", end="")
                    
                    # PFL status
                    if safety_info.max_contact_force > 0:
                        pfl_status = "🔴 VIOLATION" if safety_info.pfl_violation else "🟢 OK"
                        contact_type = safety_info.contact_type
                        region = safety_info.contact_region or "unknown"
                        
                        # Get limit for region
                        limit = PFL_LIMITS.get(region)
                        if limit:
                            force_limit = limit.get_force_limit(contact_type)
                            print(f" | PFL: {pfl_status} {safety_info.max_contact_force:.0f}N/{force_limit:.0f}N ({region}, {contact_type})")
                        else:
                            print(f" | PFL: {pfl_status} {safety_info.max_contact_force:.0f}N ({region})")
                    else:
                        print(f" | PFL: No contact")
            
            viewer.sync()
            time.sleep(0.001)
    
    print("\nDemo complete.")


if __name__ == "__main__":
    demo_safety()
