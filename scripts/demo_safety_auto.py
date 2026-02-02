"""
Visual Demo: ISO 15066 Safety Monitoring - Automated Tests

Runs three automated test scenarios to prove the safety wrapper works:
1. SSM Test - Robot approaches human until violation
2. PFL Transient - Ball drops onto human arm
3. PFL Quasi-static - Hand clamped between robot and wall

Usage:
    mjpython scripts/demo_safety_auto.py
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


# Scene optimized for guaranteed collisions
DEMO_SCENE_XML = """
<mujoco model="safety_demo_auto">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <default>
    <geom contype="1" conaffinity="1" friction="0.5 0.005 0.0001"/>
  </default>
  
  <asset>
    <material name="human_skin" rgba="0.9 0.7 0.6 1"/>
    <material name="robot_metal" rgba="0.3 0.5 0.8 1"/>
    <material name="wall_mat" rgba="0.5 0.5 0.5 1"/>
  </asset>
  
  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.9 0.9 0.9 1"/>
    
    <!-- Wall for clamping test -->
    <geom name="wall" type="box" size="0.02 0.5 0.5" pos="-0.6 0 0.8" material="wall_mat"/>
    
    <!-- Human (simplified - freejoint for free positioning) -->
    <body name="human" pos="0 0 1">
      <freejoint name="human_joint"/>
      
      <!-- Pelvis (core) -->
      <geom name="Pelvis_col" type="sphere" size="0.15" rgba="0.9 0.7 0.6 1"/>
      
      <!-- Torso -->
      <geom name="Chest_col" type="capsule" size="0.12" fromto="0 0 0 0 0 0.4" rgba="0.9 0.7 0.6 1"/>
      
      <!-- Head -->
      <geom name="Head_col" type="sphere" size="0.1" pos="0 0 0.55" rgba="0.9 0.7 0.6 1"/>
      
      <!-- Right arm - positioned to stick out -->
      <body name="r_arm" pos="0.2 0 0.3">
        <geom name="R_Shoulder_col" type="capsule" size="0.04" fromto="0 0 0 0.2 0 0" rgba="0.9 0.7 0.6 1"/>
        <geom name="R_Elbow_col" type="capsule" size="0.035" fromto="0.2 0 0 0.4 0 0" rgba="0.9 0.7 0.6 1"/>
        <geom name="R_Wrist_col" type="sphere" size="0.03" pos="0.45 0 0" rgba="0.9 0.7 0.6 1"/>
      </body>
      
      <!-- Left arm - extends toward wall for clamping -->
      <body name="l_arm" pos="-0.2 0 0.3">
        <geom name="L_Shoulder_col" type="capsule" size="0.04" fromto="0 0 0 -0.15 0 0" rgba="0.9 0.7 0.6 1"/>
        <geom name="L_Elbow_col" type="capsule" size="0.035" fromto="-0.15 0 0 -0.3 0 0" rgba="0.9 0.7 0.6 1"/>
        <geom name="L_Wrist_col" type="sphere" size="0.03" pos="-0.35 0 0" rgba="0.95 0.75 0.65 1"/>
      </body>
    </body>
    
    <!-- Robot end-effector (mocap for direct control) -->
    <body name="robot_ee" mocap="true" pos="0 1 1">
      <geom name="robot_ee_geom" type="sphere" size="0.08" rgba="0.3 0.5 0.8 1" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Impact ball (mocap for controlled drop) -->
    <body name="impact_ball" mocap="true" pos="0.6 0 2">
      <geom name="impact_sphere" type="sphere" size="0.06" rgba="1 0.4 0 1" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Clamping pusher (mocap for controlled push) -->
    <body name="clamping_pusher" mocap="true" pos="0 0 0.8">
      <geom name="clamping_geom" type="box" size="0.05 0.1 0.1" rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def run_demo():
    """Run automated safety demo with three test scenarios."""
    
    # Create scene
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(DEMO_SCENE_XML)
        scene_path = f.name
    
    # Load model
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # Create safety wrapper
    ssm_config = SSMConfig(T_r=0.1, T_s=0.05, a_max=5.0, C=0.1, v_h_max=0.0)
    wrapper = ISO15066Wrapper(model, data, ssm_config=ssm_config)
    
    # Register robot and fixture geoms
    wrapper.add_robot_geom("robot_ee_geom")
    wrapper.add_robot_geom("impact_sphere")
    wrapper.add_robot_geom("clamping_geom")
    wrapper.add_fixture_geom("wall")
    
    # Get mocap body indices
    robot_mocap_id = model.body("robot_ee").mocapid[0]
    ball_mocap_id = model.body("impact_ball").mocapid[0]
    pusher_mocap_id = model.body("clamping_pusher").mocapid[0]
    human_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human")
    
    # Initial positions
    robot_start = np.array([0, 1.5, 1.0])
    ball_start = np.array([0.65, 0, 2.5])  # Above right arm
    pusher_start = np.array([0.5, 0, 1.3])  # Away from human
    
    # Reset positions
    data.mocap_pos[robot_mocap_id] = robot_start
    data.mocap_pos[ball_mocap_id] = ball_start
    data.mocap_pos[pusher_mocap_id] = pusher_start
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*70)
    print("ISO 15066 SAFETY WRAPPER - AUTOMATED VALIDATION DEMO")
    print("="*70)
    print("This demo runs three scenarios to validate the safety wrapper:\n")
    print("  TEST 1: SSM Violation - Robot approaches human until margin < 0")
    print("  TEST 2: PFL Transient - Ball drops onto forearm (limit: 320N)")
    print("  TEST 3: PFL Quasi-static - Hand clamped between wall and robot (limit: 140N)")
    print("="*70 + "\n")
    
    # State machine
    test_phase = 0
    phase_time = 0.0
    test_results = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_time = data.time
            human_pos = data.xpos[human_body_id].copy()
            robot_pos = data.mocap_pos[robot_mocap_id].copy()
            
            # Compute robot velocity for SSM
            robot_vel = 0.5 if test_phase == 1 else 0.0
            
            # Check safety
            safety_info = wrapper.check_safety_no_step(
                robot_pos=robot_pos,
                robot_vel=robot_vel,
                human_pos=human_pos,
                human_vel=0.0,
            )
            
            # ===== TEST 1: SSM Violation =====
            if test_phase == 0:
                print(">>> TEST 1: SSM (Speed & Separation Monitoring)")
                print("    Robot will approach human at 0.5 m/s")
                print("    Expected: margin decreases, then VIOLATION when < 0")
                test_phase = 1
                phase_time = current_time
            
            elif test_phase == 1:
                # Move robot toward human
                direction = (human_pos - robot_pos)
                direction[2] = 0  # Keep same height
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                data.mocap_pos[robot_mocap_id] += direction * 0.001  # Slow approach
                
                # Print status
                if int(current_time * 10) % 10 == 0:  # Every 1 second
                    status = "🔴 VIOLATION" if safety_info.ssm_violation else "🟢 OK"
                    dist = np.linalg.norm(robot_pos - human_pos)
                    print(f"    [SSM] Distance: {dist:.2f}m | Margin: {safety_info.ssm_margin:.2f}m | {status}")
                
                # Check for violation
                if safety_info.ssm_violation:
                    print("\n    ✅ SSM VIOLATION DETECTED!")
                    print(f"    Final margin: {safety_info.ssm_margin:.3f}m (negative = violation)")
                    test_results.append(("SSM Violation", True))
                    test_phase = 2
                    phase_time = current_time
                    # Reset robot position
                    data.mocap_pos[robot_mocap_id] = robot_start
                    time.sleep(1)
                    
            # ===== TEST 2: PFL Transient (Ball Drop) =====
            elif test_phase == 2:
                print("\n>>> TEST 2: PFL Transient (Impact)")
                print("    Ball will drop onto human's forearm")
                print("    Forearm transient limit: 320N")
                test_phase = 3
                phase_time = current_time
                # Position ball above forearm
                data.mocap_pos[ball_mocap_id] = np.array([0.5, 0, 1.8])
                
            elif test_phase == 3:
                dt_test = current_time - phase_time
                
                # Drop ball with physics-like motion
                drop_speed = 3.0  # m/s downward
                data.mocap_pos[ball_mocap_id][2] -= drop_speed * 0.002
                
                # Check for contact
                if safety_info.max_contact_force > 1:
                    region = safety_info.contact_region
                    contact_type = safety_info.contact_type
                    force = safety_info.max_contact_force
                    limit = PFL_LIMITS.get(region)
                    limit_val = limit.transient_force if limit else 999
                    
                    status = "🔴 VIOLATION" if safety_info.pfl_violation else "🟢 OK"
                    print(f"    [PFL] Force: {force:.0f}N / {limit_val}N limit | Region: {region} | Type: {contact_type} | {status}")
                    
                    if force > 10:  # Significant contact
                        print(f"\n    ✅ PFL CONTACT DETECTED!")
                        print(f"    Force: {force:.1f}N on {region}")
                        print(f"    Contact type: {contact_type}")
                        print(f"    Force ratio: {safety_info.pfl_force_ratio:.2f}")
                        test_results.append(("PFL Transient", True))
                        test_phase = 4
                        phase_time = current_time
                        # Move ball away
                        data.mocap_pos[ball_mocap_id] = ball_start
                        time.sleep(1)
                
                if dt_test > 3:  # Timeout
                    print("    ⚠️ No significant contact detected (timeout)")
                    test_results.append(("PFL Transient", False))
                    test_phase = 4
                    phase_time = current_time
                    
            # ===== TEST 3: PFL Quasi-static (Clamping) =====
            elif test_phase == 4:
                print("\n>>> TEST 3: PFL Quasi-static (Clamping)")
                print("    Pusher will clamp hand against wall")
                print("    Hand quasi-static limit: 140N (stricter than 280N transient)")
                test_phase = 5
                phase_time = current_time
                # Position pusher near left hand (which is near the wall)
                # Left wrist is at human pos + (-0.55, 0, 0.3)
                data.mocap_pos[pusher_mocap_id] = np.array([-0.3, 0, 1.3])
                
            elif test_phase == 5:
                dt_test = current_time - phase_time
                
                # Move pusher toward wall (clamping the hand)
                data.mocap_pos[pusher_mocap_id][0] -= 0.001  # Move left toward wall
                
                # Check for quasi-static contact
                if safety_info.max_contact_force > 1:
                    region = safety_info.contact_region
                    contact_type = safety_info.contact_type
                    force = safety_info.max_contact_force
                    
                    status = "🔴 VIOLATION" if safety_info.pfl_violation else "🟢 OK"
                    print(f"    [PFL] Force: {force:.0f}N | Region: {region} | Type: {contact_type} | {status}")
                    
                    if contact_type == "quasi_static":
                        print(f"\n    ✅ QUASI-STATIC CLAMPING DETECTED!")
                        print(f"    This means hand is contacting BOTH wall AND robot")
                        test_results.append(("PFL Quasi-static", True))
                        test_phase = 6
                        phase_time = current_time
                        time.sleep(1)
                    elif force > 10 and dt_test > 2:
                        print(f"\n    ⚠️ Contact detected but classified as transient")
                        print(f"    (Hand may not be touching wall)")
                        test_results.append(("PFL Quasi-static", False))
                        test_phase = 6
                        phase_time = current_time
                
                if dt_test > 5:  # Timeout
                    print("    ⚠️ Test timeout")
                    test_results.append(("PFL Quasi-static", False))
                    test_phase = 6
                    phase_time = current_time
                    
            # ===== Results =====
            elif test_phase == 6:
                print("\n" + "="*70)
                print("VALIDATION RESULTS")
                print("="*70)
                for name, passed in test_results:
                    status = "✅ PASS" if passed else "❌ FAIL"
                    print(f"  {name}: {status}")
                print("="*70)
                test_phase = 7
                
            elif test_phase == 7:
                # Just keep running for observation
                pass
            
            # Step physics
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
    
    print("\nDemo complete.")


if __name__ == "__main__":
    run_demo()
