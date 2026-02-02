"""
Minimal Validation: ISO 15066 Safety Wrapper

Simple, direct test that proves the wrapper logic works.
No complex physics - just direct position manipulation.

Usage:
    mjpython scripts/test_safety_minimal.py
"""

import numpy as np
import mujoco
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safety_bigym.safety import ISO15066Wrapper, SSMConfig, PFL_LIMITS


MINIMAL_SCENE = """
<mujoco model="minimal_safety_test">
  <option timestep="0.002"/>
  
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    
    <!-- Wall fixture -->
    <geom name="wall" type="box" size="0.02 0.5 0.5" pos="-0.5 0 0.5" 
          contype="1" conaffinity="1"/>
    
    <!-- Human (fixed in place for simple test) -->
    <body name="human" pos="0 0 1">
      <geom name="Pelvis_col" type="sphere" size="0.1" contype="1" conaffinity="1"/>
      <geom name="Head_col" type="sphere" size="0.08" pos="0 0 0.4" contype="1" conaffinity="1"/>
      <geom name="R_Elbow_col" type="sphere" size="0.04" pos="0.3 0 0" contype="1" conaffinity="1"/>
      <geom name="L_Wrist_col" type="sphere" size="0.03" pos="-0.35 0 0" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Robot sphere (can be positioned anywhere) -->
    <body name="robot" pos="0 2 1">
      <freejoint name="robot_joint"/>
      <geom name="robot_geom" type="sphere" size="0.1" mass="5" contype="1" conaffinity="1" rgba="0.3 0.5 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_ssm():
    """Test SSM violation detection."""
    print("\n" + "="*60)
    print("TEST 1: SSM (Speed & Separation Monitoring)")
    print("="*60)
    
    # Create model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(MINIMAL_SCENE)
        scene_path = f.name
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    
    # Create wrapper
    ssm_config = SSMConfig(T_r=0.1, T_s=0.05, a_max=5.0, C=0.1, v_h_max=0.0)
    wrapper = ISO15066Wrapper(model, data, ssm_config=ssm_config)
    wrapper.add_robot_geom("robot_geom")
    
    # Human at origin, robot starts at (0, 2, 1)
    human_pos = np.array([0, 0, 1])
    
    print(f"\nHuman position: {human_pos}")
    print(f"SSM Safety distance formula: S_p = v_h*(T_r+T_s) + v_r*T_r + v_r²/(2*a_max) + C")
    print(f"With v_h=0, v_r=0.5: S_p = 0 + 0.5*0.1 + 0.25/10 + 0.1 = 0.175m")
    
    # Test at different robot positions
    test_positions = [
        (np.array([0, 1.5, 1]), 0.0, "Far away, v=0"),
        (np.array([0, 0.5, 1]), 0.0, "Close, v=0"),
        (np.array([0, 0.3, 1]), 0.0, "Very close, v=0"),
        (np.array([0, 0.2, 1]), 0.0, "Touching distance, v=0"),
        (np.array([0, 0.5, 1]), 0.5, "Close, v=0.5 m/s"),
        (np.array([0, 0.3, 1]), 0.5, "Very close, v=0.5 m/s"),
        (np.array([0, 0.15, 1]), 0.5, "0.15m away, v=0.5 m/s (should violate)"),
    ]
    
    print(f"\n{'Robot Position':<25} {'Dist':<8} {'v_r':<6} {'S_p':<8} {'Margin':<10} {'Status'}")
    print("-"*70)
    
    violations = 0
    for robot_pos, robot_vel, desc in test_positions:
        is_violation, margin, dist = wrapper.compute_ssm(robot_pos, robot_vel, human_pos, 0.0)
        S_p = dist - margin
        status = "🔴 VIOLATION" if is_violation else "🟢 OK"
        if is_violation:
            violations += 1
        print(f"{str(robot_pos):<25} {dist:.2f}m   {robot_vel:.1f}    {S_p:.3f}m   {margin:+.3f}m    {status}")
    
    print(f"\nResult: {violations} violation(s) detected")
    return violations > 0


def test_pfl_limits():
    """Test PFL body region limits."""
    print("\n" + "="*60)
    print("TEST 2: PFL Body Region Limits")
    print("="*60)
    
    test_regions = [
        ("Head_col", "skull", 130, 260),
        ("R_Elbow_col", "forearm", 160, 320),
        ("L_Wrist_col", "hand_palm", 140, 280),
    ]
    
    print(f"\n{'Geom Name':<15} {'Region':<12} {'QS Limit':<12} {'Trans Limit'}")
    print("-"*50)
    
    all_correct = True
    for geom_name, expected_region, expected_qs, expected_trans in test_regions:
        from safety_bigym.safety import get_region_for_geom, get_limits_for_geom
        
        region = get_region_for_geom(geom_name)
        limits = get_limits_for_geom(geom_name)
        
        if region != expected_region:
            print(f"❌ {geom_name}: Region mismatch - expected {expected_region}, got {region}")
            all_correct = False
        elif limits.quasi_static_force != expected_qs:
            print(f"❌ {geom_name}: QS limit mismatch")
            all_correct = False
        elif limits.transient_force != expected_trans:
            print(f"❌ {geom_name}: Trans limit mismatch")
            all_correct = False
        else:
            print(f"✅ {geom_name:<15} {region:<12} {limits.quasi_static_force}N          {limits.transient_force}N")
    
    return all_correct


def test_pfl_contact():
    """Test PFL contact force detection."""
    print("\n" + "="*60)
    print("TEST 3: PFL Contact Force Detection")
    print("="*60)
    
    # Create model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(MINIMAL_SCENE)
        scene_path = f.name
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # Create wrapper
    wrapper = ISO15066Wrapper(model, data)
    wrapper.add_robot_geom("robot_geom")
    wrapper.add_fixture_geom("wall")
    
    # Get robot freejoint qpos index
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    
    # Move robot to collide with forearm
    print("\nPositioning robot to collide with forearm (R_Elbow_col)...")
    data.qpos[0:3] = [0.25, 0, 1]  # Position robot at forearm location
    data.qvel[1] = -10  # Moving toward human
    
    # Step simulation
    contacts_detected = False
    max_force = 0.0
    contact_region = ""
    
    for _ in range(100):
        mujoco.mj_step(model, data)
        safety_info = wrapper.check_safety_no_step()
        
        if safety_info.max_contact_force > max_force:
            max_force = safety_info.max_contact_force
            contact_region = safety_info.contact_region
            contacts_detected = True
    
    if contacts_detected:
        print(f"✅ Contact detected!")
        print(f"   Max force: {max_force:.1f}N")
        print(f"   Body region: {contact_region}")
        print(f"   Contact type: {safety_info.contact_type}")
    else:
        print(f"⚠️  No contact detected")
        print(f"   Robot final pos: {data.qpos[0:3]}")
    
    return contacts_detected


def test_contact_classification():
    """Test quasi-static vs transient classification."""
    print("\n" + "="*60)
    print("TEST 4: Contact Classification (Transient vs Quasi-static)")
    print("="*60)
    
    # Create model
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(MINIMAL_SCENE)
        scene_path = f.name
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    wrapper = ISO15066Wrapper(model, data)
    wrapper.add_robot_geom("robot_geom")
    wrapper.add_fixture_geom("wall")
    
    # Get IDs
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "L_Wrist_col")
    robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_geom")
    
    print("\nScenario A: Robot touches wrist (NO wall contact)")
    wrapper._human_contacts_this_step.clear()
    contact_type = wrapper._classify_contact_type(wrist_id, robot_id)
    print(f"   Classification: {contact_type}")
    result_a = contact_type == "transient"
    print(f"   Expected: transient → {'✅' if result_a else '❌'}")
    
    print("\nScenario B: Robot touches wrist WHILE wrist touches wall (clamping)")
    wrapper._human_contacts_this_step.clear()
    wrapper._human_contacts_this_step["L_Wrist_col"].add("wall")  # Wrist also touching wall
    contact_type = wrapper._classify_contact_type(wrist_id, robot_id)
    print(f"   Classification: {contact_type}")
    result_b = contact_type == "quasi_static"
    print(f"   Expected: quasi_static → {'✅' if result_b else '❌'}")
    
    if result_b:
        print("\n   IMPORTANT: Quasi-static limit (140N) is stricter than transient (280N)")
    
    return result_a and result_b


def main():
    print("\n" + "="*60)
    print("ISO 15066 SAFETY WRAPPER - MINIMAL VALIDATION")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("SSM Violation Detection", test_ssm()))
    results.append(("PFL Body Region Limits", test_pfl_limits()))
    results.append(("PFL Contact Detection", test_pfl_contact()))
    results.append(("Contact Classification", test_contact_classification()))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - ISO 15066 wrapper is working correctly!")
    else:
        print("⚠️  Some tests failed - review implementation")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
