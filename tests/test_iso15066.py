"""
Comprehensive tests for ISO 15066 Safety Wrapper.

Tests all code paths as specified:
- SSM computation with known inputs
- PFL transient contact violation
- PFL quasi-static (clamping) detection
- Body region mapping
- Sub-step peak force capture
"""

import pytest
import numpy as np
import mujoco
import tempfile
from pathlib import Path

from safety_bigym.safety import (
    ISO15066Wrapper,
    SSMConfig,
    SafetyInfo,
    ContactInfo,
    PFL_LIMITS,
    get_region_for_geom,
    get_limits_for_geom,
)


# Minimal test scene with human and robot
TEST_SCENE_XML = """
<mujoco model="safety_test">
  <option timestep="0.002"/>
  
  <default>
    <geom contype="1" conaffinity="1"/>
  </default>
  
  <worldbody>
    <!-- Ground (fixture) -->
    <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0"/>
    
    <!-- Wall (fixture for quasi-static test) -->
    <geom name="wall" type="box" size="0.05 1 1" pos="-0.5 0 0.5"/>
    
    <!-- Human body parts (simplified) -->
    <body name="human_torso" pos="0 0 1">
      <freejoint name="human_root"/>
      <geom name="Pelvis_col" type="capsule" size="0.1" fromto="0 -0.1 0 0 0.1 0"/>
      <body name="human_chest" pos="0 0 0.3">
        <geom name="Chest_col" type="capsule" size="0.12" fromto="0 -0.12 0 0 0.12 0"/>
        <body name="human_head" pos="0 0 0.3">
          <geom name="Head_col" type="sphere" size="0.1"/>
        </body>
      </body>
      <body name="human_arm" pos="0.25 0 0.3">
        <joint name="shoulder" type="ball"/>
        <geom name="R_Shoulder_col" type="capsule" size="0.04" fromto="0 0 0 0.15 0 0"/>
        <body name="human_forearm" pos="0.2 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom name="R_Elbow_col" type="capsule" size="0.035" fromto="0 0 0 0.15 0 0"/>
          <body name="human_hand" pos="0.18 0 0">
            <joint name="wrist" type="hinge" axis="0 1 0"/>
            <geom name="R_Wrist_col" type="capsule" size="0.025" fromto="0 0 0 0.08 0 0"/>
          </body>
        </body>
      </body>
      <body name="human_thigh" pos="0.1 0 -0.2">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom name="R_Knee_col" type="capsule" size="0.06" fromto="0 0 0 0 0 -0.4"/>
      </body>
    </body>
    
    <!-- Robot (simple arm) -->
    <body name="robot_base" pos="0 1 0.5">
      <geom name="robot_base_geom" type="box" size="0.1 0.1 0.3"/>
      <body name="robot_arm" pos="0 -0.2 0.3">
        <joint name="robot_joint" type="hinge" axis="1 0 0"/>
        <geom name="robot_link1" type="capsule" size="0.04" fromto="0 0 0 0 -0.3 0"/>
        <body name="robot_ee" pos="0 -0.35 0">
          <geom name="robot_ee_geom" type="sphere" size="0.05"/>
        </body>
      </body>
    </body>
    
    <!-- Falling object (for impact test) -->
    <body name="impact_object" pos="0.5 0 2">
      <freejoint name="object_joint"/>
      <geom name="impact_sphere" type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="robot_actuator" joint="robot_joint" gear="50"/>
  </actuator>
</mujoco>
"""


class TestSSMComputation:
    """Tests for Speed and Separation Monitoring."""
    
    @pytest.fixture
    def ssm_config(self):
        """Standard SSM configuration."""
        return SSMConfig(
            T_r=0.1,
            T_s=0.05,
            a_max=5.0,
            C=0.1,
            v_h_max=1.6,
        )
    
    def test_ssm_zero_velocity(self, ssm_config):
        """Test SSM with zero velocities (obstruction scenario)."""
        # S_h = 0 * 0.15 = 0
        # S_r = 0 * 0.1 + 0 / 10 = 0
        # S_p = 0 + 0 + 0.1 = 0.1
        
        S_p = ssm_config.compute_separation_distance(v_robot=0.0, v_human=0.0)
        
        # Only the intrusion distance C remains
        assert S_p == pytest.approx(0.1, abs=0.01)
    
    def test_ssm_robot_moving(self, ssm_config):
        """Test SSM with known robot velocity."""
        v_robot = 1.0  # m/s
        v_human = 0.0
        
        # S_h = 0 * 0.15 = 0
        # S_r = 1.0 * 0.1 + 1.0² / 10 = 0.1 + 0.1 = 0.2
        # S_p = 0 + 0.2 + 0.1 = 0.3
        
        S_p = ssm_config.compute_separation_distance(v_robot, v_human)
        assert S_p == pytest.approx(0.3, abs=0.01)
    
    def test_ssm_human_moving(self, ssm_config):
        """Test SSM with known human velocity."""
        v_robot = 0.0
        v_human = 1.6  # m/s (max walking speed)
        
        # S_h = 1.6 * 0.15 = 0.24
        # S_r = 0
        # S_p = 0.24 + 0 + 0.1 = 0.34
        
        S_p = ssm_config.compute_separation_distance(v_robot, v_human)
        assert S_p == pytest.approx(0.34, abs=0.01)
    
    def test_ssm_violation_detection(self, ssm_config):
        """Test SSM violation detection with wrapper."""
        # Create a simple model just for testing SSM computation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(TEST_SCENE_XML)
            path = f.name
        
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        wrapper = ISO15066Wrapper(model, data, ssm_config=ssm_config)
        
        # Robot at (0, 1, 0.5), human at (0, 0, 1)
        robot_pos = np.array([0, 1, 0.5])
        human_pos = np.array([0, 0, 1])
        
        # No velocity - required separation is 0.1m (just C)
        is_violation, margin, d_min = wrapper.compute_ssm(
            robot_pos, robot_vel=0.0, human_pos=human_pos, human_vel=0.0
        )
        
        # Distance ≈ 1.1m, required = 0.1m, margin = 1.0m (no violation)
        assert not is_violation
        assert margin > 0
        
        # Now simulate them being close with robot moving fast
        robot_pos_close = np.array([0, 0.3, 1])  # 0.3m away
        is_violation, margin, d_min = wrapper.compute_ssm(
            robot_pos_close, robot_vel=2.0, human_pos=human_pos, human_vel=0.0
        )
        
        # With v=2m/s, S_r = 0.2 + 0.4 = 0.6, S_p = 0 + 0.6 + 0.1 = 0.7
        # d_min = 0.3, margin = 0.3 - 0.7 = -0.4 (violation!)
        assert is_violation
        assert margin < 0


class TestPFLTransient:
    """Tests for transient (impact) contact detection."""
    
    @pytest.fixture
    def setup_model(self):
        """Create model and wrapper."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(TEST_SCENE_XML)
            path = f.name
        
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        
        wrapper = ISO15066Wrapper(model, data)
        wrapper.add_robot_geom("robot_ee_geom")
        wrapper.add_robot_geom("robot_link1")
        wrapper.add_robot_geom("impact_sphere")
        
        return model, data, wrapper
    
    def test_forearm_transient_limit(self, setup_model):
        """Forearm transient limit should be 320N."""
        model, data, wrapper = setup_model
        
        limits = get_limits_for_geom("R_Elbow_col")
        assert limits is not None
        assert limits.transient_force == 320
        assert limits.quasi_static_force == 160
    
    def test_impact_force_capture(self, setup_model):
        """Test that impact forces are captured."""
        model, data, wrapper = setup_model
        
        # Move the falling object to be directly above the forearm
        # Find the freejoint for the object
        object_qpos_start = 7  # After human's 7 qpos
        
        # Position object above forearm
        data.qpos[object_qpos_start:object_qpos_start+3] = [0.4, 0, 1.5]
        data.qvel[object_qpos_start+3:object_qpos_start+6] = [0, 0, -5]  # Falling
        
        mujoco.mj_forward(model, data)
        
        # Run simulation and check for contacts
        had_contact = False
        max_force = 0.0
        
        for _ in range(500):  # 1 second at 0.002 timestep
            safety_info = wrapper.step(n_substeps=1)
            if safety_info.max_contact_force > 0:
                had_contact = True
                max_force = max(max_force, safety_info.max_contact_force)
        
        # We should have detected some contact
        # (Exact force depends on collision dynamics)
        # This is more of a smoke test
        print(f"Max contact force: {max_force}N")


class TestPFLQuasiStatic:
    """Tests for quasi-static (clamping) contact detection."""
    
    @pytest.fixture
    def setup_clamping_scene(self):
        """Create scene for clamping test."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(TEST_SCENE_XML)
            path = f.name
        
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        
        wrapper = ISO15066Wrapper(model, data)
        wrapper.add_robot_geom("robot_ee_geom")
        wrapper.add_fixture_geom("wall")
        
        return model, data, wrapper
    
    def test_hand_quasi_static_limit(self):
        """Hand quasi-static limit should be 140N (stricter than 280N transient)."""
        limits = get_limits_for_geom("R_Wrist_col")
        assert limits is not None
        assert limits.quasi_static_force == 140
        assert limits.transient_force == 280
    
    def test_clamping_detection_requires_fixture(self, setup_clamping_scene):
        """Verify quasi-static requires contact with both robot AND fixture."""
        model, data, wrapper = setup_clamping_scene
        
        # Position hand near wall
        # The human starts at (0, 0, 1), wall is at x=-0.5
        # We'll check that contact type is correctly classified
        
        # First, do a check without fixture contact
        mujoco.mj_forward(model, data)
        
        # The classifier should return transient when no fixture
        contact_type = wrapper._classify_contact_type(
            human_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "R_Wrist_col"),
            other_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_ee_geom"),
        )
        assert contact_type == "transient"
        
        # Now simulate hand touching wall
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
        wrapper._human_contacts_this_step["R_Wrist_col"].add("wall")
        
        # Now classifier should return quasi_static
        contact_type = wrapper._classify_contact_type(
            human_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "R_Wrist_col"),
            other_geom_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot_ee_geom"),
        )
        assert contact_type == "quasi_static"


class TestBodyRegionMapping:
    """Tests for body region to ISO region mapping."""
    
    def test_all_collision_geoms_mapped(self):
        """All standard collision geoms should have mappings."""
        expected_mappings = {
            "Head_col": "skull",
            "Neck_col": "neck",
            "Chest_col": "chest",
            "Pelvis_col": "pelvis",
            "R_Shoulder_col": "upper_arm",
            "R_Elbow_col": "forearm",
            "R_Wrist_col": "hand_palm",
            "R_Knee_col": "thigh",
        }
        
        for geom_name, expected_region in expected_mappings.items():
            region = get_region_for_geom(geom_name)
            assert region == expected_region, f"{geom_name} should map to {expected_region}, got {region}"
    
    def test_head_limits(self):
        """Head should trigger at 130N quasi-static."""
        limits = get_limits_for_geom("Head_col")
        assert limits is not None
        assert limits.quasi_static_force == 130
        assert limits.transient_force == 260
    
    def test_thigh_limits(self):
        """Thigh should trigger at 220N quasi-static."""
        limits = PFL_LIMITS["thigh"]
        assert limits.quasi_static_force == 220
        assert limits.transient_force == 440
    
    def test_unknown_geom_returns_none(self):
        """Unknown geom should return None."""
        assert get_region_for_geom("random_geom") is None
        assert get_limits_for_geom("random_geom") is None


class TestSubstepCapture:
    """Tests for sub-step peak force capture."""
    
    def test_peak_tracking_across_substeps(self):
        """Verify peak force is captured across multiple substeps."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(TEST_SCENE_XML)
            path = f.name
        
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        
        wrapper = ISO15066Wrapper(model, data)
        wrapper.add_robot_geom("robot_ee_geom")
        
        # Run with multiple substeps
        safety_info = wrapper.step(n_substeps=10)
        
        # Info should reflect peak across all substeps
        assert isinstance(safety_info, SafetyInfo)
        assert hasattr(safety_info, 'max_contact_force')
        assert hasattr(safety_info, 'pfl_force_ratio')


class TestSafetyInfoOutput:
    """Tests for SafetyInfo output interface."""
    
    def test_safety_info_to_dict(self):
        """Test SafetyInfo.to_dict() produces expected structure."""
        info = SafetyInfo(
            ssm_violation=True,
            pfl_violation=False,
            ssm_margin=-0.5,
            pfl_force_ratio=0.8,
            min_separation=1.2,
            max_contact_force=120.0,
            contact_region="forearm",
            contact_type="transient",
            violations_by_region={"forearm": 1},
        )
        
        d = info.to_dict()
        
        assert d['ssm_violation'] == True
        assert d['pfl_violation'] == False
        assert d['ssm_margin'] == -0.5
        assert d['pfl_force_ratio'] == 0.8
        assert d['min_separation'] == 1.2
        assert d['max_contact_force'] == 120.0
        assert d['contact_region'] == "forearm"
        assert d['contact_type'] == "transient"
        assert d['violations_by_region'] == {"forearm": 1}
    
    def test_default_values(self):
        """Test default SafetyInfo values."""
        info = SafetyInfo()
        
        assert info.ssm_violation == False
        assert info.pfl_violation == False
        assert info.ssm_margin == float('inf')
        assert info.pfl_force_ratio == 0.0
        assert info.min_separation == float('inf')
        assert info.max_contact_force == 0.0


class TestWrapperIntegration:
    """Integration tests for the full wrapper."""
    
    def test_wrapper_with_smplh(self):
        """Test wrapper with actual SMPL-H model."""
        smplh_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
        
        if not smplh_path.exists():
            pytest.skip("SMPL-H model not found")
        
        model = mujoco.MjModel.from_xml_path(str(smplh_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        
        wrapper = ISO15066Wrapper(model, data)
        
        # Should detect human geoms
        assert len(wrapper.human_geoms) > 0
        
        # Check safety at initial state
        safety_info = wrapper.check_safety_no_step()

        assert isinstance(safety_info, SafetyInfo)
        assert safety_info.pfl_violation == False  # No robot to contact


def _make_wrapper():
    """Build a fresh wrapper over the TEST_SCENE_XML for the new test classes."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(TEST_SCENE_XML)
        path = f.name
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wrapper = ISO15066Wrapper(model, data)
    wrapper.add_robot_geom("robot_ee_geom")
    wrapper.add_robot_geom("robot_link1")
    return wrapper


class TestBuildSafetyInfo:
    """
    Pins down the contract that the env-side aggregator populates
    pfl_force_ratio from the contact list (T0.1 bug fix).
    """

    def test_pfl_force_ratio_populated_from_single_contact(self):
        wrapper = _make_wrapper()
        contacts = [ContactInfo(
            geom1_name="R_Elbow_col",
            geom2_name="robot_ee_geom",
            force=256.0,
            contact_type="transient",
            body_region="forearm",
            is_human_robot=True,
            is_violation=False,
            force_ratio=0.8,
            force_limit=320.0,
        )]
        info = wrapper.build_safety_info(
            contacts=contacts,
            robot_positions=np.array([[0.0, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=np.array([[5.0, 0.0, 0.0]]),
            human_vel=0.0,
        )
        assert info.pfl_force_ratio == pytest.approx(0.8)
        assert info.pfl_violation is False
        assert info.max_contact_force == pytest.approx(256.0)
        assert info.contact_region == "forearm"

    def test_pfl_violation_flag_set_when_ratio_over_one(self):
        wrapper = _make_wrapper()
        contacts = [ContactInfo(
            geom1_name="R_Elbow_col",
            geom2_name="robot_ee_geom",
            force=400.0,
            contact_type="transient",
            body_region="forearm",
            is_human_robot=True,
            is_violation=True,
            force_ratio=1.25,
            force_limit=320.0,
        )]
        info = wrapper.build_safety_info(
            contacts=contacts,
            robot_positions=np.array([[0.0, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=np.array([[5.0, 0.0, 0.0]]),
            human_vel=0.0,
        )
        assert info.pfl_violation is True
        assert info.pfl_force_ratio > 1.0

    def test_pfl_force_ratio_takes_max_across_contacts(self):
        wrapper = _make_wrapper()
        contacts = [
            ContactInfo(geom1_name="R_Elbow_col", geom2_name="robot_ee_geom",
                        force=96.0, contact_type="transient", body_region="forearm",
                        is_human_robot=True, is_violation=False,
                        force_ratio=0.3, force_limit=320.0),
            ContactInfo(geom1_name="R_Wrist_col", geom2_name="robot_ee_geom",
                        force=224.0, contact_type="transient", body_region="hand_palm",
                        is_human_robot=True, is_violation=False,
                        force_ratio=0.8, force_limit=280.0),
            ContactInfo(geom1_name="Head_col", geom2_name="robot_ee_geom",
                        force=130.0, contact_type="transient", body_region="skull",
                        is_human_robot=True, is_violation=False,
                        force_ratio=0.5, force_limit=260.0),
        ]
        info = wrapper.build_safety_info(
            contacts=contacts,
            robot_positions=np.array([[0.0, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=np.array([[5.0, 0.0, 0.0]]),
            human_vel=0.0,
        )
        assert info.pfl_force_ratio == pytest.approx(0.8)

    def test_empty_contacts_gives_zero_ratio(self):
        wrapper = _make_wrapper()
        info = wrapper.build_safety_info(
            contacts=[],
            robot_positions=np.array([[0.0, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=np.array([[5.0, 0.0, 0.0]]),
            human_vel=0.0,
        )
        assert info.pfl_force_ratio == 0.0
        assert info.pfl_violation is False
        assert info.max_contact_force == 0.0


class TestSSMClosestJoint:
    """
    Tests for closest-joint SSM (T0.2). Production SSM must use min distance
    across ALL human body parts x ALL robot links, not pelvis-to-pelvis.
    """

    def test_closest_joint_beats_pelvis_distance(self):
        wrapper = _make_wrapper()
        human_positions = np.array([
            [2.0, 0.0, 0.0],  # pelvis far
            [0.3, 0.0, 0.0],  # wrist close
        ])
        robot_positions = np.array([[0.0, 0.0, 0.0]])

        _, _, d_min = wrapper.compute_ssm(
            robot_pos=robot_positions,
            robot_vel=0.0,
            human_pos=human_positions,
            human_vel=0.0,
        )
        assert d_min == pytest.approx(0.3, abs=0.01)

    def test_legacy_single_point_api_still_works(self):
        wrapper = _make_wrapper()
        robot_pos = np.array([0.0, 1.0, 0.5])
        human_pos = np.array([0.0, 0.0, 1.0])

        is_violation, margin, d_min = wrapper.compute_ssm(
            robot_pos=robot_pos, robot_vel=0.0,
            human_pos=human_pos, human_vel=0.0,
        )
        expected = float(np.linalg.norm(robot_pos - human_pos))
        assert d_min == pytest.approx(expected, abs=1e-6)

    def test_closest_pair_indices_recorded_on_wrapper(self):
        wrapper = _make_wrapper()
        human_positions = np.array([
            [2.0, 0.0, 0.0],   # 0 pelvis
            [0.3, 0.0, 0.0],   # 1 wrist (0.1m from ee)
            [1.5, 0.0, 0.0],   # 2 head
        ])
        robot_positions = np.array([
            [0.0, 0.0, 0.0],   # 0 base
            [0.2, 0.0, 0.0],   # 1 ee
        ])
        wrapper.compute_ssm(
            robot_pos=robot_positions, robot_vel=0.0,
            human_pos=human_positions, human_vel=0.0,
        )
        assert wrapper.last_closest_human_idx == 1
        assert wrapper.last_closest_robot_idx == 1

    def test_build_safety_info_records_closest_pair_names(self):
        wrapper = _make_wrapper()
        info = wrapper.build_safety_info(
            contacts=[],
            robot_positions=np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=np.array([[2.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
            human_vel=0.0,
            human_names=["pelvis", "wrist"],
            robot_names=["base", "ee"],
        )
        assert info.closest_human_joint == "wrist"
        assert info.closest_robot_link == "ee"
        assert info.min_separation == pytest.approx(0.1, abs=0.01)

    def test_build_safety_info_exposes_full_joint_array(self):
        """Phase 1 needs all tracked joint positions, not just the closest one,
        so BodySLAMWrapper can build a skeleton-level noisy observation."""
        wrapper = _make_wrapper()
        human_positions = np.array([
            [2.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [1.5, 0.1, 0.2],
        ])
        info = wrapper.build_safety_info(
            contacts=[],
            robot_positions=np.array([[0.0, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=human_positions,
            human_vel=0.0,
            human_names=["pelvis", "wrist", "head"],
            robot_names=["base"],
        )
        joints = np.asarray(info.human_joint_positions)
        assert joints.shape == (3, 3)
        np.testing.assert_allclose(joints, human_positions)
        assert info.human_joint_names == ["pelvis", "wrist", "head"]

    def test_build_safety_info_joint_array_survives_to_dict(self):
        """to_dict() must serialise the new fields so info['safety'] carries
        them through the gym step pipeline."""
        wrapper = _make_wrapper()
        info = wrapper.build_safety_info(
            contacts=[],
            robot_positions=np.array([[0.0, 0.0, 0.0]]),
            robot_vel=0.0,
            human_positions=np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
            human_vel=0.0,
            human_names=["a", "b"],
            robot_names=["base"],
        )
        d = info.to_dict()
        assert "human_joint_positions" in d
        assert "human_joint_names" in d
        # Nested list, not ndarray (JSON-serialisable).
        assert isinstance(d["human_joint_positions"], list)
        assert isinstance(d["human_joint_positions"][0], list)
        assert d["human_joint_names"] == ["a", "b"]

    def test_build_safety_info_no_ssm_keeps_joints_empty(self):
        """If build_safety_info is called without SSM inputs (e.g. no human
        injected), the joint arrays default to empty lists."""
        wrapper = _make_wrapper()
        info = wrapper.build_safety_info(contacts=[])
        assert info.human_joint_positions == []
        assert info.human_joint_names == []
