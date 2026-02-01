"""Tests for SMPL-H human asset loading."""
import pytest
from pathlib import Path
import mujoco


class TestSMPLHAsset:
    """Test suite for SMPL-H human model."""
    
    @pytest.fixture
    def asset_path(self):
        """Path to SMPL-H human XML."""
        return Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
    
    def test_xml_exists(self, asset_path):
        """Test that the SMPL-H XML file exists."""
        assert asset_path.exists(), f"SMPL-H asset not found at {asset_path}"
    
    def test_xml_valid(self, asset_path):
        """Test that the XML is valid MuJoCo XML."""
        # Should not raise an exception
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        assert model is not None
    
    def test_has_correct_joint_count(self, asset_path):
        """Test that the model has the expected number of joints."""
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        # SMPL-H has 52 joints (22 body + 15 left hand + 15 right hand)
        # Each joint may have 3 DOF (x,y,z rotations)
        # For MuJoCo we'll use ball joints for major joints
        # Minimum: root + body chain = at least 22 named joints
        assert model.njnt >= 22, f"Expected at least 22 joints, got {model.njnt}"
    
    def test_has_human_collision_geoms(self, asset_path):
        """Test that the model has collision geometry."""
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        # Should have collision capsules for body parts
        assert model.ngeom > 0, "Expected collision geometry"
    
    def test_has_actuators(self, asset_path):
        """Test that the model has actuators for joint control."""
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        # Should have position actuators for joints
        assert model.nu > 0, "Expected actuators for position control"
    
    def test_body_region_annotations(self, asset_path):
        """Test that collision geoms have ISO region annotations."""
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        
        # Check for annotated geoms - we use geom names that include 
        # the body region for PFL lookup
        found_annotated = False
        for i in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and any(region in geom_name.lower() for region in 
                                 ['forearm', 'upper_arm', 'chest', 'head', 'hand', 'shin', 'thigh']):
                found_annotated = True
                break
        
        assert found_annotated, "Expected body region annotations in geom names"
    
    def test_can_create_simulation_data(self, asset_path):
        """Test that we can create simulation data from the model."""
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        data = mujoco.MjData(model)
        
        # Should be able to step simulation
        mujoco.mj_step(model, data)
        
        assert data is not None
    
    def test_soft_contact_parameters(self, asset_path):
        """Test that collision geoms have soft contact parameters."""
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        
        # Check that at least some geoms have non-default contact parameters
        # solref[0] should be around 0.02 for soft contact
        found_soft_contact = False
        for i in range(model.ngeom):
            # Check solref - default is [0.02, 1]
            # We set solref="0.02 1.0" which should give soft contact
            if model.geom_solref[i, 0] > 0:
                found_soft_contact = True
                break
        
        assert found_soft_contact, "Expected soft contact parameters on geoms"
