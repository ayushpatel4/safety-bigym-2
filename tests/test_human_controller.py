"""Unit tests for PD controller and human controller."""

import pytest
import numpy as np
import mujoco
from pathlib import Path


class TestPDController:
    """Tests for the PD controller."""
    
    @pytest.fixture
    def model_and_data(self):
        """Load SMPL-H model."""
        model_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        return model, data
    
    def test_pd_controller_creation(self, model_and_data):
        """Test PD controller can be created."""
        from safety_bigym.human import PDController
        
        model, data = model_and_data
        controller = PDController(model, data)
        
        assert controller is not None
        assert len(controller.actuator_name_to_idx) == model.nu
    
    def test_pd_controller_set_targets(self, model_and_data):
        """Test setting joint targets."""
        from safety_bigym.human import PDController
        
        model, data = model_and_data
        controller = PDController(model, data)
        
        # Set random targets
        targets = np.random.randn(model.nq) * 0.1
        controller.set_targets(targets)
        
        np.testing.assert_array_almost_equal(controller.target_qpos, targets)
    
    def test_pd_controller_compute_control(self, model_and_data):
        """Test control computation."""
        from safety_bigym.human import PDController
        
        model, data = model_and_data
        controller = PDController(model, data)
        
        ctrl = controller.compute_control()
        
        assert ctrl.shape == (model.nu,)


class TestHumanController:
    """Tests for the human motion controller."""
    
    @pytest.fixture
    def model_and_data(self):
        """Load SMPL-H model."""
        model_path = Path(__file__).parent.parent / "safety_bigym" / "assets" / "smplh_human.xml"
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        return model, data
    
    @pytest.fixture
    def sample_clip_path(self):
        """Path to sample motion clip."""
        import os
        data_dir = os.environ.get("AMASS_DATA_DIR")
        if not data_dir:
            pytest.skip("AMASS_DATA_DIR not set")
        return f"{data_dir}/01/01_01_poses.npz"
    
    def test_human_controller_creation(self, model_and_data):
        """Test human controller can be created."""
        from safety_bigym.human import HumanController
        
        model, data = model_and_data
        controller = HumanController(model, data)
        
        assert controller is not None
        assert controller.pd_controller is not None
    
    def test_load_clip(self, model_and_data, sample_clip_path):
        """Test loading AMASS clip."""
        from safety_bigym.human import HumanController
        
        model, data = model_and_data
        controller = HumanController(model, data)
        
        controller.load_clip(sample_clip_path)
        
        assert controller.clip is not None
        assert controller.clip.num_frames > 0
    
    def test_scenario_phases(self, model_and_data, sample_clip_path):
        """Test motion controller phases."""
        from safety_bigym.human import HumanController, ScenarioParams
        
        model, data = model_and_data
        controller = HumanController(model, data)
        
        scenario = ScenarioParams(
            clip_path=sample_clip_path,
            trigger_time=1.0,
            blend_duration=0.5,
        )
        controller.set_scenario(scenario)
        controller.reset()
        
        # Phase 1: AMASS (t < 1.0)
        assert controller.current_phase == "amass"
        
        # Step to blending phase
        for _ in range(600):  # ~1.2s at 0.002s timestep
            controller.step(0.002)
        
        assert controller.current_phase == "blending"
        
        # Step to IK phase
        for _ in range(300):  # ~0.6s more
            controller.step(0.002)
        
        assert controller.current_phase == "ik"
