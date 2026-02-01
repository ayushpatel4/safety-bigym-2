"""
Tests for Scenario Sampler module.
"""

import pytest
import numpy as np
from pathlib import Path

from safety_bigym.scenarios import (
    ScenarioSampler,
    ScenarioParams,
    ParameterSpace,
    DisruptionType,
    DisruptionConfig,
)


class TestDisruptionTypes:
    """Tests for disruption type configuration."""
    
    def test_disruption_type_enum(self):
        """Test all 5 disruption types exist."""
        assert len(DisruptionType) == 5
        assert DisruptionType.INCIDENTAL
        assert DisruptionType.SHARED_GOAL
        assert DisruptionType.DIRECT
        assert DisruptionType.OBSTRUCTION
        assert DisruptionType.RANDOM_PERTURBED
    
    def test_disruption_config_requires_ik(self):
        """Test IK requirement detection."""
        # IK required
        assert DisruptionConfig(DisruptionType.DIRECT).requires_ik()
        assert DisruptionConfig(DisruptionType.SHARED_GOAL).requires_ik()
        assert DisruptionConfig(DisruptionType.OBSTRUCTION).requires_ik()
        
        # IK not required
        assert not DisruptionConfig(DisruptionType.INCIDENTAL).requires_ik()
        assert not DisruptionConfig(DisruptionType.RANDOM_PERTURBED).requires_ik()
    
    def test_ik_target_computation(self):
        """Test IK target computation from robot state."""
        config = DisruptionConfig(
            disruption_type=DisruptionType.DIRECT,
            target_noise_std=0.0,  # No noise for testing
        )
        
        robot_state = {'ee_pos': np.array([0.5, 0.3, 1.0])}
        rng = np.random.default_rng(42)
        
        target = config.get_ik_target(robot_state, rng)
        np.testing.assert_array_almost_equal(target, [0.5, 0.3, 1.0])
    
    def test_shared_goal_target(self):
        """Test shared goal IK target."""
        config = DisruptionConfig(
            disruption_type=DisruptionType.SHARED_GOAL,
            target_noise_std=0.0,
        )
        
        robot_state = {'task_object_pos': np.array([0.4, 0.2, 0.8])}
        rng = np.random.default_rng(42)
        
        target = config.get_ik_target(robot_state, rng)
        np.testing.assert_array_almost_equal(target, [0.4, 0.2, 0.8])


class TestParameterSpace:
    """Tests for parameter space configuration."""
    
    def test_default_parameter_space(self):
        """Test default parameter ranges."""
        params = ParameterSpace()
        
        assert params.trigger_time_range == (0.5, 5.0)
        assert params.speed_range == (0.5, 2.0)
        assert params.height_percentile_range == (0.05, 0.95)
        assert len(params.disruption_weights) == 5
    
    def test_disruption_weights_sum(self):
        """Test disruption weights are valid."""
        params = ParameterSpace()
        total = sum(params.disruption_weights.values())
        assert 0.99 < total < 1.01  # Allow floating point tolerance


class TestScenarioSampler:
    """Tests for scenario sampler."""
    
    def test_sampler_creation(self):
        """Test sampler can be created."""
        sampler = ScenarioSampler()
        assert sampler.params is not None
    
    def test_reproducible_sampling(self):
        """Test same seed produces same scenario."""
        sampler = ScenarioSampler()
        
        s1 = sampler.sample_scenario(42)
        s2 = sampler.sample_scenario(42)
        
        assert s1.trigger_time == s2.trigger_time
        assert s1.speed_multiplier == s2.speed_multiplier
        assert s1.disruption_type == s2.disruption_type
        assert s1.approach_angle == s2.approach_angle
    
    def test_different_seeds_different_scenarios(self):
        """Test different seeds produce different scenarios."""
        sampler = ScenarioSampler()
        
        s1 = sampler.sample_scenario(1)
        s2 = sampler.sample_scenario(2)
        
        # At least one parameter should differ
        different = (
            s1.trigger_time != s2.trigger_time or
            s1.speed_multiplier != s2.speed_multiplier or
            s1.approach_angle != s2.approach_angle
        )
        assert different
    
    def test_sample_batch(self):
        """Test batch sampling."""
        sampler = ScenarioSampler()
        batch = sampler.sample_batch(10, base_seed=0)
        
        assert len(batch) == 10
        assert all(isinstance(s, ScenarioParams) for s in batch)
    
    def test_clip_discovery(self, tmp_path):
        """Test motion clip auto-discovery."""
        # Create fake motion files
        (tmp_path / "01").mkdir()
        (tmp_path / "01" / "01_01_poses.npz").touch()
        (tmp_path / "01" / "01_02_poses.npz").touch()
        
        sampler = ScenarioSampler(motion_dir=tmp_path)
        assert len(sampler.params.clip_paths) == 2
    
    def test_scenario_params_fields(self):
        """Test all required fields are present in sampled scenario."""
        sampler = ScenarioSampler()
        s = sampler.sample_scenario(0)
        
        # Check all expected fields
        assert hasattr(s, 'disruption_type')
        assert hasattr(s, 'trigger_time')
        assert hasattr(s, 'blend_duration')
        assert hasattr(s, 'speed_multiplier')
        assert hasattr(s, 'human_height_percentile')
        assert hasattr(s, 'approach_angle')
        assert hasattr(s, 'spawn_distance')
        assert hasattr(s, 'reaching_arm')
        assert hasattr(s, 'seed')
    
    def test_reaching_arm_selection(self):
        """Test arm selection based on approach angle."""
        sampler = ScenarioSampler()
        
        # Sample multiple scenarios and check arm selection logic
        right_count = 0
        left_count = 0
        
        for seed in range(100):
            s = sampler.sample_scenario(seed)
            if s.reaching_arm == "right_arm":
                right_count += 1
            else:
                left_count += 1
        
        # Both arms should be selected at least sometimes
        assert right_count > 0
        assert left_count > 0
