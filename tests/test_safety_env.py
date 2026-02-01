"""Tests for SafetyBiGymEnv wrapper."""
import pytest
import numpy as np
from gymnasium import spaces

from bigym.action_modes import JointPositionActionMode
from bigym.envs.reach_target import ReachTarget

from safety_bigym.envs.safety_env import SafetyBiGymEnv


class TestSafetyBiGymEnv:
    """Test suite for SafetyBiGymEnv wrapper."""
    
    @pytest.fixture
    def base_env(self):
        """Create a base bigym environment."""
        action_mode = JointPositionActionMode(floating_base=True)
        env = ReachTarget(action_mode=action_mode)
        yield env
        env.close()
    
    @pytest.fixture
    def safety_env(self, base_env):
        """Create a safety-wrapped environment."""
        env = SafetyBiGymEnv(base_env)
        yield env
        env.close()
    
    def test_wrapper_creation(self, safety_env):
        """Test that wrapper can be created around bigym env."""
        assert safety_env is not None
        assert hasattr(safety_env, 'unwrapped')
    
    def test_observation_space_preserved(self, base_env, safety_env):
        """Test that observation space is preserved from base env."""
        # Base observation space should be a subset of wrapped space
        base_obs_space = base_env.observation_space
        wrapped_obs_space = safety_env.observation_space
        
        assert isinstance(wrapped_obs_space, spaces.Dict)
        for key in base_obs_space.keys():
            assert key in wrapped_obs_space.keys()
    
    def test_action_space_identical(self, base_env, safety_env):
        """Test that action space is identical to base env."""
        assert safety_env.action_space == base_env.action_space
    
    def test_reset_returns_observation(self, safety_env):
        """Test that reset returns valid observation."""
        obs, info = safety_env.reset()
        
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        # Check observation matches observation space
        for key in safety_env.observation_space.keys():
            assert key in obs
    
    def test_step_returns_valid_tuple(self, safety_env):
        """Test that step returns (obs, reward, terminated, truncated, info)."""
        safety_env.reset()
        action = safety_env.action_space.sample()
        
        result = safety_env.step(action)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_safety_info_in_step_output(self, safety_env):
        """Test that step info contains safety-related fields."""
        safety_env.reset()
        action = safety_env.action_space.sample()
        
        _, _, _, _, info = safety_env.step(action)
        
        # Safety wrapper should add safety info
        assert 'safety' in info
        safety_info = info['safety']
        assert 'contacts' in safety_info
        assert 'violations' in safety_info
    
    def test_seed_reproducibility(self, base_env):
        """Test that seeded resets are reproducible."""
        env1 = SafetyBiGymEnv(base_env)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env1.reset(seed=42)
        
        for key in obs1.keys():
            np.testing.assert_array_almost_equal(obs1[key], obs2[key])
        
        env1.close()
    
    def test_render_mode_passthrough(self):
        """Test that render mode is passed through to base env."""
        action_mode = JointPositionActionMode(floating_base=True)
        base_env = ReachTarget(action_mode=action_mode, render_mode="rgb_array")
        safety_env = SafetyBiGymEnv(base_env)
        
        img = safety_env.render()
        
        assert img is not None
        assert img.ndim == 3  # (H, W, C)
        assert img.shape[-1] == 3  # RGB
        
        safety_env.close()
