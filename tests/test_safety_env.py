"""
Tests for SafetyBiGymEnv integration.

This module tests the SafetyBiGymEnv class which subclasses BiGymEnv
to integrate human presence and ISO 15066 safety monitoring.
"""

import pytest
import numpy as np


class TestSafetyBiGymEnvImports:
    """Test that all imports work correctly."""
    
    def test_import_safety_env(self):
        """Test importing safety environment."""
        from safety_bigym.envs.safety_env import SafetyBiGymEnv, SafetyConfig, HumanConfig
        
        assert SafetyBiGymEnv is not None
        assert SafetyConfig is not None
        assert HumanConfig is not None
    
    def test_import_from_package(self):
        """Test importing from main package."""
        from safety_bigym import (
            SafetyBiGymEnv,
            SafetyConfig,
            HumanConfig,
            ISO15066Wrapper,
            SSMConfig,
            ScenarioSampler,
        )
        
        assert SafetyBiGymEnv is not None
    
    def test_safety_config_defaults(self):
        """Test SafetyConfig has sensible defaults."""
        from safety_bigym.envs.safety_env import SafetyConfig
        
        config = SafetyConfig()
        
        assert config.T_r == 0.1
        assert config.T_s == 0.05
        assert config.a_max == 5.0
        assert config.C == 0.1
        assert config.v_h_max == 1.6
        assert config.use_pfl is True
        assert config.terminate_on_violation is False
    
    def test_human_config_defaults(self):
        """Test HumanConfig has sensible defaults."""
        from safety_bigym.envs.safety_env import HumanConfig, DEFAULT_SPAWN_POSITIONS
        
        config = HumanConfig()
        
        assert config.kp == 200.0
        assert config.kd == 20.0
        assert config.use_ik_reaching is True
        assert len(config.spawn_positions) == len(DEFAULT_SPAWN_POSITIONS)


class TestSafetyEnvHelpers:
    """Test helper methods."""
    
    def test_euler_to_quat(self):
        """Test Euler to quaternion conversion."""
        from safety_bigym.envs.safety_env import SafetyBiGymEnv
        
        # Identity rotation (no rotation)
        quat = SafetyBiGymEnv._euler_to_quat(0, 0, 0)
        np.testing.assert_array_almost_equal(quat, [1, 0, 0, 0])
        
        # 90 degree yaw rotation
        quat = SafetyBiGymEnv._euler_to_quat(0, 0, np.pi/2)
        expected = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
        np.testing.assert_array_almost_equal(quat, expected)
    
    def test_spawn_positions_structure(self):
        """Test spawn positions have required keys."""
        from safety_bigym.envs.safety_env import DEFAULT_SPAWN_POSITIONS
        
        for spawn in DEFAULT_SPAWN_POSITIONS:
            assert "pos" in spawn
            assert "yaw" in spawn
            assert len(spawn["pos"]) == 3


# Integration tests require BiGym installation and graphics
@pytest.mark.skip(reason="Requires full BiGym environment setup with graphics")
class TestSafetyBiGymEnvIntegration:
    """Integration tests with actual BiGym environment."""
    
    @pytest.fixture
    def safety_env(self):
        """Create a safety environment."""
        from bigym.action_modes import JointPositionActionMode
        from safety_bigym import SafetyBiGymEnv, SafetyConfig
        
        env = SafetyBiGymEnv(
            action_mode=JointPositionActionMode(),
            safety_config=SafetyConfig(),
        )
        yield env
        env.close()
    
    def test_create_env(self, safety_env):
        """Test creating a safety environment."""
        assert safety_env is not None
        assert safety_env.observation_space is not None
        assert safety_env.action_space is not None
    
    def test_observation_space_has_safety_keys(self, safety_env):
        """Test observation space includes safety observations."""
        from gymnasium import spaces
        
        obs_space = safety_env.observation_space
        
        assert isinstance(obs_space, spaces.Dict)
        assert "safety_ssm_margin" in obs_space.spaces
        assert "safety_pfl_ratio" in obs_space.spaces
        assert "safety_human_distance" in obs_space.spaces
    
    def test_reset_returns_safety_obs(self, safety_env):
        """Test that reset returns safety observations."""
        obs, info = safety_env.reset()
        
        assert "safety_ssm_margin" in obs
        assert "safety_pfl_ratio" in obs
        assert "safety_human_distance" in obs
    
    def test_step_returns_safety_info(self, safety_env):
        """Test that step returns safety information in info dict."""
        safety_env.reset()
        action = safety_env.action_space.sample()
        
        obs, reward, terminated, truncated, info = safety_env.step(action)
        
        # Check safety info in info dict
        assert "safety" in info
        assert "safety_episode" in info
        
        # Check safety observations
        assert "safety_ssm_margin" in obs
        assert "safety_pfl_ratio" in obs
    
    def test_episode_safety_stats_reset(self, safety_env):
        """Test that episode safety stats reset on env reset."""
        safety_env.reset()
        
        # Take some steps
        for _ in range(10):
            action = safety_env.action_space.sample()
            safety_env.step(action)
        
        # Reset and check stats are cleared
        safety_env.reset()
        stats = safety_env.episode_safety_stats
        
        assert stats["ssm_violations"] == 0
        assert stats["pfl_violations"] == 0
        assert stats["max_force"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
