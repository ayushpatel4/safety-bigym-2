"""Safety-wrapped BiGym environment with human presence and ISO 15066 monitoring."""
from __future__ import annotations

from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType


class SafetyBiGymEnv(gym.Wrapper):
    """
    Safety wrapper for BiGym environments.
    
    This wrapper adds:
    - SMPL-H human presence with parameterised motion
    - ISO 15066 safety monitoring (SSM and PFL)
    - Contact detection and force measurement
    
    The wrapper follows the Gymnasium Wrapper pattern, delegating most
    operations to the underlying BiGym environment while adding safety
    monitoring at each step.
    """
    
    def __init__(
        self,
        env: gym.Env,
        human_enabled: bool = True,
        safety_monitoring: bool = True,
    ):
        """
        Initialize the safety wrapper.
        
        Args:
            env: Base BiGym environment to wrap
            human_enabled: Whether to spawn human in environment (Phase 2+)
            safety_monitoring: Whether to enable ISO 15066 monitoring (Phase 6+)
        """
        super().__init__(env)
        
        self._human_enabled = human_enabled
        self._safety_monitoring = safety_monitoring
        
        # Will be initialized in later phases
        self._human_controller = None
        self._scenario_sampler = None
        self._iso15066_monitor = None
        
        # Tracking state
        self._current_scenario = None
        self._episode_time = 0.0
        
        # Extend observation space with safety info
        self._setup_observation_space()
    
    def _setup_observation_space(self):
        """Extend observation space to include safety-related observations."""
        base_obs_space = self.env.observation_space
        
        # For now, just use the base observation space
        # In later phases, we'll add human state observations
        self.observation_space = base_obs_space
    
    @property
    def unwrapped(self):
        """Return the base unwrapped environment."""
        return self.env.unwrapped
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment.
        
        In full implementation:
        1. Sample scenario from HumanBehaviourSampler
        2. Reset base environment
        3. Spawn human with sampled motion clip
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset episode state
        self._episode_time = 0.0
        
        # Sample scenario (Phase 5+)
        if self._scenario_sampler is not None:
            self._current_scenario = self._scenario_sampler.sample_scenario(seed)
        
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Spawn human with scenario (Phase 2+)
        if self._human_controller is not None and self._current_scenario is not None:
            self._human_controller.load_clip(self._current_scenario.clip)
        
        return obs, info
    
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment.
        
        In full implementation:
        1. Step human motion controller
        2. Step base environment (with sub-step force capture)
        3. Check ISO 15066 violations
        4. Return observations with safety info
        
        Args:
            action: Action to take in environment
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Step human controller (Phase 3+)
        if self._human_controller is not None and self._current_scenario is not None:
            robot_state = self._get_robot_state()
            self._human_controller.step(
                t=self._episode_time,
                scenario=self._current_scenario,
                robot_state=robot_state,
            )
        
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode time
        dt = getattr(self.env, 'dt', 0.002)  # Default physics dt
        self._episode_time += dt
        
        # Add safety info to step output
        safety_info = self._get_safety_info()
        info['safety'] = safety_info
        
        return obs, reward, terminated, truncated, info
    
    def _get_robot_state(self) -> dict:
        """Get current robot state for IK targeting."""
        # Get robot end-effector positions for IK
        robot = getattr(self.env, 'robot', None)
        if robot is None:
            return {}
        
        return {
            'ee_positions': {},  # Will be populated with actual EE positions
            'joint_positions': np.array([]),
        }
    
    def _get_safety_info(self) -> dict:
        """
        Get safety monitoring information.
        
        Returns dict with:
        - contacts: List of human-robot contacts
        - violations: List of ISO 15066 violations
        - max_force: Maximum contact force this step
        """
        # Placeholder - will be implemented in Phase 6
        contacts = []
        violations = []
        
        if self._iso15066_monitor is not None:
            contacts, violations = self._iso15066_monitor.get_contacts_and_violations()
        
        return {
            'contacts': contacts,
            'violations': violations,
            'max_force': 0.0,
            'contact_type': None,
        }
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Clean up resources."""
        if self._human_controller is not None:
            # Clean up human controller resources
            pass
        super().close()
