from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """
    Abstract base class for visuomotor policies.
    """
    
    @abstractmethod
    def reset(self):
        """Reset policy state at the start of an episode."""
        pass
        
    @abstractmethod
    def act(self, obs: dict) -> np.ndarray:
        """
        Compute action given observation.
        
        Args:
            obs: Observation dictionary from the environment
                 (e.g., {'proprioception': ..., 'visual': ...})
                 
        Returns:
            Action array (normalized or raw, depending on environment config)
        """
        pass


class RandomPolicy(Policy):
    """
    Baseline policy that samples random actions from a given action space.
    """
    
    def __init__(self, action_space):
        """
        Args:
            action_space: The gym action space to sample from
        """
        self.action_space = action_space
        
    def reset(self):
        pass
        
    def act(self, obs: dict) -> np.ndarray:
        return self.action_space.sample()
