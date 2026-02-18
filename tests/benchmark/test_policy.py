import pytest
from gymnasium import spaces
import numpy as np
from safety_bigym.benchmark.policy import RandomPolicy

def test_random_policy():
    """Test standard RandomPolicy behavior."""
    action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
    policy = RandomPolicy(action_space=action_space)

    # Test reset
    policy.reset()

    # Test act
    act = policy.act({})
    assert action_space.contains(act)
