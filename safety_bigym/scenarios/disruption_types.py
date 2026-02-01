"""
Disruption Types for Human Behaviour Scenarios

Defines the 5 disruption types that determine how the human
interacts with the robot workspace.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np


class DisruptionType(Enum):
    """Types of human disruption behaviours."""
    
    INCIDENTAL = auto()
    """AMASS motion that happens to cross robot workspace.
    No IK - pure motion playback. The human is not intentionally
    interacting with the robot."""
    
    SHARED_GOAL = auto()
    """Human reaches toward an object near the robot's task goal.
    IK target: task object position + noise."""
    
    DIRECT = auto()
    """Human reaches toward the robot's end-effector.
    IK target: robot EE position with lag and noise."""
    
    OBSTRUCTION = auto()
    """Human moves into robot's path and holds position.
    IK target: fixed point in robot's planned trajectory."""
    
    RANDOM_PERTURBED = auto()
    """AMASS motion with Gaussian noise on trajectory.
    No IK - perturbed motion playback."""


@dataclass
class DisruptionConfig:
    """Configuration for a specific disruption type."""
    
    disruption_type: DisruptionType
    
    # For IK-based disruptions
    target_noise_std: float = 0.03  # meters
    tracking_lag: float = 0.0  # seconds (for DIRECT type)
    
    # For RANDOM_PERTURBED
    trajectory_noise_std: float = 0.02  # radians per joint
    
    # For OBSTRUCTION
    obstruction_target: Optional[np.ndarray] = None  # Fixed world position
    hold_duration: float = 2.0  # seconds to hold position
    
    def requires_ik(self) -> bool:
        """Check if this disruption type uses IK targeting."""
        return self.disruption_type in {
            DisruptionType.SHARED_GOAL,
            DisruptionType.DIRECT,
            DisruptionType.OBSTRUCTION,
        }
    
    def get_ik_target(
        self,
        robot_state: dict,
        rng: np.random.Generator,
    ) -> Optional[np.ndarray]:
        """
        Compute IK target position based on disruption type.
        
        Args:
            robot_state: Dict with 'ee_pos', 'task_object_pos', etc.
            rng: Random number generator for noise
            
        Returns:
            Target position (3,) or None if no IK needed
        """
        if self.disruption_type == DisruptionType.SHARED_GOAL:
            base = robot_state.get('task_object_pos')
            if base is None:
                return None
            noise = rng.normal(0, self.target_noise_std, 3)
            return base + noise
        
        elif self.disruption_type == DisruptionType.DIRECT:
            base = robot_state.get('ee_pos')
            if base is None:
                return None
            noise = rng.normal(0, self.target_noise_std, 3)
            return base + noise
        
        elif self.disruption_type == DisruptionType.OBSTRUCTION:
            if self.obstruction_target is not None:
                return self.obstruction_target.copy()
            # Fallback: use robot base area
            robot_base = robot_state.get('robot_base_pos', np.array([0, 0, 0]))
            return robot_base + np.array([0.3, 0, 0.8])  # In front at chest height
        
        return None


# Default configurations for each disruption type
DEFAULT_CONFIGS = {
    DisruptionType.INCIDENTAL: DisruptionConfig(
        disruption_type=DisruptionType.INCIDENTAL,
    ),
    DisruptionType.SHARED_GOAL: DisruptionConfig(
        disruption_type=DisruptionType.SHARED_GOAL,
        target_noise_std=0.05,
    ),
    DisruptionType.DIRECT: DisruptionConfig(
        disruption_type=DisruptionType.DIRECT,
        target_noise_std=0.03,
        tracking_lag=0.15,  # 150ms reaction time
    ),
    DisruptionType.OBSTRUCTION: DisruptionConfig(
        disruption_type=DisruptionType.OBSTRUCTION,
        hold_duration=2.0,
    ),
    DisruptionType.RANDOM_PERTURBED: DisruptionConfig(
        disruption_type=DisruptionType.RANDOM_PERTURBED,
        trajectory_noise_std=0.02,
    ),
}
