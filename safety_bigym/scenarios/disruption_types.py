"""
Disruption Types for Human Behaviour Scenarios

Defines the 5 disruption types that determine how the human
interacts with the robot workspace.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict
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
    """Human moves into robot's path and holds position for the loiter
    window, then departs. IK target: a fixed point inside the robot's
    task workspace. Passive intrusion — the robot is the one that has
    to wait/avoid until the human walks away."""

    RANDOM_PERTURBED = auto()
    """AMASS motion with Gaussian noise on trajectory.
    No IK - perturbed motion playback."""

    CONTACT = auto()
    """Human deliberately walks into and presses on the robot during
    the loiter phase, then departs. IK target: a specific robot link,
    offset slightly inside the surface (embed_distance)."""

    COWORKER = auto()
    """Human stays in the robot's workspace for the whole episode and
    intermittently reaches toward the robot EE or the task object,
    then retracts to a rest pose. Procedural, no AMASS dependency
    during the reach cycle. Models a sustained co-working scenario
    rather than a one-shot intrusion."""


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

    # For CONTACT (active press into robot)
    contact_target_part: str = "ee"  # "ee" | "left_forearm" | "right_forearm" | "torso"
    embed_distance: float = 0.0  # meters past the link surface (toward human)

    # For COWORKER (sustained co-working with periodic reaches)
    coworker_reach_period: float = 4.0      # seconds per reach cycle
    coworker_reach_fraction: float = 0.30   # fraction of cycle extending toward target
    coworker_hold_fraction: float = 0.15    # fraction held at target
    coworker_retract_fraction: float = 0.30 # fraction retracting back to rest
    # remainder of cycle = idle at rest pose
    coworker_target_mix: tuple = (0.5, 0.5) # (P(reach EE), P(reach task object))
    coworker_active_arm: str = "right_arm"  # "right_arm" | "left_arm"

    def requires_ik(self) -> bool:
        """Check if this disruption type uses IK targeting."""
        return self.disruption_type in {
            DisruptionType.SHARED_GOAL,
            DisruptionType.DIRECT,
            DisruptionType.OBSTRUCTION,
            DisruptionType.CONTACT,
            DisruptionType.COWORKER,
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

        elif self.disruption_type == DisruptionType.CONTACT:
            link_pos: Optional[Dict[str, np.ndarray]] = robot_state.get('link_pos')
            if link_pos and self.contact_target_part in link_pos:
                base = np.asarray(link_pos[self.contact_target_part], dtype=float)
            else:
                base = robot_state.get('ee_pos')
                if base is None:
                    return None
                base = np.asarray(base, dtype=float)

            if self.embed_distance > 0.0:
                pelvis = robot_state.get('human_pelvis_pos')
                if pelvis is not None:
                    direction = np.asarray(pelvis, dtype=float) - base
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        return base + (direction / norm) * self.embed_distance
            return base

        elif self.disruption_type == DisruptionType.COWORKER:
            # The coworker behaviour module resolves which target to use
            # per reach cycle. This method just returns the EE position
            # as a safe default; CoworkerArmController consults the cycle
            # state and re-reads robot_state directly when extending.
            base = robot_state.get('ee_pos')
            if base is None:
                return None
            return np.asarray(base, dtype=float)

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
    DisruptionType.CONTACT: DisruptionConfig(
        disruption_type=DisruptionType.CONTACT,
        target_noise_std=0.0,
        contact_target_part="ee",
        embed_distance=0.05,
    ),
    DisruptionType.COWORKER: DisruptionConfig(
        disruption_type=DisruptionType.COWORKER,
        target_noise_std=0.04,
        coworker_reach_period=5.0,
        coworker_reach_fraction=0.15,
        coworker_hold_fraction=0.20,
        coworker_retract_fraction=0.15,
        coworker_target_mix=(0.5, 0.5),
        coworker_active_arm="right_arm",
    ),
}
