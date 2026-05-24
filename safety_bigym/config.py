"""Configuration dataclasses for SafetyBiGymEnv."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class SSMConfig:
    """Speed and Separation Monitoring configuration (ISO 15066).

    Single source of truth for SSM parameters. Used by both
    SafetyConfig (high-level) and ISO15066Wrapper (low-level).
    """

    T_r: float = 0.1        # Robot reaction time (seconds)
    T_s: float = 0.05       # System response time (seconds)
    a_max: float = 5.0      # Maximum robot braking deceleration (m/s²)
    C: float = 0.1          # Intrusion distance / uncertainty (meters)
    v_h_max: float = 1.6    # Maximum assumed human velocity (m/s)

    def compute_separation_distance(
        self,
        v_robot: float,
        v_human: float = None,
    ) -> float:
        """Compute protective separation distance S_p.

        S_p = S_h + S_r + S_s + C + Z_d + Z_r

        Simplified to: S_p = S_h + S_r + C
        where:
        - S_h = v_h × (T_r + T_s)  (human contribution)
        - S_r = v_r × T_r + v_r² / (2 × a_max)  (robot stopping distance)

        Args:
            v_robot: Robot velocity magnitude (m/s)
            v_human: Human velocity magnitude (m/s), uses v_h_max if None

        Returns:
            Required protective separation distance S_p (meters)
        """
        v_h = v_human if v_human is not None else self.v_h_max

        # Human contribution
        S_h = v_h * (self.T_r + self.T_s)

        # Robot stopping distance
        S_r = v_robot * self.T_r + (v_robot ** 2) / (2 * self.a_max)

        # Total separation
        S_p = S_h + S_r + self.C

        return S_p


@dataclass  
class SafetyConfig:
    """Configuration for ISO 15066 safety monitoring."""
    
    # SSM parameters
    ssm: SSMConfig = field(default_factory=SSMConfig)
    
    # PFL settings
    use_pfl: bool = True
    
    # Behavior on violation
    terminate_on_violation: bool = False
    add_violation_penalty: bool = False
    violation_penalty: float = -1.0

    # Phase 3 workspace reward shaping
    # r_workspace = -beta * min(workspace_excess_cap, max(0, ||p_ee - p_task|| - r_ws));
    # subtracted from task reward. Prevents the Lagrangian policy from satisfying its
    # cost constraint by evacuating the workspace.
    #
    # `workspace_excess_cap` BOUNDS the per-step penalty at `-beta * cap`. This is
    # load-bearing: an unbounded dense penalty produces a discounted return
    # (-beta*excess / (1 - gamma)) that blows past the CQN-AS C51 critic's value
    # support [v_min, v_max], so the Bellman-target clamp saturates value learning
    # and the gradient that should pull the EE back is clipped away (the 2026-05-20
    # base-validation failure; see docs/phase3_base_validation_findings.md). Keep the
    # design invariant `beta * workspace_excess_cap / (1 - gamma) <= |v_min|`.
    # beta lowered 0.2 -> 0.05 and cap added 2026-05-20 for that reason.
    add_workspace_penalty: bool = False
    workspace_radius: float = 0.4
    workspace_beta: float = 0.05
    workspace_excess_cap: Optional[float] = 1.0

    # G1 bisection diagnostic (2026-05-24). When True, _configure_collision_bits
    # skips the floor's cross-paired bit promotion, so human <-> floor contacts
    # are filtered out by MuJoCo's contype/conaffinity intersection (floor stays
    # on its default bit-0 channel; human emits on bit 1, accepts bit 2 — no
    # overlap -> no contact). Tests the hypothesis that G1's foot-floor contacts
    # (which SMPL-H avoided because its pelvis was welded to world / weldid=0)
    # are injecting solver noise that propagates to the robot via the shared
    # floor contact graph and causes the policy to flee the workspace in stage 0.
    # Human <-> robot contacts are UNAFFECTED (robot still gets the cross-pair).
    # SSM is unaffected (it reads body xpos, not geom contacts).
    disable_human_floor_collision: bool = False

    # Logging
    log_violations: bool = True
    log_all_contacts: bool = False


@dataclass
class HumanConfig:
    """Configuration for human behavior and spawning."""
    
    # Motion data
    motion_clip_dir: Optional[Path] = None
    motion_clip_paths: List[str] = field(default_factory=list)
    
    # Human appearance
    height_percentile: float = 0.5  # 0.0 = short, 1.0 = tall
    
    # Spawn configuration
    spawn_positions: Optional[List[Dict[str, Any]]] = None
    spawn_distance: float = 2.0     # Default distance from robot
    spawn_angle_range: tuple = (0, 2 * np.pi)  # Full circle
    
    # Controller settings
    pd_kp: float = 200.0  # Position gain
    pd_kd: float = 20.0   # Derivative gain
    
    def __post_init__(self):
        if self.motion_clip_dir is not None:
            self.motion_clip_dir = Path(self.motion_clip_dir)


# Default spawn positions for common BiGym tasks
# These are validated to be clear of obstacles
DEFAULT_SPAWN_POSITIONS = {
    "ReachTarget": [
        {"pos": [2.0, 0.0, 0.0], "yaw": np.pi},       # Behind robot
        {"pos": [0.0, 2.0, 0.0], "yaw": -np.pi/2},    # Left of robot  
        {"pos": [0.0, -2.0, 0.0], "yaw": np.pi/2},    # Right of robot
        {"pos": [-1.5, 1.5, 0.0], "yaw": -3*np.pi/4}, # Front-left diagonal
        {"pos": [-1.5, -1.5, 0.0], "yaw": 3*np.pi/4}, # Front-right diagonal
    ],
    "PickBox": [
        {"pos": [2.0, 0.0, 0.0], "yaw": np.pi},
        {"pos": [0.0, 2.0, 0.0], "yaw": -np.pi/2},
        {"pos": [0.0, -2.0, 0.0], "yaw": np.pi/2},
    ],
    # Default for unknown tasks
    "default": [
        {"pos": [2.0, 0.0, 0.0], "yaw": np.pi},
    ],
}


def get_spawn_positions(task_name: str) -> List[Dict[str, Any]]:
    """Get valid spawn positions for a task."""
    return DEFAULT_SPAWN_POSITIONS.get(
        task_name, 
        DEFAULT_SPAWN_POSITIONS["default"]
    )
