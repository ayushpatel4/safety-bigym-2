"""Configuration dataclasses for SafetyBiGymEnv."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np


@dataclass
class SSMConfig:
    """Speed and Separation Monitoring configuration (ISO 15066)."""
    T_r: float = 0.2      # Reaction time (s) - time for system to react
    T_s: float = 0.1      # Stopping time constant (s)
    a_max: float = 5.0    # Maximum deceleration (m/s²)
    C: float = 0.1        # Safety margin / intrusion distance (m)
    v_h_max: float = 1.6  # Maximum expected human velocity (m/s)


@dataclass  
class SafetyConfig:
    """Configuration for ISO 15066 safety monitoring."""
    
    # SSM parameters
    ssm: SSMConfig = field(default_factory=SSMConfig)
    
    # PFL settings
    use_pfl: bool = True
    
    # Behavior on violation
    terminate_on_violation: bool = False
    add_violation_penalty: bool = True
    violation_penalty: float = -1.0
    
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
