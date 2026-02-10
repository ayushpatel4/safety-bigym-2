"""
Safety BiGym - Human-Aware Robot Learning Environment

Extends BiGym with ISO 15066 safety monitoring and human motion.
"""

from safety_bigym.config import (
    SSMConfig,
    SafetyConfig,
    HumanConfig,
    DEFAULT_SPAWN_POSITIONS,
    get_spawn_positions,
)
from safety_bigym.safety import (
    ISO15066Wrapper,
    SafetyInfo,
    ContactInfo,
    PFL_LIMITS,
    get_region_for_geom,
)
from safety_bigym.human import (
    HumanController,
    PDController,
    PDGains,
    HumanIK,
)
from safety_bigym.scenarios import (
    ScenarioSampler,
    ScenarioParams,
    ParameterSpace,
    DisruptionType,
    DisruptionConfig,
)
from safety_bigym.envs import SafetyBiGymEnv, make_safety_env

__version__ = "0.1.0"

__all__ = [
    # Main environment
    "SafetyBiGymEnv",
    "make_safety_env",
    # Configuration
    "SSMConfig",
    "SafetyConfig", 
    "HumanConfig",
    "DEFAULT_SPAWN_POSITIONS",
    "get_spawn_positions",
    # Safety
    "ISO15066Wrapper",
    "SafetyInfo",
    "ContactInfo",
    "PFL_LIMITS",
    "get_region_for_geom",
    # Human control
    "HumanController",
    "PDController",
    "PDGains",
    "HumanIK",
    # Scenarios
    "ScenarioSampler",
    "ScenarioParams",
    "ParameterSpace",
    "DisruptionType",
    "DisruptionConfig",
]
