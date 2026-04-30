"""Perception layer for safety_bigym.

Phase 1 of the Hybrid Safety Critic plan: a Mock BodySLAM++ observation
wrapper that adds a noisy human-pose estimate to the env's observation
dict. See .claude/HYBRID_SAFETY_CRITIC_PLAN.md.
"""

from safety_bigym.perception.bodyslam_wrapper import (
    BodySLAMWrapper,
    MujocoRayOcclusion,
    NoOcclusion,
)
from safety_bigym.perception.demo_position_provider import (
    AMASSDemoPositionProvider,
)

__all__ = [
    "AMASSDemoPositionProvider",
    "BodySLAMWrapper",
    "MujocoRayOcclusion",
    "NoOcclusion",
]
