"""Safety filter subpackage — SVF dataset, critic, runtime wrapper.

Phase 2 of the Hybrid Safety Critic plan. See
.claude/HYBRID_SAFETY_CRITIC_PLAN.md and CLAUDE.md.
"""

from safety_bigym.filters.aux_unsafe_provider import (
    AuxUnsafeProvider,
    EmptyAuxProvider,
)
from safety_bigym.filters.cql_trainer import CQLSafetyTrainer
from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.dataset import (
    SafetyTransitionDataset,
    TransitionShardWriter,
    make_oversampler,
)
from safety_bigym.filters.feature_extractor import (
    CriticFeatureSpec,
    make_critic_input,
)
from safety_bigym.filters.labeling import label_transition

__all__ = [
    "AuxUnsafeProvider",
    "CQLSafetyTrainer",
    "CriticFeatureSpec",
    "EmptyAuxProvider",
    "SafetyCritic",
    "SafetyTransitionDataset",
    "TransitionShardWriter",
    "label_transition",
    "make_critic_input",
    "make_oversampler",
]
