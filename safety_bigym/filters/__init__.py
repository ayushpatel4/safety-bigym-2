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
from safety_bigym.filters.fallback import (
    Fallback,
    FallbackRegistry,
    ZeroVelocityFallback,
)
from safety_bigym.filters.feature_extractor import (
    CriticFeatureSpec,
    make_critic_input,
)
from safety_bigym.filters.labeling import label_transition
from safety_bigym.filters.runtime_wrapper import SafetyFilterWrapper
from safety_bigym.filters.snapshots import SNAPSHOTS, resolve_snapshot
from safety_bigym.filters.threshold_sweep import (
    ThresholdEvalResult,
    evaluate_threshold,
    sweep_thresholds,
)

__all__ = [
    "AuxUnsafeProvider",
    "CQLSafetyTrainer",
    "CriticFeatureSpec",
    "EmptyAuxProvider",
    "Fallback",
    "FallbackRegistry",
    "SNAPSHOTS",
    "SafetyCritic",
    "SafetyFilterWrapper",
    "SafetyTransitionDataset",
    "ThresholdEvalResult",
    "TransitionShardWriter",
    "ZeroVelocityFallback",
    "evaluate_threshold",
    "label_transition",
    "make_critic_input",
    "make_oversampler",
    "resolve_snapshot",
    "sweep_thresholds",
]
