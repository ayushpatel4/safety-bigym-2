"""Privileged-unsafe-state provider stub.

The CQL trainer optionally adds a supervised loss on AMASS-derived ground-truth
unsafe configurations: ``L_aux = E[Q(x_unsafe, π_rand(x_unsafe))²]``. v1 ships
the inert :class:`EmptyAuxProvider`; the real provider is deferred until after
the first GPU calibration of CQL alone (per the approved Phase 2 plan).
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch

from safety_bigym.filters.feature_extractor import CriticFeatureSpec


class AuxUnsafeProvider(Protocol):
    """Sample privileged unsafe-state features for the supervised aux loss."""

    def sample(
        self,
        batch_size: int,
        spec: CriticFeatureSpec,
        device: torch.device,
    ) -> Optional[dict]:  # pragma: no cover - Protocol
        ...


class EmptyAuxProvider:
    """Default no-op provider: returns ``None``, signalling the trainer to skip."""

    def sample(
        self,
        batch_size: int,
        spec: CriticFeatureSpec,
        device: torch.device,
    ) -> Optional[dict]:
        return None


__all__ = ["AuxUnsafeProvider", "EmptyAuxProvider"]
