"""Bounded-output safety critic.

MLP ``[in → 256 → 256 → 256 → 1]`` with ReLU, output ``q = q_max * sigmoid(logit)``
where ``q_max = 1/(1-γ)``. The scaled-sigmoid output mathematically prevents
overestimation past the discounted-return ceiling, which is the standard CQL
treatment for binary-reward critics.

The critic is decoupled from any pixel encoder and the actor; it consumes the
deterministic 1-D feature vector built by :mod:`safety_bigym.filters.feature_extractor`.
"""

from __future__ import annotations

import copy
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from safety_bigym.filters.feature_extractor import (
    CriticFeatureSpec,
    make_critic_input,
)


def _build_mlp(
    input_dim: int, hidden_dims: Sequence[int], output_dim: int = 1
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU(inplace=True))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class SafetyCritic(nn.Module):
    """Bounded MLP critic for binary safety rewards.

    Output is in ``[0, q_max]`` via a scaled sigmoid. Use ``q_value(obs, action)``
    for inference at runtime; use ``forward(features)`` for training when
    features are already concatenated.
    """

    def __init__(
        self,
        *,
        spec: CriticFeatureSpec,
        gamma: float = 0.99,
        hidden_dims: Sequence[int] = (256, 256, 256),
    ):
        super().__init__()
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1); got {gamma}")
        self.spec = spec
        self.gamma = float(gamma)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.net = _build_mlp(spec.input_dim, self.hidden_dims, output_dim=1)

    @property
    def q_max(self) -> float:
        return 1.0 / (1.0 - self.gamma)

    @property
    def input_dim(self) -> int:
        return self.spec.input_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.net(features).squeeze(-1)
        return self.q_max * torch.sigmoid(logits)

    @torch.no_grad()
    def q_value(
        self,
        obs: Mapping[str, Union[np.ndarray, torch.Tensor]],
        action: Union[np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray]:
        feats = make_critic_input(obs, action, self.spec)
        # Move to same device as parameters
        device = next(self.parameters()).device
        feats = feats.to(device)
        if feats.ndim == 1:
            q = self.forward(feats.unsqueeze(0)).squeeze(0)
            return float(q.cpu().item())
        q = self.forward(feats)
        return q.detach().cpu().numpy()

    # ---------- target network helpers ----------

    def make_target(self) -> "SafetyCritic":
        """Return a deepcopy with gradients disabled — use as the target network."""
        target = copy.deepcopy(self)
        for p in target.parameters():
            p.requires_grad_(False)
        target.eval()
        return target

    @staticmethod
    @torch.no_grad()
    def polyak_update(
        target: "SafetyCritic", source: "SafetyCritic", tau: float
    ) -> None:
        """``target ← τ·source + (1−τ)·target`` for every parameter."""
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1]; got {tau}")
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

    # ---------- checkpoint round-trip ----------

    def checkpoint_payload(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "gamma": self.gamma,
            "hidden_dims": list(self.hidden_dims),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict) -> "SafetyCritic":
        spec = CriticFeatureSpec.from_dict(payload["spec"])
        critic = cls(
            spec=spec,
            gamma=float(payload["gamma"]),
            hidden_dims=tuple(int(h) for h in payload["hidden_dims"]),
        )
        critic.load_state_dict(payload["state_dict"])
        return critic


__all__ = ["SafetyCritic"]
