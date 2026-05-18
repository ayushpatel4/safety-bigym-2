"""Phase 3 cost critic ``Q_c(s, a)`` — architectural twin of ``SafetyCritic``.

Phase 2's :class:`safety_bigym.filters.critic.SafetyCritic` regresses on the
binary safety reward ``r_safe`` (1 = safe, 0 = violation); high ``Q`` means
"safe future." The Phase 3 cost critic regresses on the continuous step cost
``c_t = max(c_ssm, c_pfl) ∈ [0, 1]`` (0 = safe, 1 = at violation boundary);
high ``Q_c`` means "dangerous future." The two networks share input shape,
MLP backbone, gamma bound, and checkpoint format so that weight transfer
between them is mechanically *possible* — but their output heads disagree in
sign by construction, so :meth:`warm_start_from_svf` requires an explicit opt-in
and reinitialises the final layer when invoked.

See ``UPDATED_PROJECT_PLAN.md:308-314`` for the Option B-value-mean architecture
this critic plugs into.
"""

from __future__ import annotations

import copy
from typing import Mapping, Sequence, Union

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


class CostCritic(nn.Module):
    """Bounded MLP critic for the continuous Phase 3 cost signal.

    Output is in ``[0, q_max]`` via a scaled sigmoid with
    ``q_max = 1/(1-γ)``. Since per-step ``c_t ∈ [0, 1]`` the discounted-return
    ceiling is numerically identical to the Phase 2 SafetyCritic; only the
    semantic direction of the head differs.
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
        device = next(self.parameters()).device
        feats = feats.to(device)
        if feats.ndim == 1:
            q = self.forward(feats.unsqueeze(0)).squeeze(0)
            return float(q.cpu().item())
        q = self.forward(feats)
        return q.detach().cpu().numpy()

    # ---------- target network helpers (same Polyak machinery as SafetyCritic) ----------

    def make_target(self) -> "CostCritic":
        target = copy.deepcopy(self)
        for p in target.parameters():
            p.requires_grad_(False)
        target.eval()
        return target

    @staticmethod
    @torch.no_grad()
    def polyak_update(
        target: "CostCritic", source: "CostCritic", tau: float
    ) -> None:
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
    def from_checkpoint_payload(cls, payload: dict) -> "CostCritic":
        spec = CriticFeatureSpec.from_dict(payload["spec"])
        critic = cls(
            spec=spec,
            gamma=float(payload["gamma"]),
            hidden_dims=tuple(int(h) for h in payload["hidden_dims"]),
        )
        critic.load_state_dict(payload["state_dict"])
        return critic

    # ---------- Phase 2 → Phase 3 warm-start ----------

    def warm_start_from_svf(
        self,
        svf_payload: dict,
        *,
        force_sign_flip: bool = False,
    ) -> None:
        """Optionally seed this critic's hidden layers from a Phase 2 SVF checkpoint.

        The Phase 2 SafetyCritic was trained on ``r_safe = 1 - violation``; high
        Q means "safe future." This CostCritic regresses on ``c_t`` so high Q
        means "dangerous future." The body weights may still encode useful
        state-action structure, but the *output head* points in the wrong
        direction. We therefore:

        - Refuse to load unless ``force_sign_flip=True`` is passed explicitly
        - On opt-in, copy every parameter *except* the final ``Linear`` layer
          (whose weights and bias are left at their fresh PyTorch init).

        Whether body warm-start actually helps over fresh init is an empirical
        question deferred to P3.1+ ablation; this helper makes the experiment
        cheap to run.

        Raises:
            ValueError: if the payload spec or hidden dims don't match this
                critic, or if ``force_sign_flip`` is False.
        """
        payload_spec = CriticFeatureSpec.from_dict(svf_payload["spec"])
        if payload_spec != self.spec:
            raise ValueError(
                "Cannot warm-start: SVF payload spec does not match CostCritic spec.\n"
                f"  payload: {payload_spec}\n"
                f"  self:    {self.spec}"
            )
        payload_hidden = tuple(int(h) for h in svf_payload["hidden_dims"])
        if payload_hidden != self.hidden_dims:
            raise ValueError(
                "Cannot warm-start: SVF payload hidden_dims do not match.\n"
                f"  payload: {payload_hidden}\n"
                f"  self:    {self.hidden_dims}"
            )

        if not force_sign_flip:
            raise ValueError(
                "Refusing to warm-start CostCritic from a SafetyCritic checkpoint "
                "without force_sign_flip=True. SafetyCritic was trained on r_safe "
                "(high Q = safe); CostCritic regresses on c_t (high Q = dangerous). "
                "The output head points in the wrong direction. Pass "
                "force_sign_flip=True to copy hidden layers only (head is "
                "reinitialised); A/B against fresh init in P3.1+ ablation."
            )

        # Copy every parameter except the final Linear's weight + bias.
        # The MLP layout is: Linear -> ReLU -> ... -> Linear (terminal head).
        own_sd = self.state_dict()
        payload_sd = svf_payload["state_dict"]
        # Identify the terminal Linear's parameter prefix: it's the last
        # nn.Linear inside self.net. The MLP layers alternate Linear/ReLU so
        # the terminal layer's index is 2 * len(hidden_dims) inside self.net.
        terminal_idx = 2 * len(self.hidden_dims)
        skip = {f"net.{terminal_idx}.weight", f"net.{terminal_idx}.bias"}

        loaded = 0
        for k, v in payload_sd.items():
            if k in skip:
                continue
            if k not in own_sd:
                raise ValueError(
                    f"Cannot warm-start: payload key {k!r} not in CostCritic state_dict"
                )
            if own_sd[k].shape != v.shape:
                raise ValueError(
                    f"Cannot warm-start: shape mismatch for {k!r}: "
                    f"payload {tuple(v.shape)} vs self {tuple(own_sd[k].shape)}"
                )
            own_sd[k] = v
            loaded += 1
        self.load_state_dict(own_sd)
        # Terminal head is left at its fresh PyTorch init; record for diagnostics.
        self._warm_start_skipped = sorted(skip)
        self._warm_start_loaded_params = loaded


__all__ = ["CostCritic"]
