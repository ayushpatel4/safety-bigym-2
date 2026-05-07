"""Offline CQL trainer for :class:`SafetyCritic`.

Loss:
    L = MSE( Q(s, a),  r_safe + γ·(1−done)·Q_target(s', a') )
        + α · ( logsumexp_a' Q(s, a') − E_data[Q(s, a)] )      (CQL conservatism)
        + w_aux · E[ Q(x_unsafe, π_rand)² ]                    (optional)

Where ``a'`` for the CQL term are sampled uniformly from the action-space box
(the random target policy provides OOD action coverage). For the Bellman
target, ``a'`` is also drawn random (no learned policy in this offline setup).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from safety_bigym.filters.aux_unsafe_provider import (
    AuxUnsafeProvider,
    EmptyAuxProvider,
)
from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.feature_extractor import (
    CriticFeatureSpec,
    make_critic_input,
)


def _flatten_obs_dict(
    obs: Mapping[str, torch.Tensor], spec: CriticFeatureSpec
) -> torch.Tensor:
    pieces = []
    for key in spec.obs_keys:
        t = obs[key]
        if t.ndim == 1:
            t = t.unsqueeze(0)
        pieces.append(t.float())
    return torch.cat(pieces, dim=-1)


def _stack_features(
    obs: Mapping[str, torch.Tensor],
    action: torch.Tensor,
    spec: CriticFeatureSpec,
) -> torch.Tensor:
    obs_flat = _flatten_obs_dict(obs, spec)
    a = action.float()
    if a.ndim == 1:
        a = a.unsqueeze(0)
    return torch.cat([obs_flat, a], dim=-1)


class CQLSafetyTrainer:
    """One-step offline CQL trainer with target-network Polyak smoothing."""

    def __init__(
        self,
        *,
        critic: SafetyCritic,
        action_space: gym.spaces.Box,
        cql_alpha: float = 5.0,
        cql_n_actions: int = 10,
        aux_weight: float = 0.0,
        aux_provider: Optional[AuxUnsafeProvider] = None,
        lr: float = 3e-4,
        target_tau: float = 5e-3,
        device: Union[str, torch.device] = "cpu",
        seed: int = 0,
    ):
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"CQLSafetyTrainer requires a Box action_space; got {type(action_space).__name__}"
            )
        self.device = torch.device(device)
        self.critic = critic.to(self.device)
        self.target_critic = critic.make_target().to(self.device)
        self.spec = critic.spec
        self.gamma = critic.gamma

        self.cql_alpha = float(cql_alpha)
        self.cql_n_actions = int(cql_n_actions)
        self.aux_weight = float(aux_weight)
        self.aux_provider = aux_provider or EmptyAuxProvider()
        self.target_tau = float(target_tau)

        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.action_low = torch.as_tensor(
            action_space.low, dtype=torch.float32, device=self.device
        )
        self.action_high = torch.as_tensor(
            action_space.high, dtype=torch.float32, device=self.device
        )
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self._step_count = 0

    # ---------- helpers ----------

    def _to_device(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        return x

    def _move_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                moved[k] = {kk: self._to_device(vv) for kk, vv in v.items()}
            else:
                moved[k] = self._to_device(v)
        return moved

    def _sample_random_actions(self, batch_size: int, n_per: int) -> torch.Tensor:
        """Draw ``[batch, n_per, action_dim]`` uniform actions from the box."""
        low = self.action_low
        high = self.action_high
        u = torch.rand(
            batch_size, n_per, low.shape[0],
            generator=self.generator, device=self.device,
        )
        return low + u * (high - low)

    # ---------- loss ----------

    def _compute_loss(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        batch = self._move_batch(batch)
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"].float()
        r_safe = batch["r_safe"].float().reshape(-1)
        done = batch["done"].float().reshape(-1)

        batch_size = action.shape[0]

        # ---- Bellman target: r + γ(1-done) · Q_target(s', a'_random) ----
        with torch.no_grad():
            next_actions = self._sample_random_actions(batch_size, n_per=1).squeeze(1)
            next_features = _stack_features(next_obs, next_actions, self.spec)
            q_next = self.target_critic(next_features)
            target_q = r_safe + self.gamma * (1.0 - done) * q_next

        # ---- Bellman/MSE term ----
        features = _stack_features(obs, action, self.spec)
        q_pred = self.critic(features)
        bellman_loss = F.mse_loss(q_pred, target_q)

        # ---- CQL conservatism term ----
        ood_actions = self._sample_random_actions(batch_size, n_per=self.cql_n_actions)
        # Repeat obs across the n_per axis: [B, n_per, D]
        obs_flat = _flatten_obs_dict(obs, self.spec)
        obs_rep = obs_flat.unsqueeze(1).expand(-1, self.cql_n_actions, -1)
        features_ood = torch.cat([obs_rep, ood_actions], dim=-1)
        q_ood = self.critic(features_ood.reshape(-1, features_ood.shape[-1]))
        q_ood = q_ood.reshape(batch_size, self.cql_n_actions)
        # logsumexp over OOD actions − E_data[Q(s, a_data)]
        cql_term = (
            torch.logsumexp(q_ood, dim=1) - q_pred
        ).mean()
        cql_loss = self.cql_alpha * cql_term

        # ---- Aux loss (optional) ----
        aux_loss = torch.tensor(0.0, device=self.device)
        if self.aux_weight > 0.0:
            aux_batch = self.aux_provider.sample(batch_size, self.spec, self.device)
            if aux_batch is not None and "features" in aux_batch:
                aux_q = self.critic(aux_batch["features"].to(self.device))
                aux_loss = (aux_q ** 2).mean()
        weighted_aux = self.aux_weight * aux_loss

        total = bellman_loss + cql_loss + weighted_aux

        return {
            "loss": float(total.detach().cpu().item()),
            "bellman_loss": float(bellman_loss.detach().cpu().item()),
            "cql_term": float(cql_term.detach().cpu().item()),
            "aux_loss": float(aux_loss.detach().cpu().item()),
            "q_mean": float(q_pred.detach().mean().cpu().item()),
            "q_target_mean": float(target_q.detach().mean().cpu().item()),
            "_total_loss_tensor": total,
        }

    # ---------- step ----------

    def train_step(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        info = self._compute_loss(batch)
        loss_tensor = info.pop("_total_loss_tensor")
        self.optimizer.zero_grad(set_to_none=True)
        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.optimizer.step()
        SafetyCritic.polyak_update(self.target_critic, self.critic, tau=self.target_tau)
        self._step_count += 1
        info["step"] = self._step_count
        return info


__all__ = ["CQLSafetyTrainer"]
