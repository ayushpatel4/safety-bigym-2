"""Tests for filters/cql_trainer.py — offline CQL trainer for SafetyCritic."""

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from safety_bigym.filters.aux_unsafe_provider import (
    AuxUnsafeProvider,
    EmptyAuxProvider,
)
from safety_bigym.filters.cql_trainer import CQLSafetyTrainer
from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.dataset import (
    SafetyTransitionDataset,
    TransitionShardWriter,
)
from safety_bigym.filters.feature_extractor import CriticFeatureSpec


def _make_spec(obs_dim: int = 6, action_dim: int = 4) -> CriticFeatureSpec:
    return CriticFeatureSpec(
        obs_keys=("low_dim_state",),
        obs_dims=(obs_dim,),
        action_dim=action_dim,
    )


def _action_space(action_dim: int = 4) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)


def _toy_dataset(tmp_path: Path, *, n: int = 256, n_violations: int = 64):
    spec = _make_spec()
    rng = np.random.default_rng(0)
    obs = {"low_dim_state": rng.standard_normal((n, spec.obs_dims[0])).astype(np.float32)}
    next_obs = {"low_dim_state": rng.standard_normal((n, spec.obs_dims[0])).astype(np.float32)}
    action = rng.standard_normal((n, spec.action_dim)).astype(np.float32).clip(-1, 1)
    r_safe = np.ones(n, dtype=np.float32)
    r_safe[:n_violations] = 0.0
    done = np.zeros(n, dtype=np.bool_)
    done[:n_violations] = True
    margin = rng.standard_normal(n).astype(np.float32)
    writer = TransitionShardWriter(spec, tmp_path)
    writer.write_shard(
        name="toy",
        obs=obs,
        action=action,
        next_obs=next_obs,
        r_safe=r_safe,
        done=done,
        ssm_margin=margin,
        source=np.zeros(n, dtype=np.uint8),
        task_id=np.zeros(n, dtype=np.uint8),
    )
    return SafetyTransitionDataset(tmp_path), spec


def test_bellman_loss_decreases_on_toy_data(tmp_path):
    """200 sequential grad steps must decrease the Bellman MSE component.

    The total loss can grow with CQL because the conservatism term scales
    with how distinguishable Q is on data vs OOD actions — that's a
    *learning signal*, not a fitting error. So we test bellman_loss, which
    is the only term whose decrease implies improved fitting.

    Uses ``shuffle=False`` and ``cql_alpha=0.5`` so the Bellman dynamics
    aren't washed out by aggressive conservatism on a tiny toy dataset.
    """
    ds, spec = _toy_dataset(tmp_path)
    critic = SafetyCritic(spec=spec, gamma=0.99)
    torch.manual_seed(0)
    trainer = CQLSafetyTrainer(
        critic=critic,
        action_space=_action_space(spec.action_dim),
        cql_alpha=0.5,
        lr=3e-4,
        target_tau=5e-3,
        device="cpu",
        seed=0,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    bellman = []
    step = 0
    while step < 200:
        for batch in loader:
            if step >= 200:
                break
            info = trainer.train_step(batch)
            bellman.append(info["bellman_loss"])
            step += 1
    assert bellman, "Trainer never stepped"
    # Compare the BEST late-window loss against the WORST early-window loss
    # to reduce sensitivity to per-step noise.
    q1 = max(1, len(bellman) // 4)
    early_max = float(np.max(bellman[:q1]))
    late_min = float(np.min(bellman[-q1:]))
    assert math.isfinite(late_min)
    assert late_min < early_max, (
        f"Bellman MSE didn't decrease: early_max={early_max} late_min={late_min}"
    )


def test_cql_term_scales_with_alpha(tmp_path):
    """Bigger α ⇒ stronger conservatism ⇒ lower mean Q on data states."""
    ds, spec = _toy_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batch = next(iter(loader))

    critic = SafetyCritic(spec=spec, gamma=0.99)
    trainer_lo = CQLSafetyTrainer(
        critic=critic,
        action_space=_action_space(spec.action_dim),
        cql_alpha=0.0,
        seed=0,
    )
    info_lo = trainer_lo._compute_loss(batch)

    # Re-init critic to the same weights via state_dict round-trip; otherwise
    # the lo-α trainer's grad mutates the shared critic.
    critic2 = SafetyCritic(spec=spec, gamma=0.99)
    critic2.load_state_dict(critic.state_dict())
    trainer_hi = CQLSafetyTrainer(
        critic=critic2,
        action_space=_action_space(spec.action_dim),
        cql_alpha=10.0,
        seed=0,
    )
    info_hi = trainer_hi._compute_loss(batch)

    assert info_hi["cql_term"] >= info_lo["cql_term"] - 1e-6
    # And the total loss strictly grows when α grows (CQL term is added)
    assert info_hi["loss"] > info_lo["loss"]


def test_aux_loss_off_when_weight_zero(tmp_path):
    """Aux provider must not be called when aux_weight=0."""
    ds, spec = _toy_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    batch = next(iter(loader))

    class _CountingProvider(AuxUnsafeProvider):
        def __init__(self):
            self.calls = 0

        def sample(self, batch_size, spec, device):
            self.calls += 1
            return None

    counter = _CountingProvider()
    trainer = CQLSafetyTrainer(
        critic=SafetyCritic(spec=spec, gamma=0.99),
        action_space=_action_space(spec.action_dim),
        cql_alpha=1.0,
        aux_weight=0.0,
        aux_provider=counter,
        seed=0,
    )
    trainer._compute_loss(batch)
    assert counter.calls == 0, "Aux provider was called despite aux_weight=0"


def test_aux_loss_invoked_when_weight_positive(tmp_path):
    ds, spec = _toy_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    batch = next(iter(loader))

    class _ZeroProvider(AuxUnsafeProvider):
        def __init__(self):
            self.calls = 0

        def sample(self, batch_size, spec, device):
            self.calls += 1
            # Return a tiny synthetic unsafe batch
            return {
                "features": torch.zeros(batch_size, spec.input_dim),
            }

    provider = _ZeroProvider()
    trainer = CQLSafetyTrainer(
        critic=SafetyCritic(spec=spec, gamma=0.99),
        action_space=_action_space(spec.action_dim),
        cql_alpha=1.0,
        aux_weight=0.5,
        aux_provider=provider,
        seed=0,
    )
    info = trainer._compute_loss(batch)
    assert provider.calls == 1
    assert "aux_loss" in info
    assert info["aux_loss"] >= 0.0


def test_target_network_updates_after_step(tmp_path):
    ds, spec = _toy_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    critic = SafetyCritic(spec=spec, gamma=0.99)
    trainer = CQLSafetyTrainer(
        critic=critic,
        action_space=_action_space(spec.action_dim),
        cql_alpha=1.0,
        target_tau=0.5,  # large τ so we see motion
        seed=0,
    )
    pre_target_state = {k: v.clone() for k, v in trainer.target_critic.state_dict().items()}
    trainer.train_step(next(iter(loader)))
    post_target_state = trainer.target_critic.state_dict()
    # At least one parameter should have moved.
    moved = False
    for k in pre_target_state:
        if not torch.allclose(pre_target_state[k], post_target_state[k]):
            moved = True
            break
    assert moved, "Target network did not update after a step"


def test_default_aux_provider_is_empty(tmp_path):
    """If no aux provider is supplied, the trainer uses the inert default."""
    ds, spec = _toy_dataset(tmp_path)
    trainer = CQLSafetyTrainer(
        critic=SafetyCritic(spec=spec, gamma=0.99),
        action_space=_action_space(spec.action_dim),
        cql_alpha=1.0,
        seed=0,
    )
    assert isinstance(trainer.aux_provider, EmptyAuxProvider)


def test_train_step_returns_finite_metrics(tmp_path):
    ds, spec = _toy_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    trainer = CQLSafetyTrainer(
        critic=SafetyCritic(spec=spec, gamma=0.99),
        action_space=_action_space(spec.action_dim),
        cql_alpha=1.0,
        seed=0,
    )
    info = trainer.train_step(next(iter(loader)))
    for k, v in info.items():
        if isinstance(v, (int, float)):
            assert math.isfinite(float(v)), f"{k}={v} is not finite"
