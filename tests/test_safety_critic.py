"""Tests for filters/critic.py — bounded-output safety critic MLP."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.feature_extractor import CriticFeatureSpec


def _make_spec(obs_dim: int = 8, action_dim: int = 4) -> CriticFeatureSpec:
    return CriticFeatureSpec(
        obs_keys=("low_dim_state",),
        obs_dims=(obs_dim,),
        action_dim=action_dim,
    )


def test_output_in_bounds_0_to_qmax():
    """With γ=0.99, q_max = 1/(1-γ) = 100. Output must lie in [0, 100]."""
    spec = _make_spec()
    critic = SafetyCritic(spec=spec, gamma=0.99)
    assert critic.q_max == pytest.approx(100.0)

    feats = torch.randn(64, spec.input_dim) * 100  # extreme inputs
    q = critic(feats)
    assert q.shape == (64,)
    assert (q >= 0.0).all()
    assert (q <= critic.q_max + 1e-3).all()


def test_q_max_matches_gamma():
    spec = _make_spec()
    for gamma in (0.9, 0.95, 0.99):
        critic = SafetyCritic(spec=spec, gamma=gamma)
        assert critic.q_max == pytest.approx(1.0 / (1.0 - gamma))


def test_gradient_flows():
    spec = _make_spec()
    critic = SafetyCritic(spec=spec, gamma=0.99)
    feats = torch.randn(8, spec.input_dim, requires_grad=True)
    q = critic(feats).sum()
    q.backward()
    # All params should receive gradient
    for p in critic.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_q_value_from_obs_action():
    """High-level convenience: critic.q_value(obs_dict, action) -> scalar."""
    spec = _make_spec(obs_dim=6, action_dim=4)
    critic = SafetyCritic(spec=spec, gamma=0.99)

    obs = {"low_dim_state": np.zeros(6, dtype=np.float32)}
    action = np.zeros(4, dtype=np.float32)

    q = critic.q_value(obs, action)
    assert isinstance(q, float)
    assert 0.0 <= q <= critic.q_max + 1e-3


def test_q_value_supports_batched_inputs():
    spec = _make_spec(obs_dim=6, action_dim=4)
    critic = SafetyCritic(spec=spec, gamma=0.99)

    obs = {"low_dim_state": np.zeros((3, 6), dtype=np.float32)}
    action = np.zeros((3, 4), dtype=np.float32)

    q = critic.q_value(obs, action)
    assert isinstance(q, np.ndarray)
    assert q.shape == (3,)


def test_no_pixel_keys_in_signature():
    """The critic must reject specs that include pixel keys."""
    pixel_space = gym.spaces.Dict(
        {
            "low_dim_state": gym.spaces.Box(-1, 1, (4,), np.float32),
            "rgb_head": gym.spaces.Box(0, 255, (3, 84, 84), np.uint8),
        }
    )
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
    spec = CriticFeatureSpec.from_spaces(pixel_space, action_space)
    # Spec already drops rgb; the critic should accept this spec but its input
    # dim must match the dropped layout
    critic = SafetyCritic(spec=spec, gamma=0.99)
    assert critic.input_dim == 4 + 2


def test_state_dict_round_trip():
    """SafetyCritic checkpoint round-trip must preserve weights AND spec."""
    spec = _make_spec()
    critic = SafetyCritic(spec=spec, gamma=0.99)
    payload = critic.checkpoint_payload()

    restored = SafetyCritic.from_checkpoint_payload(payload)
    assert restored.spec.to_dict() == critic.spec.to_dict()
    assert restored.gamma == critic.gamma

    feats = torch.randn(4, spec.input_dim)
    a = critic(feats)
    b = restored(feats)
    assert torch.allclose(a, b)


def test_target_network_polyak_update():
    """make_target() deep-copies; polyak_update() pulls target toward source."""
    spec = _make_spec()
    critic = SafetyCritic(spec=spec, gamma=0.99)
    target = critic.make_target()
    # Target params start identical to source.
    for tp, sp in zip(target.parameters(), critic.parameters()):
        assert torch.allclose(tp, sp)
    # Target params have requires_grad=False.
    assert all(not p.requires_grad for p in target.parameters())

    # Snapshot target params, then move source weights, then polyak.
    target_pre = [p.detach().clone() for p in target.parameters()]
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(torch.ones_like(p))  # deterministic offset

    SafetyCritic.polyak_update(target, critic, tau=0.5)

    # target_new = 0.5 * source + 0.5 * target_pre
    for tp_new, sp, tp_pre in zip(target.parameters(), critic.parameters(), target_pre):
        expected = 0.5 * sp.detach() + 0.5 * tp_pre
        assert torch.allclose(tp_new, expected, atol=1e-6)


def test_polyak_tau_one_copies_source():
    spec = _make_spec()
    critic = SafetyCritic(spec=spec, gamma=0.99)
    target = critic.make_target()
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(torch.ones_like(p))
    SafetyCritic.polyak_update(target, critic, tau=1.0)
    for tp, sp in zip(target.parameters(), critic.parameters()):
        assert torch.allclose(tp, sp)


def test_critic_is_deterministic_in_eval_mode():
    spec = _make_spec()
    critic = SafetyCritic(spec=spec, gamma=0.99).eval()
    feats = torch.randn(8, spec.input_dim)
    q1 = critic(feats)
    q2 = critic(feats)
    assert torch.allclose(q1, q2)
