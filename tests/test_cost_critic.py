"""Tests for filters/cost_critic.py — Phase 3 cost critic Q_c.

Mirrors the SafetyCritic test suite where the architectures overlap, then
covers the warm-start-from-SVF guard and head-reinit invariant that are
unique to CostCritic.
"""

import copy

import numpy as np
import pytest
import torch

from safety_bigym.filters.cost_critic import CostCritic
from safety_bigym.filters.critic import SafetyCritic
from safety_bigym.filters.feature_extractor import CriticFeatureSpec


def _make_spec(obs_dim: int = 8, action_dim: int = 4) -> CriticFeatureSpec:
    return CriticFeatureSpec(
        obs_keys=("low_dim_state",),
        obs_dims=(obs_dim,),
        action_dim=action_dim,
    )


# ---------- shared invariants (architectural twin of SafetyCritic) ----------


def test_output_in_bounds_0_to_qmax():
    spec = _make_spec()
    critic = CostCritic(spec=spec, gamma=0.99)
    assert critic.q_max == pytest.approx(100.0)

    feats = torch.randn(64, spec.input_dim) * 100
    q = critic(feats)
    assert q.shape == (64,)
    assert (q >= 0.0).all()
    assert (q <= critic.q_max + 1e-3).all()


def test_q_max_matches_gamma():
    spec = _make_spec()
    for gamma in (0.9, 0.95, 0.99):
        critic = CostCritic(spec=spec, gamma=gamma)
        assert critic.q_max == pytest.approx(1.0 / (1.0 - gamma))


def test_input_dim_matches_safety_critic_for_same_spec():
    """Architectural twin: same spec → same input_dim → weight-transfer compatible."""
    spec = _make_spec(obs_dim=12, action_dim=16)
    a = SafetyCritic(spec=spec, gamma=0.99)
    b = CostCritic(spec=spec, gamma=0.99)
    assert a.input_dim == b.input_dim
    # State dicts have identical key structure
    assert list(a.state_dict().keys()) == list(b.state_dict().keys())
    # And identical per-parameter shapes
    for k in a.state_dict():
        assert a.state_dict()[k].shape == b.state_dict()[k].shape


def test_q_value_inference_helper():
    spec = _make_spec()
    critic = CostCritic(spec=spec, gamma=0.99)
    obs = {"low_dim_state": np.zeros(spec.obs_dims[0], dtype=np.float32)}
    action = np.zeros(spec.action_dim, dtype=np.float32)
    q = critic.q_value(obs, action)
    assert isinstance(q, float)
    assert 0.0 <= q <= critic.q_max


def test_target_network_is_independent_no_grad():
    spec = _make_spec()
    src = CostCritic(spec=spec)
    tgt = src.make_target()
    for p in tgt.parameters():
        assert not p.requires_grad
    # Mutate source weights; target must not change
    snapshot = {k: v.clone() for k, v in tgt.state_dict().items()}
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    for k, v in tgt.state_dict().items():
        assert torch.equal(v, snapshot[k])


def test_polyak_update_moves_toward_source():
    spec = _make_spec()
    src = CostCritic(spec=spec)
    tgt = src.make_target()
    # Move source far from target
    with torch.no_grad():
        for p in src.parameters():
            p.add_(10.0)
    pre = {k: v.clone() for k, v in tgt.state_dict().items()}
    CostCritic.polyak_update(tgt, src, tau=0.1)
    post = tgt.state_dict()
    src_sd = src.state_dict()
    for k in pre:
        # τ=0.1 → post ≈ 0.9*pre + 0.1*src
        expected = 0.9 * pre[k] + 0.1 * src_sd[k]
        assert torch.allclose(post[k], expected)


# ---------- checkpoint round-trip ----------


def test_checkpoint_round_trip_preserves_outputs():
    spec = _make_spec()
    a = CostCritic(spec=spec, gamma=0.95)
    payload = a.checkpoint_payload()
    b = CostCritic.from_checkpoint_payload(payload)
    feats = torch.randn(32, spec.input_dim)
    a.eval()
    b.eval()
    with torch.no_grad():
        out_a = a(feats)
        out_b = b(feats)
    assert torch.allclose(out_a, out_b)
    assert b.gamma == a.gamma
    assert b.hidden_dims == a.hidden_dims
    assert b.spec == a.spec


# ---------- warm-start from SVF: the sign-flip guard ----------


def test_warm_start_default_raises_with_sign_message():
    """Without force_sign_flip the guard must fire — sign mismatch is the whole point."""
    spec = _make_spec()
    svf = SafetyCritic(spec=spec, gamma=0.99)
    cost = CostCritic(spec=spec, gamma=0.99)
    with pytest.raises(ValueError) as exc:
        cost.warm_start_from_svf(svf.checkpoint_payload())
    msg = str(exc.value)
    assert "force_sign_flip=True" in msg
    assert "sign" in msg.lower() or "direction" in msg.lower()


def test_warm_start_spec_mismatch_raises():
    cost_spec = _make_spec(obs_dim=8, action_dim=4)
    svf_spec = _make_spec(obs_dim=12, action_dim=4)  # different obs_dim
    cost = CostCritic(spec=cost_spec, gamma=0.99)
    svf = SafetyCritic(spec=svf_spec, gamma=0.99)
    with pytest.raises(ValueError, match="spec does not match"):
        cost.warm_start_from_svf(svf.checkpoint_payload(), force_sign_flip=True)


def test_warm_start_hidden_dims_mismatch_raises():
    spec = _make_spec()
    cost = CostCritic(spec=spec, gamma=0.99, hidden_dims=(256, 256, 256))
    svf = SafetyCritic(spec=spec, gamma=0.99, hidden_dims=(128, 128, 128))
    with pytest.raises(ValueError, match="hidden_dims do not match"):
        cost.warm_start_from_svf(svf.checkpoint_payload(), force_sign_flip=True)


def test_warm_start_force_copies_body_and_reinits_head():
    """force_sign_flip=True: body params from SVF, terminal Linear stays at fresh init."""
    spec = _make_spec()
    hidden = (256, 256, 256)
    svf = SafetyCritic(spec=spec, gamma=0.99, hidden_dims=hidden)
    cost = CostCritic(spec=spec, gamma=0.99, hidden_dims=hidden)

    # Snapshot CostCritic's fresh init for the terminal head before warm-start.
    terminal_idx = 2 * len(hidden)  # net is Linear/ReLU pairs + final Linear
    head_w_key = f"net.{terminal_idx}.weight"
    head_b_key = f"net.{terminal_idx}.bias"
    fresh_head_w = cost.state_dict()[head_w_key].clone()
    fresh_head_b = cost.state_dict()[head_b_key].clone()

    # Bump SVF weights so we can detect the copy.
    with torch.no_grad():
        for p in svf.parameters():
            p.add_(1.0)
    svf_sd = svf.state_dict()

    cost.warm_start_from_svf(svf.checkpoint_payload(), force_sign_flip=True)

    cost_sd = cost.state_dict()
    # Body params: every key EXCEPT the terminal head should match SVF.
    body_keys = [k for k in cost_sd.keys() if k not in {head_w_key, head_b_key}]
    assert body_keys, "expected at least one non-head parameter to copy"
    for k in body_keys:
        assert torch.equal(cost_sd[k], svf_sd[k]), f"body param {k} did not copy from SVF"

    # Head: untouched, still at fresh init.
    assert torch.equal(cost_sd[head_w_key], fresh_head_w)
    assert torch.equal(cost_sd[head_b_key], fresh_head_b)

    # Diagnostic fields set by warm_start_from_svf
    assert cost._warm_start_skipped == sorted([head_w_key, head_b_key])
    assert cost._warm_start_loaded_params == len(body_keys)


def test_warm_start_does_not_share_storage_with_svf():
    """After warm-start, mutating SVF must not bleed into CostCritic."""
    spec = _make_spec()
    svf = SafetyCritic(spec=spec, gamma=0.99)
    cost = CostCritic(spec=spec, gamma=0.99)
    cost.warm_start_from_svf(svf.checkpoint_payload(), force_sign_flip=True)
    cost_sd_before = {k: v.clone() for k, v in cost.state_dict().items()}
    with torch.no_grad():
        for p in svf.parameters():
            p.add_(100.0)
    cost_sd_after = cost.state_dict()
    for k in cost_sd_before:
        assert torch.equal(cost_sd_before[k], cost_sd_after[k])
