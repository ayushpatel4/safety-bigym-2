"""Tests for LagrangianCQNASAgent (Phase 3 P3.1).

These instantiate the real agent (small dims, CPU) so they exercise the cost
critic + dual-Q + lambda wiring end to end. The vendored agent imports
``tensordict``; skip cleanly when it is not installed (local dev box) -- the
2000-frame smoke on the GPU box covers the same path with the dependency present.
"""

import pytest

pytest.importorskip("tensordict")

import torch  # noqa: E402

from safety_bigym.agents.cqn_as.agent import CQNASAgent  # noqa: E402
from safety_bigym.agents.cqn_as.lagrangian_agent import (  # noqa: E402
    LagrangianCQNASAgent,
)

# 84x84 is load-bearing: MultiViewCNNEncoder hard-codes repr_dim for that size.
_V, _C, _H, _W = 1, 3, 84, 84
_LOW_DIM = 8
_ACTION_SEQ, _ACTION_DIM = 2, 4
_ACTOR_DIM = _ACTION_SEQ * _ACTION_DIM


def _base_kwargs():
    return dict(
        rgb_obs_shape=(_V, _C, _H, _W),
        low_dim_obs_shape=(_LOW_DIM,),
        action_shape=[_ACTION_SEQ, _ACTION_DIM],
        device="cpu",
        lr=1e-3,
        feature_dim=32,
        hidden_dim=64,
        levels=3,
        bins=5,
        atoms=11,
        v_min=-2.0,
        v_max=2.0,
        bc_lambda=1.0,
        bc_margin=0.1,
        gru_layers=1,
        rgb_encoder_layers=2,
        use_parallel_impl=False,
        critic_lambda=0.1,
        critic_target_tau=0.02,
        critic_target_interval=1,
        weight_decay=0.0,
        num_expl_steps=0,
        update_every_steps=1,
        stddev_schedule="0.01",
    )


def _make_batch(batch_size=4, cost_value=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.randint(0, 256, (batch_size, _V, _C, _H, _W), generator=g).float()
    nrgb = torch.randint(0, 256, (batch_size, _V, _C, _H, _W), generator=g).float()
    return {
        "rgb_obs": rgb,
        "low_dim_obs": torch.randn(batch_size, _LOW_DIM, generator=g),
        "action": torch.empty(batch_size, _ACTOR_DIM).uniform_(-1, 1, generator=g),
        "reward": torch.randn(batch_size, 1, generator=g),
        "discount": torch.full((batch_size, 1), 0.99),
        "next_rgb_obs": nrgb,
        "next_low_dim_obs": torch.randn(batch_size, _LOW_DIM, generator=g),
        "demos": torch.ones(batch_size, 1),
        "cost": torch.full((batch_size, 1), float(cost_value)),
        "max_cost": torch.full((batch_size, 1), float(cost_value)),
    }


def test_base_cqn_as_still_constructs_without_cost_q():
    """Plain CQN-AS (agent=cqn_as) is untouched: no cost critic attribute."""
    agent = CQNASAgent(**_base_kwargs())
    assert not hasattr(agent, "cost_critic")
    metrics = agent.update(_make_batch())
    assert "q_critic_loss" in metrics
    assert "q_c_loss" not in metrics


def test_lagrangian_agent_constructs_with_cost_critic():
    agent = LagrangianCQNASAgent(**_base_kwargs())
    assert hasattr(agent, "cost_critic")
    assert hasattr(agent, "cost_critic_target")
    assert hasattr(agent, "cost_encoder")
    # Cost critic uses its own (cost-range) support, distinct from reward critic.
    assert float(agent.cost_critic.v_min) == 0.0
    assert float(agent.cost_critic.v_max) == 10.0


def test_update_consumes_cost_and_logs_lagrangian_metrics():
    agent = LagrangianCQNASAgent(**_base_kwargs())
    batch = _make_batch(cost_value=0.5)
    metrics = agent.update(batch)
    for key in ("q_c_loss", "lambda", "rolling_cost", "cost_violation", "batch_cost"):
        assert key in metrics, f"missing {key}"
    assert torch.isfinite(metrics["q_c_loss"]).all()
    assert metrics["batch_cost"].item() == pytest.approx(0.5)


def test_cost_critic_params_change_on_update():
    """Confirms batch['cost'] actually drives a gradient step on Q_c."""
    agent = LagrangianCQNASAgent(**_base_kwargs())
    before = [p.detach().clone() for p in agent.cost_critic.parameters()]
    agent.update(_make_batch(cost_value=0.5))
    after = list(agent.cost_critic.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_q_c_loss_finite_and_decreases_on_fixed_batch():
    torch.manual_seed(0)
    agent = LagrangianCQNASAgent(**_base_kwargs())
    batch = _make_batch(cost_value=0.5)
    losses = []
    for _ in range(20):
        m = agent.update(batch)  # target nets NOT soft-updated -> stationary target
        losses.append(m["q_c_loss"].item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_lambda_rises_when_cost_exceeds_budget():
    agent = LagrangianCQNASAgent(**_base_kwargs())
    assert agent.lam == 0.0
    for _ in range(10):
        agent.update(_make_batch(cost_value=0.8))  # 0.8 >> budget 0.01
    assert agent.lam > 0.0


def test_update_target_critic_soft_updates_cost_target():
    agent = LagrangianCQNASAgent(**_base_kwargs())
    agent.update(_make_batch(cost_value=0.5))  # move online cost critic off target
    before = [p.detach().clone() for p in agent.cost_critic_target.parameters()]
    agent.update_target_critic(step=1)  # interval == 1 -> fires
    after = list(agent.cost_critic_target.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_state_dict_roundtrip_includes_cost_nets_and_lambda():
    agent = LagrangianCQNASAgent(**_base_kwargs())
    for _ in range(3):
        agent.update(_make_batch(cost_value=0.5))
    sd = agent.state_dict()
    for key in ("cost_critic", "cost_critic_target", "cost_encoder", "lambda"):
        assert key in sd
    fresh = LagrangianCQNASAgent(**_base_kwargs())
    fresh.load_state_dict(sd)
    assert fresh.lam == agent.lam
    for a, b in zip(
        agent.cost_critic.parameters(), fresh.cost_critic.parameters()
    ):
        assert torch.equal(a, b)
