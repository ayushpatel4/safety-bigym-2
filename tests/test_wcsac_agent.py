"""Unit tests for the WCSAC agent (E3.7 / P9 external safe-RL baseline).

Torch-only, CPU, tiny dims -- no env, no AMASS, no GPU. Covers:
  * the Gaussian-CVaR coefficient (closed form vs known values + Monte Carlo),
  * actor output shapes / squashing / log-prob finiteness,
  * one full update() step returns finite losses and the expected metric keys,
  * the Lagrange multiplier moves up when CVaR > budget and down when below,
  * state_dict / load_state_dict round-trip,
  * act() shapes and the uniform-random seeding branch.
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
from tensordict import TensorDict  # noqa: E402

from safety_bigym.agents.wcsac.nets import SquashedGaussianActor  # noqa: E402
from safety_bigym.agents.wcsac.wcsac_agent import (  # noqa: E402
    WCSACAgent,
    gaussian_cvar_coefficient,
)

# 84x84 is hard-coded into MultiViewCNNEncoder (repr_dim = V*256*5*5).
RGB_SHAPE = [1, 3, 84, 84]
LOW_DIM = 8
ACTION_SEQ = 1
ACT_DIM = 4
TOTAL_ACT = ACTION_SEQ * ACT_DIM


def _make_agent(num_expl_steps=0, cost_budget=1.0, lambda_init=0.0):
    return WCSACAgent(
        rgb_obs_shape=RGB_SHAPE,
        low_dim_obs_shape=[LOW_DIM],
        action_shape=[ACTION_SEQ, ACT_DIM],
        device="cpu",
        actor_lr=1e-3,
        critic_lr=1e-3,
        safety_lr=1e-3,
        alpha_lr=1e-3,
        init_temperature=0.1,
        critic_target_tau=0.01,
        critic_target_interval=1,
        update_every_steps=1,
        feature_dim=32,
        hidden_dim=32,
        weight_decay=0.0,
        cvar_alpha=0.9,
        cost_budget=cost_budget,
        lambda_lr=1e-2,
        lambda_init=lambda_init,
        lambda_max=100.0,
        num_expl_steps=num_expl_steps,
    )


def _make_batch(bsz=2):
    g = torch.Generator().manual_seed(0)
    rgb = torch.randint(0, 256, (bsz, *RGB_SHAPE), generator=g).float()
    next_rgb = torch.randint(0, 256, (bsz, *RGB_SHAPE), generator=g).float()
    return TensorDict(
        {
            "rgb_obs": rgb,
            "next_rgb_obs": next_rgb,
            "low_dim_obs": torch.randn(bsz, LOW_DIM, generator=g),
            "next_low_dim_obs": torch.randn(bsz, LOW_DIM, generator=g),
            "action": torch.empty(bsz, TOTAL_ACT).uniform_(-1, 1, generator=g),
            "reward": torch.randn(bsz, 1, generator=g),
            "cost": torch.rand(bsz, 1, generator=g),  # per-step cost in [0,1]
            "discount": torch.full((bsz, 1), 0.99),
            "demos": torch.zeros(bsz, 1),
            "max_cost": torch.rand(bsz, 1, generator=g),
        },
        batch_size=[bsz],
    )


# --------------------------------------------------------------------------
# Gaussian CVaR coefficient
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "alpha,expected",
    [
        (0.90, 1.7550),  # phi(Phi^-1(.9))/.1
        (0.95, 2.0627),
        (0.99, 2.6652),
    ],
)
def test_cvar_coefficient_known_values(alpha, expected):
    assert gaussian_cvar_coefficient(alpha) == pytest.approx(expected, abs=1e-3)


def test_cvar_coefficient_matches_monte_carlo():
    alpha = 0.9
    coef = gaussian_cvar_coefficient(alpha)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2_000_000, generator=g)  # standard normal: CVaR == coef
    q = torch.quantile(x, alpha)
    cvar_emp = x[x >= q].mean().item()
    assert cvar_emp == pytest.approx(coef, abs=2e-2)


def test_cvar_coefficient_rejects_bad_alpha():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            gaussian_cvar_coefficient(bad)


# --------------------------------------------------------------------------
# Actor
# --------------------------------------------------------------------------
def test_actor_shapes_and_squashing():
    actor = SquashedGaussianActor(
        repr_dim=64, low_dim=LOW_DIM, action_dim=TOTAL_ACT, feature_dim=16, hidden_dim=16
    )
    feats = torch.randn(5, 64)
    low = torch.randn(5, LOW_DIM)
    action, log_prob = actor(feats, low, sample=True)
    assert action.shape == (5, TOTAL_ACT)
    assert log_prob.shape == (5, 1)
    assert torch.isfinite(action).all() and torch.isfinite(log_prob).all()
    assert action.abs().max() < 1.0  # tanh-squashed strictly inside (-1, 1)

    mean_a = actor.mean_action(feats, low)
    assert mean_a.shape == (5, TOTAL_ACT)
    assert mean_a.abs().max() < 1.0


# --------------------------------------------------------------------------
# Agent update
# --------------------------------------------------------------------------
def test_update_returns_finite_metrics():
    agent = _make_agent()
    batch = _make_batch()
    metrics = agent.update(batch)
    expected = {
        "reward_critic_loss", "safety_mean_loss", "safety_var_loss",
        "batch_reward", "batch_cost", "qc_mean", "vc_mean",
        "actor_loss", "alpha_loss", "alpha", "entropy", "cvar",
        "lambda", "cost_violation",
    }
    assert expected.issubset(set(metrics.keys()))
    for k in expected:
        assert torch.isfinite(torch.as_tensor(metrics[k])).all(), f"{k} not finite"
    # A second step must also run cleanly (target nets, opts persist).
    agent.update_target_critic(1)
    metrics2 = agent.update(batch)
    assert torch.isfinite(torch.as_tensor(metrics2["actor_loss"])).all()


def test_update_rejects_wrong_action_dim():
    agent = _make_agent()
    batch = _make_batch()
    # Simulate action_sequence>1 leaking through (flattened dim mismatch).
    batch["action"] = torch.empty(batch["action"].shape[0], TOTAL_ACT * 16).uniform_(-1, 1)
    with pytest.raises(ValueError, match="action_sequence=1"):
        agent.update(batch)


# --------------------------------------------------------------------------
# Lagrange multiplier dual update
# --------------------------------------------------------------------------
def test_lambda_increases_when_cvar_exceeds_budget():
    agent = _make_agent(cost_budget=1.0, lambda_init=0.5)
    agent._last_cvar = 5.0  # well above budget
    before = agent._lambda
    agent._update_lambda()
    assert agent._lambda > before


def test_lambda_decreases_when_cvar_below_budget():
    agent = _make_agent(cost_budget=1.0, lambda_init=5.0)
    agent._last_cvar = 0.0  # below budget
    before = agent._lambda
    agent._update_lambda()
    assert agent._lambda < before


def test_lambda_stays_nonnegative_and_capped():
    agent = _make_agent(cost_budget=1.0, lambda_init=0.0)
    agent._last_cvar = -100.0  # huge negative violation
    agent._update_lambda()
    assert agent._lambda == 0.0  # projected to >= 0
    agent._lambda = 99.999
    agent.lambda_max = 100.0
    agent._last_cvar = 1e9
    agent._update_lambda()
    assert agent._lambda <= 100.0


# --------------------------------------------------------------------------
# Snapshot persistence
# --------------------------------------------------------------------------
def test_state_dict_round_trip():
    agent = _make_agent(lambda_init=0.0)
    # Take a couple of steps so weights / lambda / log_alpha move off init.
    batch = _make_batch()
    agent.update(batch)
    agent._lambda = 3.14
    sd = agent.state_dict()

    agent2 = _make_agent()
    agent2.load_state_dict(sd)

    # A reference parameter from each sub-network matches.
    for net_name in ("actor", "critic", "safety_critic", "encoder", "cost_encoder"):
        p1 = next(getattr(agent, net_name).parameters())
        p2 = next(getattr(agent2, net_name).parameters())
        assert torch.equal(p1, p2), f"{net_name} params differ after load"
    assert agent2._lambda == pytest.approx(3.14)
    assert torch.equal(agent.log_alpha.detach(), agent2.log_alpha.detach())


# --------------------------------------------------------------------------
# Acting
# --------------------------------------------------------------------------
def test_act_eval_is_deterministic_and_in_range():
    agent = _make_agent(num_expl_steps=0)
    rgb = np.random.randint(0, 256, size=RGB_SHAPE).astype(np.float32)
    low = np.random.randn(LOW_DIM).astype(np.float32)
    a1 = agent.act(rgb, low, step=10_000, eval_mode=True)
    a2 = agent.act(rgb, low, step=10_000, eval_mode=True)
    assert a1.shape == (TOTAL_ACT,)
    assert np.all(np.abs(a1) <= 1.0)
    np.testing.assert_allclose(a1, a2, atol=1e-6)  # mean action is deterministic


def test_act_seed_phase_is_uniform_random():
    agent = _make_agent(num_expl_steps=100)
    rgb = np.random.randint(0, 256, size=RGB_SHAPE).astype(np.float32)
    low = np.random.randn(LOW_DIM).astype(np.float32)
    a = agent.act(rgb, low, step=5, eval_mode=False)  # step < num_expl_steps
    assert a.shape == (TOTAL_ACT,)
    assert np.all(np.abs(a) <= 1.0)


def test_stochastic_act_flag_is_set():
    # train_cqn_as.py reads this to sample during collection.
    assert _make_agent().stochastic_act is True
