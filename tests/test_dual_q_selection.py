"""Tests for the dual-Q bin selector (Phase 3 P3.1).

``dual_select`` is the cost-aware argmax injected at every coarse-to-fine
level. Pure torch -- no ``tensordict`` / MuJoCo.
"""

import torch

from safety_bigym.agents.cqn_as.lagrangian import dual_select


def test_lambda_zero_matches_reward_argmax():
    # Reward best at bin 1; cost irrelevant when lambda == 0.
    qs_r = torch.tensor([[[0.0, 1.0, 0.5]]])  # [B=1, D=1, bins=3]
    qs_c = torch.tensor([[[0.0, 5.0, 2.0]]])
    out = dual_select(qs_r, qs_c, lam=0.0)
    assert torch.equal(out, qs_r.max(-1)[1])
    assert out.item() == 1


def test_large_lambda_shifts_to_low_cost_bin():
    # Reward prefers bin 1, but bin 1 is the most dangerous; bin 0 is cheapest.
    qs_r = torch.tensor([[[0.0, 1.0, 0.5]]])
    qs_c = torch.tensor([[[0.0, 5.0, 2.0]]])
    out = dual_select(qs_r, qs_c, lam=100.0)
    assert out.item() == 0  # low-cost bin wins under heavy penalty


def test_shape_preserved_over_batch_and_dims():
    B, D, bins = 4, 6, 5
    qs_r = torch.randn(B, D, bins)
    qs_c = torch.randn(B, D, bins)
    out = dual_select(qs_r, qs_c, lam=0.3)
    assert out.shape == (B, D)
    expected = (qs_r - 0.3 * qs_c).max(-1)[1]
    assert torch.equal(out, expected)


def test_monotone_in_lambda_picks_cheaper_as_lambda_grows():
    qs_r = torch.tensor([[[1.0, 1.0]]])  # tie on reward
    qs_c = torch.tensor([[[2.0, 0.0]]])  # bin 1 strictly cheaper
    assert dual_select(qs_r, qs_c, lam=0.0).item() == 0  # tie -> first index
    assert dual_select(qs_r, qs_c, lam=1.0).item() == 1  # cost breaks the tie
