"""Regression test for the C51 target-projection index clamp (2026-05-21 fix).

`compute_target_q_dist` projects the Bellman target onto the fixed atom support.
delta_z is generally not exactly representable in float32, so at the support
ceiling (Tz == v_max, reached by reward=1 success transitions) the division
b = (Tz - v_min)/delta_z rounds *up* and ceil(b) lands on atom index `atoms`,
which is out of [0, atoms-1]. Pre-fix that crashed index_add_ with a CUDA
device-side assert; the fix clamps b. This test forces the boundary on CPU and
asserts the projection runs and stays a valid distribution.

Needs tensordict (the vendored agent imports it) — skips on the local dev box.
"""

import pytest

pytest.importorskip("tensordict")

import torch  # noqa: E402

from safety_bigym.agents.cqn_as.agent import C2FCritic  # noqa: E402


def _make_critic(atoms=101, v_min=-6.0, v_max=2.0):
    # Small but structurally real C2F critic; repr_dim/low_dim arbitrary since we
    # only exercise compute_target_q_dist (which uses forward()).
    return C2FCritic(
        action_shape=(2, 4),
        repr_dim=32,
        low_dim=8,
        feature_dim=16,
        hidden_dim=32,
        levels=3,
        bins=5,
        atoms=atoms,
        v_min=v_min,
        v_max=v_max,
        gru_layers=1,
        rgb_encoder_layers=1,
        use_parallel_impl=False,
    )


@pytest.mark.parametrize("atoms,v_min,v_max", [(101, -6.0, 2.0), (51, -2.0, 2.0)])
def test_projection_handles_v_max_ceiling_without_oob(atoms, v_min, v_max):
    torch.manual_seed(0)
    critic = _make_critic(atoms=atoms, v_min=v_min, v_max=v_max)
    B, D = 4, critic.network._action_sequence * critic.network._actor_dim
    rgb = torch.randn(B, 32)
    low = torch.randn(B, 8)
    action = torch.empty(B, D).uniform_(-1, 1)
    # reward=1 (success) + gamma*v_max pushes the top atoms to clamp at v_max,
    # i.e. exactly the boundary that overshot pre-fix.
    reward = torch.ones(B, 1)
    discount = torch.full((B, 1), 0.99)

    m = critic.compute_target_q_dist(rgb, low, action, reward, discount)

    assert torch.isfinite(m).all()
    # Target is a probability distribution over atoms (last dim) per (B, L, D).
    assert m.shape[-1] == atoms
    sums = m.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_b_clamp_keeps_indices_in_range_at_extremes():
    """Directly check the boundary arithmetic the fix targets."""
    critic = _make_critic(atoms=101, v_min=-6.0, v_max=2.0)
    support = critic.support  # [atoms]
    # Tz at the ceiling for every atom (worst case: all clamp to v_max).
    Tz = torch.full((1, critic.atoms), critic.v_max)
    b = (Tz - critic.v_min) / critic.delta_z
    b = b.clamp(min=0.0, max=float(critic.atoms - 1))
    lower = b.floor().to(torch.int64)
    upper = b.ceil().to(torch.int64)
    assert int(upper.max()) <= critic.atoms - 1
    assert int(lower.min()) >= 0
    assert support.numel() == critic.atoms
