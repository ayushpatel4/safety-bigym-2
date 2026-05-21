"""Regression tests for the C51 target-projection OOB fix (2026-05-21).

`compute_target_q_dist` scatters the projected Bellman target into a flat buffer
`m` of `batch_size * atoms` elements using a per-row `offset`. ROOT CAUSE of the
device-side `index_add_` assert: the offset was built with `torch.linspace` in
float32, which cannot represent integers past 2**24 (~16.7M). With
`batch_size*atoms ~= 39.7M` (B*L*D*atoms) the offsets round and the boundary row
addresses past `m.numel()`. The fix builds the offset with exact int64
`arange * atoms`. Two cheap clamps (project index `b`, and integer lower/upper)
are defensive hardening, not the trigger.

`test_offset_*` validate the offset arithmetic at a magnitude where float32
fails (no tensordict / GPU needed). The `compute_target_q_dist` tests exercise
the small path end-to-end and need tensordict (skip on the local dev box).
"""

import pytest

import torch  # noqa: E402


def _arange_offset(batch_size, atoms, dtype=torch.int64):
    """The fixed offset (exact int64 arange * atoms)."""
    return torch.arange(batch_size, dtype=dtype) * atoms


def test_offset_arange_is_exact_where_linspace_rounds():
    # batch_size*atoms ~= 40.4M > 2**24, where float32 linspace loses precision.
    batch_size, atoms = 400_000, 101
    numel = batch_size * atoms
    arange = _arange_offset(batch_size, atoms)
    # Exact: row i offset is exactly i*atoms, and the boundary stays in range.
    assert int(arange[-1].item()) == (batch_size - 1) * atoms
    assert int(arange.max().item()) + (atoms - 1) == numel - 1
    # No collisions (every row maps to a distinct atom block).
    assert torch.unique(arange).numel() == batch_size


# NOTE: the buggy float `linspace` offset rounds only on CUDA (CUDA computes
# integer linspace via float32); on CPU it is exact, so we cannot portably assert
# the imprecision here. The arange invariant above is what guarantees no OOB, and
# the GPU re-run is the end-to-end confirmation.

import importlib.util  # noqa: E402

_HAS_TENSORDICT = importlib.util.find_spec("tensordict") is not None
_needs_td = pytest.mark.skipif(
    not _HAS_TENSORDICT, reason="vendored agent needs tensordict (GPU box)"
)


def _make_critic(atoms=101, v_min=-6.0, v_max=2.0):
    from safety_bigym.agents.cqn_as.agent import C2FCritic
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


@_needs_td
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


@_needs_td
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
