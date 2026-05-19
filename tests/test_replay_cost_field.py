"""P3.0c round-trip: synthetic episode on disk → ReplayBuffer._sample → cost preserved.

The plan's load-bearing correctness gate (``UPDATED_PROJECT_PLAN.md:348``,
pre-port smoke 3) is that the per-step cost ``c_t`` reaches the batch
*per-env-step*, not aggregated per K-action chunk. This test builds an
episode with a known per-step cost sequence, points ReplayBuffer at it,
and asserts:

1. The accumulated ``cost`` over the n-step window matches the n-step
   discounted-cost return computed by hand.
2. The ``max_cost`` field equals the per-env-step max within the same window.
3. Pre-P3.0c shards (no ``cost`` key on disk) yield zeros without crashing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from safety_bigym.agents.cqn_as.replay_buffer import (
    ReplayBuffer,
    save_episode,
)


def _make_episode(
    *,
    cost_pattern: np.ndarray,
    reward_pattern: np.ndarray,
    rgb_shape: tuple = (1, 3, 8, 8),
    low_dim_dim: int = 6,
    action_dim: int = 4,
) -> dict:
    """Build an episode dict with the schema ReplayBuffer expects.

    Length is inferred from ``cost_pattern`` (must equal reward_pattern length);
    the +1 dummy first transition convention matches episode_len() in the buffer.
    """
    total = cost_pattern.shape[0]
    assert reward_pattern.shape[0] == total
    return {
        "rgb_obs": np.zeros((total, *rgb_shape[1:]), dtype=np.uint8),
        "low_dim_obs": np.zeros((total, low_dim_dim), dtype=np.float32),
        "action": np.zeros((total, action_dim), dtype=np.float32),
        "reward": reward_pattern.reshape(total, 1).astype(np.float32),
        "discount": np.ones((total, 1), dtype=np.float32),
        "demo": np.zeros((total, 1), dtype=np.float32),
        "cost": cost_pattern.reshape(total, 1).astype(np.float32),
    }


def _save(tmp_path: Path, episode: dict, eps_idx: int = 0) -> None:
    eps_len = next(iter(episode.values())).shape[0] - 1
    fn = tmp_path / f"20260518T000000_{eps_idx}_{eps_len}.npz"
    save_episode(episode, fn)


def _drain_one_sample(buf: ReplayBuffer):
    """Force the buffer to load on-disk episodes then return one sampled tuple."""
    buf._samples_since_last_fetch = buf._fetch_every  # force fetch
    return buf._sample()


# ---------- the two load-bearing P3.0c assertions ----------


def test_cost_accumulates_as_nstep_discounted_return(tmp_path):
    """Cost over the n-step window must equal sum_{i=0..n-1} γ^i * c_{t+i}.

    With γ=1 and nstep=4, the accumulator is a plain sum; with γ<1 it's
    discounted. Pinning both clarifies the contract for Phase 3 Q_c.
    """
    eps_len = 12
    # Cost pattern: a tall spike at step 3 (index 4 incl. dummy), zero elsewhere
    cost = np.zeros(eps_len + 1, dtype=np.float32)
    cost[4] = 0.9
    cost[5] = 0.3
    cost[7] = 0.5
    reward = np.zeros(eps_len + 1, dtype=np.float32)
    reward[2:6] = 1.0

    ep = _make_episode(cost_pattern=cost, reward_pattern=reward)
    _save(tmp_path, ep)

    # nstep=4, discount=0.9 to make per-step contributions distinct
    buf = ReplayBuffer(
        replay_dir=tmp_path,
        max_size=1000,
        num_workers=1,
        nstep=4,
        discount=0.9,
        action_sequence=1,
        frame_stack=1,
        fetch_every=1,
        save_snapshot=True,
    )
    # Seed numpy so we deterministically hit idx=4 (the spike window start).
    # _sample picks idx = np.random.randint(0, episode_len - nstep + 1) + 1.
    # episode_len = 12, nstep = 4 → randint(0, 9) → +1 ∈ [1, 9].
    rng = np.random.default_rng(0)
    np.random.seed(0)  # _sample reads global state, not a passed RNG
    # Try a few seeds until we land on idx=4 (the most informative window).
    cost_acc = None
    max_cost_acc = None
    for seed in range(50):
        np.random.seed(seed)
        sample = _drain_one_sample(buf)
        # tuple indices: ..., 8=cost, 9=max_cost
        # also infer idx by re-sampling: but we instead verify against any
        # valid n-step window from the buf's deterministic accumulator.
        # Compute the ground truth for every possible idx and pick the one
        # that matches the sample's cost.
        episode_len_val = eps_len
        nstep = 4
        gamma = 0.9
        gt_cost_per_idx = {}
        gt_max_per_idx = {}
        for idx in range(1, episode_len_val - nstep + 2):
            acc = 0.0
            mx = 0.0
            for i in range(nstep):
                acc += (gamma ** i) * cost[idx + i]
                mx = max(mx, cost[idx + i])
            gt_cost_per_idx[idx] = acc
            gt_max_per_idx[idx] = mx
        # Check sample's cost matches some valid idx's gt cost
        sample_cost = float(sample[8].reshape(-1)[0])
        sample_max = float(sample[9].reshape(-1)[0])
        matched_idx = None
        for idx, gt in gt_cost_per_idx.items():
            if abs(sample_cost - gt) < 1e-5:
                matched_idx = idx
                break
        assert matched_idx is not None, (
            f"sample cost {sample_cost} matches no valid n-step window for "
            f"episode_len={episode_len_val}, nstep={nstep}, gamma={gamma}"
        )
        # And max_cost must match the same idx's per-step max
        assert sample_max == pytest.approx(gt_max_per_idx[matched_idx]), (
            f"sample max_cost {sample_max} != gt {gt_max_per_idx[matched_idx]} "
            f"at matched idx={matched_idx}"
        )
        cost_acc = sample_cost
        max_cost_acc = sample_max
        if matched_idx == 4:  # the most informative window — spike at t
            break

    assert cost_acc is not None
    assert max_cost_acc is not None


def test_max_cost_captures_per_step_spike_within_window(tmp_path):
    """A single 1.0 spike anywhere in the n-step window must surface as max_cost=1.0.

    This is the per-env-step granularity guarantee: a spike in any one
    env-step within the K-action chunk reaches the batch dict — it is NOT
    diluted by being averaged with surrounding zero-cost steps.
    """
    eps_len = 8
    cost = np.zeros(eps_len + 1, dtype=np.float32)
    cost[3] = 1.0  # one and only spike
    reward = np.zeros(eps_len + 1, dtype=np.float32)

    ep = _make_episode(cost_pattern=cost, reward_pattern=reward)
    _save(tmp_path, ep)

    buf = ReplayBuffer(
        replay_dir=tmp_path,
        max_size=1000,
        num_workers=1,
        nstep=4,  # window large enough to capture the spike from idx 1..3
        discount=1.0,
        action_sequence=1,
        frame_stack=1,
        fetch_every=1,
        save_snapshot=True,
    )

    # Sample many times; for any window that overlaps step 3, max_cost must be 1.0.
    spike_window_seen = False
    for seed in range(200):
        np.random.seed(seed)
        sample = _drain_one_sample(buf)
        # Reconstruct which idx was sampled by matching n-step cost sum.
        sample_cost = float(sample[8].reshape(-1)[0])
        sample_max = float(sample[9].reshape(-1)[0])
        # Try every valid idx; if one of them overlaps step 3, max_cost should be 1.0.
        for idx in range(1, eps_len - 3 + 1):  # nstep=4, so idx + 3 must be ≤ eps_len
            window = cost[idx : idx + 4]
            if abs(sample_cost - window.sum()) < 1e-5:
                if 3 in range(idx, idx + 4):
                    spike_window_seen = True
                    assert sample_max == pytest.approx(1.0), (
                        f"window starting at idx={idx} overlaps spike (cost[3]=1.0) "
                        f"but max_cost={sample_max}"
                    )
                else:
                    assert sample_max == pytest.approx(0.0)
                break
    assert spike_window_seen, "200 samples and no window overlapping idx=3 — RNG broken?"


def test_legacy_episode_without_cost_field_returns_zero(tmp_path):
    """Pre-P3.0c shards on disk must still load — emit zero cost / max_cost."""
    eps_len = 6
    cost = np.zeros(eps_len + 1, dtype=np.float32)
    reward = np.ones(eps_len + 1, dtype=np.float32)
    ep = _make_episode(cost_pattern=cost, reward_pattern=reward)
    # Strip the cost key to simulate a legacy shard
    ep.pop("cost")
    _save(tmp_path, ep)

    buf = ReplayBuffer(
        replay_dir=tmp_path,
        max_size=1000,
        num_workers=1,
        nstep=3,
        discount=0.99,
        action_sequence=1,
        frame_stack=1,
        fetch_every=1,
        save_snapshot=True,
    )

    sample = _drain_one_sample(buf)
    assert len(sample) == 10, "tuple shape must remain stable for legacy shards"
    assert float(sample[8].reshape(-1)[0]) == 0.0
    assert float(sample[9].reshape(-1)[0]) == 0.0


def test_sample_tuple_has_ten_elements(tmp_path):
    eps_len = 5
    cost = np.full(eps_len + 1, 0.2, dtype=np.float32)
    reward = np.zeros(eps_len + 1, dtype=np.float32)
    ep = _make_episode(cost_pattern=cost, reward_pattern=reward)
    _save(tmp_path, ep)

    buf = ReplayBuffer(
        replay_dir=tmp_path,
        max_size=1000,
        num_workers=1,
        nstep=2,
        discount=0.99,
        action_sequence=1,
        frame_stack=1,
        fetch_every=1,
        save_snapshot=True,
    )
    sample = _drain_one_sample(buf)
    # rgb_obs, low_dim_obs, action, reward, discount, next_rgb_obs,
    # next_low_dim_obs, demo, cost, max_cost
    assert len(sample) == 10
