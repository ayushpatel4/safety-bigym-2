"""Tests for the P6 benchmark harness (safety_bigym/benchmark/*).

These cover the PURE pieces (stats, records, schema, aggregate) with no snapshot, no
model load, and no env construction — they run in well under a second each. The filter
attachment + apply_veto tests touch the real SafetyFilterWrapper with a stub critic
(still no MuJoCo). Mirrors the stub conventions in tests/test_threshold_sweep.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from safety_bigym.benchmark.stats import bootstrap_ci, cvar, percentile
from safety_bigym.benchmark.records import EpisodeRecord, read_parquet, write_parquet
from safety_bigym.benchmark.schema import COLUMNS, FILTER_COLUMNS, assemble_row
from safety_bigym.benchmark.aggregate import aggregate_cell


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _ep_safety(prox=0.1, ssm=0.2, ssm_actual=0.05, min_sep=0.6, steps=50):
    """A representative info["episode_safety"] dict (subset that aggregate reads)."""
    return {
        "ep_steps": steps,
        "ep_ssm_violation_rate": ssm,
        "ep_ssm_violation_actual_rate": ssm_actual,
        "ep_proximity_violation_rate": prox,
        "ep_pfl_violation_rate": 0.0,
        "ep_min_ssm_margin": -0.1,
        "ep_min_ssm_margin_actual": 0.2,
        "ep_min_separation": min_sep,
        "ep_mean_separation": min_sep + 0.3,
        "ep_p5_separation": min_sep + 0.05,
        "ep_p25_separation": min_sep + 0.15,
        "ep_max_pfl_force_ratio": 0.0,
        "ep_max_contact_force": 0.0,
        "ep_max_robot_vel": 1.2,
        "ep_mean_robot_vel": 0.4,
        "ep_time_to_first_violation": 7,
        "ep_time_in_proximity_0p3m": 0.0,
        "ep_time_in_proximity_0p5m": prox,
        "ep_time_in_proximity_1p0m": 0.5,
    }


def _record(i, *, seed=0, success=True, reward=1.0, min_sep=0.6, prox=0.1, cost=2.5,
            filtered=False, n_interventions=0, filter_steps=0, sum_q=0.0):
    return EpisodeRecord(
        seed=seed,
        episode_index=i,
        success=success,
        episode_reward=reward,
        n_steps=50,
        steps_to_completion=(20.0 if success else float("nan")),
        ep_safety=_ep_safety(prox=prox, min_sep=min_sep),
        ep_cost_integral=cost,
        filtered=filtered,
        n_interventions=n_interventions,
        filter_steps=filter_steps,
        sum_q_value=sum_q,
    )


def _dicts_equal(a, b):
    assert set(a.keys()) == set(b.keys())
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, float) and isinstance(bv, float) and math.isnan(av) and math.isnan(bv):
            continue
        if isinstance(av, float) or isinstance(bv, float):
            assert av == pytest.approx(bv), k
        else:
            assert av == bv, k


# --------------------------------------------------------------------------------------
# Test 1 — bootstrap CI matches a manual numpy reimplementation at a fixed seed
# --------------------------------------------------------------------------------------

def test_bootstrap_ci_matches_manual():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    point, lo, hi = bootstrap_ci(samples, agg=np.mean, n_resamples=1000, alpha=0.05, seed=0)

    arr = np.asarray(samples, dtype=float)
    n = arr.size
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(1000, n))
    boot = arr[idx].mean(axis=1)

    assert point == pytest.approx(arr.mean())
    assert lo == pytest.approx(np.percentile(boot, 2.5))
    assert hi == pytest.approx(np.percentile(boot, 97.5))
    assert lo <= point <= hi


def test_bootstrap_ci_degenerate():
    # Empty -> all nan.
    assert all(math.isnan(x) for x in bootstrap_ci([], seed=0))
    # Single sample -> degenerate point CI (no spread).
    p, lo, hi = bootstrap_ci([4.2], seed=0)
    assert p == lo == hi == pytest.approx(4.2)


# --------------------------------------------------------------------------------------
# Test 2 — CVaR + percentile against hand-computed constants
# --------------------------------------------------------------------------------------

def test_cvar_and_percentile_hand_computed():
    a = list(range(1, 101))  # 1..100
    # upper tail: q-quantile(0.95) = 95.05 → mean of {96,97,98,99,100} = 98
    assert cvar(a, q=0.95, tail="upper") == pytest.approx(98.0)
    # lower tail: (1-q)-quantile(0.05) = 5.95 → mean of {1,2,3,4,5} = 3
    assert cvar(a, q=0.95, tail="lower") == pytest.approx(3.0)
    # 1st percentile = 1 + 0.01*99 = 1.99
    assert percentile(a, 1.0) == pytest.approx(1.99)
    assert math.isnan(cvar([], q=0.95))
    assert math.isnan(percentile([], 50))
    with pytest.raises(ValueError):
        cvar(a, tail="sideways")


# --------------------------------------------------------------------------------------
# Test 4 — parquet roundtrip reproduces the aggregated CSV row
# --------------------------------------------------------------------------------------

def test_parquet_roundtrip_reaggregation(tmp_path):
    records = [
        _record(0, success=True, reward=1.0, min_sep=0.55, prox=0.12, cost=3.0),
        _record(1, success=False, reward=0.0, min_sep=0.40, prox=0.30, cost=5.0),
        _record(2, success=True, reward=1.0, min_sep=0.70, prox=0.05, cost=1.5),
    ]
    agg_inmem = aggregate_cell(records, stats_seed=7, n_resamples=500)

    path = tmp_path / "raw_episodes.parquet"
    write_parquet(path, records)
    reloaded = read_parquet(path)
    agg_reloaded = aggregate_cell(reloaded, stats_seed=7, n_resamples=500)

    _dicts_equal(agg_inmem, agg_reloaded)


# --------------------------------------------------------------------------------------
# Test 5 — assembled row spans exactly COLUMNS; filter cols empty iff no filter
# --------------------------------------------------------------------------------------

def _identification_meta():
    return {
        "task": "saucepan_to_hob",
        "disruption": "coworker_train",
        "obs_mode": "noisy",
        "human_model": "g1",
        "policy_kind": "random",
        "snapshot": "",
        "filter_snapshot": "",
        "filter_threshold": "",
        "seeds": "0",
        "episodes_per_seed": 3,
        "git_sha": "deadbeef",
        "timestamp_utc": "2026-05-29T00:00:00Z",
    }


def test_csv_schema_completeness():
    records = [_record(i, success=(i % 2 == 0)) for i in range(4)]

    # No filter → filter columns must be "".
    agg = aggregate_cell(records, stats_seed=1, n_resamples=200)
    row = assemble_row({**_identification_meta(), **agg}, filtered=False)
    assert set(row.keys()) == set(COLUMNS)
    assert list(row.keys()) == COLUMNS  # order preserved
    for col in FILTER_COLUMNS:
        assert row[col] == ""
    # Every column aggregate_cell computes must be populated (not the "" sentinel).
    # Identification fields that depend on optional CLI inputs may legitimately be "".
    optional_empty = set(FILTER_COLUMNS) | {"snapshot", "filter_snapshot", "filter_threshold"}
    for col in COLUMNS:
        if col not in optional_empty:
            assert row[col] != "", col

    # With filter → filter columns populated.
    frecords = [
        _record(i, filtered=True, n_interventions=2, filter_steps=50, sum_q=120.0)
        for i in range(4)
    ]
    fagg = aggregate_cell(frecords, filter_meta={"fallback": "zero_velocity"},
                          stats_seed=1, n_resamples=200)
    frow = assemble_row({**_identification_meta(), **fagg}, filtered=True)
    assert set(frow.keys()) == set(COLUMNS)
    assert frow["filter_intervention_rate"] == pytest.approx(2 * 4 / (50 * 4))
    assert frow["filter_passthrough_rate"] == pytest.approx(1 - 2 / 50)
    assert frow["filter_fallback"] == "zero_velocity"
    assert frow["n_interventions"] == 8


# --------------------------------------------------------------------------------------
# Test 3 — real SafetyFilterWrapper attaches, intervenes, and populates filter columns
# --------------------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402


class _SafetyStubEnv(gym.Env):
    """Pure gym env (no MuJoCo) emitting info["safety"] EpisodeSafetyMetrics can read."""

    def __init__(self, episode_length: int = 10):
        self.observation_space = spaces.Dict(
            {"low_dim_state": spaces.Box(-1, 1, (4,), np.float32)}
        )
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.episode_length = episode_length
        self._step = 0

    def _safety(self):
        return {
            "ssm_violation": False,
            "ssm_violation_actual": False,
            "proximity_violation": (self._step % 3 == 0),
            "min_separation": 0.6,
            "ssm_margin": 0.1,
            "ssm_margin_actual": 0.3,
            "robot_vel": 0.5,
        }

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return {"low_dim_state": np.zeros(4, np.float32)}, {
            "safety": self._safety(), "task_success": 0.0
        }

    def step(self, action):
        self._step += 1
        done = self._step >= self.episode_length
        return (
            {"low_dim_state": np.zeros(4, np.float32)},
            0.0,
            False,
            done,
            {"safety": self._safety(), "task_success": 0.0},
        )


def _stub_critic(q: float):
    import torch
    from safety_bigym.filters.critic import SafetyCritic
    from safety_bigym.filters.feature_extractor import CriticFeatureSpec

    class _StubCritic(SafetyCritic):
        def __init__(self):
            super().__init__(
                spec=CriticFeatureSpec(obs_keys=("low_dim_state",), obs_dims=(4,), action_dim=2),
                gamma=0.99,
            )

        def forward(self, features):
            if features.ndim == 1:
                return torch.tensor(q)
            return torch.full((features.shape[0],), float(q))

    return _StubCritic()


def test_filter_attaches_and_emits_columns():
    from safety_bigym.safety.episode_metrics_wrapper import EpisodeSafetyMetrics
    from safety_bigym.benchmark.filter_attach import attach_filter_gym
    from safety_bigym.benchmark.runners import GymRunner, run_episode, make_random_policy

    env = EpisodeSafetyMetrics(_SafetyStubEnv(episode_length=10))
    critic = _stub_critic(q=1.0)  # always below R → always intervene
    wrapped = attach_filter_gym(env, critic=critic, threshold_R=50.0)
    runner = GymRunner(wrapped, make_random_policy(wrapped.action_space, seed=0))

    rec = run_episode(runner, seed=0, episode_index=0, max_steps=10, filtered=True)
    assert rec.filtered
    assert rec.n_interventions == rec.filter_steps == 10
    # every step must carry the filter info dict
    # (verified indirectly: sum_q accumulated and interventions counted)

    agg = aggregate_cell([rec], filter_meta={"fallback": "zero_velocity"},
                         stats_seed=0, n_resamples=100)
    row = assemble_row({**_identification_meta(), **agg}, filtered=True)
    assert row["filter_intervention_rate"] == pytest.approx(1.0)
    assert row["filter_passthrough_rate"] == pytest.approx(0.0)
    assert row["filter_fallback"] == "zero_velocity"
    assert row["mean_q_value"] == pytest.approx(1.0)


def test_apply_veto_pure():
    from safety_bigym.filters.fallback import ZeroVelocityFallback
    from safety_bigym.benchmark.runners import apply_veto

    obs = {"low_dim_state": np.zeros(4, np.float32)}
    sub = np.array([0.5, -0.5], np.float32)
    ident = lambda a: a  # noqa: E731
    fb = ZeroVelocityFallback(gym.spaces.Box(-1, 1, (2,), np.float32))

    executed, info = apply_veto(
        critic=_stub_critic(q=1.0), fallback=fb, obs=obs, sub_action=sub,
        to_raw=ident, from_raw=ident, threshold_R=50.0,
    )
    assert info["intervened"] is True
    assert np.allclose(executed, 0.0)  # zero-velocity fallback

    executed2, info2 = apply_veto(
        critic=_stub_critic(q=99.0), fallback=fb, obs=obs, sub_action=sub,
        to_raw=ident, from_raw=ident, threshold_R=50.0,
    )
    assert info2["intervened"] is False
    assert np.allclose(executed2, sub)


# --------------------------------------------------------------------------------------
# Loader dispatch (pure — on payload key sets, no real snapshot)
# --------------------------------------------------------------------------------------

def test_detect_kind():
    from safety_bigym.benchmark.loader import detect_kind, load_policy, PolicyMeta

    assert detect_kind({"agent_state", "config", "step", "episode"}) == "cqn_as"
    assert detect_kind({"agent", "cfg", "actor_ema", "action_stats"}) == "act"
    with pytest.raises(ValueError):
        detect_kind({"weights", "optimizer"})

    # random path returns (meta, payload) with no snapshot load
    meta, payload = load_policy(None)
    assert isinstance(meta, PolicyMeta) and meta.kind == "random"
    assert payload is None
