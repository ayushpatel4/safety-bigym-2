"""Tests for filters/dataset.py — sharded SVF transition dataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from safety_bigym.filters.dataset import (
    SafetyTransitionDataset,
    TransitionShardWriter,
    make_oversampler,
)
from safety_bigym.filters.feature_extractor import CriticFeatureSpec


def _make_spec(obs_dim: int = 6, action_dim: int = 4) -> CriticFeatureSpec:
    return CriticFeatureSpec(
        obs_keys=("low_dim_state",),
        obs_dims=(obs_dim,),
        action_dim=action_dim,
    )


def _write_shard(
    tmp_path: Path,
    spec: CriticFeatureSpec,
    n: int,
    *,
    n_violations: int,
    source: int = 0,
    task_id: int = 0,
    name: str = "shard_0001",
) -> Path:
    rng = np.random.default_rng(0)
    obs = {"low_dim_state": rng.standard_normal((n, spec.obs_dims[0])).astype(np.float32)}
    next_obs = {"low_dim_state": rng.standard_normal((n, spec.obs_dims[0])).astype(np.float32)}
    action = rng.standard_normal((n, spec.action_dim)).astype(np.float32)
    r_safe = np.ones(n, dtype=np.float32)
    r_safe[:n_violations] = 0.0
    done = np.zeros(n, dtype=np.bool_)
    done[:n_violations] = True
    ssm_margin = rng.standard_normal(n).astype(np.float32)

    writer = TransitionShardWriter(spec, tmp_path)
    writer.write_shard(
        name=name,
        obs=obs,
        action=action,
        next_obs=next_obs,
        r_safe=r_safe,
        done=done,
        ssm_margin=ssm_margin,
        source=np.full(n, source, dtype=np.uint8),
        task_id=np.full(n, task_id, dtype=np.uint8),
    )
    return tmp_path / f"{name}.npz"


def test_write_then_load_round_trip(tmp_path):
    spec = _make_spec()
    _write_shard(tmp_path, spec, n=20, n_violations=2)

    ds = SafetyTransitionDataset(tmp_path)
    assert len(ds) == 20

    sample = ds[0]
    assert set(sample) >= {
        "obs", "action", "next_obs", "r_safe", "done", "ssm_margin",
        "source", "task_id",
    }
    assert sample["obs"]["low_dim_state"].shape == (6,)
    assert sample["action"].shape == (4,)
    assert sample["next_obs"]["low_dim_state"].shape == (6,)
    # r_safe / done are scalars at sample level
    assert np.asarray(sample["r_safe"]).shape == ()
    assert sample["done"].shape == () if hasattr(sample["done"], "shape") else True


def test_manifest_written(tmp_path):
    spec = _make_spec()
    _write_shard(tmp_path, spec, n=10, n_violations=1, name="a")
    _write_shard(tmp_path, spec, n=15, n_violations=3, name="b")

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    assert manifest["total_transitions"] == 25
    assert sorted(manifest["shards"]) == ["a.npz", "b.npz"]
    assert manifest["spec"]["obs_keys"] == ["low_dim_state"]
    assert manifest["spec"]["action_dim"] == 4
    # Per-source violation rate present
    assert "violation_rate_total" in manifest
    assert pytest.approx(manifest["violation_rate_total"], rel=0.01) == 4 / 25


def test_dataset_concatenates_multiple_shards(tmp_path):
    spec = _make_spec()
    _write_shard(tmp_path, spec, n=10, n_violations=1, name="s1")
    _write_shard(tmp_path, spec, n=12, n_violations=2, name="s2")

    ds = SafetyTransitionDataset(tmp_path)
    assert len(ds) == 22


def test_empty_directory_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        SafetyTransitionDataset(tmp_path)


def test_oversampler_hits_target_violation_rate(tmp_path):
    spec = _make_spec()
    # 100 transitions, 5 violations (5% raw rate)
    _write_shard(tmp_path, spec, n=100, n_violations=5)
    ds = SafetyTransitionDataset(tmp_path)

    sampler = make_oversampler(ds, target_violation_rate=0.3, seed=0)
    drawn = list(iter(sampler))
    # Sampler emits indices; check that the drawn r_safe distribution sits
    # close to the target rate (Monte Carlo, generous tolerance)
    n_violations_drawn = sum(1 for i in drawn if ds[i]["r_safe"] == 0.0)
    rate = n_violations_drawn / len(drawn)
    assert 0.20 < rate < 0.40, f"Got {rate:.3f}, expected ~0.30"


def test_oversampler_handles_no_violations_gracefully(tmp_path):
    """If a shard somehow has zero violations, the sampler must not blow up
    (uniform fallback)."""
    spec = _make_spec()
    _write_shard(tmp_path, spec, n=20, n_violations=0)
    ds = SafetyTransitionDataset(tmp_path)
    sampler = make_oversampler(ds, target_violation_rate=0.3, seed=0)
    drawn = list(iter(sampler))
    assert len(drawn) == len(ds)


def test_torch_dataloader_compat(tmp_path):
    spec = _make_spec()
    _write_shard(tmp_path, spec, n=16, n_violations=4)
    ds = SafetyTransitionDataset(tmp_path)

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["obs"]["low_dim_state"].shape == (4, 6)
    assert batch["action"].shape == (4, 4)
    assert batch["r_safe"].shape == (4,)
    # default_collate gives torch tensors
    assert isinstance(batch["action"], torch.Tensor)


def test_violation_count_matches_label(tmp_path):
    """The dataset's bookkeeping (violation indices) must match the ground-truth
    r_safe == 0 count loaded from disk."""
    spec = _make_spec()
    _write_shard(tmp_path, spec, n=50, n_violations=7)
    ds = SafetyTransitionDataset(tmp_path)

    n_zeros = sum(1 for i in range(len(ds)) if ds[i]["r_safe"] == 0.0)
    assert n_zeros == 7
    assert len(ds.violation_indices) == 7
    assert len(ds.safe_indices) == 50 - 7
