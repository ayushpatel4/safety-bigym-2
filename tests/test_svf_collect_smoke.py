"""End-to-end smoke test for ``scripts/svf_collect_dataset.py``.

Runs the collector in ``--smoke`` mode in-process (skips when AMASS is unset,
following the convention from ``test_safety_env.py``) and asserts the shard +
manifest are written with the expected schema.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


pytestmark = pytest.mark.skipif(
    os.environ.get("AMASS_DATA_DIR") is None,
    reason="AMASS_DATA_DIR not set",
)


def _import_script():
    import importlib

    return importlib.import_module("svf_collect_dataset")


def test_smoke_writes_shard_and_manifest(tmp_path):
    mod = _import_script()
    plan = mod.CollectionPlan.smoke(tmp_path)
    out_dir = mod.run_collection(plan)

    assert out_dir == tmp_path
    shards = sorted(p.name for p in tmp_path.glob("*.npz"))
    assert shards, f"No shards written to {tmp_path}"
    assert (tmp_path / "manifest.json").exists()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total_transitions"] > 0
    assert manifest["spec"]["action_dim"] == 15  # H1 with floating-base
    # Critic input should be 1-D non-pixel only — check no pixel-flavoured key
    obs_keys = manifest["spec"]["obs_keys"]
    assert all("rgb" not in k and "pixel" not in k and "cam" not in k for k in obs_keys)
    assert "human_pos_estimate" in obs_keys


def test_smoke_shard_schema(tmp_path):
    mod = _import_script()
    plan = mod.CollectionPlan.smoke(tmp_path)
    mod.run_collection(plan)

    shard_path = next(tmp_path.glob("*.npz"))
    with np.load(shard_path) as data:
        assert "action" in data.files
        assert "r_safe" in data.files
        assert "done" in data.files
        assert "ssm_margin" in data.files
        assert "source" in data.files
        assert "task_id" in data.files
        n = data["action"].shape[0]
        assert n > 0
        assert data["action"].shape == (n, 15)
        assert data["r_safe"].shape == (n,)
        assert data["done"].dtype == np.bool_
        assert data["source"].dtype == np.uint8
        # All r_safe values must be 0 or 1 (binary label invariant)
        unique = set(np.unique(data["r_safe"]).tolist())
        assert unique <= {0.0, 1.0}


def test_smoke_dataset_loadable(tmp_path):
    """End-to-end: collected shards must round-trip through SafetyTransitionDataset."""
    mod = _import_script()
    plan = mod.CollectionPlan.smoke(tmp_path)
    mod.run_collection(plan)

    from safety_bigym.filters.dataset import SafetyTransitionDataset

    ds = SafetyTransitionDataset(tmp_path)
    assert len(ds) > 0
    sample = ds[0]
    assert "action" in sample
    assert sample["action"].shape == (15,)
    assert "human_pos_estimate" in sample["obs"]
    assert sample["obs"]["human_pos_estimate"].shape == (6,)


def test_snapshot_source_missing_path_raises_file_not_found(tmp_path):
    """Snapshot source requires --snapshot-path. Until Phase-0 retrain lands
    the path will typically be missing; the script must fail loudly."""
    mod = _import_script()
    plan = mod.CollectionPlan(
        sources=("snapshot",),
        tasks=("reach_target_single",),
        disruptions=("INCIDENTAL",),
        episodes_per_cell=1,
        max_steps=10,
        bodyslam_mode="oracle",
        output_dir=tmp_path,
        snapshot_path=None,
    )
    with pytest.raises(FileNotFoundError):
        mod.run_collection(plan)


def test_snapshot_source_nonexistent_path_raises_file_not_found(tmp_path):
    mod = _import_script()
    plan = mod.CollectionPlan(
        sources=("snapshot",),
        tasks=("reach_target_single",),
        disruptions=("INCIDENTAL",),
        episodes_per_cell=1,
        max_steps=10,
        bodyslam_mode="oracle",
        output_dir=tmp_path,
        snapshot_path=tmp_path / "nope.pt",
    )
    with pytest.raises(FileNotFoundError):
        mod.run_collection(plan)


def test_demo_source_writes_safe_transitions(tmp_path):
    """Demo source must produce r_safe=1 transitions on every step (demos are
    safe by construction; live safety physics is not run)."""
    mod = _import_script()
    plan = mod.CollectionPlan(
        sources=("demo",),
        tasks=("reach_target_single",),
        disruptions=(),  # demos don't iterate disruptions
        episodes_per_cell=0,  # ignored for demo source
        max_steps=0,         # ignored
        bodyslam_mode="oracle",
        output_dir=tmp_path,
        seed=0,
        demos_per_task=2,
    )
    mod.run_collection(plan)

    # Expect at least one shard
    shards = sorted(tmp_path.glob("demo__*.npz"))
    assert shards, f"No demo shards produced under {tmp_path}"

    # All transitions in those shards must be labelled safe.
    for shard in shards:
        with np.load(shard) as data:
            assert (data["r_safe"] == 1.0).all(), (
                f"{shard} contains a non-safe label; demos must be safe-by-construction"
            )
            # source code 0 = demo
            assert (data["source"] == 0).all()
            # last step done flag set
            assert data["done"][-1] is np.True_ or bool(data["done"][-1]) is True


def test_demo_source_human_pos_estimate_is_synthesised(tmp_path):
    """Demos have no live human; the wrapper must synthesise a non-zero
    ``human_pos_estimate`` so the channel is not a constant the critic ignores."""
    mod = _import_script()
    plan = mod.CollectionPlan(
        sources=("demo",),
        tasks=("reach_target_single",),
        disruptions=(),
        episodes_per_cell=0,
        max_steps=0,
        bodyslam_mode="oracle",
        output_dir=tmp_path,
        seed=0,
        demos_per_task=1,
    )
    mod.run_collection(plan)

    shard = next(tmp_path.glob("demo__*.npz"))
    with np.load(shard) as data:
        hpe = data["obs__human_pos_estimate"]
        assert hpe.shape[1] == 6
        # x or y coord should vary across the episode (AMASS playback)
        coord_var = hpe[:, :3].std(axis=0).max()
        assert coord_var > 1e-3, f"human_pos_estimate looks constant: std={coord_var}"
