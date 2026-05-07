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


def test_unknown_source_raises_not_implemented(tmp_path):
    mod = _import_script()
    plan = mod.CollectionPlan(
        sources=("snapshot",),
        tasks=("reach_target_single",),
        disruptions=("INCIDENTAL",),
        episodes_per_cell=1,
        max_steps=10,
        bodyslam_mode="oracle",
        output_dir=tmp_path,
    )
    with pytest.raises(NotImplementedError):
        mod.run_collection(plan)
