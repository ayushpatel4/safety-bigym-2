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


def test_snapshot_source_skips_when_dict_unset(tmp_path, monkeypatch):
    """If SNAPSHOTS[task] is None and no override is given, the snapshot
    source skips that task with a warning. Total transitions then = 0,
    so the run errors at the "0 transitions produced" guard."""
    from safety_bigym.filters import snapshots as snapmod

    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"reach_target_single": None})

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
    with pytest.raises(RuntimeError, match="0 transitions"):
        mod.run_collection(plan)


def test_snapshot_source_invalid_path_in_dict_raises(tmp_path, monkeypatch):
    """If SNAPSHOTS[task] points to a missing file (typo / stale path), the
    resolver raises FileNotFoundError — that's a config bug, not deliberate skip."""
    from safety_bigym.filters import snapshots as snapmod

    monkeypatch.setattr(
        snapmod, "SNAPSHOTS", {"reach_target_single": "/nope/missing.pt"}
    )

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
    with pytest.raises(FileNotFoundError):
        mod.run_collection(plan)


def test_snapshot_source_invalid_override_path_raises(tmp_path, monkeypatch):
    from safety_bigym.filters import snapshots as snapmod

    monkeypatch.setattr(snapmod, "SNAPSHOTS", {"reach_target_single": None})

    mod = _import_script()
    plan = mod.CollectionPlan(
        sources=("snapshot",),
        tasks=("reach_target_single",),
        disruptions=("INCIDENTAL",),
        episodes_per_cell=1,
        max_steps=10,
        bodyslam_mode="oracle",
        output_dir=tmp_path,
        snapshot_overrides={"reach_target_single": "/nope/missing.pt"},
    )
    with pytest.raises(FileNotFoundError):
        mod.run_collection(plan)


def test_parse_snapshot_overrides():
    mod = _import_script()
    parsed = mod._parse_snapshot_overrides([
        "reach_target_single=/path/a.pt",
        "dishwasher_close=/path/b.pt",
    ])
    assert parsed == {
        "reach_target_single": "/path/a.pt",
        "dishwasher_close": "/path/b.pt",
    }


def test_parse_snapshot_overrides_rejects_malformed():
    mod = _import_script()
    with pytest.raises(SystemExit):
        mod._parse_snapshot_overrides(["no_equals_sign"])
    with pytest.raises(SystemExit):
        mod._parse_snapshot_overrides(["=missing_task"])
    with pytest.raises(SystemExit):
        mod._parse_snapshot_overrides(["task="])


def test_build_live_env_with_cameras_emits_rgb_keys(tmp_path):
    """When cameras are configured, the bare env must emit rgb_<name> keys
    with HWC uint8 arrays — that's what the snapshot policy adapter consumes."""
    mod = _import_script()
    env = mod._build_live_env(
        task_key="reach_target_single",
        disruption="INCIDENTAL",
        mode="oracle",
        motion_clips=mod.DEFAULT_CLIPS,
        cameras=("head",),
        camera_resolution=(64, 64),
    )
    obs, _info = env.reset()
    assert "rgb_head" in obs, f"expected rgb_head in obs, got {sorted(obs.keys())}"
    rgb = obs["rgb_head"]
    assert rgb.shape == (64, 64, 3) or rgb.shape == (3, 64, 64), (
        f"unexpected rgb_head shape {rgb.shape}"
    )
    assert rgb.dtype == np.uint8


def test_build_live_env_without_cameras_emits_no_rgb_keys(tmp_path):
    """Default (no cameras) bare env must NOT emit rgb_* keys — random/demo
    sources avoid the render cost."""
    mod = _import_script()
    env = mod._build_live_env(
        task_key="reach_target_single",
        disruption="INCIDENTAL",
        mode="oracle",
        motion_clips=mod.DEFAULT_CLIPS,
    )
    obs, _info = env.reset()
    rgb_keys = [k for k in obs if k.startswith("rgb")]
    assert not rgb_keys, f"expected no rgb_* keys, got {rgb_keys}"


def test_peek_snapshot_cameras_missing_file_raises(tmp_path):
    mod = _import_script()
    with pytest.raises(FileNotFoundError):
        mod.peek_snapshot_cameras(tmp_path / "nope.pt")


def test_peek_snapshot_cameras_extracts_cfg_fields(tmp_path):
    """Synthesize a tiny payload with just the cfg fields peek_snapshot_cameras reads."""
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "pixels": True,
        "visual_observation_shape": [84, 84],
        "env": {"cameras": ["head", "right_wrist"]},
    })
    payload = {"cfg": cfg}
    snap_path = tmp_path / "synth.pt"
    torch.save(payload, snap_path)

    mod = _import_script()
    cameras, resolution = mod.peek_snapshot_cameras(snap_path)
    assert cameras == ("head", "right_wrist")
    assert resolution == (84, 84)


def test_peek_snapshot_cameras_returns_empty_for_no_pixel_snapshot(tmp_path):
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"pixels": False, "env": {"cameras": []}})
    payload = {"cfg": cfg}
    snap_path = tmp_path / "no_pixel.pt"
    torch.save(payload, snap_path)

    mod = _import_script()
    cameras, _ = mod.peek_snapshot_cameras(snap_path)
    assert cameras == ()


def test_peek_snapshot_bodyslam_mode_phase0(tmp_path):
    """Phase 0 ACT: cfg either lacks `env.bodyslam` or has mode=off."""
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"pixels": True, "env": {"cameras": ["head"]}})
    torch.save({"cfg": cfg}, tmp_path / "p0_a.pt")

    cfg2 = OmegaConf.create({
        "pixels": True,
        "env": {"cameras": ["head"], "bodyslam": {"mode": "off"}},
    })
    torch.save({"cfg": cfg2}, tmp_path / "p0_b.pt")

    mod = _import_script()
    assert mod.peek_snapshot_bodyslam_mode(tmp_path / "p0_a.pt") == "off"
    assert mod.peek_snapshot_bodyslam_mode(tmp_path / "p0_b.pt") == "off"


def test_peek_snapshot_bodyslam_mode_phase1_variants(tmp_path):
    import torch
    from omegaconf import OmegaConf

    for mode in ("oracle", "noisy"):
        cfg = OmegaConf.create({
            "pixels": True,
            "env": {"cameras": ["head"], "bodyslam": {"mode": mode}},
        })
        snap_path = tmp_path / f"p1_{mode}.pt"
        torch.save({"cfg": cfg}, snap_path)

    mod = _import_script()
    assert mod.peek_snapshot_bodyslam_mode(tmp_path / "p1_oracle.pt") == "oracle"
    assert mod.peek_snapshot_bodyslam_mode(tmp_path / "p1_noisy.pt") == "noisy"


def test_peek_snapshot_bodyslam_mode_rejects_unknown(tmp_path):
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "pixels": True,
        "env": {"cameras": ["head"], "bodyslam": {"mode": "garbage"}},
    })
    torch.save({"cfg": cfg}, tmp_path / "bad.pt")

    mod = _import_script()
    with pytest.raises(ValueError, match="bodyslam.mode"):
        mod.peek_snapshot_bodyslam_mode(tmp_path / "bad.pt")


def test_build_live_env_bodyslam_off_skips_wrapper(tmp_path):
    """mode='off' must NOT wrap with BodySLAMWrapper — obs has no human_pos_estimate."""
    mod = _import_script()
    env = mod._build_live_env(
        task_key="reach_target_single",
        disruption="INCIDENTAL",
        mode="off",
        motion_clips=mod.DEFAULT_CLIPS,
    )
    obs, _info = env.reset()
    assert "human_pos_estimate" not in obs, (
        "bodyslam mode=off must not add human_pos_estimate to obs"
    )


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
