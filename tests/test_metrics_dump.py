"""Tests for the streaming metrics.jsonl writer + final_metrics.json summary
in train_cqn_as.Workspace.

These are unit-level: they exercise _append_metrics_jsonl + _write_final_summary
directly on a stub Workspace, no MuJoCo / RoboBase needed.
"""
import json
import tempfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def stub_workspace(tmp_path):
    # Late import: train_cqn_as pulls in MuJoCo. Pytest-discovery does this
    # cleanly; running the script standalone with MUJOCO_GL=egl set on
    # darwin fails (egl is Linux-only), so we lean on pytest's env.
    from train_cqn_as import Workspace

    ws = Workspace.__new__(Workspace)
    ws.work_dir = tmp_path
    ws._wandb_run = None
    ws._global_step = 100
    ws._global_episode = 5
    ws.cfg = OmegaConf.create(
        {
            "env": {"task_name": "saucepan_to_hob"},
            "disruption": "coworker_train",
            "num_train_frames": 2000,
            "num_demos": 36,
            "agent": {"v_min": -6.0, "v_max": 2.0},
            "wandb": {
                "name": "smoke",
                "tags": ["stage0", "method=unconstrained"],
            },
        }
    )
    return ws


def test_jsonl_writer_appends_one_row_per_call(stub_workspace):
    ws = stub_workspace
    ws._append_metrics_jsonl(10, "train", {"train/episode_reward": -7.2})
    ws._append_metrics_jsonl(
        20, "episode",
        {"episode/episode_safety/ep_proximity_violation_rate": 0.4},
    )

    rows = (ws.work_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2
    parsed = [json.loads(r) for r in rows]
    assert parsed[0]["step"] == 10 and parsed[0]["ty"] == "train"
    assert parsed[0]["train/episode_reward"] == pytest.approx(-7.2)
    assert parsed[1]["ty"] == "episode"
    assert parsed[1][
        "episode/episode_safety/ep_proximity_violation_rate"
    ] == pytest.approx(0.4)


def test_final_summary_picks_best_and_last(stub_workspace):
    ws = stub_workspace
    # Two eval rows: row 2 has higher success / reward, lower proximity rate.
    ws._append_metrics_jsonl(
        30, "eval",
        {
            "eval/success_rate": 0.2,
            "eval/episode_reward": -8.1,
            "eval/ep_proximity_violation_rate": 0.35,
            "eval/ep_min_separation": 0.42,
        },
    )
    ws._append_metrics_jsonl(
        40, "eval",
        {
            "eval/success_rate": 0.4,
            "eval/episode_reward": -5.5,
            "eval/ep_proximity_violation_rate": 0.28,
            "eval/ep_min_separation": 0.31,
        },
    )
    ws._append_metrics_jsonl(
        41, "train",
        {"train/episode_reward": -5.4, "train/episode_success": 1.0},
    )

    ws._write_final_summary()
    out = json.loads((ws.work_dir / "final_metrics.json").read_text())

    # Best (eval): max-prefer for reward/success; min-prefer for safety axes.
    assert out["best_eval"]["success_rate"] == pytest.approx(0.4)
    assert out["best_eval"]["episode_reward"] == pytest.approx(-5.5)
    assert out["best_eval"]["ep_proximity_violation_rate"] == pytest.approx(0.28)
    assert out["best_eval"]["ep_min_separation_lowest"] == pytest.approx(0.31)

    # Last train + last eval rows captured.
    assert out["last_train_episode"]["train/episode_reward"] == pytest.approx(-5.4)
    assert out["last_eval"]["eval/success_rate"] == pytest.approx(0.4)

    # Config block carries the headline launch args + W&B tags.
    assert out["config"]["task"] == "saucepan_to_hob"
    assert out["config"]["disruption"] == "coworker_train"
    assert "stage0" in out["config"]["wandb_tags"]


def test_summary_no_op_when_jsonl_missing(stub_workspace, caplog):
    ws = stub_workspace
    # No JSONL on disk → summary should silently no-op (not crash).
    ws._write_final_summary()
    assert not (ws.work_dir / "final_metrics.json").exists()


def test_jsonl_writer_coerces_unserializable_values(stub_workspace):
    """A stray non-JSON value (e.g. a numpy array, repr-coerced) should not
    kill the trace. _append_metrics_jsonl repr's it instead."""
    import numpy as np
    ws = stub_workspace
    ws._append_metrics_jsonl(
        5, "train", {"train/episode_reward": -7.2, "train/extra": np.array([1, 2])},
    )
    rows = (ws.work_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["train/episode_reward"] == pytest.approx(-7.2)
    # The array gets repr-coerced.
    assert isinstance(parsed["train/extra"], str)
