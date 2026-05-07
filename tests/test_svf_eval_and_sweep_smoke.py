"""End-to-end smoke tests for svf_eval_filter.py and svf_threshold_sweep.py.

Both scripts depend on a live ``SafetyBiGymEnv`` so they skip cleanly when
``AMASS_DATA_DIR`` is unset.

The fixtures stage a tiny synthetic dataset, run the training script in
smoke mode to produce ``_smoke_critic.pt``, then exercise the eval and sweep
scripts against it. This is the closest local approximation to the GPU
pipeline.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


pytestmark = pytest.mark.skipif(
    os.environ.get("AMASS_DATA_DIR") is None,
    reason="AMASS_DATA_DIR not set",
)


def _seed_smoke_critic(tmp_path: Path) -> Path:
    """Run svf_collect_dataset --smoke + svf_train_critic --smoke."""
    import importlib

    collect = importlib.import_module("svf_collect_dataset")
    train = importlib.import_module("svf_train_critic")

    dataset_dir = tmp_path / "ds"
    plan = collect.CollectionPlan.smoke(dataset_dir)
    collect.run_collection(plan)

    critic_path = tmp_path / "critic.pt"
    train.run_training(train.TrainPlan.smoke(dataset_dir, critic_path))
    return critic_path


def test_eval_filter_smoke_emits_metrics(tmp_path):
    critic_path = _seed_smoke_critic(tmp_path)

    import importlib

    mod = importlib.import_module("svf_eval_filter")
    args = mod.parse_args([
        "--smoke",
        "--critic-path", str(critic_path),
        "--output-csv", str(tmp_path / "eval.csv"),
    ])
    args.tasks = ("reach_target_single",)
    args.disruptions = ("INCIDENTAL",)
    args.episodes_per_cell = 1
    args.max_steps = 30
    args.policy = "random"
    args.bodyslam_mode = "oracle"
    rows = mod.run_eval(args)
    assert len(rows) >= 1
    assert 0.0 <= rows[0].intervention_rate <= 1.0
    assert 0.0 <= rows[0].residual_violation_rate <= 1.0


def test_threshold_sweep_smoke_writes_csv(tmp_path):
    critic_path = _seed_smoke_critic(tmp_path)

    import importlib

    mod = importlib.import_module("svf_threshold_sweep")
    csv_path = tmp_path / "sweep.csv"
    args = mod.parse_args([
        "--smoke",
        "--critic-path", str(critic_path),
        "--output-csv", str(csv_path),
    ])
    mod.apply_smoke_overrides(args)
    rows = mod.run_sweep(args)
    mod.write_csv(rows, csv_path)
    assert csv_path.exists()

    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows_csv = list(reader)
    assert len(rows_csv) == 2  # smoke uses 2 thresholds
    # CSV must surface the Pareto-relevant columns
    assert {"threshold_R", "intervention_rate", "residual_violation_rate"}.issubset(
        rows_csv[0].keys()
    )
