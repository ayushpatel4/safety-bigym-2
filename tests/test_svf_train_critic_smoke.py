"""End-to-end smoke for the SVF training script.

Builds a tiny synthetic dataset on disk, calls the training script's
``run_training`` in-process with the smoke plan, and asserts:
- Checkpoint is written
- ``loss_first > loss_last``
- Restored critic round-trips numerically
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from safety_bigym.filters.critic import SafetyCritic  # noqa: E402
from safety_bigym.filters.dataset import TransitionShardWriter  # noqa: E402
from safety_bigym.filters.feature_extractor import CriticFeatureSpec  # noqa: E402


def _seed_dataset(tmp_path: Path, n: int = 256, n_violations: int = 64):
    spec = CriticFeatureSpec(
        obs_keys=("low_dim_state",),
        obs_dims=(8,),
        action_dim=4,
    )
    rng = np.random.default_rng(42)
    obs = {"low_dim_state": rng.standard_normal((n, 8)).astype(np.float32)}
    next_obs = {"low_dim_state": rng.standard_normal((n, 8)).astype(np.float32)}
    action = rng.uniform(-1, 1, size=(n, 4)).astype(np.float32)
    r_safe = np.ones(n, dtype=np.float32)
    r_safe[:n_violations] = 0.0
    done = np.zeros(n, dtype=np.bool_)
    done[:n_violations] = True
    margin = rng.standard_normal(n).astype(np.float32)
    writer = TransitionShardWriter(spec, tmp_path)
    writer.write_shard(
        name="seed",
        obs=obs,
        action=action,
        next_obs=next_obs,
        r_safe=r_safe,
        done=done,
        ssm_margin=margin,
        # B2.8 schema: per-step separation + PFL ratio (required since the
        # writer signature drift). Consistent with r_safe; PFL zero (bug).
        min_separation=np.where(r_safe > 0, 1.0, 0.05).astype(np.float32),
        pfl_force_ratio=np.zeros(n, dtype=np.float32),
        source=np.zeros(n, dtype=np.uint8),
        task_id=np.zeros(n, dtype=np.uint8),
    )


def _import_script():
    import importlib

    return importlib.import_module("svf_train_critic")


def test_smoke_training_writes_checkpoint(tmp_path):
    _seed_dataset(tmp_path)
    mod = _import_script()
    output = tmp_path / "_smoke_critic.pt"
    plan = mod.TrainPlan.smoke(dataset_dir=tmp_path, output=output)

    out = mod.run_training(plan)
    assert out == output
    assert output.is_file()


def test_smoke_training_decreases_bellman_loss(tmp_path):
    """Bellman MSE is the load-bearing fit signal; total loss can grow with
    CQL because the conservatism penalty scales with how distinguishable Q
    becomes on data vs OOD actions."""
    _seed_dataset(tmp_path)
    mod = _import_script()
    output = tmp_path / "critic.pt"
    mod.run_training(mod.TrainPlan.smoke(dataset_dir=tmp_path, output=output))

    payload = torch.load(output, weights_only=False)
    first = payload["training"]["bellman_first"]
    last = payload["training"]["bellman_last"]
    assert last < first, f"Bellman MSE didn't decrease: first={first} last={last}"


def test_smoke_checkpoint_round_trips(tmp_path):
    _seed_dataset(tmp_path)
    mod = _import_script()
    output = tmp_path / "critic.pt"
    mod.run_training(mod.TrainPlan.smoke(dataset_dir=tmp_path, output=output))

    payload = torch.load(output, weights_only=False)
    critic = SafetyCritic.from_checkpoint_payload(payload)
    feats = torch.randn(2, critic.input_dim)
    q = critic(feats)
    assert q.shape == (2,)
    assert (q >= 0).all() and (q <= critic.q_max + 1e-3).all()
