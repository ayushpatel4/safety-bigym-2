"""Per-step / per-episode record dataclasses + JSONL & parquet persistence.

The benchmark harness records one :class:`EpisodeRecord` per rolled-out episode. These
are streamed to a ``.jsonl`` sidecar **as they complete** (crash-resilient, matches the
``train_cqn_as.py metrics.jsonl`` convention) and, at cell completion, consolidated into
``raw_episodes.parquet`` — the canonical raw-roll artifact the report refers to.

Re-aggregating the per-cell CSV from ``raw_episodes.parquet`` (without re-rolling out) is
the whole point of persisting raw rolls: threshold/percentile re-cuts are then free.

Depends only on numpy / pandas / stdlib — no torch / mujoco — so the persistence tests
run in milliseconds.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

__all__ = ["StepRecord", "EpisodeRecord", "write_jsonl_line", "write_parquet", "read_parquet"]


@dataclass
class StepRecord:
    """One environment step, uniform across the random / ACT / CQN-AS runner paths."""

    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    # Cached out of info["safety"] for the cross-episode tail-risk columns.
    min_separation: float = float("inf")
    c_t: float = 0.0

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)


# Scalar fields stored directly on EpisodeRecord (everything else in a row is a
# wrapper-emitted ``ep_*`` safety field). Used to split a flat row back into the
# record on read. ``ep_cost_integral`` is harness-computed (NOT from the wrapper),
# so it is listed here explicitly to keep it out of ``ep_safety``.
_SCALAR_FIELDS = (
    "seed",
    "episode_index",
    "success",
    "task_success_raw",
    "episode_reward",
    "n_steps",
    "steps_to_completion",
    "ep_cost_integral",
    "filtered",
    "n_interventions",
    "filter_steps",
    "sum_q_value",
)


@dataclass
class EpisodeRecord:
    """Aggregate of one episode — the row persisted to JSONL / parquet."""

    seed: int
    episode_index: int
    # Task
    success: bool
    episode_reward: float
    n_steps: int
    steps_to_completion: float  # env-steps to first success; nan if unsuccessful
    # The full info["episode_safety"] dict at the terminal step (all ep_* fields).
    ep_safety: Dict[str, float] = field(default_factory=dict)
    # Harness-computed per-episode cost integral (Σ_t c_t).
    ep_cost_integral: float = 0.0
    # Filter mechanics (per episode); filter_steps == n_steps when filtered.
    filtered: bool = False
    n_interventions: int = 0
    filter_steps: int = 0
    sum_q_value: float = 0.0
    # Raw env-reported success flag when present (cross-check; may be None).
    task_success_raw: float | None = None

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten to a single dict (scalar fields + ep_safety) for a parquet/JSONL row."""
        row: Dict[str, Any] = {
            "seed": int(self.seed),
            "episode_index": int(self.episode_index),
            "success": bool(self.success),
            "task_success_raw": self.task_success_raw,
            "episode_reward": float(self.episode_reward),
            "n_steps": int(self.n_steps),
            "steps_to_completion": float(self.steps_to_completion),
            "ep_cost_integral": float(self.ep_cost_integral),
            "filtered": bool(self.filtered),
            "n_interventions": int(self.n_interventions),
            "filter_steps": int(self.filter_steps),
            "sum_q_value": float(self.sum_q_value),
        }
        # ep_safety keys all start with "ep_" and never collide with the scalars above.
        for k, v in self.ep_safety.items():
            row[k] = float(v) if isinstance(v, (int, float, bool)) else v
        return row

    @classmethod
    def from_flat_dict(cls, d: Dict[str, Any]) -> "EpisodeRecord":
        ep_safety = {k: v for k, v in d.items() if k not in _SCALAR_FIELDS}
        raw = d.get("task_success_raw", None)
        return cls(
            seed=int(d["seed"]),
            episode_index=int(d["episode_index"]),
            success=bool(d["success"]),
            episode_reward=float(d["episode_reward"]),
            n_steps=int(d["n_steps"]),
            steps_to_completion=float(d.get("steps_to_completion", float("nan"))),
            ep_safety={k: v for k, v in ep_safety.items()},
            ep_cost_integral=float(d.get("ep_cost_integral", 0.0)),
            filtered=bool(d.get("filtered", False)),
            n_interventions=int(d.get("n_interventions", 0)),
            filter_steps=int(d.get("filter_steps", 0)),
            sum_q_value=float(d.get("sum_q_value", 0.0)),
            task_success_raw=(None if raw is None or (isinstance(raw, float) and math.isnan(raw)) else float(raw)),
        )


def _jsonl_safe(v: Any) -> Any:
    """nan/inf -> None (portable JSON, matching train_cqn_as.metrics.jsonl)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        fv = float(v)
        return fv if math.isfinite(fv) else None
    return v


def write_jsonl_line(path: Path, record: EpisodeRecord) -> None:
    """Append one episode as a JSON line (live, crash-resilient sidecar)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: _jsonl_safe(v) for k, v in record.to_flat_dict().items()}
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_parquet(path: Path, records: List[EpisodeRecord]) -> None:
    """Write all episode records to parquet atomically (temp + os.replace).

    Empty ``records`` writes nothing (no schema to infer) and returns silently.
    """
    if not records:
        return
    import pandas as pd  # local import keeps module import light for non-IO tests

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_flat_dict() for r in records])
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet(path: Path) -> List[EpisodeRecord]:
    """Read a raw-rolls parquet back into EpisodeRecords (for re-aggregation)."""
    import pandas as pd

    df = pd.read_parquet(path)
    return [EpisodeRecord.from_flat_dict(row) for row in df.to_dict("records")]
