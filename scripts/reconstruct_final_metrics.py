#!/usr/bin/env python3
"""Reconstruct final_metrics.json for crashed / killed CQN-AS cells.

A run that is killed (OOM, pkill, disk-full) never reaches the end of
``train()`` and so never writes ``final_metrics.json`` (see
``train_cqn_as.py::_write_final_metrics``). The pool dispatcher
(``scripts/dispatch_p3p4_pool.sh::cell_done``) uses the *presence* of
``final_metrics.json`` as its only "this cell is finished" sentinel, so a
crashed cell looks un-done and gets relaunched. This script regenerates the
file *offline* from the artefacts that survive a crash, so a salvageable run
can be marked done without retraining:

  - ``metrics.jsonl``      one JSON row per ``_log()`` call (train / episode /
                           safety / eval). Carries everything the in-memory
                           ``_last_*`` / ``_best_*`` trackers would have held.
  - ``snapshot_best.pt``   the peak-success checkpoint. This script does NOT
                           pick or rename it — it must ALREADY be the correct
                           checkpoint on disk (curate it from W&B first).
  - ``.hydra/config.yaml`` the resolved run config, for the ``config`` block.

It mirrors ``_write_final_metrics()`` field-for-field, with two deliberate
differences for salvaged (partial) runs:

  1. ``config.num_train_frames`` is set to the frames ACTUALLY reached (the max
     step of the primary run), not the requested value, so the file does not
     silently claim a complete run.
  2. a top-level ``_reconstructed`` block records the provenance and the
     frames_reached-vs-requested gap, so the salvage is auditable.

``metrics.jsonl`` may concatenate several runs when crash-restarts wrote into
the same hydra dir (a killed run, then dispatcher retries). Runs are split on a
step reset; the PRIMARY run (highest max step) supplies every ``last_*`` /
``best_*`` field, so the file describes one coherent run rather than a mix.

Usage:
    python scripts/reconstruct_final_metrics.py DIR [DIR ...]
    python scripts/reconstruct_final_metrics.py --dry-run DIR   # preview only
    python scripts/reconstruct_final_metrics.py --force DIR     # overwrite existing
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

INF = float("inf")
# A step drop larger than this marks a crash-restart boundary in metrics.jsonl.
# Within a single run the step is monotonic non-decreasing; a restart resets it
# to ~0, so any threshold well above the eval cadence (2500) is safe.
RESET_DROP = 1000


def load_rows(jsonl: Path) -> list[dict]:
    """Parse metrics.jsonl defensively (skip blank / truncated tail lines)."""
    rows: list[dict] = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written final line after a hard kill — drop it
        if isinstance(r, dict) and "step" in r:
            rows.append(r)
    return rows


def split_runs(rows: list[dict]) -> list[list[dict]]:
    """Split rows into runs on a step reset (concatenated crash-restarts)."""
    if not rows:
        return []
    segments: list[list[dict]] = []
    current = [rows[0]]
    for prev, row in zip(rows, rows[1:]):
        if row["step"] < prev["step"] - RESET_DROP:
            segments.append(current)
            current = [row]
        else:
            current.append(row)
    segments.append(current)
    return segments


def _strip(row: dict) -> dict:
    """Drop the streaming-only keys so the block matches the in-memory dict
    that ``_log`` stored (``_last_* = dict(prefixed)``, no step / ty)."""
    return {k: v for k, v in row.items() if k not in ("step", "ty")}


def reconstruct_blocks(run: list[dict]) -> dict:
    """Replay one run's rows into the last_* / best_* blocks, exactly as the
    live trackers in train_cqn_as.py would have accumulated them."""
    last_train: dict = {}
    last_episode_safety: dict = {}
    last_eval: dict = {}
    evals: list[dict] = []
    for r in run:
        ty = r.get("ty")
        if ty == "train" and "train/episode_reward" in r:
            last_train = _strip(r)
        elif ty == "episode":
            last_episode_safety = _strip(r)
        elif ty == "eval":
            last_eval = _strip(r)
            evals.append(r)

    # best_eval: max-prefer reward/success, min-prefer violation rates, and the
    # lowest per-eval ep_min_separation. Keys are UNPREFIXED (mirrors
    # _update_best_eval, which runs on the raw eval_row before _log prefixes it).
    best = {
        "success_rate": -INF,
        "episode_reward": -INF,
        "ep_proximity_violation_rate": INF,
        "ep_ssm_violation_actual_rate": INF,
        "ep_min_separation_lowest": INF,
    }
    for e in evals:
        raw = {k[len("eval/"):]: v for k, v in e.items() if k.startswith("eval/")}
        for k in ("success_rate", "episode_reward"):
            v = raw.get(k)
            if isinstance(v, (int, float)) and float(v) > best[k]:
                best[k] = float(v)
        for k in ("ep_proximity_violation_rate", "ep_ssm_violation_actual_rate"):
            v = raw.get(k)
            if isinstance(v, (int, float)) and float(v) < best[k]:
                best[k] = float(v)
        v = raw.get("ep_min_separation")
        if isinstance(v, (int, float)) and float(v) < best["ep_min_separation_lowest"]:
            best["ep_min_separation_lowest"] = float(v)
    best_eval = {k: (None if v in (INF, -INF) else v) for k, v in best.items()}

    # best_snapshot: the single eval with the highest success_rate (episode_reward
    # tie-break), mirroring _mark_best_snapshot. step == that eval's logged step,
    # which is also the snapshot_<step>.pt that _finalize_best_snapshot copied.
    b_sr, b_er, b_step = -INF, -INF, None
    for e in evals:
        sr = e.get("eval/success_rate")
        if not isinstance(sr, (int, float)) or not math.isfinite(float(sr)):
            continue
        er_raw = e.get("eval/episode_reward")
        er = (
            float(er_raw)
            if isinstance(er_raw, (int, float)) and math.isfinite(float(er_raw))
            else -INF
        )
        sr = float(sr)
        if sr > b_sr or (sr == b_sr and er > b_er):
            b_sr, b_er, b_step = sr, er, int(e["step"])
    best_snapshot = (
        None
        if b_step is None
        else {
            "path": "snapshot_best.pt",
            "source": f"snapshot_{b_step}.pt",
            "step": b_step,
            "success_rate": None if b_sr == -INF else b_sr,
            "episode_reward": None if b_er == -INF else b_er,
        }
    )
    return {
        "last_train_episode": last_train,
        "last_episode_safety": last_episode_safety,
        "last_eval": last_eval,
        "best_eval": best_eval,
        "best_snapshot": best_snapshot,
        "_n_evals": len(evals),
    }


def read_config(cell: Path, frames_reached: int) -> tuple[dict, int]:
    """Build the config block from .hydra/config.yaml, mirroring the access
    paths in _write_final_metrics. num_train_frames is overridden to the
    frames actually reached. Returns (config_block, requested_frames)."""
    cfg: dict = {}
    cfgp = cell / ".hydra" / "config.yaml"
    if cfgp.is_file():
        cfg = yaml.safe_load(cfgp.read_text()) or {}
    env = cfg.get("env") or {}
    wb = cfg.get("wandb") or {}
    agent = cfg.get("agent") or {}
    requested = int(cfg.get("num_train_frames", 0) or 0)
    block = {
        "task": str(env.get("env_name", "")),
        "disruption": str(cfg.get("disruption", "") or ""),
        "num_train_frames": int(frames_reached),
        "num_demos": int(cfg.get("num_demos", 0) or 0),
        "agent_v_min": float(agent.get("v_min", float("nan"))),
        "agent_v_max": float(agent.get("v_max", float("nan"))),
        "wandb_name": str(wb.get("name", "") or ""),
        "wandb_tags": list(wb.get("tags", []) or []),
    }
    return block, requested


def build_final_metrics(cell: Path) -> dict | None:
    """Assemble the full final_metrics.json payload for one cell dir, or None
    if there is nothing to reconstruct from."""
    jsonl = cell / "metrics.jsonl"
    if not jsonl.is_file():
        print(f"  SKIP {cell.name}: no metrics.jsonl")
        return None
    rows = load_rows(jsonl)
    if not rows:
        print(f"  SKIP {cell.name}: metrics.jsonl has no usable rows")
        return None

    runs = split_runs(rows)
    primary = max(runs, key=lambda s: max(r["step"] for r in s))
    steps = [r["step"] for r in primary]
    lo, hi = min(steps), max(steps)

    blocks = reconstruct_blocks(primary)
    n_evals = blocks.pop("_n_evals")
    config, requested = read_config(cell, frames_reached=hi)

    if not (cell / "snapshot_best.pt").is_file():
        print(f"  WARN {cell.name}: snapshot_best.pt missing — "
              f"best_snapshot.path will dangle")

    out = {
        "config": config,
        "last_train_episode": blocks["last_train_episode"],
        "last_episode_safety": blocks["last_episode_safety"],
        "last_eval": blocks["last_eval"],
        "best_eval": blocks["best_eval"],
        "best_snapshot": blocks["best_snapshot"],
        "_reconstructed": {
            "tool": "scripts/reconstruct_final_metrics.py",
            "source": "metrics.jsonl + snapshot_best.pt (offline salvage)",
            "num_runs_in_metrics": len(runs),
            "primary_run_step_range": [lo, hi],
            "primary_run_eval_count": n_evals,
            "frames_reached": hi,
            "frames_requested": requested,
        },
    }
    # One-line audit so the operator can eyeball the salvage.
    le = out["last_eval"]
    bs = out["best_snapshot"] or {}
    print(
        f"  OK   {cell.name}: runs={len(runs)} primary={lo}->{hi} "
        f"(req {requested}) evals={n_evals} "
        f"last_succ={le.get('eval/success_rate')} "
        f"best_succ={bs.get('success_rate')}@{bs.get('step')}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path,
                    help="cell directories to reconstruct final_metrics.json in")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing final_metrics.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written; do not write files")
    args = ap.parse_args()

    written = 0
    for cell in args.dirs:
        cell = cell.resolve()
        if not cell.is_dir():
            print(f"  SKIP {cell}: not a directory")
            continue
        target = cell / "final_metrics.json"
        if target.is_file() and not args.force:
            print(f"  SKIP {cell.name}: final_metrics.json exists (use --force)")
            continue
        payload = build_final_metrics(cell)
        if payload is None:
            continue
        if args.dry_run:
            continue
        target.write_text(json.dumps(payload, indent=2))
        written += 1
    print(f"\n{'(dry-run) would write' if args.dry_run else 'wrote'} "
          f"{written if not args.dry_run else ''} final_metrics.json file(s)")


if __name__ == "__main__":
    main()
