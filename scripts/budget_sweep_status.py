#!/usr/bin/env python3
"""Status of the cost-budget retarget sweep — does the constraint BIND?

Reads each cell's latest snapshot for the Lagrangian PID state (final λ +
rolling-cost ≈ budget + prev_violation) and its last in-training eval
(success / proximity). The headline question per cell:
  - λ > 0  -> constraint is active (vs the λ=0 inert cells of the first sweep)
  - λ near λ_max(100) -> over-tight (likely collapsing the task)
  - success preserved AND proximity < baseline -> a usable safer operating point

  venv/bin/python scripts/budget_sweep_status.py
"""
import glob
import json
import os
import subprocess
from pathlib import Path

import sys

import torch

# Optional positional arg picks the campaign dir (default budget_sweep; pass
# "budget_warmstart" for the warm-started variant).
_SUB = sys.argv[1] if len(sys.argv) > 1 else "budget_sweep"
ROOT = Path(__file__).resolve().parent.parent / "exp_local" / _SUB
BASE_PROX = {"dishwasher_close": 0.246, "drawers_open_all": 0.211}


def latest_snapshot(run):
    snaps = glob.glob(str(run / "snapshot_[0-9]*.pt"))
    if not snaps:
        return None
    return max(snaps, key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))


def pid_state(snap):
    try:
        d = torch.load(snap, map_location="cpu", weights_only=False)
    except Exception:
        return None

    def find(o, depth=0):
        if depth > 6:
            return None
        if isinstance(o, dict):
            if "lam" in o and "cost_budget" in o:
                return o
            for v in o.values():
                r = find(v, depth + 1)
                if r:
                    return r
        return None
    return find(d)


def last_eval(run):
    f = run / "metrics.jsonl"
    if not f.exists():
        return None, 0
    ev, frame = None, 0
    for l in f.read_text().splitlines():
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("ty") == "eval":
            ev = d
        if d.get("ty") == "train":
            frame = d.get("step", frame)
    return ev, frame


def running(run):
    return subprocess.run(["pgrep", "-f", f"hydra.run.dir={run}($| )"],
                          capture_output=True).returncode == 0


def task_of(name):
    return "dishwasher_close" if name.startswith("dish") else "drawers_open_all"


def main():
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"]) \
        if ROOT.exists() else []
    if not runs:
        print("no budget_sweep runs yet under", ROOT)
        return
    print(f"{'cell':<14} {'state':<8} {'frame':>6} | {'lambda':>8} {'rollcost':>8} {'budget':>6} | "
          f"{'succ':>5} {'prox':>5} {'baseProx':>8}")
    print("-" * 86)
    for run in runs:
        done = (run / "final_metrics.json").exists()
        st = "DONE" if done else ("running" if running(run) else "idle")
        snap = latest_snapshot(run)
        ps = pid_state(snap) if snap else None
        ev, frame = last_eval(run)
        task = task_of(run.name)
        if ps:
            lam = ps.get("lam", float("nan"))
            budget = ps.get("cost_budget", float("nan"))
            roll = budget + ps.get("prev_violation", float("nan"))
            lam_s, roll_s, bud_s = f"{lam:8.3f}", f"{roll:8.3f}", f"{budget:6.2f}"
        else:
            lam_s = roll_s = bud_s = "     —"
        if ev:
            succ = ev.get("eval/success_rate", 0.0)
            prox = ev.get("eval/ep_proximity_violation_rate", 0.0)
            ev_s = f"{succ:5.2f} {prox:5.2f}"
        else:
            ev_s = "  —     —"
        print(f"{run.name:<14} {st:<8} {frame:>6} | {lam_s} {roll_s} {bud_s} | "
              f"{ev_s} {BASE_PROX[task]:8.3f}")
    if (ROOT / "dispatch_summary.json").exists():
        print("\n[dispatch_summary.json present — sweep training complete]")


if __name__ == "__main__":
    main()
