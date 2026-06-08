#!/usr/bin/env python3
"""Snapshot of the base-task isolation campaign (rung1 vs rung3).

  venv/bin/python scripts/isolation_status.py

Prints, per run under exp_local/isolation/, the eval success/reward curve, the
peak success, current frame, and whether it has finished — grouped so the
dishwasher / drawers  rung1-vs-rung3  A/Bs are easy to read.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "exp_local" / "isolation"


def evals(run):
    f = run / "metrics.jsonl"
    if not f.exists():
        return [], None
    ev, last_step = [], None
    for line in f.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("ty") == "eval":
            ev.append((d["step"], d.get("eval/success_rate", 0.0),
                       d.get("eval/episode_reward", 0.0)))
        if d.get("ty") == "train":
            last_step = d["step"]
    return ev, last_step


def running(run):
    # Match by run-dir basename so it works whether the job was launched with a
    # relative or absolute hydra.run.dir.
    r = subprocess.run(["pgrep", "-f", f"hydra.run.dir=.*{run.name}($| )"],
                       capture_output=True)
    return r.returncode == 0


def main():
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir()]) if ROOT.exists() else []
    if not runs:
        print("no isolation runs yet under", ROOT)
        return
    print(f"{'run':<34} {'state':<8} {'frame':>7} {'peak':>5}  eval success curve")
    print("-" * 100)
    for run in runs:
        ev, last = evals(run)
        done = (run / "final_metrics.json").exists()
        state = "DONE" if done else ("running" if running(run) else "stopped")
        peak = max((s for _, s, _ in ev), default=0.0)
        curve = " ".join(f"{st//1000}k:{s:.1f}" for st, s, _ in ev[-12:])
        print(f"{run.name:<34} {state:<8} {(last or 0):>7} {peak:>5.2f}  {curve}")
    summ = ROOT / "dispatch_summary.json"
    if summ.exists():
        print("\n[dispatch_summary.json present]")


if __name__ == "__main__":
    main()
