#!/usr/bin/env python3
"""Status of the adaptive-lambda safety phase — task success vs safety tradeoff.

  venv/bin/python scripts/safety_status.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "exp_local" / "safety"


def last_eval(run):
    f = run / "metrics.jsonl"
    if not f.exists():
        return None, 0
    ev, frame = [], 0
    for l in f.read_text().splitlines():
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("ty") == "eval":
            ev.append(d)
        if d.get("ty") == "train":
            frame = d.get("step", frame)
    return (ev[-1] if ev else None), frame


def running(run):
    return subprocess.run(["pgrep", "-f", f"hydra.run.dir={run}($| )"],
                          capture_output=True).returncode == 0


def main():
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"]) \
        if ROOT.exists() else []
    if not runs:
        print("no safety runs yet under", ROOT)
        return
    print(f"{'cell':<14} {'state':<8} {'frame':>6} | {'succ':>5} {'proxViol':>8} {'minSep':>6} {'meanSep':>7}")
    print("-" * 70)
    for run in runs:
        ev, frame = last_eval(run)
        done = (run / "final_metrics.json").exists()
        st = "DONE" if done else ("running" if running(run) else "idle")
        if ev:
            print(f"{run.name:<14} {st:<8} {frame:>6} | "
                  f"{ev.get('eval/success_rate',0):>5.2f} "
                  f"{ev.get('eval/ep_proximity_violation_rate',0):>8.2f} "
                  f"{ev.get('eval/ep_min_separation',0):>6.2f} "
                  f"{ev.get('eval/ep_mean_separation',0):>7.2f}")
        else:
            print(f"{run.name:<14} {st:<8} {frame:>6} | (no eval yet)")
    if (ROOT / "dispatch_summary.json").exists():
        print("\n[safety dispatch_summary.json present — phase complete]")


if __name__ == "__main__":
    main()
