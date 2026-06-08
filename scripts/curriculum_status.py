#!/usr/bin/env python3
"""Status of the human-curriculum phase (stage1_easy -> stage2_full per cell).

  venv/bin/python scripts/curriculum_status.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "exp_local" / "curriculum"


def stage(run, name):
    f = run / name / "metrics.jsonl"
    if not f.exists():
        return None
    succ, frame = [], 0
    for l in f.read_text().splitlines():
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("ty") == "eval":
            succ.append(d.get("eval/success_rate", 0.0))
        if d.get("ty") == "train":
            frame = d.get("step", frame)
    done = (run / name / "final_metrics.json").exists()
    return {"frame": frame, "peak": max(succ, default=0.0),
            "last": succ[-1] if succ else 0.0, "done": done, "n": len(succ)}


def running(run):
    return subprocess.run(["pgrep", "-f", f"exp_local/curriculum/{run.name}/"],
                          capture_output=True).returncode == 0


def main():
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name != "__pycache__"]) \
        if ROOT.exists() else []
    if not runs:
        print("no curriculum runs yet under", ROOT)
        return
    print(f"{'cell':<18} {'state':<8} | {'stage1_easy':<22} | {'stage2_full':<22}")
    print("-" * 80)
    for run in runs:
        s1, s2 = stage(run, "stage1_easy"), stage(run, "stage2_full")
        full_done = bool(s2 and s2["done"])
        st = "DONE" if full_done else ("running" if running(run) else "idle")

        def fmt(s):
            if not s:
                return "—"
            tag = "✓" if s["done"] else f"{s['frame']//1000}k"
            return f"{tag} peak{s['peak']:.2f} last{s['last']:.1f}"
        print(f"{run.name:<18} {st:<8} | {fmt(s1):<22} | {fmt(s2):<22}")
    summ = ROOT / "dispatch_summary.json"
    if summ.exists():
        print("\n[dispatch_summary.json present — curriculum phase complete]")


if __name__ == "__main__":
    main()
