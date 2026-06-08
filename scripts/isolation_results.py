#!/usr/bin/env python3
"""Aggregate the base-task isolation campaign into an A/B table.

For each run under exp_local/isolation/, computes peak / final / mean-of-last-4
eval success_rate and locates snapshot_best.pt, then groups by
(task x rung x seed) so the dishwasher / drawers, rung1-vs-rung3 comparison is
explicit. Prints a markdown table and writes isolation_results.json.

  venv/bin/python scripts/isolation_results.py
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "exp_local" / "isolation"


def classify(name):
    task = "dishwasher" if ("dish" in name) else ("drawers" if "drawers" in name else "?")
    rung = "rung3" if ("rung3" in name or "progress" in name) else "rung1"
    seed = "s2" if "seed2" in name else "s1"
    return task, rung, seed


def run_stats(run):
    f = run / "metrics.jsonl"
    succ = []
    last_frame = 0
    if f.exists():
        for line in f.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ty") == "eval":
                succ.append(d.get("eval/success_rate", 0.0))
            if d.get("ty") == "train":
                last_frame = d.get("step", last_frame)
    done = (run / "final_metrics.json").exists()
    best = run / "snapshot_best.pt"
    return {
        "peak": max(succ, default=0.0),
        "final": succ[-1] if succ else 0.0,
        "mean_last4": round(statistics.mean(succ[-4:]), 3) if succ else 0.0,
        "n_evals": len(succ),
        "frame": last_frame,
        "done": done,
        "snapshot_best": str(best) if best.exists() else None,
    }


def main():
    runs = sorted([p for p in ROOT.iterdir() if p.is_dir()]) if ROOT.exists() else []
    cells = {}
    for run in runs:
        t, r, s = classify(run.name)
        cells[(t, r, s)] = {"run": run.name, **run_stats(run)}

    print("\n## Base-task isolation: rung1 (demos) vs rung3 (+progress reward)\n")
    print("| task | rung | seed | peak | mean_last4 | final | frame | done |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for t in ("dishwasher", "drawers"):
        for r in ("rung1", "rung3"):
            for s in ("s1", "s2"):
                c = cells.get((t, r, s))
                if not c:
                    continue
                rows.append((t, r, s, c))
                print(f"| {t} | {r} | {s} | {c['peak']:.2f} | {c['mean_last4']:.2f} "
                      f"| {c['final']:.2f} | {c['frame']} | {'Y' if c['done'] else '·'} |")

    print("\n## Per-(task,rung) summary (mean across seeds)\n")
    print("| task | rung | mean peak | mean mean_last4 |")
    print("|---|---|---|---|")
    agg = {}
    for t in ("dishwasher", "drawers"):
        for r in ("rung1", "rung3"):
            cs = [c for (tt, rr, _), c in cells.items() if tt == t and rr == r]
            if not cs:
                continue
            mp = statistics.mean(c["peak"] for c in cs)
            ml = statistics.mean(c["mean_last4"] for c in cs)
            agg[f"{t}/{r}"] = {"mean_peak": round(mp, 3), "mean_last4": round(ml, 3),
                               "n_seeds": len(cs)}
            print(f"| {t} | {r} | {mp:.2f} | {ml:.2f} |")

    out = {"cells": {f"{t}/{r}/{s}": c for (t, r, s), c in cells.items()}, "agg": agg}
    (ROOT / "isolation_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ROOT/'isolation_results.json'}")
    ndone = sum(1 for c in cells.values() if c["done"])
    print(f"runs done: {ndone}/{len(cells)}")


if __name__ == "__main__":
    main()
