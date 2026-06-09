#!/usr/bin/env python3
"""Pool the safety-phase operating-point benchmarks and emit the results doc.

For each (task, budget) it pools the 3 train-seed cells' per-episode rolls
(<cell>.episodes.jsonl, 60 ep each -> 180 ep) and reports success / proximity /
min-separation / robot-velocity with bootstrap CIs, plus the proximity delta vs
the unconstrained curriculum baseline. Writes docs/safety_phase_results.md.

Pure post-processing of results/safety_eval/*.episodes.jsonl — no GPU, no rollout.

  venv/bin/python scripts/summarize_safety_eval.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "results" / "safety_eval"
DOC = REPO / "docs" / "safety_phase_results.md"

GROUPS = [
    ("dishwasher_close", 0.3, ["dish_b03", "dish_b03_s1", "dish_b03_s2"]),
    ("dishwasher_close", 0.5, ["dish_b05", "dish_b05_s1", "dish_b05_s2"]),
    ("drawers_open_all", 0.3, ["drawers_b03", "drawers_b03_s1", "drawers_b03_s2"]),
    ("drawers_open_all", 0.5, ["drawers_b05", "drawers_b05_s1", "drawers_b05_s2"]),
]
BASELINE = {  # from results/safety_eval/baseline_<task>.csv (noisy, 60 ep)
    "dishwasher_close": dict(csv="baseline_dishwasher", succ=0.767, prox=0.246, minsep=0.403, vel=0.444),
    "drawers_open_all": dict(csv="baseline_drawers", succ=0.817, prox=0.211, minsep=0.099, vel=0.192),
}
B01 = {  # collapsed cells (budget too tight) — in-training success, documented not benchmarked
    "dishwasher_close": "dish_b01", "drawers_open_all": "drawers_b01",
}
FIELDS = [("success", "success"), ("ep_proximity_violation_rate", "prox (τ=0.3)"),
          ("ep_min_separation", "min-sep (m)"), ("ep_mean_robot_vel", "robot vel")]


def bootstrap_ci(v, n=10000, seed=12345):
    v = np.asarray([x for x in v if x is not None and not pd.isna(x)], float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, size=(n, v.size))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(cells):
    frames = []
    for c in cells:
        p = EVAL / f"{c}.episodes.jsonl"
        if not p.exists():
            print(f"  [warn] missing {p.name}")
            continue
        frames.append(pd.DataFrame(json.loads(l) for l in p.read_text().splitlines() if l.strip()))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


PREAMBLE = """# Safety pipeline — dishwasher_close & drawers_open_all

End-to-end pipeline taking two BiGym manipulation tasks from untrainable to a
constrained-RL safety phase, mirroring the saucepan_to_hob pipeline. Regenerate
this doc with `venv/bin/python scripts/summarize_safety_eval.py`.

## Pipeline

1. **Base task is untrainable under the live human.** With BiGym's sparse terminal
   reward (`float(success)`), CQN-AS never leaves 0% success on either task under a
   moving coworker — the agent never stumbles onto the success state to bootstrap
   from. (See memory `safety-bigym-base-untrainability`; this is a reward-sparsity
   wall, not a Lagrangian bug.)

2. **Human curriculum solves the task.** Demos + a 3-stage disruption curriculum
   (`coworker_idle` → `coworker_easy` → `coworker_train`), oracle bodyslam, widened
   critic support (v_min=-6, v_max=2, atoms=101). Both tasks reach **peak 1.00**
   success under the full `coworker_train` human (`dish_rung1_a`, `drawers_rung1_a`,
   stage2 `snapshot_best.pt`). These are the unconstrained baselines below.

3. **Adaptive-λ constrained fine-tune (this eval).** Each stage-2 policy is
   fine-tuned with the `cqn_as_lagrangian` dual reward+cost C51 critic under an
   ADAPTIVE-λ PID at a feasible `cost_budget` (workspace penalty OFF — the e3_2
   recipe). This is the fix for the original collapse, where a **frozen** λ=0.1 /
   budget=0 from a collapsed warm-start drove dishwasher success to **0%**. Sweeps
   `cost_budget ∈ {0.1, 0.3, 0.5}`; budgets 0.3/0.5 run 3 train-seeds.

## Operating-point evaluation

Each cell benchmarked at its **final deployed checkpoint** under deployment-realistic
noisy obs (`coworker_train`, g1 human, 3 train-seeds × 20 ep = 180 pooled episodes
per operating point). Baseline = the unconstrained stage-2 curriculum policy (60 ep).
Final-checkpoint (not basin-pick) selection is deliberate: it is the honest deployed
policy, apples-to-apples across cells, and free of the selection bias that nominating
off noisy 10-ep in-training evals would introduce.

Proximity = fraction of episode time within 0.3 m human–robot separation (lower =
safer). Δ is vs the same task's unconstrained baseline; CIs are 10k-sample bootstrap.
"""


def main():
    lines = [PREAMBLE]
    for task in ("dishwasher_close", "drawers_open_all"):
        b = BASELINE[task]
        lines.append(f"\n## {task}\n")
        lines.append(f"| operating point | n | success | prox (τ=0.3) | Δprox vs base | min-sep (m) | robot vel |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(f"| **baseline (unconstrained)** | 60 | {b['succ']:.3f} | {b['prox']:.3f} | — "
                     f"| {b['minsep']:.3f} | {b['vel']:.3f} |")
        for t, budget, cells in GROUPS:
            if t != task:
                continue
            df = load(cells)
            if df.empty:
                lines.append(f"| budget {budget} | 0 | (pending) | | | | |")
                continue
            vals = {k: bootstrap_ci(df[k].tolist()) for k, _ in FIELDS if k in df.columns}
            sm = vals["success"][0]
            pm, plo, phi = vals["ep_proximity_violation_rate"]
            dp = pm - b["prox"]
            mm = vals.get("ep_min_separation", (float('nan'),))[0]
            vm = vals.get("ep_mean_robot_vel", (float('nan'),))[0]
            lines.append(f"| budget {budget} | {len(df)} | {sm:.3f} | {pm:.3f} [{plo:.2f},{phi:.2f}] "
                         f"| {dp:+.3f} ({100*dp/b['prox']:+.0f}%) | {mm:.3f} | {vm:.3f} |")
        lines.append(f"\n*budget 0.1 collapsed to 0% task success in-training (constraint too tight "
                     f"to leave the task feasible) — see `{B01[task]}`; not benchmarked.*")

    lines.append("""
## Findings

1. **Adaptive-λ preserves the task — the collapse is fixed.** At every deployable
   budget (0.3, 0.5) success holds at or above the unconstrained baseline (drawers
   even improves, 0.93/0.87 vs 0.82). This is the headline win over the original
   **frozen**-λ regime, which drove dishwasher to 0%. Adaptive-λ + a feasible budget
   keeps the policy on-task.

2. **…but it buys ~no proximity safety at deployment.** Final-checkpoint proximity
   moves within ±5% of baseline on both tasks, every CI overlapping the baseline.
   The Lagrangian constraint does not meaningfully reduce time-in-proximity at these
   budgets. This confirms the standing `safety-task-tradeoff-finding`: no current
   λ-based method delivers a large safety gain *and* task success.

3. **Budget 0.1 is infeasible** — the constraint is tight enough to make the task
   itself unsolvable (0% success), so it is not a usable operating point.

### Caveat — the basin is masked by final-checkpoint selection

In-training curves show *lower*-proximity checkpoints mid-training (e.g. several
budget-0.3/0.5 seeds dip to in-training prox 0.00–0.10 around 30k frames) that late
reward-chasing erodes back to baseline by the final checkpoint — the documented ROW3
"avoidance basin". A proper **basin sweep** (benchmark every checkpoint, pick the
lowest-deploy-proximity one at acceptable success — `run_basin_sweep.sh` +
`analyze_row3.py pick`) could recover a genuinely safer deployable operating point
that this final-checkpoint table does not capture. It is the rigorous follow-up; it
is **not** run here (multi-hour, and early snapshots for the seed-0 cells were rotated
off disk, so a clean basin pool is only available for the seed-1/2 cells).

## Provenance

- Baselines: `exp_local/curriculum/{dish_rung1_a,drawers_rung1_a}/stage2_full/snapshot_best.pt`
  (curriculum stage-2 peak 1.00), benchmarked to `results/safety_eval/baseline_{dishwasher,drawers}.csv`.
- Safety cells: `exp_local/safety/<cell>/` final `snapshot_<~37-40k>.pt`; per-cell
  benchmarks in `results/safety_eval/<cell>.{csv,episodes.jsonl}`.
- Training dispatcher: `scripts/dispatch_safety.py` (adaptive-λ, workspace OFF).
- Eval dispatcher: `scripts/dispatch_safety_eval.py` (lock-based, race-free, no double-launch).
- This table: `scripts/summarize_safety_eval.py` over `results/safety_eval/*.episodes.jsonl`.
""")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote] {DOC}")


if __name__ == "__main__":
    main()
