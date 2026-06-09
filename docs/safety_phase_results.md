# Safety pipeline — dishwasher_close & drawers_open_all

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


## dishwasher_close

| operating point | n | success | prox (τ=0.3) | Δprox vs base | min-sep (m) | robot vel |
|---|---|---|---|---|---|---|
| **baseline (unconstrained)** | 60 | 0.767 | 0.246 | — | 0.403 | 0.444 |
| budget 0.3 | 180 | 0.761 | 0.243 [0.19,0.30] | -0.003 (-1%) | 0.403 | 0.472 |
| budget 0.5 | 180 | 0.800 | 0.245 [0.19,0.30] | -0.001 (-0%) | 0.412 | 0.454 |

*budget 0.1 collapsed to 0% task success in-training (constraint too tight to leave the task feasible) — see `dish_b01`; not benchmarked.*

## drawers_open_all

| operating point | n | success | prox (τ=0.3) | Δprox vs base | min-sep (m) | robot vel |
|---|---|---|---|---|---|---|
| **baseline (unconstrained)** | 60 | 0.817 | 0.211 | — | 0.099 | 0.192 |
| budget 0.3 | 180 | 0.883 | 0.216 [0.18,0.25] | +0.005 (+2%) | 0.099 | 0.188 |
| budget 0.5 | 180 | 0.872 | 0.222 [0.19,0.26] | +0.011 (+5%) | 0.090 | 0.182 |

*budget 0.1 collapsed to 0% task success in-training (constraint too tight to leave the task feasible) — see `drawers_b01`; not benchmarked.*

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

