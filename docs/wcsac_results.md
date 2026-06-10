# WCSAC external baseline (E3.7 / P9) — results

Faithful Worst-Case SAC (Yang et al., AAAI 2021) reimplemented on the 76-DOF
humanoid SafetyBiGym tasks as `agent=wcsac` on the CQN-AS stack. This is the
**external** distributional safe-RL reference (distinct from the project's
value-based B-value-CVaR / Lagrangian methods). Code: `safety_bigym/agents/wcsac/`.

## Setup

- **Sweep**: 2 tasks (`dishwasher_close`, `drawers_open_all`) × 3 CVaR-budgets
  `d ∈ {5, 15, 30}` (ceiling on CVaR_α of the discounted cost **return**,
  α=0.9) × seed 0. 150k env-frames each, **trained from scratch** (`num_demos=0`).
- **Why from scratch**: WCSAC is actor-critic (SAC); it cannot load the
  CQN-AS C2F warm-start snapshots the Lagrangian fine-tunes from. This is the
  documented honest-failure path — a from-scratch pixel-SAC baseline.
- **Eval**: `benchmark_policy.py`, final (converged) snapshot per cell, 30
  episodes, `disruption=coworker_train`, `obs-mode=oracle`, `human-model=g1`.
  cvar95 + bootstrap CIs, identical harness to the Lagrangian P5 rows.
- **Validity fix**: WCSAC trains demo-free, so its action de-normalisation uses
  the adapter's *identity* stats. `benchmark/env_build.build_cqn_adapter` was
  forcing 5-demo-derived stats on every snapshot (a train/deploy mismatch that
  produced plausible-but-wrong numbers); fixed to keep identity stats for
  `num_demos=0` snapshots. Cross-validated against `train_cqn_as` eval
  (proximity-violation-rate and success agree). CQN-AS (demo-trained) eval is
  unaffected.

## Results (30 episodes/cell, final snapshot)

### dishwasher_close
| budget | λ (train end) | success | prox-viol rate | mean min-sep (m) | worst min-sep (m) | CVaR95 min-sep (m) | CVaR95 cost |
|--------|---------------|---------|----------------|------------------|-------------------|--------------------|-------------|
| 5      | 100 (max)     | 0.47    | 0.036          | 1.12             | 0.025             | 0.037              | 89.4        |
| 15     | 4.5           | 0.43    | 0.102          | 0.62             | 0.009             | 0.016              | 97.2        |
| 30     | 0             | 0.03    | 0.021          | 1.21             | 0.031             | 0.035              | 44.4        |

### drawers_open_all
| budget | λ (train end) | success | prox-viol rate | mean min-sep (m) | worst min-sep (m) | CVaR95 min-sep (m) | CVaR95 cost |
|--------|---------------|---------|----------------|------------------|-------------------|--------------------|-------------|
| 5      | 92            | 0.00    | 0.000          | 2.18             | 1.31              | 1.36               | 14.2        |
| 15     | 0             | 0.00    | 0.016          | 1.77             | 0.035             | 0.038              | 21.0        |
| 30     | 0             | 0.00    | 0.000          | 1.63             | 0.015             | 0.028              | 30.2        |

CSVs: `results/wcsac_eval/wcsac_<task>_b<budget>_s0.csv` (full schema + CIs +
`*.raw_episodes.parquet` per-episode rolls).

## Reading

1. **dishwasher_close: WCSAC learns the task** — ~0.43–0.47 success at the
   tighter budgets with low proximity-violation (3.6% at b5), i.e. the external
   baseline is non-degenerate on the easier task.
2. **drawers_open_all: 0% success at every budget** — the documented base-task
   untrainability (sparse reward, hard from scratch). The low proximity-violation
   there is trivial: the policy never engages the task, so it stays far from the
   human (b5 keeps ≥1.3 m even worst-case).
3. **Single-seed variance is visible** — the b30/dishwasher cell (unconstrained,
   λ=0) underperforms the constrained cells on success (0.03 vs 0.47). From-scratch
   SAC is high-variance run-to-run; the budget→success ordering within dishwasher
   is within noise. A clean tradeoff curve would need ≥3 seeds (future work).
4. **vs the value-based Lagrangian** — from-scratch WCSAC tops out ~0.47 success
   on dishwasher and 0 on drawers, below the warm-started CQN-AS Lagrangian. The
   intended contribution stands: the canonical actor-critic safe-RL reference,
   reimplemented honestly on the humanoid, underperforms the warm-started
   value-based method — consistent with §disc:wcsac-honest.

## Reproduce

```bash
# train (multi-GPU, polls for idle GPUs):
AMASS_DATA_DIR=/path/CMU/CMU FRAMES=150000 venv/bin/python scripts/dispatch_wcsac.py
# eval:
GPU=0 EPISODES=30 scripts/eval_wcsac.sh
```
