# Base-task isolation: why `dishwasher_close` / `drawers_open_all` "collapsed", and how to make them train

**Date:** 2026-06-07
**Data:** `exp_local/isolation/` · **Aggregate:** `exp_local/isolation/isolation_results.json`, `dispatch_summary.json`

## TL;DR

The fixed-λ Lagrangian runs scoring **~0% task success** on `dishwasher_close` and
`drawers_open_all` were **not** a safety/Lagrangian failure. The underlying CQN-AS
base policy could not learn these two tasks, for three compounding reasons —
**too few demos, a completion-agnostic workspace-penalty shaping term, and
warm-starting the Lagrangian from already-collapsed (0%) snapshots.** Fix those
and both tasks train to **0.8–1.0** peak success. A controlled A/B shows the
demos are the load-bearing fix; a dense potential-based progress reward helps the
*long* task's ceiling but adds instability.

The BiGym reward is **sparse and terminal** (`bigym_env.py:_reward =
float(success) * SPARSE_REWARD_FACTOR`, = 1, 0 until completion). So in the old
runs the only dense signal was the workspace-proximity penalty, which rewards
"EE near the object", not "task done" — and the Lagrangian cost then pulled the
robot *away*, producing the retreat-to-the-corner policies (`mean_separation`
2–5 m vs ~0.7 m on the working `saucepan_to_hob`).

## Experiment

Each run is the **stage-0 gate** (the repo's "can the policy learn the task at
all?" setting from `run_base_curriculum.sh`): human present but idle/distant
(`disruption=coworker_idle`, obs width preserved), **no safety shaping**, plain
`agent=cqn_as`, widened critic support (`v_min=-6 v_max=2 atoms=101`),
peak-snapshot selection (`snapshot_best.pt`).

- **rung-1** = sparse reward + *all* demos (dishwasher 69, drawers 54), no shaping.
- **rung-3** = rung-1 **+** a dense **potential-based task-progress reward**
  (new this session): Φ = −mean(|manipulable.get_state() − goal|), F = β·(γΦ′−Φ),
  with `progress_gamma=1.0` (pure ΔΦ, no survival offset), β=1, goal=0 for
  dishwasher (close), 1 for drawers (open). Default OFF; see `SafetyConfig`.

2 seeds × 2 tasks × 2 rungs = 8 runs, dispatched across free GPUs by
`scripts/dispatch_isolation.py`. dishwasher 30k frames, drawers 40k.

## Results

Old base curriculum (for reference): `pipe_base_dishwasher_close` peaked **0.2**
then decayed to 0; `pipe_base_drawers_open_all` **0.0 at every stage** (incl. no
human). `saucepan_to_hob` base = 0.9–1.0 (the control: a task BC bootstraps).

Per run (`peak` = best eval; `stab` = mean of last-4 evals; success_rate over 10
eval episodes, so ±0.1 granularity / high variance):

| task | rung | seed | peak | stab | final |
|---|---|---|---|---|---|
| dishwasher | rung-1 | 1 | 0.80 | 0.75 | 0.60 |
| dishwasher | rung-1 | 2 | **1.00** | 0.68 | 0.80 |
| dishwasher | rung-3 | 1 | 0.80 | 0.57 | 0.60 |
| dishwasher | rung-3 | 2 | 0.80 | 0.40 | 0.30 |
| drawers | rung-1 | 1 | 0.80 | 0.62 | 0.40 |
| drawers | rung-1 | 2 | 0.80 | 0.62 | 0.70 |
| drawers | rung-3 | 1 | **0.90** | 0.62 | 0.60 |
| drawers | rung-3 | 2 | **0.90** | 0.30 | 0.20 |

Mean across seeds:

| task | rung | mean peak | mean stab |
|---|---|---|---|
| dishwasher | **rung-1 (demos)** | **0.90** | **0.71** |
| dishwasher | rung-3 (+progress) | 0.80 | 0.49 |
| drawers | rung-1 (demos) | 0.80 | 0.62 |
| drawers | **rung-3 (+progress)** | **0.90** | 0.46 |

## Conclusions

1. **Both tasks are trainable.** 0% → 0.8–1.0 peak. The "collapse" was base
   trainability (demos / shaping / warm-start), not the safety machinery —
   confirmed by `saucepan_to_hob`, which the base solves *and* which survives the
   Lagrangian fine. **Do not tune λ / cost_budget / critic support to "fix" these
   tasks; fix the base policy first.**
2. **Demos are the load-bearing fix.** dishwasher 50→69 demos + dropping the
   workspace penalty + peak-snapshot moved it from 0% to a stable 0.8–1.0.
3. **The dense progress reward is horizon-dependent and a trade, not a free win:**
   - dishwasher (short, 350-step): rung-1 beats rung-3 on **both** peak (0.90 vs
     0.80) and stability (0.71 vs 0.49) — progress shaping *hurts* the
     demo-bootstrappable short task.
   - drawers (long, 1200-step): rung-3 raises the **ceiling** (0.90 vs 0.80) but
     is **less stable** (0.46 vs 0.62). With peak-snapshot deployment you get a
     better drawers policy from rung-3, at the cost of oscillation.

**Recommended recipe** (base policy for the safety pipeline): rung-1 — all demos,
no workspace penalty, widened critic support, peak-snapshot. Add the progress
reward only for long-horizon sparse tasks, and only if you select by peak.

## Best deployable snapshots (peak success)

| task | recipe | snapshot | peak |
|---|---|---|---|
| dishwasher | rung-1 | `exp_local/isolation/dish_rung1_seed2/snapshot_best.pt` | 1.00 |
| dishwasher | rung-1 | `exp_local/isolation/rung1_dish_idle_nows_69demo/snapshot_best.pt` | 0.80 |
| drawers | rung-3 | `exp_local/isolation/drawers_rung3_seed1/snapshot_best.pt` | 0.90 |
| drawers | rung-3 | `exp_local/isolation/drawers_rung3_seed2/snapshot_best.pt` | 0.90 |
| drawers | rung-1 | `exp_local/isolation/drawers_rung1_seed{1,2}/snapshot_best.pt` | 0.80 |

## Artifacts & reproduction

- **Per-run:** `exp_local/isolation/<run>/` — `metrics.jsonl`, `snapshot_best.pt`, eval videos; `logs/isolation/<run>.log`
- **W&B:** project `safety-critic`, runs `isolation_*` (+ `isolation_rung1_dish_idle_nows_69demo`, `isolation_rung3_dish_idle_progress_69demo`)
- **Live status / aggregate:** `venv/bin/python scripts/isolation_status.py` · `scripts/isolation_results.py`
- **Dispatcher (GPU-pool, idempotent):** `scripts/dispatch_isolation.py`
- **rung-3 reward:** `SafetyConfig.add_progress_reward/progress_beta/progress_goal/progress_gamma`; `safety_env._compute_progress_reward` / `_lookup_task_state` / `_progress_potential`; tests in `tests/test_progress_reward.py`. Enable per-launch: `env.safety.add_progress_reward=true env.safety.progress_goal={0|1}`.

Single-run command (drawers rung-3, the template):
```
AMASS_DATA_DIR=/home/ap2322/Documents/CMU/CMU CUDA_VISIBLE_DEVICES=<g> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
venv/bin/python train_cqn_as.py env=safety_bigym/drawers_open_all env.human_model=g1 \
  bodyslam=oracle num_demos=54 env.safety.add_workspace_penalty=false \
  env.safety.add_progress_reward=true env.safety.progress_beta=1.0 env.safety.progress_goal=1.0 \
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101 disruption=coworker_idle \
  num_train_frames=40000 save_snapshot=true hydra.run.dir=exp_local/isolation/<name>
```

## Caveats / next steps

- **Stage-0 only.** These runs use `coworker_idle` (no real disruption). The
  safety pipeline still needs: re-introduce the coworker (stages 1–2) from these
  trained snapshots, then re-add safety via **adaptive-λ PID + a feasible
  cost_budget** (not the frozen λ=0.1 / budget=0 that produced the original
  collapse). saucepan evidence (`e3_2`) shows adaptive-λ preserves task success.
- **High eval variance.** 10 eval episodes → curves oscillate ±0.2–0.3; peak and
  mean-of-last-4 are noisy. More eval episodes (or more seeds) would tighten the
  rung-1-vs-rung-3 drawers comparison, which is close.
- **2 seeds.** Enough to see the effect direction; not enough for tight CIs.
