# Phase 2 — GPU Handoff Guide

This guide takes you from the smoke-validated state at end of session to a closed Phase 2 deliverable (non-empty Pareto curve + writeup). It's meant to be followed top-to-bottom on the GPU box (`swirl`).

Plan: [.claude/plans/read-claude-md-and-claude-hybrid-safety-ethereal-hummingbird.md](../.claude/plans/read-claude-md-and-claude-hybrid-safety-ethereal-hummingbird.md)
Master Phase 2 spec: [.claude/HYBRID_SAFETY_CRITIC_PLAN.md §Phase 2](../.claude/HYBRID_SAFETY_CRITIC_PLAN.md)

---

## 0. One-time environment setup on swirl

```bash
# Adjust if your AMASS dir lives elsewhere
export AMASS_DATA_DIR=/home/ap2322/Documents/CMU/CMU

# Headless rendering — avoids the GLFW X11 warning spam
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd /home/ap2322/Documents/safety_bigym
source venv/bin/activate
```

Sanity check the env loads:

```bash
python -c "from safety_bigym.filters import SafetyCritic; print('ok')"
```

---

## 1. Push branches and (optionally) open PRs

You currently have three sub-branches stacked on `phase2`:

```
phase2 (no commits beyond main)
└── phase-2-dataset
    └── phase-2-critic-training
        └── phase-2-runtime-wrapper  ← HEAD
```

```bash
cd /home/ap2322/Documents/safety_bigym
git push -u origin phase-2-dataset phase-2-critic-training phase-2-runtime-wrapper
```

PR strategy (your call):

- **Strict per-CLAUDE.md branch policy:** open three PRs into `phase2` (dataset → critic-training → runtime-wrapper, in order). Merge each as the prior closes.
- **Pragmatic:** fast-forward `phase2` to `phase-2-runtime-wrapper` and open one PR `phase2 → main` once GPU results are in. Saves three reviews; loses checkpointing.

---

## 2. Smoke pipeline (~75s, run once before kicking off the real jobs)

Confirms env vars + venv on the GPU box are wired correctly.

```bash
SMOKE=/tmp/svf_smoke
rm -rf $SMOKE && mkdir -p $SMOKE

python scripts/svf_collect_dataset.py --smoke --output-dir $SMOKE
python scripts/svf_train_critic.py     --smoke --dataset-dir $SMOKE --output $SMOKE/critic.pt
python scripts/svf_eval_filter.py      --smoke --critic-path $SMOKE/critic.pt
python scripts/svf_threshold_sweep.py  --smoke --critic-path $SMOKE/critic.pt
```

Expected:
- `_smoke_shard.npz` + `manifest.json` in `$SMOKE`
- `critic.pt` checkpoint
- Eval prints one row of metrics
- Sweep prints two rows (R=10, R=90)

If anything fails, fix it here — don't burn GPU time debugging plumbing.

---

## 3. Full dataset collection (~2-3 hours, single-threaded)

All three sources now usable (Phase-0 ACT retrain complete; populate
[`SNAPSHOTS`](../safety_bigym/filters/snapshots.py) before kicking off
the snapshot pass — see §7a).

```bash
DATASET=/home/ap2322/Documents/safety_bigym/datasets/svf_v1
mkdir -p $DATASET

python scripts/svf_collect_dataset.py \
    --source random --source demo --source snapshot \
    --tasks reach_target_single dishwasher_close \
    --disruptions INCIDENTAL SHARED_GOAL DIRECT OBSTRUCTION RANDOM_PERTURBED \
    --episodes-per-cell 100 \
    --max-steps 300 \
    --demos-per-task 30 \
    --bodyslam-mode oracle \
    --output-dir $DATASET \
    --seed 0 \
    2>&1 | tee $DATASET/collect.log
```

Per-task snapshot paths come from `safety_bigym/filters/snapshots.py`. Tasks
whose `SNAPSHOTS` entry is `None` will skip the snapshot source with a
warning; demos that aren't in DemoStore will skip with a warning too. Both
are non-fatal — other sources continue.

**Camera config is auto-detected.** When `--source snapshot` is requested,
`peek_snapshot_cameras` reads `cfg.env.cameras` + `cfg.visual_observation_shape`
from the snapshot's payload, and the env is rebuilt with those cameras
enabled so the actor's encoder gets the rgb keys it was trained with. No
extra flag needed; random/demo sources stay camera-free so they don't pay
the render cost.

Expected scale at these flags:
- Random: 100 eps × 5 disruptions × 2 tasks × ~300 steps ≈ 300k transitions
- Demo: 30 demos × 2 tasks × ~200 steps ≈ 12k transitions
- Total ≈ **~310k transitions** (close to the plan's 500k target; bump `--episodes-per-cell` to 150 if you want closer to the full target)

Sanity check after:

```bash
python -c "
from safety_bigym.filters.dataset import SafetyTransitionDataset
ds = SafetyTransitionDataset('$DATASET')
print(f'transitions: {len(ds)}')
print(f'violations: {len(ds.violation_indices)}')
print(f'safe:       {len(ds.safe_indices)}')
print(f'violation rate: {len(ds.violation_indices) / len(ds):.3f}')
"
```

You want the violation rate in [0.05, 0.20]. If it's <0.05 the random policy isn't tripping SSM enough — bump `--max-steps` so episodes run longer, or add more disruptions. If it's >0.30 something's off (the Phase-1 `oracle` mode shouldn't be that violent).

If a cell crashes mid-collection, the manifest stays consistent — re-run with the same args and the writer will append rather than restart.

---

## 4. Full CQL training (~30-60 min on a single GPU)

```bash
CRITIC=/home/ap2322/Documents/safety_bigym/checkpoints/svf_v1.pt
mkdir -p $(dirname $CRITIC)

python scripts/svf_train_critic.py \
    --dataset-dir $DATASET \
    --output $CRITIC \
    --num-steps 200000 \
    --batch-size 512 \
    --cql-alpha 5.0 \
    --target-tau 5e-3 \
    --lr 3e-4 \
    --gamma 0.99 \
    --target-violation-rate 0.3 \
    --log-every 1000 \
    --device cuda \
    --seed 0 \
    2>&1 | tee $DATASET/train.log
```

What to monitor in `train.log`:
- `bellman` should decrease monotonically (with noise)
- `cql_term` may grow — that's fine, it's the conservatism penalty
- `q_mean` should settle in `[0.3, 0.95] * 100` — if it sits at 50 (the q_max/2 sigmoid initial point), the critic isn't learning
- Final `loss` is irrelevant for sign of fit; rely on `bellman_last < bellman_first` check that the script prints at the end

Inspect the saved payload:

```bash
python -c "
import torch
p = torch.load('$CRITIC', weights_only=False)
print(p['training'])
"
```

Expected `training.bellman_last < training.bellman_first` — if not, training didn't converge (try `--lr 1e-4` or fewer steps).

---

## 5. Full eval — intervention rate × residual violation rate

> Phase-0 ACT retrain is **complete** and the camera-correct snapshot
> adapter landed in `728e0ef`. The load-bearing eval is `--policy
> snapshot` against Phase-1 noisy ACT (see §7). Run the random-policy
> eval below first only as a sanity check that the pipeline is wired.

```bash
RESULTS=/home/ap2322/Documents/safety_bigym/results/svf_v1
mkdir -p $RESULTS

python scripts/svf_eval_filter.py \
    --critic-path $CRITIC \
    --threshold-R 50.0 \
    --policy random \
    --tasks reach_target_single dishwasher_close \
    --disruptions INCIDENTAL SHARED_GOAL DIRECT OBSTRUCTION RANDOM_PERTURBED \
    --episodes-per-cell 10 \
    --max-steps 300 \
    --bodyslam-mode oracle \
    --output-csv $RESULTS/eval_random_R50.csv \
    --seed 0 \
    2>&1 | tee $RESULTS/eval_random_R50.log
```

This produces 10 rows (2 tasks × 5 disruptions). Each row has `intervention_rate` and `residual_violation_rate`. For a healthy v1:
- `intervention_rate` should not be 1.0 across the board (would mean R is too strict for the trained critic)
- `intervention_rate` should not be 0.0 either (would mean the critic doesn't see anything as unsafe)
- `residual_violation_rate < no_filter_baseline` — if you want that comparison, run the same eval with `--threshold-R 0.0` (filter never triggers) for the unfiltered baseline.

---

## 6. Threshold sweep — the Phase 2 deliverable

Trace the Pareto curve for one task at a time. After populating `SNAPSHOTS`
(§7a) use `--policy snapshot` for the headline curve:

```bash
for TASK in reach_target_single dishwasher_close; do
  python scripts/svf_threshold_sweep.py \
      --critic-path $CRITIC \
      --task $TASK \
      --disruption INCIDENTAL \
      --thresholds 5 10 25 50 75 90 95 \
      --policy snapshot \
      --episodes-per-R 10 \
      --max-steps 300 \
      --output-csv $RESULTS/sweep_${TASK}.csv \
      --seed 0 \
      2>&1 | tee $RESULTS/sweep_${TASK}.log
done
```

For an apples-to-apples baseline, also run with `--policy random` and
compare the two curves side-by-side in the Phase 2 writeup.

Each CSV is one Pareto curve (rows = R values; columns include `intervention_rate` and `residual_violation_rate`). The "knee" of the curve — where increasing R no longer reduces residual violations — is the operating point you'd take into Phase 4.

Quick plot:

```bash
python -c "
import csv, sys
with open('$RESULTS/sweep_reach_target_single.csv') as f:
    rows = list(csv.DictReader(f))
print(f'{\"R\":>6} {\"interv\":>8} {\"resid\":>8}')
for r in rows:
    print(f'{float(r[\"threshold_R\"]):6.1f} {float(r[\"intervention_rate\"]):8.3f} {float(r[\"residual_violation_rate\"]):8.3f}')
"
```

---

## 7. Snapshot-policy eval (Phase-0 ACT retrain complete — 2026-05-07)

ACT snapshots now exist on the GPU box. Per-task paths live in
[safety_bigym/filters/snapshots.py](../safety_bigym/filters/snapshots.py)
as the `SNAPSHOTS` dict — this is the single source of truth for both the
collection script (`--source snapshot`) and the eval/sweep scripts
(`--policy snapshot`).

### Step 7a — populate the SNAPSHOTS dict

Identify the W&B `pretrain_eval/episode_success` peak per task, then edit
the dict:

```python
# safety_bigym/filters/snapshots.py
SNAPSHOTS: Dict[str, Optional[str]] = {
    "reach_target_single":   "exp_local/act_safety/reach_target_single_<ts>/snapshots/<peak>_snapshot.pt",
    "dishwasher_close":      "exp_local/act_safety/dishwasher_close_<ts>/snapshots/<peak>_snapshot.pt",
    "dishwasher_load_plates": None,  # leave None for tasks without snapshots
    "saucepan_to_hob":       None,
}
```

Paths can be relative to the repo root (portable) or absolute. Tasks left
as `None` are deliberately skipped by both the snapshot collection path and
the snapshot eval path (warning logged, run continues).

### Step 7b — sanity-check the pixel pipeline (1 cell, ~30s)

Before kicking off a multi-cell eval, confirm the camera adapter actually
feeds non-degenerate pixels through ACT. After populating SNAPSHOTS, run
a 1-episode snapshot collection and check action variance:

```bash
python scripts/svf_collect_dataset.py \
    --source snapshot \
    --tasks reach_target_single --disruptions INCIDENTAL \
    --episodes-per-cell 1 --max-steps 50 \
    --output-dir /tmp/svf_pixel_smoke

python -c "
import numpy as np
shard = next(__import__('pathlib').Path('/tmp/svf_pixel_smoke').glob('snapshot__*.npz'))
data = np.load(shard)
print('action std:', data['action'].std(axis=0).round(2))
print('action range:', (data['action'].max() - data['action'].min()).round(2))
"
```

Expected: per-dim std well below the action-space half-range. If actions look uniform-distributed across the action box, the pixel adapter isn't feeding real images — re-check `cfg.env.cameras` in the snapshot payload and `peek_snapshot_cameras` output.

### Step 7c — eval against snapshot policy

```bash
python scripts/svf_eval_filter.py \
    --critic-path $CRITIC \
    --threshold-R 50.0 \
    --policy snapshot \
    --tasks reach_target_single dishwasher_close \
    --disruptions INCIDENTAL SHARED_GOAL DIRECT OBSTRUCTION RANDOM_PERTURBED \
    --episodes-per-cell 10 \
    --output-csv $RESULTS/eval_act_R50.csv
```

No more per-task invocations or `--snapshot-path` flags — the resolver
looks up each task's snapshot automatically, and cameras are auto-detected
from each snapshot's embedded cfg.

### Step 7d — for one-off testing

If you want to test a specific snapshot without editing the dict:

```bash
python scripts/svf_eval_filter.py \
    --critic-path $CRITIC \
    --policy snapshot \
    --tasks reach_target_single \
    --snapshot-override reach_target_single=/path/to/specific/snapshot.pt \
    ...
```

`--snapshot-override` is repeatable and takes precedence over `SNAPSHOTS`.

### Why eval-against-snapshot matters

It's the load-bearing measurement for Phase 2. Random-policy eval was a
stand-in: it shows "filter triggers when random is unsafe", which is
trivial. Snapshot eval shows "filter triggers on the policy we'd actually
deploy" — that's the question the deliverable has to answer.

---

## 8. Close Phase 2 — writeup + CLAUDE.md update

Phase 2 deliverable per the plan: `safety_bigym/docs/phase2_svf_results.md` showing a non-empty Pareto curve. Sketch:

```bash
cat > docs/phase2_svf_results.md <<'EOF'
# Phase 2 — Offline SVF Safety Filter Results

## Summary
SSM-only binary safety critic, trained on ${DATASET_SIZE} transitions
(random + demo sources), evaluated against the unwrapped policy on
${TASKS} × ${DISRUPTIONS}. Pareto curve at .../sweep_*.csv.

## Critic
- γ=0.99, q_max=100
- CQL α=5.0, 200k grad steps
- bellman MSE: ${first} → ${last}

## Pareto curve (reach_target_single, INCIDENTAL)
| R    | intervention_rate | residual_violation_rate |
|------|-------------------|--------------------------|
... [from sweep_*.csv] ...

## Operating point
R=${PICK} sits at the knee of the curve, with intervention_rate=${X}
and residual_violation_rate=${Y}.

## Caveats
- PFL still broken (use_pfl=False); SSM-only labels.
- Snapshot-policy eval pending Phase-0 ACT retrain.
- CQL α=5.0 single-value — full {1, 5, 10} sweep is the E2.1 follow-up.
EOF
```

Then update CLAUDE.md's Phase-status section with a Phase-2 closed bullet (mirror the Phase-0 / Phase-1 entries already there).

---

## 9. Decision points after Phase 2 closes

| Decision | If yes | If defer |
|---|---|---|
| **CQL α full sweep** ({1, 5, 10}) | ~3× compute, completes E2.1, gives a defensible α-vs-Pareto plot | Carry α=5.0 into Phase 3, revisit if Phase 4 hybrid filter is too conservative |
| **Proportional damping fallback** | Smoother task behaviour during interventions; one extra fallback class | Phase-4 work; zero-velocity is fine for Phase-2 deliverable |
| **AMASS-driven aux loss** | Tighter bound on Q at known-unsafe states | Defer; the trainer flag is wired but currently inert |
| **Move to Phase 3 immediately** | Lagrangian constrained-RL on top of the Phase-2 filter | Hold for tail-risk metrics from a longer eval |

The plan's risk register lists "Phase 1 shows no benefit from human state" as the trigger for prioritising Phases 2 and 3 — which is exactly the path you're on. After Phase 2 closes, **Phase 3** is the next high-impact step, with the option to revisit the deferred items above as fast follow-ups.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: AMASS_DATA_DIR is not set` | env var missing on shell | `export AMASS_DATA_DIR=/home/ap2322/Documents/CMU/CMU` |
| GLFW X11 warning spam | `MUJOCO_GL=glfw` on headless box | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` |
| Collection's first cell hangs | DemoStore download (first-time only) | wait — it's pulling demos to `~/.cache/...` |
| Training OOM | batch too large for your GPU | `--batch-size 256` (or 128) |
| `bellman_last >= bellman_first` | underfit / lr issue | `--lr 1e-4` and/or `--num-steps 400000` |
| `intervention_rate=1.0` everywhere | critic Q is uniformly low; either undertrained or R too strict | longer training; or sweep R lower (start the sweep at R=2) |
| `intervention_rate=0.0` everywhere | critic learned to map everything to ~q_max | suggests CQL α needs to be higher; try α=10 |
| `residual_violation_rate` >> baseline | filter intervenes but human still walks into stationary robot | expected — SSM is a *separation* metric; zero-vel can't dodge a moving human. Phase 4's retreat fallback fixes this |

---

## Quick reference — file paths after the run

- Dataset: `datasets/svf_v1/*.npz` + `manifest.json`
- Critic: `checkpoints/svf_v1.pt`
- Eval CSVs: `results/svf_v1/eval_*.csv`
- Sweep CSVs: `results/svf_v1/sweep_*.csv`
- Logs: `results/svf_v1/*.log` and `datasets/svf_v1/*.log`
- Writeup: `docs/phase2_svf_results.md`
