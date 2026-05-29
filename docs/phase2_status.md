# Phase 2 — Status

> **Historical reference.** This was the module-level Phase 2 status snapshot before the
> COWORKER dataset collection, training, eval, and B5.5 follow-up closed. Do
> not use this as the current Phase 2 run guide. Current result/writeup:
> [phase2_results.md](phase2_results.md). Live next action:
> [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

**Goal** (from [.claude/HYBRID_SAFETY_CRITIC_PLAN.md §Phase 2](../.claude/HYBRID_SAFETY_CRITIC_PLAN.md)):
build the runtime safety filter as a standalone, decoupled module — a CQL-trained
`Q_safe(s, a)` that vetoes unsafe proposed actions through a `gym.Wrapper`,
giving ISO-15066 hard guarantees independent of the task policy.

**Current state:** all code + tests + smoke pipeline complete. GPU work (full
collection / training / eval / Pareto sweep) remains. Phase-0 ACT retrain is
done — snapshots are usable end-to-end.

---

## ✅ Done

### Branches and commits

Phase branch: `phase2` (off main). Three sub-branches stacked, plus follow-up commits:

| Sub-branch | Commit | Title |
|---|---|---|
| `phase-2-dataset` | `aaa5e0a` | labeller, feature extractor, sharded dataset, collector |
| `phase-2-dataset` | `876b3fd` | demo + snapshot sources wired |
| `phase-2-dataset` | `af0b9b9` | demo source skips gracefully when DemoStore has no match |
| `phase-2-critic-training` | `866bc71` | bounded MLP critic, CQL trainer, train script |
| `phase-2-runtime-wrapper` | `57a33da` | fallback, runtime wrapper, threshold sweep, eval/sweep scripts |
| `phase-2-runtime-wrapper` | `f0d93b6` | per-task SNAPSHOTS dict refactor |
| `phase-2-runtime-wrapper` | `cf65410` | docs: Phase 2 status report |
| (merged via PR #7) | `494238f` | merge `phase-2-runtime-wrapper` |
| (post-merge) | `728e0ef` | camera-correct snapshot policy adapter |

All Phase-2 work is merged to `main` via PR #7 (`494238f`). The camera adapter (`728e0ef`) landed afterwards on `main` directly.

### Modules

```
safety_bigym/filters/
├── __init__.py                  re-exports
├── labeling.py                  r_safe = 0 if ssm_violation else 1 (use_pfl flag, currently SSM-only)
├── feature_extractor.py         CriticFeatureSpec + make_critic_input (no pixels)
├── dataset.py                   SafetyTransitionDataset + TransitionShardWriter + WeightedRandomSampler
├── critic.py                    Bounded-output MLP, q_max = 1/(1-γ) = 100, target + Polyak
├── cql_trainer.py               Bellman MSE + α·CQL + optional aux loss (inert at α_aux=0)
├── aux_unsafe_provider.py       Stub Protocol + EmptyAuxProvider (deferred)
├── fallback.py                  Fallback ABC + ZeroVelocityFallback + FallbackRegistry
├── runtime_wrapper.py           SafetyFilterWrapper(gym.Wrapper)
├── threshold_sweep.py           evaluate_threshold + sweep_thresholds (Pareto util)
└── snapshots.py                 Per-task SNAPSHOTS dict + resolve_snapshot()
```

### Scripts (all argparse-driven, all have `--smoke`)

| Script | Purpose |
|---|---|
| `scripts/svf_collect_dataset.py` | random + demo + snapshot sources → sharded `.npz` |
| `scripts/svf_train_critic.py` | offline CQL training; saves portable `critic.pt` |
| `scripts/svf_eval_filter.py` | wraps a policy with the filter, reports per-cell metrics CSV |
| `scripts/svf_threshold_sweep.py` | Pareto curve over R values, CSV out |

### Configs

| Config | Purpose |
|---|---|
| `cfgs/filter/dataset.yaml` | Operator-facing collection defaults |
| `cfgs/filter/safety_critic.yaml` | Critic + CQL trainer hyperparams |
| `cfgs/filter/runtime.yaml` | Filter R + sweep grid |
| `cfgs/launch/svf_filter_train.yaml` | Composite launch (W&B tags) |

> Scripts are still argparse-driven; YAMLs document canonical defaults so the
> Hydra promotion will be mechanical.

### Tests — 94 green

| File | Cases | Coverage |
|---|---|---|
| `test_safety_labeling.py` | 7 | SSM-only label, PFL flag, error paths |
| `test_critic_features.py` | 9 | Spec construction, no-pixels, batch dims, round-trip |
| `test_svf_dataset.py` | 8 | Schema, oversampler, multi-shard concat |
| `test_svf_collect_smoke.py` | 15 | E2E for random + demo + snapshot; rgb-key emission with/without cameras; peek_snapshot_cameras round-trip |
| `test_safety_critic.py` | 10 | Bounds, gradients, target, Polyak |
| `test_cql_trainer.py` | 7 | Bellman decrease, α scaling, aux gating |
| `test_svf_train_critic_smoke.py` | 3 | Training script E2E |
| `test_safety_filter_fallback.py` | 5 | ZeroVelocityFallback + registry |
| `test_safety_filter_wrapper.py` | 6 | Pass-through, fallback, info logging |
| `test_threshold_sweep.py` | 4 | Pareto monotonicity invariant |
| `test_svf_eval_and_sweep_smoke.py` | 2 | Eval + sweep scripts E2E |
| `test_snapshots_resolver.py` | 9 | resolver: None, missing, override, relative paths |
| `test_svf_snapshot_pixel_adapter.py` | 9 | Single/multi-camera shape, HWC↔CHW, missing-camera raises, no-pixels mode |

Local CPU full sweep: ~38s.

### Smoke pipeline (validated on swirl 2026-05-07)

```bash
export AMASS_DATA_DIR=/home/ap2322/Documents/CMU/CMU
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl  # headless GPU box

SMOKE=/tmp/svf_smoke
rm -rf $SMOKE && mkdir -p $SMOKE

python scripts/svf_collect_dataset.py --smoke --output-dir $SMOKE
python scripts/svf_train_critic.py     --smoke --dataset-dir $SMOKE --output $SMOKE/critic.pt
python scripts/svf_eval_filter.py      --smoke --critic-path $SMOKE/critic.pt
python scripts/svf_threshold_sweep.py  --smoke --critic-path $SMOKE/critic.pt
```

End-to-end ~75s. Confirmed working on swirl.

### Documentation

- `docs/phase2_gpu_handoff.md` — top-to-bottom GPU operator guide
- `docs/phase2_status.md` — this file
- `cfgs/launch/svf_filter_train.yaml` + `cfgs/filter/*.yaml` — operator defaults

---

## 🔜 Left to do

### Blocking the Phase 2 deliverable

#### 1. ~~Push branches~~ — done

All Phase 2 code merged via PR #7 (`494238f`) plus the post-merge camera adapter (`728e0ef`). See the commits table above.

#### 2. Populate SNAPSHOTS dict

Phase-0 ACT retrain is done. Pick the W&B `pretrain_eval/episode_success` peak per task and update [safety_bigym/filters/snapshots.py](../safety_bigym/filters/snapshots.py):

```python
SNAPSHOTS: Dict[str, Optional[str]] = {
    "reach_target_single":   "exp_local/act_safety/<run_dir>/snapshots/<peak>_snapshot.pt",
    "dishwasher_close":      "exp_local/act_safety/<run_dir>/snapshots/<peak>_snapshot.pt",
    "dishwasher_load_plates": None,  # leave None for tasks without snapshots
    "saucepan_to_hob":       None,
}
```

Tasks with `None` are deliberately skipped by both `--source snapshot` and `--policy snapshot`.

Camera config (`cfg.env.cameras` + `cfg.visual_observation_shape`) is **auto-detected** from each snapshot's embedded cfg by `peek_snapshot_cameras` — no extra flag needed. The env is rebuilt per-task with the rgb keys the actor's encoder expects.

#### 3. Full dataset collection (~2-3 hours, GPU)

```bash
DATASET=/home/ap2322/Documents/safety_bigym/datasets/svf_v1
python scripts/svf_collect_dataset.py \
    --source random --source demo --source snapshot \
    --tasks reach_target_single dishwasher_close \
    --disruptions INCIDENTAL SHARED_GOAL DIRECT OBSTRUCTION RANDOM_PERTURBED \
    --episodes-per-cell 100 --max-steps 300 --demos-per-task 30 \
    --bodyslam-mode oracle --output-dir $DATASET --seed 0
```

Target: ~310k transitions (close to the plan's 500k spec).

#### 4. Full CQL training (~30-60 min, single GPU)

```bash
python scripts/svf_train_critic.py \
    --dataset-dir $DATASET --output checkpoints/svf_v1.pt \
    --num-steps 200000 --batch-size 512 --cql-alpha 5.0 \
    --device cuda --seed 0
```

Verify `bellman_last < bellman_first` in the saved payload's `training` field.

#### 5. Full eval against ACT snapshot policy

```bash
python scripts/svf_eval_filter.py \
    --critic-path checkpoints/svf_v1.pt \
    --threshold-R 50.0 --policy snapshot \
    --tasks reach_target_single dishwasher_close \
    --disruptions INCIDENTAL SHARED_GOAL DIRECT OBSTRUCTION RANDOM_PERTURBED \
    --episodes-per-cell 10 \
    --output-csv results/svf_v1/eval_act_R50.csv
```

This is the load-bearing measurement — random-policy eval was a stand-in.

**Pixel-pipeline sanity check** (run once before the full eval): the camera adapter passes actor outputs through real pixels, but it's worth confirming the action distribution differs from random before committing 1+ hour to the full grid. After a 1-cell smoke collection:

```bash
python -c "
import numpy as np
snap = np.load('<output_dir>/snapshot__<task>__<disruption>__0000.npz')
print('action std:', snap['action'].std(axis=0).round(2))
"
# Expected: per-dim std well below the uniform-distribution std for the
# corresponding action_space range. Uniform-like std means pixels aren't
# reaching the encoder — see test_svf_snapshot_pixel_adapter.py for the
# invariants the adapter enforces.
```

#### 6. Threshold sweep — the Pareto curve

```bash
for TASK in reach_target_single dishwasher_close; do
  python scripts/svf_threshold_sweep.py \
      --critic-path checkpoints/svf_v1.pt \
      --task $TASK --disruption INCIDENTAL --policy snapshot \
      --thresholds 5 10 25 50 75 90 95 \
      --episodes-per-R 10 \
      --output-csv results/svf_v1/sweep_${TASK}.csv
done
```

#### 7. Phase 2 writeup

Create `safety_bigym/docs/phase2_svf_results.md` with the Pareto curve + intervention rate × residual violation rate table per task. Closes Phase 2 per the master plan.

#### 8. Update CLAUDE.md

Add a Phase-2 closed-state bullet alongside the Phase-0 / Phase-1 entries already there. Update the "Phase-0 retrain prep" gotcha to reflect retrain is complete.

---

### Carried-forward (non-blocking) items

These are deferred from the approved plan, all explicitly accepted as v1 scope cuts:

| Item | Why deferred | When to revisit |
|---|---|---|
| **PFL contact-detection bug fix** | Outside Phase 2's scope; root cause is in BiGym/mojo runtime robot attachment. Labeller has a `use_pfl` flag wired but inert. | When someone unblocks the contact bug; flip `use_pfl=True` and re-collect |
| **AMASS-driven aux loss** | Adds AMASS coupling at training time; risk of muddy CQL calibration | After first GPU Pareto curve looks sane; treat as ablation |
| **Full CQL α∈{1, 5, 10} sweep (E2.1)** | ~3× compute; single α=5.0 gives one usable Pareto curve | After v1 results land; treat as Phase 2 fast-follow |
| **Proportional damping fallback** | Phase 4 work per the master plan; zero-vel is fine for v1 | Phase 4 / hybrid-deployment phase |
| **Hydra promotion of scripts** | argparse is sufficient for v1; YAMLs document defaults | When workflow demands it (e.g. for sweep config composition) |

---

### Known limitations carried into Phase 3

- **Zero-velocity fallback may behave badly mid-trajectory** on dishwasher tasks. Surface in the eval writeup; Phase 4 is when proportional damping lands.
- **Random-policy "noise" during dataset collection** can produce huge SSM `Required:` distances in the warning logs (because `S_p = v·T_r + ...` blows up at high human velocities). Cosmetic noise, not a bug. Goes away with snapshot-policy collection.
- **`dishwasher_close` has no recorded BiGym demos** (DemoStore lookup fails). The collector skips with a warning. That task contributes only `random` and (now) `snapshot` transitions to the dataset.
- **`frame_stack > 1` is not supported** by the snapshot policy adapter. `load_snapshot_policy` raises `NotImplementedError` early. ACT's default launch uses `frame_stack=1` so this isn't a blocker; the adapter would need a per-key deque to support stacking.

---

## Quick reference — file paths

| Type | Path |
|---|---|
| Dataset (after §3) | `datasets/svf_v1/*.npz` + `manifest.json` |
| Critic checkpoint (after §4) | `checkpoints/svf_v1.pt` |
| Eval CSVs (after §5) | `results/svf_v1/eval_*.csv` |
| Pareto CSVs (after §6) | `results/svf_v1/sweep_*.csv` |
| Final writeup (after §7) | `docs/phase2_svf_results.md` |
| Plan (authoritative) | `.claude/plans/read-claude-md-and-claude-hybrid-safety-ethereal-hummingbird.md` |
| Master plan | `.claude/HYBRID_SAFETY_CRITIC_PLAN.md` |
| GPU operator guide | `docs/phase2_gpu_handoff.md` |
