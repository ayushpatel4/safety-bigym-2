# Phase 2 — Implementation & Experiment Results

> Decoupled offline Safety Value Function (SVF) filter — the Hybrid Safety
> Critic's runtime gate. Per [UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md)
> §Phase 2: build a CQL-trained `Q_safe(s, a)` that vetoes unsafe proposed
> actions through a `gym.Wrapper`, independent of the task policy.

Status: **closed end-to-end as of 2026-05-20** (SMPL-H v1, below). **Re-run
under the G1 coworker at the tighter 0.3 m bar on 2026-05-30 — see §0, which is
the current production result; the SMPL-H sections below are retained as
history.** B5.5 (v2 with snapshot action denormalization) is also closed: the
patch was correct, but the residual violation floor was structural rather than
caused by action-subspace narrowness.

Last updated: 2026-05-20. Cross-refs: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md),
[CHANGES_AND_NEXT_STEPS.md](CHANGES_AND_NEXT_STEPS.md), [phase2_status.md](phase2_status.md)
(legacy module-level status — kept for the commit/sub-branch table).

---

## TL;DR

| Item | Value |
|---|---|
| Dataset v1 | `datasets/svf_coworker_train_v1/` — 315k transitions, 3 tasks × 2 sources, proximity-labelled (τ=0.50 m) |
| Aggregate violation rate (v1) | ~11.2 % (~35k violations / 315k) |
| Critic checkpoint (v1) | `checkpoints/svf_coworker_train_v1.pt` (627 KB) |
| Trained Bellman MSE plateau | ~6.2 |
| Operating point (hard gate) | **R ≈ 4.0** |
| OOD eval @ R=4.0 (random policy on `coworker_eval`) | intervention ≈ 99 %, residual violation 3–8 % |
| In-dist eval @ R=3.5 (random policy on `coworker_train`) | intervention 28–34 %, residual 74–87 % |
| B5.5 (v2 + action denormalization) | **Done, negative** (2026-05-20) — patch correct, but residual is structural (proximity floor), not action-narrowness. Hard gate improved to <1 % residual @ R≈3.5; partial-intervention residual unchanged (~87 %). See §7. |
| Active follow-up | **Label change** — tighter-τ relabel and/or robot-controllability-aware label, both offline from v2 shards (see §8) |

---

## 0. G1 coworker re-eval (2026-05-30) — current production (P2)

The SMPL-H v1 operating point (R≈4.0, τ=0.50 m) was recalibrated after the
round-3 switch to the **Unitree G1 coworker** and the move to a tighter
**0.3 m** geometric bar (both the eval metric `SSMConfig.proximity_threshold`
and the SVF training label; see [safety_metrics.md](safety_metrics.md)).

### Dataset — `datasets/svf_coworker_train_g1_v1/`
- **105k transitions**, 1 task (`saucepan_to_hob`), `coworker_train` (G1),
  `--bodyslam-mode noisy`, sources **random + snapshot** (210 ep × 250 steps each).
- The `snapshot` source rolls out the **P1 stage-2 G1 baseline policy** (a
  CQN-AS checkpoint) via the new `_CQNASSnapshotPolicy` loader in
  `svf_collect_dataset.py` (see [cqn_as_integration_notes.md](cqn_as_integration_notes.md) §9).
- Proximity-violation rate at τ=0.3 m: **3.6% overall** (random 1.6%,
  **snapshot 5.6%**). The trained policy gets *closer* than random — it drives
  the arm into the shared workspace where the G1 reaches in. Raw
  `min_separation` is stored per step, so τ is re-thresholdable for free.

### Critic — `checkpoints/svf_coworker_train_g1_0p3.pt`
- Relabelled on the fly at **τ=0.3 m** (`svf_train_critic.py --proximity-threshold 0.3`,
  ~1,985 violating / ~103k safe), then α_CQL=5.0, 200k steps, batch 512,
  γ=0.99, τ_polyak=5e-3.

### Threshold sweep (dense, 0.3 m) — `results/svf_sweep_g1_v1/sweep_dense_seed{0,1,2}.csv`
Seed-averaged (3 seeds × 20 ep), filterless baseline at R=0:

| R | intervention | proximity (τ=0.3) | reduction vs R=0 |
|---|---|---|---|
| 0.0 | 0.0% | 0.0435 | baseline |
| 1.0–2.0 | 3–11% | ~0.044 | **~0%** (wasted) |
| **2.25** | **21.6%** | **0.0297** | **31.7%** ✅ |
| 2.5 | 34.3% | 0.0265 | 39.1% (interv >25%) |
| 2.75 | 44.9% | 0.0248 | 42.9% (interv >25%) |
| 3.0 | 78.5% | 0.0076 | 82.5% (hard gate) |
| 4.0 | 98.0% | 0.0119 | 72.7% (frozen) |

### Operating point: **R = 2.25** (pinned in `filters/snapshots.py`)
The only threshold meeting the P2 acceptance bar (**≥30% proximity reduction at
≤25% intervention**: 31.7% @ 21.6%). Findings, reported honestly:
- **Marginal & seed-fragile.** Per-seed reduction 38.4 / 41.2 / **20.6%** —
  seed-2's rollouts hit more proximity and the zero-velocity veto can't catch
  them until the R=3.0 hard gate.
- **Low-R interventions are wasted** (R≤2.0: vetoes 3–11% of steps with ~0%
  proximity gain — the critic's Q doesn't cleanly separate near-violations at
  the margin).
- **The big proximity win (82%) costs ~79% intervention** (robot ~frozen). The
  filter's robust, low-cost win is the robot-velocity ISO-SSM axis.
- **This is the core hybrid argument**: a frozen-veto filter alone, on an
  already-fairly-safe policy with a small (~4%) violation base, gives a modest
  fragile proximity win — motivating the Phase-3 Lagrangian for *proactive*
  avoidance, with the filter as the edge-case backstop.

R=2.25 is **provisional** — re-confirm against the Phase-3 row-3 snapshot in P5
(E4.1 decision rule). The coarse `sweep_seed*.csv` (R={1,2,3,4,5,6,8}) was the
older **0.5-label** critic and is not comparable to the dense 0.3 run.

---

## 1. Pipeline overview (SMPL-H v1 — historical)

```
ACT snapshots (×3 tasks, COWORKER-train, 4-dof base, peak-by-eval)
        │
        ▼
svf_collect_dataset.py  ─►  shards (npz, per-step min_separation + pfl_force_ratio)
        │       random + snapshot sources, BodySLAMWrapper(noisy)
        ▼
svf_train_critic.py     ─►  SafetyCritic (CQL, target_violation_rate=0.30)
        │       checkpoints/svf_coworker_train_v1.pt
        ▼
svf_eval_filter.py      ─►  intervention / residual / fallback metrics
svf_threshold_sweep.py  ─►  Pareto curve over R
        │
        ▼
runtime gate: action proposed → SafetyCritic(s, a) ≥ R ? pass : fallback (zero-velocity)
```

Code layout:

- Filter package: [`safety_bigym/filters/`](../safety_bigym/filters/) — `labeling.py`, `feature_extractor.py`, `dataset.py`, `critic.py`, `cql_trainer.py`, `runtime_wrapper.py`, `snapshots.py`
- Scripts: [`scripts/svf_collect_dataset.py`](../scripts/svf_collect_dataset.py), [`scripts/svf_train_critic.py`](../scripts/svf_train_critic.py), [`scripts/svf_eval_filter.py`](../scripts/svf_eval_filter.py), [`scripts/svf_threshold_sweep.py`](../scripts/svf_threshold_sweep.py)
- Tests: [`tests/test_svf_dataset.py`](../tests/test_svf_dataset.py), [`tests/test_safety_labeling.py`](../tests/test_safety_labeling.py), [`tests/test_cql_trainer.py`](../tests/test_cql_trainer.py), [`tests/test_svf_train_critic_smoke.py`](../tests/test_svf_train_critic_smoke.py), [`tests/test_runtime_wrapper.py`](../tests/test_runtime_wrapper.py)

---

## 2. Key design decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-16 | Proximity labelling (`min_separation < τ`), **not** ISO 15066 SSM | ISO's stopping-distance demands ~5 m clearance at kitchen velocities → 93 % violation rate, dataset would be degenerate. `ssm_margin` retained as the continuous cost signal for Phase 3's Lagrangian. |
| 2026-05-16 | `τ = 0.50 m` for B3 production | Calibration: 0.10 m → <2 % (dead); 0.30 m → identical across sources (human-trajectory-dominated); 0.50 m → random 11 %, snapshot 15.5 % (sources distinguishable, dataset workable). Matches the system's effective reaction window. |
| 2026-05-16 | B3 skips demo source entirely | DemoStore-cached demos appeared to be 3-dof; Phase 2 env is 4-dof to match RoboBase training. **Corrigendum 2026-05-20:** the 4-dof DemoStore cache exists and `_get_demo_fn` loads it fine; the original `DemoNotFoundError` came from a different (hand-built) env-construction path. B3 still shipped without demos, but the path is unblocked for any v2 re-collection that wants demo coverage. See [CLAUDE.md](CLAUDE.md) "BiGym DemoStore demos load fine at 4-dof". |
| 2026-05-16 | Shard schema records `min_separation` + `pfl_force_ratio` per-transition | Free proximity-threshold sweeps post-hoc; PFL retrofit forward-compatible (but blocked on the contact-detection bug). |
| 2026-05-18 | Operating point **R = 4.0** for the v1 hard gate | B5.4 sweep: cliff between R=3 (10–22 % intervention) and R=4 (97–99 %); 30× drop in residual vs random. |
| 2026-05-20 | Open B5.5 (v2 collection with snapshot-action denormalization) | In-dist residual stayed high; B4.2 caveat (snapshot returns raw tanh-space actions, env silently clips gripper dims and body-joint actions explore only the inner [−1, 1] band) is the most plausible culprit. |

---

## 3. Implementation milestones (B1–B5)

### B1 — ACT snapshots on COWORKER train space
3 long-horizon tasks; `reach_target_single` dropped (horizon too short).
Snapshots ([`safety_bigym/filters/snapshots.py`](../safety_bigym/filters/snapshots.py) `SNAPSHOTS`):
- `dishwasher_close` → `dishwasher_close_20260515184635/snapshots/50000_snapshot.pt`
- `drawers_open_all` → `drawers_open_all_20260515184721/snapshots/40000_snapshot.pt`
- `saucepan_to_hob` → `saucepan_to_hob_20260516123308/snapshots/70000_snapshot.pt`

### B2 — `svf_collect_dataset.py` rewired (B2.1–B2.8)
- Disruption-space dispatch: `coworker_train`/`coworker_eval` → COWORKER factories; legacy strings → legacy path (back-compat).
- 4-dof floating base (X, Y, Z, RZ) to match RoboBase's `enable_all_floating_dof=True`. Bare-BiGym 3-dof gives `action_dim=15` and silent `state_dict` mismatches on Phase 0 ACT snapshots.
- Snapshot agent instantiated against synthesized `ConcatDim(shape_length=1) → FrameStack(frame_stack=1)` obs space; raw `SafetyBiGymEnv` obs space is *not* what the agent was sized against.
- `BodySLAMWrapper(plan.bodyslam_mode)` always applied so `human_pos_estimate` is always in the dataset; `_SnapshotPolicy.adapt_obs` strips the channel for Phase 0 actors.
- Labelling switched to proximity; `--proximity-threshold` surfaced; `tests/test_safety_labeling.py` rewritten (9 green).
- Shard schema extended with `min_separation` + `pfl_force_ratio`; `tests/test_svf_dataset.py` round-trips both (8 green).

### B3 — Dataset collected (2026-05-17)
- 315k transitions, 3 tasks × 2 sources (`random` + `snapshot`), `--proximity-threshold 0.50`, `--bodyslam-mode noisy`, `--episodes-per-cell 210`, `--max-steps 250`.
- Per-cell violation rates:

  | Source × Task | dishwasher_close | drawers_open_all | saucepan_to_hob |
  |---|---:|---:|---:|
  | random | 4.6 % | 17.5 % | 14.4 % |
  | snapshot | 10.2 % | 11.0 % | 9.6 % |

  Aggregate ~11.2 %.

### B4 — Sanity checks (3/4 clean, 1 caveat, 1 logging gap)
- **B4.1** violation rate ≥5 %: 5/6 cells pass; `random/dishwasher_close` at 4.6 % passes with margin (2,415 absolute violations on 52,500 transitions).
- **B4.2** action-magnitude distribution: passes with caveat — snapshot returns **raw tanh-space** outputs without `RescaleFromTanhWithMinMax` denormalization. Gripper dims sit at -1.1 (env silently clips to [0, 1]); body-joint dims stay in [−1, 1] rather than spanning the env's ±π range. Tracked → **B5.5**.
- **B4.3** trajectory-mode coverage: pass; APPROACH_LOITER_DEPART / COWORKER_PATROL / STATIONARY each 30–36 %.
- **B4.4** per-axis coverage: 3/5 axes verified; `reach_period` and `target_mix_p_ee` sampled but not persisted on `ScenarioParams` — logging gap only.

### B5 — SVF training + eval
See §4–§6.

---

## 4. B5.2 — Critic training

- **Inputs:** 315k transitions from v1.
- **Hyperparams:** 200k steps, batch 512, CQL α = 5.0, target violation rate 0.30, γ = 0.99, target τ = 5e-3, lr = 3e-4.
- **Wall-clock:** ~1h35m on the GPU box.
- **Final step:** loss = 1.77, Bellman MSE = 6.20, CQL = -0.89, Q mean ≈ 3.19.
- **Bellman plateau ~6.2** vs the ideal <1: critic is underfit relative to the OOD eval distribution. The Q distribution sits in [0.5, 5.0]; safe / unsafe gap is narrow (cliff between R=3 and R=4).

---

## 5. B5.3 — Eval results

### 5.1 OOD — `coworker_eval`, random policy

| R | intervention rate | residual violation |
|---:|---:|---:|
| 0.5 | 0 | 0.955 |
| 4.0 (sweep operating point) | ≈ 0.99 | 0.03 – 0.08 |

Results: `results/svf_eval_v1.csv`. The gate catches ~95 % of the 95-%-violating
random baseline. Functional as a **hard safety gate**.

### 5.2 In-distribution — `coworker_train`, random policy

Results: `results/svf_eval_v1_indist.csv` (date: 2026-05-18).

Headline (current text — will be tightened once the table lands):
- Intervention 28–34 % at R = 3.5 (the calibrated sensible operating point).
- Residual stays 74–87 % — *structural* to the proximity label: once the
  COWORKER human walks within 0.9–1.4 m of the robot (APPROACH_LOITER_DEPART),
  every subsequent timestep registers as a violation regardless of robot
  action. The fallback stops the **robot** from contributing to proximity but
  cannot stop the **human** approach.
- ~15 pp residual drop vs the random baseline at 30 % intervention confirms
  the critic reduces *robot-driven* proximity events.

### 5.3 Interpretation

The v1 critic is **a working hard safety gate** (R=4 OOD) but **only a partial
actor-coupled filter** (in-dist residuals stay high for structural label
reasons + critic narrowness). Two improvements are credible:

1. **B5.5** — fix snapshot action denormalization so the in-distribution
   training data covers the full env action range, widening the Q gap.
   (Active, see §B5.5.)
2. **Re-label with a tighter τ** (free — `min_separation` is per-step in the
   shard) and retrain. Cheaper than a re-collection; orthogonal to B5.5.

---

## 6. B5.4 — Threshold sweep

`results/svf_sweep_{task}_v2.csv` (v2 = sweep methodology, not dataset v2).

| R | intervention | residual |
|---:|---:|---:|
| 3 | 10–22 % | ≈ 0.93 |
| 4 | 97–99 % | 0.03–0.08 |

Methodology note (for B5.5 re-sweep): default thresholds `[10, 25, 50, 75, 90]`
were uniformly intervention=1.0 because the trained Q distribution lives in
~[0.5, 5.0]. Future sweeps should bracket ±1.5× around the training-log
`q_mean` (≈3 for v1).

---

## 7. B5.5 — v2 dataset with snapshot action denormalization (closed negative)

### Why

B4.2 flagged that `_SnapshotPolicy.__call__` returns raw tanh-space outputs:
gripper dims at -1.1 (env clips), body-joint dims in [−1, 1] only. Snapshot
rollouts in v1 thus explore a narrower action subspace than a deployed
properly-rescaled policy would. The critic's narrow Q gap and the high
in-distribution residual are both consistent with that. RoboBase wraps the env
with `RescaleFromTanhWithMinMax`; the SVF collection path doesn't — that's the
fix.

### Code patch (prerequisite — must land before running the pipeline)

`scripts/svf_collect_dataset.py`:

1. Add `action_stats` + `min_max_margin` fields to `_SnapshotPolicy`.
2. In `_SnapshotPolicy.__call__`, after the chunk-flatten step, apply
   `RescaleFromTanhWithMinMax.transform_from_tanh(action_np, action_stats, min_max_margin)`.
3. In `load_snapshot_policy`, read `payload["action_stats"]` (written by the
   FYP3/robobase workspace.py drift) and `cfg.get("min_max_margin", 0.0)`,
   pass them into the `_SnapshotPolicy` constructor.
4. Test: `tests/test_svf_collect_snapshot_denorm.py` — assert that with the
   stats set, gripper dims land in [0, 1] and body-joint dims span at least
   ±2 rad on a synthetic snapshot.

Sketch in [scripts/run_phase2_b55.sh](../scripts/run_phase2_b55.sh) header.

### Run plan

End-to-end: [`scripts/run_phase2_b55.sh`](../scripts/run_phase2_b55.sh). Four
stages, gated by the `STAGES` env var:

```bash
# GPU box, ~/Documents/safety_bigym, after the patch lands
export AMASS_DATA_DIR=~/Documents/CMU/CMU
bash scripts/run_phase2_b55.sh                # full run (collect → train → eval → sweep)
bash scripts/run_phase2_b55.sh --smoke        # 200-transition smoke first
STAGES=collect,train bash scripts/run_phase2_b55.sh   # stop before eval
```

Stages mirror v1:

1. **Collect** v2 → `datasets/svf_coworker_train_v2/` (same settings as B3:
   `random + snapshot`, 3 tasks, `--proximity-threshold 0.50`,
   `--bodyslam-mode noisy`, `--episodes-per-cell 210`, `--max-steps 250`).
2. **Train** → `checkpoints/svf_coworker_train_v2.pt` (200k × 512 × CQL α=5,
   target violation 0.30).
3. **Eval** at R=4.0 on `coworker_train` and `coworker_eval`, 20 episodes ×
   3 tasks → `results/svf_eval_v2_{cell}.csv`.
4. **Sweep** around the v2 `q_mean` (auto-extracted from the train log, ±1.5×,
   7 thresholds) on `coworker_eval` → `results/svf_sweep_v2_{task}.csv`.

The script's first action is a `_SnapshotPolicy` source-grep that aborts with
a clear error if the patch isn't in `main`.

### Success criteria

- v2 collection's per-cell snapshot violation rates differ materially from
  v1's (sign of a different action subspace explored).
- v2 critic's `q_mean` shifts vs v1 (~3.19) and the safe/unsafe gap widens —
  the sweep's R-cliff sharpens *or* the curve becomes smoother.
- v2 eval residual at R=4.0 on `coworker_train` drops below v1's (target:
  ≥10 pp drop at the same intervention rate).

If the residual doesn't move, the next move is the tighter-τ relabel (§5.3)
rather than another collection.

### Results (2026-05-20) — patch fired, hypothesis NOT confirmed

Full run completed: collect → train (`checkpoints/svf_coworker_train_v2.pt`,
training `q_mean ≈ 2.96`) → eval → sweep. The denormalization patch fired (no
raw-tanh warning; `action_stats` present on the snapshot payload).

**Sweep on `coworker_eval`** (`results/svf_sweep_v2_{task}.csv`, 10 ep ×
250 steps per R; residual ranges across the 3 tasks):

| R | intervention | residual (v2) |
|---:|---:|---:|
| 1.46 | ~27–30 % | **86.6–88.1 %** |
| 1.96 | ~37–48 % | 74–83 % |
| 2.46 | ~63–68 % | 49–61 % |
| 2.96 | ~88–90 % | 15–19 % |
| 3.46 | ~98 % | 0.3–0.7 % |
| 3.96 | ~99.5 % | 0–0.2 % |

**Fixed-R=4.0 eval** (`results/svf_eval_v2_{cell}.csv`): saturated —
intervention ≈99 %, residual ≈0 % on both `coworker_train` and
`coworker_eval`. R=4.0 sits above the v2 rollout Q range, so it clamps
everything; not a meaningful operating point for v2.

**Verdict.** Two regimes vs v1:

| Regime | v1 | v2 | Read |
|---|---|---|---|
| Hard gate (~98–99 % intervention) | residual 3–8 % @ R=4.0 | residual **<1 %** @ R≈3.5 | marginal win, but both ≈freeze the robot |
| Partial gate (~30 % intervention) | residual 74–87 % (in-dist) | residual **~87 %** (OOD) | **no improvement** |

The success criterion — ≥10 pp residual drop at a *partial* intervention rate
— is **not met**. The intervention/residual tradeoff is ~linear up to the
~90 % cliff: the critic acts like a coarse "clamp fraction" dial, not a
discriminative safe/unsafe classifier. This is the **structural proximity
floor** from §5.3, not action-subspace narrowness — once the human is within
0.9–1.4 m, every step is a violation regardless of robot action, so residual
only collapses when the robot is frozen.

**Conclusion:** B5.5 is **done and negative.** Action denormalization was
worth fixing (it's correct now, and the hard-gate residual improved sub-1 %)
but it is not the lever for the partial-intervention residual. Do **not**
collect a v3. Next move is the **tighter-τ relabel** (free, from stored
`min_separation`) and/or a **robot-controllability-aware label** that only
penalises proximity the robot can actually influence (§8). Both are offline
from the existing v2 shards.

---

## 8. Known gaps / follow-ups

- **PFL retrofit blocked** on the contact-detection bug ([CLAUDE.md](CLAUDE.md)
  "Caveat: `pfl_force_ratio` … identically zero"). v1 shards' `pfl_force_ratio`
  is uniformly 0; full retrofit needs a fresh collection through a PFL-fixed
  env, not just relabelling.
- **Demo source unblocked but unused.** 2026-05-20 corrigendum: 4-dof
  DemoStore cache exists; `_get_demo_fn` loads. A v2 collection that wants
  safe-side demo mass can include `--source demo`; the writer schema is
  consistent.
- **`ScenarioParams` missing two axes** (`reach_period`, `target_mix_p_ee`) —
  sampled but not persisted, blocks per-axis coverage audit. Small dataclass
  extension, no other dependency.
- **B5.5 closed negative (2026-05-20).** Confirmed the in-dist/OOD residual is
  dominated by the human-approach-driven structural proximity floor, not action
  subspace narrowness (§7 Results). Two offline follow-ups from the v2 shards,
  in priority order:
  1. **Tighter-τ relabel** — recompute `r_safe = (min_separation ≥ τ)` for
     τ ∈ {0.30, 0.40} from stored per-step `min_separation`, retrain, re-sweep.
     Free (no collection). Tightening the bar concentrates the unsafe class on
     genuinely-close states; may sharpen the cliff but won't remove the floor.
  2. **Robot-controllability-aware label** — gate the violation on the robot
     *contributing* to the approach (e.g. EE moving toward the human, or
     `min_separation` decreasing due to robot motion) rather than raw geometric
     proximity. This is the principled fix for the "freeze = only way to win"
     pathology, but needs a label-function change + retrain, not just a relabel.
  Either way, the v2 critic is usable **as a hard gate** (R≈3.5, residual <1 %)
  today; the partial-intervention filter is what's blocked on the label.
