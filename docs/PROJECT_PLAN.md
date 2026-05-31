# Detailed Project Plan: Hybrid Safety Architecture for `safety_bigym`

## Overview

This plan supersedes `HYBRID_SAFETY_CRITIC_PLAN.md` and the earlier
`UPDATED_PROJECT_PLAN.md`. It incorporates all completed work through
the Phase 2 SVF training, the CQN-AS adapter integration, the G1
coworker swap, and the Phase 3 P3.0/P3.1 smoke validation. It pairs
with `IMPLEMENTATION_STATUS.md` (high-level overview) and `report.tex`
(the deliverable). This document is the granular reference: every
pending task names files, interfaces, acceptance criteria, and
decision rules.

The hybrid architecture combines two mechanisms: a value-based
Lagrangian constrained-RL policy on top of the demo-driven CQN-AS
backbone that internalises safety during training, and a frozen
offline-trained SVF safety filter that vetoes catastrophic actions
at runtime. The policy handles smooth safe behaviour in normal
operation; the filter catches edge cases and provides the runtime
guarantees needed for ISO 15066 compliance.

---

## Project Status Summary

| Phase | Status | Key Outcome |
|---|---|---|
| Phase 0 | **COMPLETE** | Collision-channel fix, SSM velocity fix (mocap pelvis), eval regression fix, ACT retrain on 4 tasks |
| Phase 1 wrapper | **COMPLETE** | `BodySLAMWrapper` with off / oracle / noisy modes, calibrated against BodySLAM++ characteristics |
| Phase 1 E1.1 | **COMPLETE (negative)** | BC obs-ablation: oracle doesn't help under pure BC |
| Phase 2 (SMPL-H) | **COMPLETE** | SVF α_CQL=5.0, R=4.0 — **did not transfer to G1** |
| Phase 2 (G1) | **✅ P2 CLOSED (2026-05-30)** | `svf_coworker_train_g1_0p3.pt` on `noisy`+G1, proximity τ=0.3 m. Dense sweep (R=0 baseline + knee) done → **R=2.25** (`snapshots.py`): meets acceptance (31.7% proximity reduction @ 21.6% intervention), marginal/seed-fragile, provisional pending P5. Write-up: `phase2_results.md` §0 |
| CQN-AS adapter | **COMPLETE** | 8 bugs documented and fixed; demo conversion + action-stat sharing + per-env-step cost path validated |
| G1 coworker swap + **P1 curriculum** | **✅ DONE (2026-05-30)** | Curriculum ran; stage-2 G1 baseline snapshot in hand (row-1 reference + P3/P5 warm-start) |
| Phase 3 code | **CODE COMPLETE** | B-value-mean Lagrangian agent; P3.0/P3.1 smokes pass. **All 3 E3.1 cost forms wired (2026-05-30)**: continuous / binary (`env.safety.cost_form`) / fixed (`add_violation_penalty`) |
| Phase 3 experiments | **PENDING (launch-ready)** | E3.1 (cost-signal form — launcher built), E3.2 (budget Pareto), E3.6 (obs) — **P3, P4, P7 below** |
| Phase 4 harness | **✅ DONE (2026-05-30)** | `benchmark_policy.py` built, 8 unit tests, validated on real CQN-AS snapshot (± filter). Docs: `docs/benchmark_harness.md` — **P6 below** |
| Phase 4 headline | **TOOLING READY** | E4.1 eval driver `run_e4_1_headline.sh` + LaTeX aggregator `aggregate_e4_1.py` built; E4.3 internalisation hook in `train_cqn_as` (`filter_passive`). Rows 1/4 runnable now; rest await P3 d_knee — **P5, P8 below** |
| Phase 5 evaluation | **NOT STARTED** | E5.1 tail risk + E5.2 OOD generalisation — **P10 below** |

---

## Round-3 Decisions (recap, locked)

1. **Coworker: Unitree G1 only.** No SMPL-H pipeline. The
   `HumanConfig.human_model` selector still routes everything
   through a single switch, but the only headline embodiment is G1.
   SMPL-H is future work (§fw:embodiment of the report).
2. **Lagrangian architecture: B-value-mean.** Dual Q-functions on
   reward and mean cost; action selection
   $\mathrm{argmax}_a [Q_r(s,a) - \lambda \cdot Q_c(s,a)]$. B-value-CVaR
   is future work (§fw:cvar).
3. **Primary task: `saucepan_to_hob`.** Optional secondary:
   `drawers_open_all`.
4. **External baseline: WCSAC** with Safety-Gym Lagrangian fallback
   if reimplementation fails the matching gate.
5. **Page limit: 60 A4 content pages** (between contents and
   references, exclusive). Limit, not target.

---

## Perception Mode Policy (oracle vs. noisy)

The `BodySLAMWrapper` exposes three modes — `off`, `oracle`, `noisy`
— for the `human_pos_estimate` channel. The mode used during
training and during evaluation materially affects what each
experiment is testing, so we adopt this per-experiment policy
(cross-referenced from §setup:perception-mode of the report):

| Experiment | Purpose | Train | Eval |
|---|---|---|---|
| P1 — base-policy curriculum | unconstrained baseline reference | `oracle` | `oracle` |
| P2 — Phase 2 SVF re-eval | filter operating point | (already `noisy`)¹ | `noisy` |
| P3 — E3.1 cost-signal | methodological architecture comparison | `oracle` | `oracle` |
| P4 — E3.2 budget Pareto | operating-point selection | `oracle` | `oracle` |
| **P5 — E4.1 headline** | does the hybrid work? | **`oracle`** (policy) | **`noisy`** — whole table (filter is noisy-native; see ² + Rationale) |
| P7 — E3.6 obs-channel ablation | **measure the perception gap** | `oracle` | `off / oracle / noisy` (the sweep) |
| P10 — E5.1 tail risk | post-hoc on P5 rolls | — | — |
| P10 — E5.2 OOD | wider disruption band | `oracle` | `oracle` on `coworker_eval` |

¹ The Phase 2 SVF dataset was collected with `bodyslam=noisy`
**by design** — the filter is a runtime safety guarantee, so its
training distribution must match its deployment distribution.

² **P5 eval moved to `noisy` (2026-05-30, empirically forced).** The
original plan evaluated the headline on `oracle` + a single `noisy`
diagnostic. But running the SVF filter on `oracle` makes its
Q-values collapse (`mean_q ≈ 0.016 ≪ R=2.25`) → **100% intervention,
robot frozen, success 0.78 → 0.0** (results/e4_1/..._190001). The
filter is trained on `noisy`; oracle obs is out-of-distribution for
the critic, exactly the CQL-pessimism-on-OOD-observations failure
this policy warns about. So the **entire E4.1 table is now evaluated
on `noisy`** (`OBS_MODE=noisy`, the driver default): the filter
operates in-distribution and the comparison stays apples-to-apples
(the policy's `oracle`→`noisy` degradation is identical across rows,
so it cancels). `OBS_MODE=oracle` is retained as a policy-only
clean-perception reference (rows 1–3); the filter rows are
meaningless there and that fact is itself a reportable result.

**Rationale.** The policy still *trains* on `oracle` to isolate the
architecture comparison (no "noisy training regularises" confound).
The headline *eval* is `noisy` because that is the only condition in
which the runtime filter is in-distribution — and because it is also
the realistic sim-to-real condition. The dedicated perception-gap
experiment is E3.6 (P7), which sweeps all three modes.

**The Phase 2 SVF is the one exception: it trains and evaluates
on `noisy`.** The asymmetry is deliberate. The policy is a soft
component — it's optimised to do well in expectation, and small
perception perturbations at deployment degrade its performance
gracefully (and we measure that degradation explicitly). The
runtime filter is a hard component — its job is to provide a
deployment-time veto guarantee. A safety mechanism calibrated to
observation conditions other than the ones it operates under is
not providing the guarantee it claims to: CQL's pessimism bounds
action-conditional error, not observation-conditional error.
Train-deploy distribution match is therefore load-bearing for the
filter in a way it is not for the policy. The harness finding on
the existing SVF checkpoint (`mean_q_value ≈ 0.31 ≪ R=4.0` on G1
→ ~100% intervention) is the empirical confirmation of exactly
this: a filter trained on one distribution does not transfer
cleanly to another, even when both are noisy in the same way.

**Implementation note for the coding agent.** Every long-training
launch CLI for P1, P3, P4, and P5 row 2 must pass `bodyslam=oracle`
explicitly (do not rely on a Hydra default that might drift). The
P7 E3.6 launches are the only ones that train with
`bodyslam=oracle` and then sweep three eval modes via the
`benchmark_policy.py` harness. The P2 SVF re-eval inherits its
`noisy` setting from the existing dataset and must not be changed
without retraining. **P5's policies also train on `oracle`, but the
E4.1 *eval* is `noisy`** (`run_e4_1_headline.sh` defaults
`OBS_MODE=noisy`) because the filter is noisy-native — see footnote ².

---

## Phase 0: Preparation and Baselines — COMPLETE

(Carried forward from the original plan; summary only.)

### What was done

- **Collision-channel fix.** The SMPL-H human (now replaced by G1)
  and scene geometry shared the same MuJoCo collision channel,
  producing 220 kN spurious forces and crashing the simulator
  within ~1s. Fixed by cross-pairing bits (coworker emits bit 1,
  accepts bit 2; H1 and scene the reverse). The fix applies
  unchanged to the G1 swap.
- **SSM velocity fix.** `data.cvel` reflected the implicit velocity
  of the human freejoint being teleported (~120 m/s phantom
  velocity), inflating required separation to ~18m. Resolved by
  capping at `SSMConfig.v_h_max = 1.6 m/s` first, then by
  converting the coworker pelvis to a `mocap="true"` body.
- **Eval regression fix.** Diffusion Policy snapshots producing
  0% task success at eval — `diffusers.EMAModel.shadow_params` not
  in `state_dict()`. Fixed; legacy-snapshot fallback path included.
  (Not in the critical path now since we use CQN-AS, but the fix
  is still in the codebase.)
- **ACT retrain.** Snapshots for `reach_target_single`,
  `dishwasher_close`, `drawers_open_all`, `saucepan_to_hob`. These
  feed Phase 2's training-data mixture and the Phase 0 unconstrained
  baseline reference.

### Deliverable

Working ACT snapshots × 4 tasks, physically stable simulator,
correct SSM computation. All subsequent phases build on this.

---

## Phase 1: Mock BodySLAM++ Observation Wrapper — COMPLETE

### What was built

- **`BodySLAMWrapper`** (`safety_bigym/perception/bodyslam_wrapper.py`)
  as a `gym.Wrapper` injecting `human_pos_estimate` (6D: position
  + occluded flag + staleness counter + confidence) under three
  modes: `off`, `oracle`, `noisy`. Hydra-configurable.
- **Noise pipeline (noisy mode):**
  - Ornstein-Uhlenbeck temporally-correlated position noise (α =
    0.9, σ = 0.05 m, stationary std ≈ 0.115 m), calibrated against
    BodySLAM++'s reported MPJPE characteristics.
  - 3-step latency buffer (~60 ms at 50 Hz, matching the 15+ FPS
    perception clock).
  - Stochastic tracking dropout (p = 0.02 / step); OU continues
    updating during dropout to avoid recovery discontinuity.
  - Optional ray-cast occlusion (opt-in: `bodyslam.use_occlusion =
    true`).
- **AMASS-driven demo replay** via `AMASSDemoPositionProvider` so BC
  pretraining sees realistic human-state without train→eval channel
  shift.
- **19 tests passing** covering OU statistics, latency, dropout
  recovery, confidence, demo replay, factory integration.

### E1.1 — BC Observation Ablation — COMPLETE (Negative)

Reported in §results:obs-bc of the report. ACT, 4 tasks, off /
oracle / noisy, 6 disruption types, n = 10 episodes per cell. **No
cell clears the 20% bar.** Most striking: `saucepan_to_hob` oracle
improves task success (0.22 → 0.58) but worsens SSM violations
(0.135 → 0.203) — policy uses human state for task completion, not
safety, because there's no safety reward gradient under BC.

**E1.4 (RL reward-on pilot) is DEPRECATED** — the project has moved
from DrQ-V2+ to CQN-AS, and the channel question is now answered by
E3.6 (obs-channel under the constrained policy) instead. E1.4 code
is left in the repo but not run.

### Deliverable

`BodySLAMWrapper` + the E1.1 negative result + the E3.6 follow-up
experiment design.

---

## Phase 2: Offline SVF Safety Filter — COMPLETE (G1 re-eval done 2026-05-30)

> **P2 closed (2026-05-30).** Recollected on the G1 coworker, retrained at the
> 0.3 m label, dense threshold sweep run. **Operating point R = 2.25** on
> `checkpoints/svf_coworker_train_g1_0p3.pt` — the only threshold meeting the
> acceptance bar (≥30% proximity reduction at ≤25% intervention: 31.7% @ 21.6%),
> pinned in `filters/snapshots.py`. Marginal/seed-fragile; provisional pending
> P5 re-confirmation. Full write-up: [phase2_results.md](phase2_results.md) §0.
> The R=4.0 reference below is the superseded SMPL-H operating point.

### What was built

11 modules under `safety_bigym/filters/`:

| Module | Purpose |
|---|---|
| `labeling.py` | `r_safe = 1 - 1[proximity_violation]`; PFL flag wired but inert |
| `feature_extractor.py` | `CriticFeatureSpec`, `make_critic_input` — proprioception + BodySLAM++ + action, no pixels |
| `dataset.py` | `SafetyTransitionDataset`, `TransitionShardWriter`, `WeightedRandomSampler` for violation oversampling (1:1 target) |
| `critic.py` | Bounded-output MLP [256, 256, 256], q_max = 1/(1-γ) = 100, target network + Polyak (τ = 0.005) |
| `cql_trainer.py` | Bellman MSE + α · CQL regulariser; auxiliary-loss hook (inert) |
| `fallback.py` | `ZeroVelocityFallback`, `FallbackRegistry` |
| `runtime_wrapper.py` | `SafetyFilterWrapper(gym.Wrapper)` — vetoes action if Q_safe(s, a) < R |
| `threshold_sweep.py` | `evaluate_threshold`, `sweep_thresholds` — Pareto trace |
| `snapshots.py` | Per-task `SNAPSHOTS` dict + `resolve_snapshot()` |

Four CLI scripts with `--smoke`: collect, train, eval, sweep.
80 tests passing across 12 test files (~38s on CPU). End-to-end
smoke pipeline runs ~75s.

### What was achieved

- **Dataset:** ~310k transitions, three sources (~30% random,
  ~30% BiGym demos, ~40% Phase 0 ACT snapshots). Violation rate in
  the labelled subset balanced to 1:1 via sampler.
- **Training:** α_CQL = 5.0 chosen as headline; Bellman loss
  converged within 200k steps; Q-value distributions inspected and
  sane.
- **Pareto sweep:** R = 4.0 identified as the knee for α_CQL = 5.0
  on the previous SMPL-H coworker distribution.
- **Snapshot:** `checkpoints/svf_coworker_train_v1.pt`.

### P2 — re-eval + retrain under G1 — ✅ CLOSED (2026-05-30)

**Authoritative write-up: `phase2_results.md` §0.** The diagnosis held (old
checkpoint over-fires on G1), so we recollected on `coworker_train`+`noisy`
(random + snapshot, 105k transitions) and retrained at **τ=0.3 m** →
`svf_coworker_train_g1_0p3.pt`, then ran the **dense 0.3 m sweep** (R=0 baseline
+ fine grid around the knee, 3 seeds × 20 ep, `sweep_dense_seed{0,1,2}.csv`).

**Seed-averaged dense sweep** (filterless baseline at R=0):

| R | intervention | proximity (τ=0.3) | reduction vs R=0 |
|---|---|---|---|
| 0.0 | 0% | 0.0435 | baseline |
| **2.25** | **21.6%** | **0.0297** | **31.7%** ✅ |
| 2.5 | 34.3% | 0.0265 | 39.1% (interv >25%) |
| 3.0 | 78.5% | 0.0076 | 82.5% (hard gate, ~frozen) |

**Operating point R = 2.25** — pinned in `snapshots.py::SVF_FILTER_THRESHOLD_R`
(the launchers/driver read it as the single source of truth). It is the **only**
threshold meeting the P2 bar (**≥30% reduction at ≤25% intervention**: 31.7% @
21.6%), and the acceptance bar **IS met** — but **marginally and seed-fragile**
(per-seed 38.4 / 41.2 / **20.6**%). Low-R interventions (≤2.0) are wasted (~0%
gain); the big 82% proximity win only arrives at the R=3.0 hard gate (~79%
intervention, robot ~frozen).

**Headline finding — the filter is axis-asymmetric.** Its robust, low-cost win
is the robot-velocity-driven **ISO-SSM** axis; on the thesis-primary **geometric
proximity** axis it gives only a marginal win at usable intervention, because a
veto→zero-velocity filter cannot stop the G1 coworker — which actively reaches
into the workspace — from approaching a *stationary* robot. This is the core
empirical argument for the **hybrid**: filter = edge-case backstop, Lagrangian
policy = proactive avoidance. R=2.25 is provisional — re-confirm against the
Phase-3 row-3 snapshot in P5 (E4.1 decision rule). The coarse
`sweep_seed{0,1,2}.csv` (R={1,2,3,4,5,6,8}) was the OLD 0.5-label critic and is
**not comparable** to the dense run.

---

#### Historical context (pre-retrain rationale, carried forward)

The R = 4.0 operating point was calibrated on the SMPL-H coworker
distribution. The G1 coworker has different geometric extent
(larger forearm capsules) and movement dynamics (PD gains kp=200,
kv=20). **The operating point must be re-verified before any
downstream experiment uses the filter.**

**Update (2026-05-30): the P6 harness has already given us empirical
evidence the filter does not transfer cleanly.** Running the existing
`svf_coworker_train_v1.pt` on a G1-coworker rollout produces
`mean_q_value ≈ 0.31 ≪ R = 4.0`, leading to ~100% filter
intervention — the filter rejects essentially every proposed
action. This is exactly the "Intervention rate is 100% everywhere"
branch of the decision rule below, which means **the expected path
is now (1) confirm with a brief sweep, then (2) retrain the SVF on
a freshly-collected G1 + noisy dataset.** The 100%-intervention
finding rules out a simple threshold re-tune.

#### Tasks

1. **Sweep R on the existing checkpoint** (~1 GPU-h, was 3 h before
   we had the harness telling us the answer). Just confirms the
   harness diagnosis at multiple R values: sweep
   `R ∈ {0.5, 1, 2, 3, 4}` on `disruption=coworker_train` with the
   G1 coworker. Expected outcome: no R gives an acceptable
   intervention-rate / safety trade-off — the filter has to be
   retrained.
2. **Recollect dataset on G1 + noisy** (~3 GPU-h). Re-run
   `python -m safety_bigym.filters.dataset.collect_dataset` with
   `disruption=coworker_train`, `bodyslam=noisy`, and the same
   three policy sources (random, BiGym demos, Phase 0 ACT). The
   label function is unchanged — `proximity_violation` is still
   computed against MuJoCo's `data.geom_xpos` for the current
   bodies, so it re-labels correctly against G1 capsules.
3. **Retrain SVF** (~1 GPU-h). Same hyperparameters as the existing
   checkpoint: 3-layer MLP [256, 256, 256], α_CQL = 5.0, Polyak
   τ = 0.005, 200k steps, batch 512, violation-balanced sampler.
4. **Re-sweep R on the retrained checkpoint** (~1 GPU-h). Identify
   the new knee. Document in `safety_bigym/filters/snapshots.py`.

#### Acceptance criteria

- After retraining: Pareto curve shows a clear knee with
  intervention rate ≤ 25% and `ep_proximity_violation_rate`
  reduction ≥ 30% relative to the filterless baseline. If not,
  the issue is deeper than dataset shift (e.g. critic capacity,
  network architecture) and requires investigation.
- The new R operating point is documented in
  `safety_bigym/filters/snapshots.py` and consumed by the headline
  E4.1 row 4 and row 5.
- Filter snapshot file naming: `svf_coworker_train_g1_0p3.pt`
  (preserve `v1` for SMPL-H reproducibility).

#### Decision rule

| Sweep outcome | Action |
|---|---|
| Knee still at R = 4.0 (within ±0.5) — unlikely given the harness finding | Use existing checkpoint as-is |
| Knee shifts but exists at some R ≤ 4 — possible | Use existing checkpoint with updated R, skip retrain |
| Knee disappears / curve is flat — likely | Retrain SVF on G1 data (tasks 2–4 above) |
| Intervention rate is 100% everywhere — already observed by harness | Retrain SVF on G1 data (tasks 2–4 above) |

#### Compute budget

**~6 GPU-hours expected** (sweep ~1 h + recollect ~3 h + retrain
~1 h + re-sweep ~1 h), revised up from the original ~3-h
optimistic estimate after the harness finding ruled out a
threshold-only re-tune. Best case (existing checkpoint somehow
works at some lower R): ~1 GPU-h.

---

## Phase 2.5: CQN-AS Adapter Integration — COMPLETE

Documented in §impl:integration of the report and in
`docs/cqn_as_integration_notes.md`. Eight non-trivial bugs fixed
during integration; the most informative four:

| Bug | Diagnosis | Fix |
|---|---|---|
| Empty `human_pos_estimate` in cached demos | BiGym demos recorded without humans | `mode=demo_replay` injects AMASS clip during conversion |
| Demo–live action-stat mismatch | CQN-AS bins normalised against different stats per source | Override `self._action_stats` with demo-derived stats at construction |
| CUDA `linspace` overflow at batch 393216 | `torch.linspace(0, (B-1)*N_atoms, dtype=int64)` computed via float32 ($2^{24}$ limit) | Use `torch.arange(B) * N_atoms` exactly; clamp projection extents defensively |
| Worker-aware update gate | Replay stripes by `eps_idx mod num_workers`; cold-start can hit empty worker | Gate `agent.update()` on `global_episode >= num_replay_workers` and pre-fill demos |

Plus four smaller gotchas (tensordict==0.6.0 hidden dep, Python
3.12 `random.seed(numpy_int)` breakage, `logging.basicConfig
force=True`, CQNASAgent not being `nn.Module` requiring custom
state_dict / load_state_dict round-trip).

Carried forward unchanged.

---

## Phase 2.6: G1 Coworker Swap — CODE COMPLETE, CURRICULUM PENDING

### What was built

Documented in `docs/g1_coworker_swap.md`. Implementation:

- **`HumanConfig.human_model = "g1"`** is now the default.
  `safety_bigym/disruption/coworker.py` routes through a single
  selector; `human_model = "smplh"` is still accepted as a
  supplementary path.
- **G1 MJCF asset** integrated from the Unitree open-source
  release. Pelvis converted to `mocap="true"` (matching the SSM
  velocity fix). All body joints driven by position actuators
  (kp=200, kv=20).
- **Cross-paired collision channels** applied (G1 emits bit 1,
  accepts bit 2). No G1-self-collision; no G1-scene penetration.
- **Trajectory planner adapted** for the G1's reach kinematics
  (slightly shorter forearm than SMPL-H, slightly different shoulder
  pivot). The three trajectory modes (`stationary`,
  `approach-loiter-depart`, `coworker-patrol`) operate unchanged at
  the planner level — only the underlying IK chain changed.
- **Tests:** 7 new tests in `tests/test_g1_coworker.py` covering
  collision-channel correctness, pelvis velocity sanity, reach IK
  convergence, and disruption sampling. End-to-end smoke validated
  on GPU.

### P1 — base-policy curriculum on G1 — ✅ DONE (2026-05-30)

Ran via `scripts/run_base_curriculum.sh` (`HUMAN_MODEL=g1`, stages
idle→easy→`coworker_train`, snapshot-resume chained). The **stage-2
snapshot** is the unconstrained baseline (row-1 reference) and the
warm-start for P3/P5. **Record the stage-1 and stage-2 snapshot paths**:
P3 (E3.1) warm-starts from **stage 1**; P5 row-1/row-4 eval uses **stage 2**.
The original task spec is retained below for reference (its inline `task=`/
`frames=` keys are old sketches — `run_base_curriculum.sh` uses the real
`env=safety_bigym/<task>` / `num_train_frames=`).

**Everything downstream depends on the stage-2 snapshot from this.**

#### Tasks

Run the three-stage curriculum on `saucepan_to_hob` with the G1
coworker, resuming from each prior stage's snapshot:

```bash
# Stage 0: idle / distant G1 (sanity: can the agent learn the task?)
python train_cqn_as.py \
  task=saucepan_to_hob \
  disruption=coworker_idle \
  bodyslam=oracle \
  frames=20000 \
  +snapshot_path=null

# Stage 1: gentle G1 coworker
python train_cqn_as.py \
  task=saucepan_to_hob \
  disruption=coworker_easy \
  bodyslam=oracle \
  frames=15000 \
  +snapshot_path=runs/stage0/final.pt

# Stage 2: full G1 coworker_train
python train_cqn_as.py \
  task=saucepan_to_hob \
  disruption=coworker_train \
  bodyslam=oracle \
  frames=60000 \
  +snapshot_path=runs/stage1/final.pt
```

#### Acceptance criteria

- Stage 0: `ep_reward` ≥ +1.5 by end of training (positive,
  approaching the +2 support ceiling). `success_rate` averaged
  over last 20 eval episodes ≥ 0.5.
- Stage 1: same as stage 0 but on `coworker_easy` distribution.
- Stage 2: `ep_reward` ≥ +1.0 (some degradation expected under
  full coworker). `success_rate` ≥ 0.4. **No value-support
  saturation** — `ep_reward` curve must be monotone non-decreasing
  in expectation; a divergent trace ($-78 \to -775$) would
  indicate Theorem 5.1 invariant is violated and shaping
  hyperparameters need recheck.

#### Sanity invariants

Before launching, confirm:
- β = 0.05, c_ws = 1.0, v_min = -6, v_max = +2 in the env config.
  Invariant: β · c_ws / (1−γ) = 0.05 · 1.0 / 0.01 = 5 ≤ |v_min| = 6
  (20% headroom). See §method:lagrangian:support of the report for
  the lemma.
- `disruption=coworker_idle` has the coworker at ~3m and never
  reaching; this is the "no human" baseline that doubles as
  task-discovery sanity check.
- The `BodySLAMWrapper` channel and obs width are identical across
  stages (G1 present in all three, just distant in stage 0).
  Otherwise the policy network input shape changes between stages.

#### Compute budget

- Stage 0: 20k frames × ~0.7s/frame ≈ 4 GPU-hours
- Stage 1: 15k frames ≈ 3 GPU-hours
- Stage 2: 60k frames ≈ 12 GPU-hours
- **Total: ~19 GPU-hours**

#### Decision rule

| Outcome | Interpretation | Next step |
|---|---|---|
| Stages 0/1/2 hit acceptance | G1 curriculum works as expected | Proceed to P3 (E3.1) using stage-2 snapshot |
| Stage 0 hits acceptance, stage 1 plateaus | Curriculum gap too large between idle and easy | Insert intermediate stage (closer coworker, no reach) |
| Stage 2 collapses (ep_reward declines) | Either workspace shaping wrong or coworker_train too hard | Check invariant; if OK, loosen `coworker_train` (raise closest-approach floor) |
| Value support saturated mid-stage | Theorem 5.1 invariant violated | Widen v_min/v_max further; reduce β |
| Task discovery fails in stage 0 | Workspace shaping is masking task reward | Reduce β toward 0 in stage 0; re-introduce in stage 1 |

### Deliverable

Stage 2 snapshot `runs/saucepan_to_hob_g1_coworker_train/final.pt`,
the unconstrained-baseline reference for everything downstream.
This populates Table~\ref{tab:results:baseline} and row 1 of
Table~\ref{tab:e4.1-feature-incremental}.

---

## Phase 3: Value-Based Lagrangian Constrained RL — CODE COMPLETE, EXPERIMENTS PENDING

### What was built

The B-value-mean architecture (§method:lagrangian:Bmean of the
report) on top of the CQN-AS critic-only backbone. Key modules:

| File | Purpose |
|---|---|
| `safety_bigym/agents/cqn_as/lagrangian_agent.py` | `LagrangianCQNASAgent` — wraps two CQN-AS critics (Q_r, Q_c), one shared encoder, action selection `argmax_a [Q_r(s,a) - λ Q_c(s,a)]` |
| `safety_bigym/agents/cqn_as/pid_controller.py` | `PIDLagrangeUpdater` — running cost mean, episode-boundary λ update with clamp `[0, λ_max=100]`, gains K_I=1e-3, K_P=1e-2, K_D=0 |
| `safety_bigym/filters/cost_signal.py` | `compute_continuous_cost(safety_info) → c_t` — Equations 4.2–4.4 of the report (continuous SSM + PFL aggregator) |
| `train_cqn_as.py` | Top-level training script; Hydra config; loads warm-start weights for Q_c from Phase 2 SVF |

**Smoke validation:**

- **P3.0 — cost-flow smoke** (`scripts/phase3_p30_smoke.py`):
  verifies that `c_t` is non-zero on at least one step per episode
  under `coworker_train`, that the cost critic receives it at
  per-env-step granularity (not chunk-averaged), and that W&B logs
  `episode_cost_integral` and `episode_lambda`. **Passes.**
- **P3.1 — training smoke**: 500-frame training run produces a
  valid snapshot, no NaN losses, λ initialised at 0 and incrementing
  with positive constraint violation. **Passes.**

### What's pending — **P3, P4, P7** (the three Phase 3 experiments)

#### P3 — Experiment E3.1: Cost-Signal Form Ablation

Three cells under the B-value-mean Lagrangian backbone, 3 seeds each:

| Cell | `cost_signal` config | What it tests |
|---|---|---|
| Fixed penalty (no Lagrangian) | `r_modified = r_task - 0.05 · 1[violation]`; λ disabled | Baseline: can a fixed-magnitude binary penalty suffice? |
| Binary + λ | `c_t ∈ {0, 1}` from `1[ssm_violation]`; PID λ active | Does Lagrangian adaptivity help when cost signal is binary? |
| **Continuous + λ** (ours) | `c_t = max(c_ssm, c_pfl)` per Equation 4.4 | Headline configuration |

##### ✅ Cost-form selector LANDED (2026-05-30) — all 3 cells wired

The `cost_signal={fixed,binary,continuous}` selector in the pseudocode below was
a sketch and did **not** exist; it has now been implemented (smaller than first
estimated — the `fixed` cell needed no new code):
1. **`filters/cost_signal.py`** — new `select_cost(safety_info, cost_form=...)`
   dispatches `continuous` → graded `compute_cost` (the headline) / `binary` →
   `1[ssm_violation]`; exports `COST_FORMS=("continuous","binary")`.
2. **`agents/cqn_as/env_adapter.py`** — reads `env.safety.cost_form` (default
   `continuous`, validated) and calls `select_cost` at the per-step cost site
   (replacing the hardcoded `compute_cost`).
3. **`cfgs/env/safety_bigym.yaml`** — declares `cost_form: continuous` so
   `env.safety.cost_form=binary` overrides cleanly (no `+`).
4. **`fixed` cell** reuses the **pre-existing**, already-factory-threaded and
   already-tested `env.safety.add_violation_penalty` / `violation_penalty=0.05`
   reward penalty (`SafetyBiGymEnv._reward`) under plain `agent=cqn_as` (no Q_c /
   no λ). No new code — same path `phase1_reward_pilot_cqn_as.py` uses.
5. **Tests**: +8 `select_cost` cases in `tests/test_cost_signal.py` (31 cost-path
   tests pass); Hydra composition verified for all three cells.

Launcher `scripts/run_e3_1_cost_signal.sh` runs the full 3×3 matrix. **E3.1 is
launch-ready** — set `WARMSTART` to the P1 stage-1 snapshot.

##### Acceptance criteria

- Continuous row dominates binary row on
  `ep_proximity_violation_rate` with **non-overlapping 95% bootstrap CIs**.
- All cells preserve `success_rate ≥ 0.3` (a row that collapses
  task success is uninformative as a safety comparison).

##### Implementation notes

Use the launcher `scripts/run_e3_1_cost_signal.sh` (warm-starts from the P1
stage-1 snapshot, so each cell shares row-1's training protocol and differs
only in cost signal):

```bash
WARMSTART=exp_local/cqn_as_base_curriculum/<run>/stage1_easy/snapshot_XXXXX.pt \
  scripts/run_e3_1_cost_signal.sh           # continuous x seeds {0,1,2} (runnable today)
COST_FORMS="fixed binary continuous" WARMSTART=... scripts/run_e3_1_cost_signal.sh
  # launches continuous; reports fixed+binary as BLOCKED until the selector lands
```

The **real** per-cell command (the plan's old `task=`/`frames=`/`cost_signal=`
were sketches — verified keys against `train_cqn_as.py` + `cqn_as_config.yaml`):

```bash
python train_cqn_as.py \
  env=safety_bigym/saucepan_to_hob \      # NOT task=
  disruption=coworker_train \
  bodyslam=oracle \
  num_train_frames=60000 \                # NOT frames=
  agent=cqn_as_lagrangian \               # selects Q_c + λ-PID (continuous cost)
  agent.cost_budget=0.01 \                # the constraint target d
  num_demos=36 \
  env.safety.add_workspace_penalty=true env.safety.workspace_beta=0.05 \
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101 \
  seed=0 \
  +snapshot_path=$WARMSTART               # P1 stage-1 snapshot
```

The `cost_signal={fixed,binary,continuous}` pseudocode below was the original
sketch; the **implemented** dispatch (see "Cost-form selector LANDED" above) is:
`continuous` → `agent=cqn_as_lagrangian`; `binary` → same agent +
`env.safety.cost_form=binary`; `fixed` → `agent=cqn_as` +
`env.safety.add_violation_penalty=true env.safety.violation_penalty=0.05`
(λ disabled). `scripts/run_e3_1_cost_signal.sh` applies exactly these.

##### Compute budget

9 cells × ~2 GPU-hours per cell = **~18 GPU-hours**.

##### Decision rule

| Outcome | Action |
|---|---|
| Continuous dominates as expected | Proceed; continuous is the headline cost signal |
| Continuous matches binary | Continuous formulation is no better; report honestly, but it's still defensible as more principled |
| Continuous worse than binary | Bug — investigate. Possible cause: `d_buffer = 0.3` is wrong scale; check that `c_t` is in [0, 1] over the relevant operating range |
| All cells collapse to evacuation | Workspace shaping is being marginalised; reduce λ_max or tighten cost budget |

#### P4 — Experiment E3.2: Cost-Budget Pareto Sweep

Sweep cost budget `d ∈ {0.001, 0.01, 0.05, 0.1}`, 3 seeds each =
12 cells.

##### Acceptance criteria

- The Pareto curve (`success_rate` vs `ep_proximity_violation_rate`)
  is monotonic between the over-tight regime (d = 0.001, task
  collapse) and the over-loose regime (d = 0.1, safety saturates).
- A clear knee exists. **The knee defines the headline operating
  point** for P5 row 3.

##### Implementation notes

Use the launcher `scripts/run_e3_2_cost_budget.sh` (warm-starts from the P1
stage-1 snapshot; `SMOKE=1` for a composition check):

```bash
WARMSTART=exp_local/cqn_as_base_curriculum/<run>/stage1_easy/snapshot_XXXXX.pt \
  scripts/run_e3_2_cost_budget.sh           # d in {0.001,0.01,0.05,0.1} x seeds {0,1,2}
```

The real per-cell command it runs (env=, num_train_frames=, agent=cqn_as_lagrangian,
agent.cost_budget=). cost_signal=continuous is implicit (E3.1 compares forms;
here d = agent.cost_budget is the only treatment variable):

```bash
python train_cqn_as.py \
  env=safety_bigym/saucepan_to_hob \
  disruption=coworker_train \
  bodyslam=oracle \
  num_train_frames=60000 \
  +snapshot_path=$WARMSTART \             # P1 stage-1 snapshot
  agent=cqn_as_lagrangian \
  agent.cost_budget={0.001,0.01,0.05,0.1} \
  num_demos=36 env.safety.add_workspace_penalty=true env.safety.workspace_beta=0.05 \
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101 \
  seed={0,1,2}
```

Logging: each run must emit `episode_lambda`, `episode_cost_integral`,
`episode_cost_mean` so the PID λ trajectory and the running cost
mean against the budget can be plotted as supplementary
diagnostics.

##### Compute budget

12 cells × ~2 GPU-hours = **~24 GPU-hours**.

##### Decision rule

| Outcome | Action |
|---|---|
| Smooth Pareto with clear knee | Headline d = knee value |
| Curve has multiple knees | Pick the operating point with `success_rate` ≥ 0.5 closest to the safety axis |
| All cells collapse | Curriculum stage 2 snapshot is not converged; rerun P1 stage 2 |
| All cells have similar safety | λ is not constraining behaviour; investigate PID gains or `λ_max` |

#### P7 — Experiment E3.6: Obs-Channel Under the Constrained Policy

Three cells: `bodyslam ∈ {off, oracle, noisy}` under the
P4-winning configuration. Closes RQ1.

##### Acceptance criteria

- At least one of (oracle, noisy) reduces
  `ep_proximity_violation_rate` by ≥ 10% relative to `off`,
  under the constrained policy.
- If neither does, report the negative result honestly — it means
  the cost signal carries the safety burden and the channel is
  redundant under our setup. Still valuable for the literature.

##### Implementation notes

The training side trains a single policy on `bodyslam=oracle`
(matching the P3/P4 architecture-comparison policy), and the eval
side sweeps all three modes via the (now-built) `benchmark_policy.py`
harness. This isolates the perception-gap measurement from
training-time confounds.

```bash
# Train one policy under oracle (the same policy used in P5 row 3,
# so we can reuse the P3 continuous + d_knee run if it exists). Real keys.
python train_cqn_as.py \
  env=safety_bigym/saucepan_to_hob \
  disruption=coworker_train \
  bodyslam=oracle \
  num_train_frames=60000 \
  +snapshot_path=$WARMSTART \             # P1 stage-1 snapshot
  agent=cqn_as_lagrangian \
  agent.cost_budget=$D_HEADLINE \
  num_demos=36 env.safety.add_workspace_penalty=true env.safety.workspace_beta=0.05 \
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101 \
  seed={0,1,2}

# Then evaluate the same snapshots across three perception modes
for MODE in off oracle noisy; do
  python scripts/benchmark_policy.py \
    --snapshot runs/p7_e3.6_seed{0,1,2}/final.pt \
    --task saucepan_to_hob --disruption coworker_train \
    --obs-mode $MODE --seeds 0,1,2 --episodes 20 \
    --out results/e3.6_${MODE}.csv
done
```

##### Compute budget

3 trained policies (one per seed, all `bodyslam=oracle`) × ~2 GPU-h
= **~6 GPU-h for training**; 9 eval cells (3 seeds × 3 modes) are
pure eval via the harness, < 1 h total. **Total: ~7 GPU-hours**
(revised down from the original ~18 h estimate after consolidating
to one training run per seed). **De-prioritise if budget runs out**
— the question is independent of the headline hybrid result.

### Phase 3 — Possible outcomes

| Outcome | Interpretation | Next step |
|---|---|---|
| Continuous cost reduces SSM violations >50% vs P1 baseline with <10% task-reward loss | Constrained RL internalises safety; the load-bearing claim is validated | Proceed to Phase 4 hybrid (P5) |
| Violations reduce but task reward collapses (>30% loss) | λ too aggressive or cost budget too tight | Loosen d via P4 sweep; clamp λ_max lower |
| Oscillation — policy swings between safe-frozen and unsafe-productive | Lagrangian update is unstable | Increase K_D from 0 to ~1e-3; reduce K_P |
| No improvement over fixed penalty baseline | Cost signal too sparse OR Lagrangian update broken | Check `c_t` distribution; verify PID controller logs |
| Continuous helps, binary doesn't | Confirms the gradient-richness hypothesis — load-bearing for the thesis |
| Smoke gates pass but real training diverges | C51 support saturation likely; check Theorem 5.1 invariant on current β, c_ws |

### Deliverable

(i) Three trained snapshots from E3.1 (one per cell); (ii) the
Pareto curve from E3.2 with the headline d operating point;
(iii) one snapshot at the headline (continuous + d_knee + `bodyslam=oracle`)
that becomes the P5 row 3 input. (E3.6 separately sweeps
`bodyslam ∈ {off, oracle, noisy}` to measure the perception gap.)

---

## Phase 4: Benchmark Harness + Full Hybrid — NOT STARTED

### Goal

(i) Build `benchmark_policy.py`, the snapshot eval harness that
emits the canonical metrics schema for every results table in the
report. (ii) Run the headline E4.1 five-row feature-incremental
comparison and the E4.3 internalisation curve.

### **P6 — `scripts/benchmark_policy.py` (the benchmark harness)** — ✅ DONE (2026-05-30)

**Status**: built + unit-tested (8 tests) + validated end-to-end on the real
CQN-AS snapshot `snapshot_17826.pt` (saucepan_to_hob/G1), filter off and on.
Code: CLI `scripts/benchmark_policy.py` + `safety_bigym/benchmark/` package
(`stats`/`records`/`schema`/`aggregate`/`env_build`/`filter_attach`/`runners`/
`loader`) + `scripts/benchmark_visualize.py` + `scripts/benchmark_demo.sh`.
Usage doc: `docs/benchmark_harness.md`. Deviations + portability fixes are
recorded in `IMPLEMENTATION_STATUS.md` (P6) and the usage doc — notably: raw
rolls are **parquet** (`pandas`+`pyarrow` added to `setup.py`) + JSONL sidecar;
`success` = `info["task_success"]`; `build_cqn_cfg` rebases the snapshot's baked
AMASS path onto the local `AMASS_DATA_DIR`; `--num-demos-for-stats` caps the
CQN-AS action-stat demo load to avoid laptop OOM. The text below is the original
spec, retained for reference.

This is the load-bearing deliverable for the C1 benchmark
contribution (§bench:harness of the report). Without it, every
results table in the report is unreproducible.

#### Goal

Given any policy snapshot (a CQN-AS checkpoint, an ACT snapshot,
or a user-supplied `Policy` subclass), produce a CSV row per
(task, disruption, obs-mode, seed) cell with the full safety
metric schema. Used by P5 rows 4 and 5 as pure eval (no
training), and as the canonical data source for **every numerical
entry** in Tables 8.1, 8.3, 8.5, 8.7, the headline 8.9, and 8.11.

#### CLI

```bash
python scripts/benchmark_policy.py \
  --snapshot path/to/policy.pt \
  --filter-snapshot path/to/svf.pt \   # optional — wraps the policy with SafetyFilterWrapper(R=...)
  --filter-threshold 2.25 \             # optional — only used if --filter-snapshot given
  --task saucepan_to_hob \
  --disruption coworker_eval \         # or coworker_train; or any other ParameterSpace key
  --obs-mode oracle \                  # oracle for headline cells; noisy only for E3.6 sweep + sim-to-real diagnostic — see Perception Mode Policy above
  --seeds 0,1,2 \
  --episodes 20 \
  --out results/cell.csv \
  --smoke                              # optional — 1 seed, 2 episodes, < 5 min CPU
```

#### CSV schema (must emit ALL these columns)

**Per-episode metrics, aggregated across all (seed × episode) rolls:**

```
# Identification
task, disruption, obs_mode, snapshot, filter_snapshot, filter_threshold, seed

# Safety (canonical)
ep_proximity_violation_rate_mean
ep_proximity_violation_rate_ci_low
ep_proximity_violation_rate_ci_high
ep_ssm_violation_rate_mean
ep_ssm_violation_actual_rate_mean
ep_min_separation_mean
ep_min_separation_ci_low
ep_min_separation_ci_high

# Safety (tail-risk — populates §results:tail / Table 8.11)
cvar95_ep_cost_integral
cvar99_ep_cost_integral
cvar95_ep_min_separation
p99_ep_min_separation

# Task (populates §results:task-tax framing of headline Table 8.9)
success_rate_mean
success_rate_ci_low
success_rate_ci_high
episode_reward_mean
steps_to_completion_mean       # mean env-steps to terminal, among successful episodes
steps_to_completion_ci_low
steps_to_completion_ci_high

# Filter mechanics (only populated when --filter-snapshot is given)
filter_intervention_rate_mean
filter_intervention_rate_ci_low
filter_intervention_rate_ci_high
filter_passthrough_rate_mean   # = 1 - intervention_rate_mean; emitted explicitly for the table
mean_per_episode_interventions

# Per-region PFL counts (currently inert under the open BiGym contact bug)
pfl_violations_per_region_json
```

#### Implementation interface

```python
# scripts/benchmark_policy.py

def benchmark_policy(
    snapshot_path: Path,
    filter_snapshot_path: Path | None,
    filter_threshold: float,
    task: str,
    disruption: str,
    obs_mode: str,
    seeds: list[int],
    n_episodes: int,
    bootstrap_resamples: int = 10_000,
) -> dict:
    """
    Returns one dict matching the CSV schema. Writes per-episode
    raw rolls to `{out_dir}/raw_episodes.parquet` for later
    re-aggregation without re-rollout.
    """
```

The harness must:

1. **Load the snapshot** robustly: detect ACT vs CQN-AS vs custom
   `Policy` subclass via the snapshot's top-level keys.
2. **Wrap with the filter** if `--filter-snapshot` is given, using
   `SafetyFilterWrapper(critic, threshold=R)` from
   `safety_bigym/filters/runtime_wrapper.py`.
3. **Roll out** `n_episodes` per seed, recording per-step
   `info["safety"]` for the full episode (not just terminal).
4. **Compute aggregates** with bootstrap 95% CIs (10k resamples)
   on every `_mean` column.
5. **Compute CVaR / percentiles** over the union of all
   (seed × episode) rolls.
6. **Write CSV + parquet** atomically (write-temp, fsync, rename).

#### Acceptance criteria

- `--smoke` finishes in < 5 min on CPU on a freshly-cloned repo.
- Running on the Phase 0 ACT snapshot reproduces the
  baseline numbers reported in E1.1 within ±5% (regression test).
- Schema is forward-compatible: adding a new metric requires only
  a new column, not a script rewrite.
- 5 new tests in `tests/test_benchmark_harness.py` covering
  bootstrap correctness, CVaR formula, filter-wrapper attachment,
  raw-roll persistence, and CSV/parquet schema.

#### Compute budget

Engineering: ~2 days. Runtime: pure eval, minimal GPU
(~5 min per cell of 3 seeds × 20 episodes).

#### Decision rule

| Outcome | Action |
|---|---|
| All acceptance criteria pass | Use for P5 |
| Bootstrap CIs disagree with manual numpy reference | Bug in resampling; freeze the version that disagrees least |
| Filter wrapper attachment fails on some snapshots | Standardise the policy-loader interface in `safety_bigym/agents/_loader.py` |

### P5 — Experiment E4.1: The Headline Feature-Incremental Table

The central thesis result. Populates
Table~\ref{tab:e4.1-feature-incremental}. Five rows, three seeds
each, evaluated on `saucepan_to_hob` under `coworker_train`.

| Row | Configuration | Source |
|---|---|---|
| 1 | Unconstrained baseline | **From P1** (stage 2 snapshot of G1 curriculum) — pure eval |
| 2 | + workspace shaping | New training run (P1 had shaping enabled; row 2 disables it then re-enables for ablation) |
| 3 | + Lagrangian (continuous cost) | **From P3** continuous + d=d_knee cell — pure eval |
| 4 | Baseline + runtime filter (filter-alone) | **Pure eval** — P1 snapshot wrapped with P2 SVF via `--filter-snapshot` |
| 5 | **Full hybrid** | **Pure eval** — P3 snapshot wrapped with P2 SVF |

**Most rows are pure eval.** Only row 2 is a new training run.

#### Tasks

**Driver: `scripts/run_e4_1_headline.sh`** (built 2026-05-30) runs all rows via
`benchmark_policy.py` on **one obs mode** (`OBS_MODE`, default `noisy` — the
filter's native distribution; see Perception Mode Policy footnote ²), one CSV
per row, skipping rows whose snapshot env var is unset so it's usable
incrementally:

```bash
# Rows 1 & 4 now, noisy (need only STAGE2 + the SVF filter):
STAGE2=.../stage2_full/snapshot_28203.pt bash scripts/run_e4_1_headline.sh
# Full noisy headline once the P3 d_knee snapshot exists:
STAGE2=.../stage2_full/snapshot_28203.pt ROW3=runs/.../snapshot_*.pt \
  bash scripts/run_e4_1_headline.sh
# Policy-only oracle reference (rows 1-3; the filter rows are meaningless on oracle):
OBS_MODE=oracle STAGE2=... ROW3=... bash scripts/run_e4_1_headline.sh
```

The explicit per-row commands the driver wraps (retained for reference; the
whole table runs on `--obs-mode noisy`):

```bash
# Row 1 — eval P1 snapshot
python scripts/benchmark_policy.py \
  --snapshot runs/saucepan_to_hob_g1_coworker_train/final.pt \
  --task saucepan_to_hob --disruption coworker_train \
  --obs-mode noisy --seeds 0,1,2 --episodes 20 \
  --out results/e4.1_row1.csv

# Row 2 — NEW training run with workspace shaping (plain agent=cqn_as, no λ).
# TRAINS on bodyslam=oracle (architecture comparison); EVALUATED on noisy below.
python train_cqn_as.py \
  env=safety_bigym/saucepan_to_hob disruption=coworker_train bodyslam=oracle \
  num_train_frames=60000 +snapshot_path=$WARMSTART \
  env.safety.add_workspace_penalty=true env.safety.workspace_beta=0.05 \
  env.safety.workspace_excess_cap=1.0 num_demos=36 \
  agent.v_min=-6.0 agent.v_max=2.0 agent.atoms=101 \
  seed={0,1,2}
# Then eval
python scripts/benchmark_policy.py \
  --snapshot runs/row2_seed{0,1,2}/final.pt ... \
  --obs-mode noisy \
  --out results/e4.1_row2.csv

# Row 3 — eval P3 continuous-cost snapshot
python scripts/benchmark_policy.py \
  --snapshot runs/p3_continuous_d_knee/final.pt \
  --obs-mode noisy \
  --out results/e4.1_row3.csv

# Row 4 — eval P1 snapshot WITH filter
python scripts/benchmark_policy.py \
  --snapshot runs/saucepan_to_hob_g1_coworker_train/final.pt \
  --filter-snapshot checkpoints/svf_coworker_train_g1_0p3.pt \
  --filter-threshold 2.25 \
  --obs-mode noisy \
  --out results/e4.1_row4.csv

# Row 5 — eval P3 snapshot WITH filter
python scripts/benchmark_policy.py \
  --snapshot runs/p3_continuous_d_knee/final.pt \
  --filter-snapshot checkpoints/svf_coworker_train_g1_0p3.pt \
  --filter-threshold 2.25 \
  --obs-mode noisy \
  --out results/e4.1_row5.csv

# Aggregate the per-row CSVs into the headline LaTeX table (scripts/aggregate_e4_1.py
# built 2026-05-30; reads run_e4_1_headline.sh's row CSVs, bolds the lowest
# proximity-violation row, emits a booktabs table):
python scripts/aggregate_e4_1.py \
  --in-dir results/e4_1/<run_tag> \
  --out report_tables/e4.1_feature_incremental.tex
```

#### Acceptance criteria

- **Row 5 dominates each of rows 1–4 on
  `ep_proximity_violation_rate` with non-overlapping 95% CIs.**
- Row 5 `success_rate` is within 30% of row 1
  (the "safety tax" is acceptable; below 70% suggests the trade-off
  is too aggressive and motivates loosening the cost budget).
- Row 4 intervention rate is substantially higher than row 5
  intervention rate. This is the load-bearing evidence for
  policy/filter complementarity.

#### Compute budget

- Row 2 training: 3 seeds × ~2 GPU-hours = **~6 GPU-hours**
- Rows 1, 3, 4, 5 eval: 4 × 5 min per cell = **negligible**
- Aggregation script: ~30 min engineering

#### Decision rule

| Outcome | Interpretation | Action |
|---|---|---|
| Row 5 dominates as expected, CIs clean | Headline confirmed | Write up §results:rq3 with concrete numbers |
| Row 5 ≈ row 3 (filter is redundant on the trained policy) | Components not complementary at this disruption level | Report honestly; the conclusion shifts to "Lagrangian alone is sufficient under our \texttt{coworker\_train} distribution"; filter advantage may appear at the OOD eval band (E5.2) |
| Row 5 < row 3 (filter HURTS the trained policy) | Filter is mis-calibrated for the trained policy | Re-sweep R against the row 3 snapshot (it needs a different operating point than the untrained baseline) |
| Row 2 ≈ row 1 (workspace shaping does nothing) | Shaping was not load-bearing; can be dropped from future-work argument | Revise §bench:workspace |
| Row 4 has very low intervention rate (<5%) | Filter rarely fires on the baseline; either threshold too low or baseline is unexpectedly safe | Check P2 calibration; may indicate violations are clustered in a way the filter can't catch |

### P8 — Experiment E4.3: Filter Internalisation Curve

The filter intervention rate falls as the policy is Lagrangian-trained.

#### Implementation — ✅ POST-HOC ON NOISY (2026-05-31)

**The original "free, piggybacks on P3" plan does NOT work.** It assumed an
in-training passive hook (`filter_passive.snapshot=...`, still in
`train_cqn_as.py`, harmless and off by default) logging
`eval/filter_intervention_rate` each eval cycle. But P3/P4 train on `oracle`, so
those eval cycles run on oracle — where the SVF filter's Q collapses (the same
finding that moved E4.1 to noisy, footnote ²). A FILTER_PASSIVE curve logged
during oracle training is a **flat ~100%** line, not the internalisation signal.

**Real path: `scripts/run_e4_3_internalisation.sh` — post-hoc, on `noisy`.**
For each saved `snapshot_<N>.pt` in a P3/P4 training cell, it evaluates the
policy WITH the filter on noisy via `benchmark_policy.py` and records
`filter_intervention_rate` vs training frame:

```bash
RUN_DIR=exp_local/e3_2_cost_budget/<run>/d0pXX_seed0 \
  bash scripts/run_e4_3_internalisation.sh
# -> results/e4_3/<tag>/internalisation_curve.csv
#    (frame, filter_intervention_rate, success_rate, ep_proximity_violation_rate)
```

On noisy the filter is in-distribution, so the rate is meaningful and should
fall as the frame grows. **Not free** (~few min/snapshot, ~30 min/run) — the
oracle-collapse turned this from a logging side-product into a post-hoc eval.

#### Acceptance criteria

- The intervention rate at training start should match the
  random-policy intervention rate (≥ 30%); the intervention rate
  at convergence should be substantially lower (≤ 15%).
- The decrease should be monotonic in expectation — not strictly,
  but the trend over 60k frames should be unambiguously downward.

#### Compute budget

**~30 min post-hoc per run** (≈6–7 saved snapshots × a few min each, eval-only
on noisy). NOT free — the oracle-collapse moved this off the in-training path.

### Deliverable for Phase 4

The headline E4.1 table populated, the E4.3 figure plotted, and
the benchmark harness packaged for re-use.

---

## Phase 5: Evaluation and Stress Testing — NOT STARTED

### P10 — Experiments E5.1 and E5.2

#### E5.1 — Tail-Risk Extraction

Pure post-hoc analysis of the P5 episode rolls. The
`benchmark_policy.py` harness already emits `cvar95_*` and
`p99_*` columns; the only work is the LaTeX-table aggregation
(`scripts/aggregate_e5_1.py`).

##### Compute budget

Zero — re-uses raw rolls from P5.

#### E5.2 — OOD Generalisation on the Wider Eval Band

Re-run the P5 row 1 and row 5 evals on `coworker_eval` (the wider
ParameterSpace from Table~\ref{tab:coworker-params}) instead of
`coworker_train`. **Pure eval** — no new training.

Easiest: re-run the P5 driver with the disruption flipped (it emits all rows;
use rows 1 and 5 for the OOD comparison):

```bash
SVF_FILTER=checkpoints/svf_coworker_train_g1_0p3.pt \
STAGE2=$STAGE2 ROW3=$ROW3 DISRUPTION=coworker_eval RUN_TAG=e5_2_ood \
  bash scripts/run_e4_1_headline.sh
```

The explicit per-row commands (retained for reference):

```bash
python scripts/benchmark_policy.py \
  --snapshot runs/saucepan_to_hob_g1_coworker_train/final.pt \
  --disruption coworker_eval \
  --out results/e5.2_row1_ood.csv

python scripts/benchmark_policy.py \
  --snapshot runs/p3_continuous_d_knee/final.pt \
  --filter-snapshot checkpoints/svf_coworker_train_g1_0p3.pt \
  --disruption coworker_eval \
  --out results/e5.2_row5_ood.csv
```

##### Acceptance criteria

- The OOD degradation (drop in `ep_proximity_violation_rate`
  performance from train to eval band) is smaller for row 5 than
  row 1. If equal or worse, the hybrid does not generalise better
  than the unconstrained baseline — a significant finding worth
  reporting.

##### Compute budget

~2 GPU-hours (pure eval).

### P9 — External Baseline: WCSAC

Documented honest-failure path in §disc:wcsac-honest of the
report. **Decision gate before launching the full run**: a 2-day
sanity check that the reimplementation matches the original
WCSAC paper's Safety-Gym numbers within ±5%.

#### Tasks

1. **Sanity gate (2 days):** Run the WCSAC reimplementation on
   Safety-Gym `Safexp-PointGoal1-v0`. Compare to the Yang et al.
   2021 reported numbers. If within ±5%, the reimplementation is
   defensible.
2. **Full run on safety_bigym:** Train WCSAC on `saucepan_to_hob`
   under `coworker_train`. 3 seeds × 200k frames each.
3. **Eval via the harness:** Use `benchmark_policy.py` (the harness
   supports any `Policy` subclass — wrap WCSAC's actor).

#### Acceptance criteria

If sanity gate fails: report as "best-effort reimplementation;
gap may close with further tuning." If sanity gate passes: report
numbers at face value and compare in Table~\ref{tab:e3.7-wcsac}.

#### Compute budget

- Sanity gate: ~3 GPU-hours
- Full run: ~7 GPU-hours
- **Total: ~10 GPU-hours**

#### Decision rule

| Outcome | Action |
|---|---|
| Sanity gate passes; WCSAC underperforms our hybrid on safety_bigym | Strong external validation — report directly |
| Sanity gate passes; WCSAC matches our hybrid on safety_bigym | The contribution of our work is the benchmark, not the method. Report honestly |
| Sanity gate fails | Cut the experiment; report as future work |

### Deliverable for Phase 5

E5.1 and E5.2 tables populated, P9 outcome decided and reported,
and the report's `\result{X}` markers are fully replaced.

---

## Critical Path and Dependencies

```
Phase 0–2.6 (DONE)
    │
    ├── P1 (G1 curriculum) ─────┐
    │       │                   │
    │       └── stage-2 snapshot├── P3 (E3.1 cost-signal) ──┐
    │                           │       │                   │
    │   ┌── P2 (SVF re-eval)    │       └── P4 (E3.2 budget)│
    │   │                       │               │           │
    │   │                       │               └── P7 (E3.6 obs)
    │   │                       │                           │
    │   └── R operating point ──┘                           │
    │                                                       │
    └── P6 (benchmark harness) ── (engineering)             │
            │                                               │
            └── P5 (E4.1 headline) ─────────────────────────┤
                    │                                       │
                    ├── P8 (E4.3 internalisation — post-hoc) │
                    │                                       │
                    └── P10 (E5.1 + E5.2 — pure eval) ──────┤
                                                            │
                                                  P9 (WCSAC, parallel)
```

**P1 is the critical path bottleneck.** Everything downstream
needs the stage-2 snapshot. P6 can be built in parallel
(engineering only, no GPU). P2 can be run as soon as P1 stage 2
finishes.

---

## Open Bugs and Limitations

### PFL contact-detection bug (HIGH PRIORITY but NON-BLOCKING)

`ep_pfl_violation_rate`, `ep_max_pfl_force_ratio`, and
`ep_max_contact_force` are identically zero across every
experimental cell — including cases where the coworker pelvis is
geometrically inside the H1's collision volume. Root cause is in
BiGym's runtime robot-attachment suppressing `data.ncon` for
human↔robot pairs.

**Impact:** All safety labels and the cost signal are SSM-only.
The `use_pfl` flag is wired through the labeller, the Phase 2
critic, the Phase 3 cost signal, and the runtime filter — all
inert until the bug is fixed.

**Estimated fix effort:** 1–2 weeks of BiGym-internals work
(§fw:pfl of the report). Schema is forward-compatible: flip the
flag, re-collect Phase 2 dataset, retrain SVF. No architectural
change required.

**Non-blocking for:** P1 through P10. All claims qualified as
"geometric / SSM-only" (§disc:pfl-bug of the report).

### SSM margin outliers

Occasional `ep_min_ssm_margin` values of -16m or -25m appear in
eval data. These are single-episode artefacts where the SSM
formula's robot-velocity term produces physically unreasonable
required separation when the H1 is tumbling. The `v_h_max` cap
handles the human side; consider clamping `v_r` at a sane upper
bound (or reporting the conservative-bound flavour as auxiliary
data only).

### CQN-AS C51 support saturation

Resolved (Theorem 5.1, §method:lagrangian:support). The invariant
`β · c_ws / (1−γ) ≤ |v_min|` must hold for any new shaped reward.
Current config (β=0.05, c_ws=1.0, v_min=-6) leaves 20% headroom.

---

## Risk Register

### Active high-severity risks

| Risk | Trigger | Mitigation |
|---|---|---|
| P1 stage 2 collapses on G1 | `coworker_train` is harder than `coworker_easy` | Add intermediate stage; loosen `coworker_train` parameters (raise closest-approach floor to 0.65 m); check Theorem 5.1 invariant |
| P3 continuous cell collapses to evacuation | Cost signal too aggressive | Increase `r_workspace_shaping.beta` from 0.05 toward 0.08 (re-check invariant); tighten λ_max from 100 toward 20 |
| P5 row 5 doesn't dominate row 3 (filter is redundant on trained policy) | The trained policy is so safe the filter has no work | Report honestly. Frame as "Lagrangian-alone is sufficient at this disruption level; filter advantage appears at OOD eval (E5.2)." This is still a publishable finding |
| P6 benchmark harness has bootstrap CI bugs | New code without battle-testing | 5 unit tests; regression-check against manual numpy on a sample cell |

### Active medium-severity risks

| Risk | Trigger | Mitigation |
|---|---|---|
| WCSAC sanity gate fails | Reimplementation has bugs | Cut P9, report as future work |
| P2 SVF needs retraining on G1 data | Operating point shifted | ~3 GPU-hours extra; accept in budget |
| `steps_to_completion` is noisy due to small successful-episode denominator | Some cells have low `success_rate` | Report with wide CIs and flag in table footnote |
| PFL bug fix takes longer than 2 weeks | Deep BiGym-internals issue | Document the fix path in §fw:pfl; final submission qualifies all claims as SSM-only |

### Inactive risks (resolved or not materialised)

| Risk | Status |
|---|---|
| BodySLAM++ wrapper doesn't help BC | Materialised (E1.1 negative); reframed as motivation for constrained-RL formulation |
| RoboBase surgery proves too invasive | Avoided — we switched from RoboBase/DrQ-V2+ to vendored CQN-AS |
| C51 support saturation under shaped rewards | Resolved (Theorem 5.1) |
| G1 swap breaks the rest of the pipeline | Resolved — code complete, smoke validated |

---

## Compute Budget — Total Remaining

| Task | GPU-hours (approx) |
|---|---|
| **P1** — G1 base-policy curriculum (stages 0/1/2) | 19 |
| **P2** — Phase 2 SVF re-eval + retrain under G1 (expected) | 6 |
| **P3** — E3.1 cost-signal form ablation | 18 |
| **P4** — E3.2 cost-budget Pareto sweep | 24 |
| **P5** — E4.1 headline (row 2 train only; rows 1/3/4/5 are eval) | 6 |
| **P6** — Benchmark harness (engineering, not GPU) | 0 |
| **P7** — E3.6 obs-channel under constrained policy (de-prioritise) | 7 |
| **P8** — E4.3 internalisation curve (post-hoc on noisy, ~0.5 h/run) | ~1 |
| **P9** — WCSAC external baseline (sanity gate + full run) | 10 |
| **P10** — E5.1 + E5.2 (pure eval) | 2 |
| **Total — mandatory (P1, P2, P3, P4, P5, P6, P8)** | **~73** |
| **Total — with strengthening (P7, P9, P10)** | **~92** |

---

## Smoke-Gate Checklist (run before EVERY multi-hour launch)

```bash
# Phase 2 SVF pipeline
python -m safety_bigym.filters.dataset --smoke
python -m safety_bigym.filters.train --smoke
python -m safety_bigym.filters.sweep --smoke

# Phase 3 training composition — `--smoke` does NOT exist (hydra rejects it);
# the real gate is a short num_train_frames run with no demos/W&B.
python train_cqn_as.py env=safety_bigym/saucepan_to_hob \
  num_train_frames=100 num_demos=0 wandb.use=false
# Lagrangian agent composition (the P3 path)
python train_cqn_as.py env=safety_bigym/saucepan_to_hob \
  agent=cqn_as_lagrangian num_train_frames=100 num_demos=0 wandb.use=false

# P3.0 cost-flow smoke (verify c_t flows end-to-end) — Hydra-based, NEEDS a task.
# (+phase3_p30_smoke.dry_run=true runs only the warm-start guard, no MuJoCo.)
python scripts/phase3_p30_smoke.py env=safety_bigym/saucepan_to_hob \
  disruption=coworker_train bodyslam=oracle pixels=false

# P6 benchmark harness (P6)
python scripts/benchmark_policy.py --smoke
```

All gates < 5 min on CPU. **If any fail, do not launch the long
run.**

---

## Suggested Execution Order (assuming ~70 GPU-hours)

| Day | Tasks | Notes |
|---|---|---|
| 0 (today) | Smoke gates × all; **P6** implementation begins | ~2 days engineering, parallel with GPU runs |
| 1 | P1 stage 0 launch (4 h) | Sanity check: ep_reward positive, no support saturation |
| 1 | P1 stage 1 launch resuming from stage 0 (3 h) | |
| 2 | P1 stage 2 launch resuming from stage 1 (12 h) | Long run; baby-sit |
| 3 | P6 finishes; P6 smoke + regression test on ACT snapshot | Engineering deliverable |
| 3 | P2 (SVF re-eval, ~3 h) launches as soon as P1 stage 2 finishes | |
| 4 | P3 launch (9 cells, parallel as compute allows) | E3.1 cost-signal ablation |
| 5 | P3 finishes; **diagnostic**: continuous > binary > fixed? | If not, investigate |
| 5 | P4 launch (12 cells) | E3.2 budget Pareto |
| 6 | P4 finishes; identify d_knee | Operating point fixed |
| 6 | P5 row 2 training launches (3 seeds, ~6 h) | Only new training in P5 |
| 7 | P5 rows 1/3/4/5 eval via P6 harness; P10 eval | All fast (~0.5 h total) |
| 7 | P8 internalisation curve plotted from P3 logs | Free |
| 8 | P9 WCSAC sanity gate | Decision: full run or cut |
| 9 | P9 full run if gate passes | |
| 10 | P7 obs-channel under constrained policy if budget remains | De-prioritise; valuable but not headline |
| 11–14 | Report writing: fill `\result{X}` markers; finalise figures | |

If compute slips, the cuts in priority order are: P7 → P9 → P3.6.
P1, P2, P3 (E3.1), P4 (E3.2), P5, P6, P8, P10 are mandatory.

---

## Deliverable Summary

At project completion, the following artefacts populate the report:

| Report element | Source data | Status |
|---|---|---|
| Table~\ref{tab:results:baseline} | P1 stage 2 eval via P6 | Pending P1 |
| Table~\ref{tab:e1.1-bc-ablation} | Phase 1 E1.1 data | **Already populated** |
| Table~\ref{tab:e3.1-cost-signal} | P3 outputs | Pending P3 |
| Figure~\ref{fig:e3.2-pareto} | P4 outputs | Pending P4 |
| Table~\ref{tab:e3.6-obs-rl} | P7 outputs | Pending P7 (de-prioritisable) |
| Table~\ref{tab:e3.7-wcsac} | P9 outputs | Pending P9 (cuttable) |
| **Table~\ref{tab:e4.1-feature-incremental}** (the headline) | P5 outputs | Pending P5 (depends on P1, P3, P6) |
| Figure~\ref{fig:e4.3-internalisation} | P8 post-hoc (`run_e4_3_internalisation.sh`, noisy) | Pending P3/P4 |
| Table~\ref{tab:e5.1-tail} | P10 post-hoc on P5 rolls | Pending P10 (cheap) |
| Figure~\ref{fig:e5.2-ood} | P10 OOD eval | Pending P10 (cheap) |

Each `\result{X}` marker in `report.tex` maps to a specific table
cell above; the populate-the-results pass after P10 is a mechanical
substitution.
