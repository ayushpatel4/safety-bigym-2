# Updated Project Plan: Hybrid Safety Critic for safety_bigym

## Overview

This plan supersedes the original `HYBRID_SAFETY_CRITIC_PLAN.md`. It incorporates all completed work through Phase 2 code completion, actual experimental results, revised interpretations, and the forward path through Phases 3–5.

The hybrid approach combines two mechanisms: a constrained-RL-trained policy that internalises safety through training, and a decoupled SVF safety filter that provides a runtime backup. The policy handles smooth safe behaviour in normal operation; the filter catches edge cases and provides the hard guarantees needed for ISO 15066 compliance.

---

## Plan Change Log

This revision incorporates three load-bearing changes. They cascade through every phase, so they are summarised here once.

### Change 1 — Single disruption type: `COWORKER`

The original plan trained and evaluated across five disruption types (`static`, `cross`, `reach`, `approach`, `occlude`). The revised plan trains and evaluates on a single `DisruptionType.COWORKER` (implemented and tested — see `safety_bigym/scenarios/coworker_behavior.py` and `test_coworker_disruption.py`). COWORKER itself contains internal variation via three trajectory modes (`STATIONARY`, `APPROACH_LOITER_DEPART`, `COWORKER_PATROL`) and an arm state machine with extend/hold/retract/idle phases.

Breadth of evaluation comes from **five continuous parameter axes** with strict superset eval-vs-train ranges (`make_coworker_train_space` / `make_coworker_eval_space`):

| Axis | Train (moderate) | Eval (wider) |
|---|---|---|
| `closest_approach` | 0.9–1.4 m | 0.6–1.8 m |
| `reach_period` | 4.5–6.5 s | 3.0–9.0 s |
| `target_mix_p_ee` | 0.4–0.6 | 0.1–0.9 |
| `near_loiter` | 7–11 s | 4–16 s |
| `walk_speed` | 1.0–1.6 m/s | 0.6–2.2 m/s |

The eval range is a strict superset of train on every axis (test-enforced); ≥10% of eval samples land OOD per axis. The "OOD robustness" story in Phase 5 is now parameter-axis-based, not disruption-type-based.

### Change 2 — Switch from DrQ-V2+ to CQN-AS

CQN-AS (Seo et al., 2024, "Coarse-to-fine Q-Network with Action Sequence", arXiv:2411.12155) replaces DrQ-V2+ as the backbone RL algorithm for Phase 1.4 and Phase 3. Reasons:

- **Critic-only**: no actor network, so the entire RoboBase shared-encoder constraint that justified earlier architectural decisions disappears
- **BiGym-native**: CQN-AS was designed and benchmarked on BiGym and outperforms DrQ-V2+ and ACT on 45 BiGym/RLBench tasks in the published results
- **Humanoid-capable**: validated on HumanoidBench, which is the closest published analogue to our H1 setup
- **Action sequences**: predicts Q-values over K-step action chunks rather than single actions; aligns with how ACT and Diffusion Policy structure their outputs

**Architectural consequence:** The Phase 3 Lagrangian formulation is no longer actor-critic. It is value-based: action selection is `argmax_a [Q_r(s,a) − λ · Q_c(s,a)]` over the coarse-to-fine bins, not a separate actor maximising a weighted gradient. The three-option structure (A / B-mean / B-CVaR) survives but is re-expressed in value-based form. See Phase 3 below.

**Engineering risk:** Confirm before full port that (a) CQN-AS composes cleanly with `BodySLAMWrapper` (the noisy human-state observation key must be consumable by the CQN-AS encoder), and (b) a reference implementation is available. Do a one-task smoke run on `reach_target_single` with current wrappers *before* rewriting Phase 3 in full.

### Change 3 — Workspace reward shaping

Without a task-region bias, the Lagrangian-trained policy can satisfy the cost constraint by *evacuating the workspace* — drifting far from the task object and refusing to engage. This is a known local optimum in safe RL and is incompatible with the HRC realism the project is supposed to demonstrate ("work alongside the human, not run away from them"). A workspace-bound penalty is added to the task reward:

```
r_workspace = -β · max(0, ‖p_ee − p_task‖ − r_ws)
r_total = r_task + r_workspace − λ · c_t        (Option A-value form)
```

Zero penalty inside radius `r_ws` so the policy can still dodge; quadratic outside so evacuation is expensive. `β` is calibrated by sweep in Phase 3 (new sub-experiment E3.X.workspace). The Lagrangian λ already provides an *implicit* curriculum on safety, so no separate stage-based curriculum is introduced.

**What was rejected:**
- Manual stage-based curriculum (stay-away → close → both): risks training the "evacuate" prior we are trying to avoid. Implicit λ schedule via Lagrangian PID is preferred.
- Three-policy switching architecture (task / avoid / return): conceptually overlapping with the Phase 2 filter; adds three policies' worth of training cost and gradient discontinuities; muddies the Lagrangian + filter narrative. If pursued, treated as a Phase 4 fallback upgrade (E4.2), not a Phase 3 architecture.

---

## Project Status Summary

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| Phase 0 | **COMPLETE** | Human collision fix, SSM velocity fix, eval regression fix, ACT retrain done |
| Phase 0.5 (NEW) | **COMPLETE** | `DisruptionType.COWORKER` + 3 trajectory modes + 5 parameter axes + train/eval factories |
| Phase 1 E1.1 | **COMPLETE** (legacy, multi-disruption) | BC obs-ablation negative — kept as historical; new coworker-only re-run needed only if used as report evidence |
| Phase 1.4 | **REWRITE REQUIRED** | Was DrQ-V2+ on 5 disruptions; redo as CQN-AS on COWORKER train space |
| Phase 2 | **CODE COMPLETE, DATASET REGEN REQUIRED** | SVF filter pipeline ready; existing dataset is multi-disruption, must re-collect on COWORKER train space |
| Phase 3 | **NOT STARTED, REFORMULATED** | Value-based constrained RL on CQN-AS + workspace reward shaping |
| Phase 4 | **NOT STARTED** | Full hybrid deployment + fallback upgrade |
| Phase 5 | **NOT STARTED, REFRAMED** | OOD axes are now coworker parameter ranges, not held-out disruption types |

---

## Phase 0: Preparation and Baselines — COMPLETE

### What was done

**Human collision fix:** The SMPL-H human and scene geometry shared the same MuJoCo collision channel, causing the human to penetrate dishwasher geometry and crash the simulator within ~1s. Fixed by separating collision channels using a cross-paired bit scheme (human emits on bit 1, accepts bit 2; robot/floor the reverse). Also reduced PD actuator gains from kp=2000 to kp=200 to match the HumanConfig contract. Eliminated SMPL-H self-collision (adjacent body parts producing 220 kN spurious forces).

**SSM velocity fix:** ISO 15066 SSM was reporting required separation distances of 18m because `data.cvel` reflected the implicit velocity of the human freejoint being teleported at physics timestep resolution (~120 m/s phantom velocity). Fixed by capping human velocity at `SSMConfig.v_h_max` (1.6 m/s, the ISO 15066 prescribed bound). Root cause later superseded by converting the pelvis freejoint to a `mocap="true"` body.

**Eval regression fix:** Diffusion Policy snapshots were producing 0% task success at evaluation. Root cause: `diffusers.EMAModel` is not an `nn.Module`, so its `shadow_params` (the actual inference weights) were invisible to `state_dict()` and never saved in snapshots. On reload, `EMAModel` re-initialised from the untrained actor weights. Fixed by explicitly persisting `actor.ema.state_dict()` in snapshots, with a legacy-snapshot fallback path.

**ACT retrain:** Retrained ACT policies on `reach_target_single`, `dishwasher_close`, `drawers_open_all`, and `saucepan_to_hob` with the fixes above. Snapshots validated end-to-end. Task swap: `dishwasher_load_plates` (never learned) replaced with `saucepan_to_hob`.

**Diagnostic and regression tests:** 9 new tests across `test_collision_groups.py` and `test_safety_preserved.py`. 4 new diagnostic scripts (`diagnose_truncation.py`, `diagnose_contacts.py`, `diagnose_no_human.py`, `diagnose_spawn_geometry.py`).

### Deliverable

Working ACT snapshots across 4 tasks, physically stable simulator, correct SSM computation, verified eval pipeline. All downstream phases build on this foundation.

---

## Phase 0.5: COWORKER Disruption Infrastructure — COMPLETE

### What was built

Single sustained-coworker disruption replaces the previous six-type catalogue as the only disruption used in training and evaluation. Implementation lives in `safety_bigym/scenarios/coworker_behavior.py`, `scenario_sampler.py`, and the trajectory builders in `human/trajectory_planner.py`. 19 tests in `test_coworker_disruption.py` (`tests/test_coworker_disruption.py`) cover parameter-space invariants, train/eval superset relationships, arm gating, and rest-pose geometry.

**Three trajectory modes** (uniformly sampled per episode by `_select_trajectory_type`):
- `STATIONARY` — human spawns at NEAR, stays put for the whole episode
- `APPROACH_LOITER_DEPART` — human spawns far, walks in to NEAR, stays
- `COWORKER_PATROL` — walk in to NEAR, depart to AWAY (90°–270° offset, 2–3 m), loiter, return to a *new* NEAR (different angle, different distance), repeat for 1–2 excursions

**Arm state machine** (independent of trajectory phase): EXTEND (15%) → HOLD (20%) → RETRACT (15%) → IDLE (50%) within each cycle, period `T ∈ [4.5, 6.5] s`. Each cycle samples a target — EE or task object — via `coworker_target_mix` (default 0.5/0.5).

**Reach gate** at 0.75 m shoulder-to-target distance suppresses IK when the human is too far (i.e. during AWAY phase of patrol), producing a natural arms-down stance instead of waving at an unreachable target.

**Five parameter axes** with strict-superset train/eval ranges enforced by tests:

| Axis | Train | Eval |
|---|---|---|
| `coworker_closest_approach_range` | 0.9–1.4 m | 0.6–1.8 m |
| `coworker_reach_period_range` | 4.5–6.5 s | 3.0–9.0 s |
| `coworker_target_mix_p_ee_range` | 0.4–0.6 | 0.1–0.9 |
| `coworker_near_loiter_range` | 7–11 s | 4–16 s |
| `coworker_walk_speed_range` | 1.0–1.6 m/s | 0.6–2.2 m/s |

`make_coworker_train_space()` / `make_coworker_eval_space()` factories force `disruption_weights = {COWORKER: 1.0}`. Hydra presets ship at `cfgs/disruptions/coworker_{train,eval}.yaml`.

### Implications downstream

- Phase 2 dataset is regenerated on the COWORKER train space — old multi-disruption transitions are stale.
- Phase 1 E1.1 multi-disruption results remain reportable as historical context but are no longer directly comparable; Phase 1.4 / Phase 3 eval rows use COWORKER only.
- Phase 5 OOD axes are the five eval-vs-train deltas above, plus possibly evaluation-only patrol-frequency / spawn-distance probes.
- Failure-mode breakdown is now by trajectory mode (`STATIONARY` / `APPROACH_LOITER_DEPART` / `COWORKER_PATROL`) and/or by parameter-axis bin (in-distribution vs OOD), not by disruption type.

### Deliverable

A single, parameterised, sustained-coworker disruption with verified train/eval distributions and visualisation tooling (`mjpython scripts/demo_coworker.py`, `scripts/record_coworker_videos.py`). All downstream phases consume the same `ParameterSpace`.

---

## Phase 1: Mock BodySLAM++ Observation Wrapper — PARTIALLY COMPLETE

### Phase 1 Wrapper Build — COMPLETE

**`BodySLAMWrapper`** implemented as a `gym.Wrapper` injecting `human_pos_estimate` (6D: xyz position + occluded flag + staleness counter + confidence score) into the observation space. Three modes via Hydra config: `bodyslam=off` (baseline), `bodyslam=oracle` (ground truth), `bodyslam=noisy` (full noise pipeline).

**Noise pipeline (mode=noisy):**
- Ornstein-Uhlenbeck temporally-correlated position noise (α=0.9, σ=0.05m, stationary std ≈ 0.115m)
- 3-step latency buffer (~60ms lag at 50 Hz control)
- Stochastic tracking dropout (p=0.02/step) with staleness tracking — OU internal state continues updating during dropout to avoid recovery discontinuity
- Optional ray-cast occlusion detection from the H1 head camera (opt-in via `bodyslam.use_occlusion=true`)

**AMASS-driven demo replay:** During BC pretraining (where no live human exists), the wrapper plays back real AMASS motion clips via `AMASSDemoPositionProvider` so the policy sees realistic human state during pretraining, avoiding a train-test distribution shift on the observation channel.

**19 tests passing** covering OU statistics, latency, dropout recovery, confidence derivation, demo replay, factory integration, and occlusion via minimal custom MJCF.

### E1.1: BC Observation Ablation — COMPLETE (Negative) [LEGACY MULTI-DISRUPTION]

**Status note.** E1.1 was run before the COWORKER simplification, across 6 disruption types. The qualitative finding (oracle does not help under pure BC) is robust to the disruption change — the underlying cause is the absence of a reward gradient, not the disruption mix. The numbers below remain reportable as historical context. No re-run on COWORKER-only is planned unless E1.4 / Phase 3 produces a result that requires direct comparison.

**Question:** Does adding human state observation to a pure BC policy (ACT) reduce SSM violation rate?

**Setup:** ACT, 4 tasks, 3 obs modes (off / oracle / noisy), 6 disruption types, n=10 episodes per cell. No violation penalty active during training.

**Result:**

| Task | off SSM | oracle SSM | Δ off→oracle |
|------|---------|------------|--------------|
| reach_target_single | 0.534 | 0.548 | −2.7% (worse) |
| dishwasher_close | 0.533 | 0.537 | −0.8% (worse) |
| drawers_open_all | 0.277 | 0.239 | +13.7% (best, below 20% bar) |
| saucepan_to_hob | 0.135 | 0.203 | −49.9% (much worse) |

**No cell clears the 20% success criterion.**

**Critical side-finding on saucepan:** Oracle dramatically improves task success (0.22 → 0.58) but worsens SSM violations (0.135 → 0.203). The policy uses human state to route around obstructions and complete the task, not to maintain safe distance. This reveals that the policy prioritises task completion over safety when given human state — because there was no safety incentive in the training objective.

**Interpretation caveat:** E1.1 ran without any violation penalty. BC has no reward gradient, and the demos were collected without a human, so `human_pos_estimate` and demo actions are statistically uncorrelated. BC marginalises the channel away. E1.1 cannot distinguish "the channel is useless" from "BC cannot learn to use it without a reward signal." This ambiguity motivates E1.4.

### E1.4: RL Reward-On Pilot — REWRITE REQUIRED (CQN-AS + COWORKER)

**Status note.** The DrQ-V2+ launch config (`drqv2plus_pixel_safety_bigym.yaml`), violation-penalty wiring, and sweep script (`phase1_reward_pilot.py`) remain useful as templates. The core experimental setup transfers; only the backbone algorithm and disruption sampler change. Existing DrQ-V2+ code is not deleted — kept as a fallback if CQN-AS integration encounters blockers.

**Question:** Does the `human_pos_estimate` channel help when the training algorithm has access to a safety reward signal? (Unchanged from original framing.)

**Setup (revised):** CQN-AS (critic-only, coarse-to-fine discrete-action RL with K-step action sequences), `reach_target_single`, 3 obs modes (off / oracle / noisy), `violation_penalty=0.05` active for all cells, COWORKER train ParameterSpace, ~200k env frames per cell.

**Pre-port checklist (do these before any full training run):**

1. **Reference implementation.** Pull CQN-AS reference code (Younggyo Seo / project page `younggyo.me/cqn-as`) and confirm it runs end-to-end on stock BiGym `reach_target_single`.
2. **Wrapper composition smoke.** Run CQN-AS on `reach_target_single` with `BodySLAMWrapper` + COWORKER scenario sampler active. Confirm the encoder accepts the `human_pos_estimate` observation key (may require minor encoder modification to fuse the 6D vector with proprio).
3. **Action-sequence vs SSM cost.** Verify the cost signal `c_t` is logged per-step inside the K-step chunk, not only at chunk boundaries. If only chunk-level, the policy will optimise mean-over-chunk cost and miss spikes — adjust the cost aggregator to use max-over-chunk or per-step.
4. **Action space.** CQN-AS discretises each of the 76 joint-position dimensions into B bins per coarse-to-fine level. Confirm `B`, the number of levels, and the resulting per-step action vocabulary are tractable for the H1 setup. Default CQN-AS values from the BiGym experiments should work directly.

Only after the smoke run passes (steps 1–3 above) commit to the full 3-cell GPU sweep.

**Decision rule:** unchanged from the original plan — channel helps under RL with reward signal, or doesn't; treat the answer as gating Phase 3's observation configuration.

### E1.2 and E1.3 — PARKED

Unchanged: predicated on a strong (method, task) cell, which E1.1 didn't produce. Unblocked only by E1.4 or Phase 3 surfacing one.

### GPU Work Required (revised)

- Pre-port smoke: ~2–4 GPU-hours to confirm CQN-AS + BodySLAMWrapper + COWORKER compose
- E1.4 full: 3 cells × CQN-AS training (~10–15 GPU-hours total — CQN-AS is more sample-efficient than DrQ-V2+ per BiGym results, so allow similar wall-clock to original plan)
- Eval against COWORKER eval ParameterSpace (20 episodes × seeds)
- Author decision based on table in original plan

---

## Phase 2: Offline SVF Safety Filter — CODE COMPLETE, GPU PENDING

### What was built

**Full pipeline implemented** across 11 modules in `safety_bigym/filters/`:

| Module | Purpose |
|--------|---------|
| `labeling.py` | `r_safe = 0 if ssm_violation else 1` (PFL flag wired but inert until contact bug is fixed) |
| `feature_extractor.py` | `CriticFeatureSpec` + `make_critic_input` — proprioception + BodySLAM++ estimate + action, no pixels |
| `dataset.py` | `SafetyTransitionDataset` + `TransitionShardWriter` + `WeightedRandomSampler` for violation oversampling |
| `critic.py` | Bounded-output MLP [256, 256, 256], `q_max = 1/(1-γ) = 100`, target network + Polyak averaging |
| `cql_trainer.py` | Bellman MSE + α·CQL regulariser + optional auxiliary loss (currently inert) |
| `fallback.py` | `ZeroVelocityFallback` + `FallbackRegistry` |
| `runtime_wrapper.py` | `SafetyFilterWrapper(gym.Wrapper)` — intercepts actions, vetoes if `Q_safe(s,a) < R` |
| `threshold_sweep.py` | `evaluate_threshold` + `sweep_thresholds` — traces the conservatism-performance Pareto frontier |
| `snapshots.py` | Per-task `SNAPSHOTS` dict + `resolve_snapshot()` |

**Four CLI scripts** (all with `--smoke`): dataset collection, CQL training, filter evaluation, threshold sweep.

**80 tests passing** across 12 test files (~38s on CPU).

**End-to-end smoke pipeline validated on GPU box** — collect → train → eval → sweep runs in ~75s.

**Three data sources for the safety dataset:**
- Random policy — ensures coverage of unsafe state-action regions
- BiGym demonstrations — relabelled with binary safety reward
- Phase 0 ACT snapshots — coverage of deployment-realistic robot behaviour

### How CQL Training Works

**Input to the critic:** Concatenated vector of robot proprioception + BodySLAM++ human pose estimate (6D) + the proposed 76-DOF robot action. No pixels.

**Output:** Single scalar `Q_safe(s, a)` bounded to [0, 100] via scaled sigmoid. Represents expected sum of future safety rewards. 100 = perfectly safe; 0 = violation imminent.

**Training objective:**
```
L_total = L_bellman + α · L_cql

L_bellman = (Q(s,a) - [r_safe + γ · Q_target(s', a')])²
L_cql = α · [E_{a~uniform}[Q(s,a)] - E_{a~dataset}[Q(s,a)]]
```
CQL pushes down Q-values for out-of-distribution actions so the critic never falsely certifies an unseen action as safe. The bounded output prevents overestimation mathematically.

**Runtime:** The filter compares `Q_safe(s, a)` against threshold R. If `Q_safe ≥ R`: pass through. If `Q_safe < R`: veto and substitute zero-velocity braking. R controls the conservatism-performance trade-off; the threshold sweep traces the Pareto frontier.

**Filter under CQN-AS (no change to the filter itself).** CQN-AS outputs a K-step action sequence; at execution the first action of the sequence is filtered. The filter input is still `(s, a_t)` with `a_t` a single 76-DOF action — Phase 2 code does not change. Sequence-aware filtering (`Q_safe(s, a_{t:t+K})`) is a Phase 4 option if needed; not in the v1 plan.

### GPU Work Required (revised for COWORKER)

1. **Populate SNAPSHOTS dict** — pick W&B peak-by-eval-success checkpoints from Phase 0 ACT retrain (unchanged)
2. **Dataset regen on COWORKER train space** — previous 5-disruption dataset is stale. New collection: ~310k transitions across 2 tasks × COWORKER train ParameterSpace × 3 sources (random / BiGym demos / Phase 0 ACT snapshots) (~2–3 hours). Sampler-side change is small: drop the multi-disruption mix, swap in `make_coworker_train_space()` factory.
3. **CQL training** — unchanged: 200k gradient steps, batch size 512, α=5.0 (~30–60 min single GPU)
4. **Evaluation against ACT policy** — evaluate on COWORKER eval ParameterSpace (wider ranges) so the filter is stress-tested on OOD parameter values from its first measurement
5. **Threshold Pareto sweep** — R ∈ {5, 10, 25, 50, 75, 90, 95}, plot intervention rate vs residual violation rate; report both in-distribution (train ranges) and OOD (eval-only ranges) cells

### Experiments

**E2.1 — CQL α sweep (deferred):** α ∈ {1.0, 5.0, 10.0}. Starting with α=5.0 only; full sweep is a fast-follow if v1 Pareto curve looks sane.

**E2.2 — Filter effectiveness:** Apply the filter to the Phase 0 ACT policy (without retraining) and measure SSM violation rate reduction vs unfiltered baseline. This is the core Phase 2 deliverable.

**E2.3 — Threshold Pareto curve:** For the α=5.0 critic, sweep R and plot violation rate vs intervention rate. Identify the knee of the curve as the operating point.

**E2.4 — Robustness to perception noise (deferred):** Test with σ_test > σ_train. Deferred to Phase 5 evaluation.

### Possible outcomes

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| Pareto curve shows clear knee; filter reduces violations with <20% intervention rate | SVF is a viable runtime safety net | Proceed to Phase 3, carry filter into Phase 4 hybrid |
| Filter reduces violations but intervention rate is >50% | Critic is too conservative or threshold is wrong | Sweep wider R range; try α=1.0; check if critic is undertrained |
| Filter has no effect on violation rate regardless of R | Critic hasn't learned the safety landscape | Debug: check dataset violation rate, Q-value distributions, Bellman convergence |
| Intervention rate = 1.0 everywhere | Q-values uniformly low; either undertrained or R too high | Longer training; sweep R down to R=2 |

### Deliverable

A frozen safety filter module + runtime wrapper + measured Pareto frontier showing the intervention rate vs residual violation rate trade-off. At this point we have a working safety mechanism independent of the task policy.

---

## Phase 3: Constrained RL Integration — NOT STARTED

### Goal

Train a task policy that internalises safety via a Lagrangian cost constraint, using continuous cost signals and (optionally) the Phase 2 filter to prevent unsafe exploration during training.

### Design decisions gated on E1.4

| E1.4 outcome | Phase 3 obs configuration | Rationale |
|---|---|---|
| Channel helps under RL | `bodyslam=noisy` for actor | Policy can exploit human state with reward gradient |
| Channel doesn't help | `bodyslam=off` for actor | Channel is consumed only by Phase 2 filter, not the policy |
| Oracle helps but noisy doesn't | `bodyslam=oracle` during training, investigate noise model | Noise model may be too aggressive |

### Integration strategy (rewritten for CQN-AS — value-based)

CQN-AS is critic-only. There is no actor network; the policy is implicit as `argmax_a Q(s, a_bin)` over the coarse-to-fine discretised action bins. The three Phase 3 options of the original plan (single-critic A; dual-critic B-mean; distributional B-CVaR) survive *as a conceptual structure*, but their implementation is re-expressed in value-based form.

**Option A-value — Reward-shaping Lagrangian (prototype only).** Single Q-network on a modified scalar reward `r' = r_task + r_workspace − λ · c_t`. Action selection `argmax_a Q(s, a)` over CQN-AS bins. PID on λ updates as before. Same conceptual non-stationarity issue as the original Option A — each λ update changes the reward target the critic is regressing toward.

**Option B-value-mean — Dual Q-networks on expected cost (recommended starting headline).**
- Task Q-network `Q_r(s, a)` learned from `r_task + r_workspace` (standard CQN-AS objective)
- Cost Q-network `Q_c(s, a)` learned from per-step cost `c_t` (Bellman regression, max-of-twins pessimism)
- Action selection at each coarse-to-fine level: `argmax_a [Q_r(s, a) − λ · Q_c(s, a)]`
- λ updates via PID on rolling mean cost (unchanged)

`Q_c` is architecturally identical to the Phase 2 offline safety value function (same input shape: proprio + BodySLAM++ estimate + action; same MLP backbone); weight transfer from Phase 2 at Phase 3 initialisation is supported by construction. Each Q-network faces a stationary prediction target (its own scalar). This is the value-based analogue of the original B-mean option.

**Option B-value-CVaR — Distributional cost Q-network on tail cost (recommended final headline).**
Same as B-value-mean, but `Q_c` is distributional — either a Gaussian (WCSAC-style) or a quantile (QR-DQN-style) head. At each level, action selection is `argmax_a [Q_r(s, a) − λ · CVaR_α(Z_c(s, a))]` with α ∈ {0.95, 0.99}, and the cost budget `d` is now a target on rolling CVaR rather than rolling mean. Same justification as before: aligns training objective with the Phase 5 tail-risk metric, and answers the standard critique of mean-cost safe RL.

### What the shared-encoder discussion becomes

The original plan's framing — "RoboBase's ActorCritic shares an encoder, so we keep the cost critic decoupled to avoid coupled gradient updates" — is no longer the justification, because CQN-AS has no actor. The decoupled cost critic is now justified on two stronger grounds:

1. **Stationary targets.** Each Q-network regresses toward its own scalar (task reward, mean cost, or cost-return quantiles). λ enters only at action selection time, not in any critic's regression target. This eliminates non-stationarity by construction, not as a workaround.
2. **Weight transfer.** The Phase 2 offline SVF is architecturally identical to the Phase 3 mean-cost critic, enabling warm-start. The two safety critics serve different runtime roles (one shapes training, one filters at deployment) but share the same representational substrate.

The report's `Section~\ref{sec:method:asymmetry}` ("Asymmetric Observation Handling") needs to be re-justified on these grounds, not on RoboBase architectural constraints.

### Workspace reward shaping (new sub-component)

Added to `r_task` *before* it reaches any Q-network:

```python
r_workspace = -β * max(0, ‖p_ee − p_task‖ − r_ws)
r_task' = r_task + r_workspace
```

Defaults: `r_ws = 0.4 m`, `β` swept in {0.0, 0.05, 0.2, 0.5, 1.0} as part of E3.X.workspace below. `r_ws` chosen so the policy can still dodge the human up to ~30 cm without paying workspace tax; β calibrated against the cost weight to ensure that "small workspace tax to dodge a violation" beats "large workspace tax to permanently evacuate."

### Continuous cost signal design (unchanged)

```python
d_buffer = 0.3  # metres — activate cost before violation
c_ssm = max(0, 1.0 - ssm_margin / d_buffer)    # 0 far away, 1 at violation boundary
c_pfl = max(0, pfl_force_ratio - 0.8)           # activate before threshold 1.0
c_t = max(c_ssm, c_pfl)                          # worst-case across both ISO criteria
```

**Per-step aggregation under action sequences.** CQN-AS executes K-step sequences. The Bellman target for `Q_c` must be computed per executed step, not per K-step chunk, or the policy will satisfy the mean-over-chunk cost while spiking violations within chunks. Confirm during the pre-port smoke that the CQN-AS Bellman backup runs at single-step resolution for the cost critic.

### PID-controlled λ update (unchanged)

```python
cost_violation = rolling_mean_cost - d           # rolling CVaR_α for B-value-CVaR
λ = max(0, λ + K_I · cost_violation + K_P · cost_violation + K_D · Δcost_violation)
λ = min(λ, λ_max)
```

Starting hyperparameters: `K_I = 1e-3, K_P = 1e-2, K_D = 0, λ_max = 100, d = 0.01`.

### Training with Phase 2 filter

Unchanged in concept: the SVF filter sits between action selection and the environment during training rollouts, vetoing catastrophically unsafe actions. Under CQN-AS this means: after the coarse-to-fine argmax produces a candidate action, check `Q_safe(s, a)` and substitute fallback if below threshold.

### Experiments (revised)

**E3.1 — Cost signal comparison:** unchanged in intent.
- Fixed -0.05 penalty (E1.4 baseline)
- Binary 0/1 cost with Lagrangian λ
- Continuous smooth cost with Lagrangian λ

**E3.2 — Cost budget Pareto sweep:** d ∈ {0.001, 0.01, 0.05, 0.1} (unchanged).

**E3.3 — λ update method:** gradient ascent vs PID (unchanged).

**E3.4 — Filter during training:** filter on/off (unchanged).

**E3.5 — Architecture comparison (revised):** Option A-value vs B-value-mean vs B-value-CVaR. Headline metrics on COWORKER eval ParameterSpace.

**E3.6 — Observation channel (if E1.4 results allow):** `bodyslam=off` vs `bodyslam=noisy` under B-value-CVaR (unchanged).

**E3.7 — External baseline: WCSAC (unchanged):** the comparison remains useful even though we've moved to value-based RL — WCSAC is *the* distributional safe-RL reference. Reimplementing it on a 76-DOF humanoid is the relevant contribution; whether WCSAC itself is actor-critic is a methodological footnote.

**E3.X.workspace (new) — Workspace shaping β sweep:** β ∈ {0.0, 0.05, 0.2, 0.5, 1.0} on `reach_target_single` under B-value-mean. Measures: task success, mean SSM violation, mean distance-to-task-object across episode, fraction of episodes where end-effector left the workspace radius. Default β is chosen at the knee — smallest β that prevents evacuation without measurable task-success loss. This is the experiment that defends the workspace-penalty choice against an examiner asking "why this β?".

### Pre-port smoke gates (must pass before full Phase 3 GPU sweep)

1. CQN-AS reference code runs end-to-end on stock `reach_target_single` (no wrappers).
2. CQN-AS + BodySLAMWrapper + COWORKER scenario composes (a few k env steps, no crashes, observation key consumed).
3. Per-step cost backup confirmed for `Q_c` (not chunk-level).
4. Workspace-penalty wiring lands in `r_task` before Q-learning, with the right sign.

### Possible outcomes

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| Continuous cost + Lagrangian reduces SSM violations >50% vs baseline with <10% task reward loss | Constrained RL works; policy internalises safety | Proceed to Phase 4 hybrid |
| Violations reduce but task reward collapses (>30% loss) | λ too aggressive or cost budget too tight | Loosen budget d; clamp λ_max lower; investigate reward scaling |
| Oscillation — policy swings between safe-and-frozen and unsafe-and-productive | Lagrangian update is unstable | Switch to PID updates if not already; increase K_D; reduce K_P |
| No improvement over fixed penalty baseline | The wrapper-level approach (Option A) is too crude | Consider Option B (dual-critic with separate safety head inside RoboBase) |
| Continuous cost helps but binary doesn't | Confirms the gradient richness hypothesis — binary penalty is too sparse | Strong evidence for the continuous formulation; validates the plan |

### Deliverable

A Lagrangian-trained policy that achieves baseline task performance with significantly reduced SSM violation rate, without the runtime filter active at evaluation time.

---

## Phase 4: Full Hybrid Deployment and Fallback Upgrade — NOT STARTED

### Goal

Combine the Phase 3 Lagrangian-trained policy with the frozen Phase 2 safety filter, and improve the fallback action for smooth recovery.

### Architecture

```
Observation → Lagrangian-trained actor → proposed action u_nom
                                                  │
                                                  ▼
                                          Safety filter
                                          Q_safe(s, u_nom) ≥ R?
                                                  │
                                         yes  ────┴──── no
                                          │              │
                                       execute        execute
                                       u_nom          u_safe
```

### Fallback action upgrade

Replace zero-velocity braking with one of:
- **Proportional damping:** `u_safe = (Q_safe / R) · u_nom` — preserves direction, scales magnitude with predicted safety
- **Trajectory replay:** Cache last N=10 safe actions; on filter trigger, replay most recent. Momentum-preserving
- **Retreat controller:** Repulsive velocity component pointing away from nearest human body part
- **Learned retreat policy (new, optional):** A small dedicated CQN-AS policy trained to maximise `Q_safe`, invoked when the filter triggers. Concept overlap with Recovery RL; included here only if simpler fallbacks underperform on smoothness or recovery time. Decision gate: only train this if E4.2 shows the three simpler fallbacks all produce >X% task failure on filter trigger.

### Threshold re-calibration

Because the Lagrangian-trained policy is already mostly safe, the filter triggers rarely. This means a more aggressive (less conservative) R than Phase 2's calibration. Re-run the Pareto sweep with the new policy.

### Experiments

**E4.1 — Full hybrid comparison:** Four configurations across all tasks and disruption types:
- Baseline (ACT + no penalty)
- Lagrangian policy alone (no filter)
- Safety filter alone (with baseline ACT)
- Full hybrid (Lagrangian policy + filter)

**E4.2 — Fallback action ablation:** Sweep across the three fallback options. Measure task success, smoothness (jerk), and recovery time after intervention.

**E4.3 — Intervention rate analysis:** Confirm that the Lagrangian policy's internalised safety reduces filter dependence vs filter-alone.

### Deliverable

The full hybrid system operating smoothly on all tasks with measurably better safety than any single approach.

---

## Phase 5: Evaluation and Stress Testing — NOT STARTED

### Goal

Rigorous final evaluation including tail-risk metrics and robustness testing.

### Metrics

- CVaR(0.95) of max contact force across episodes
- 95th and 99th percentile of SSM margin at closest approach
- Distribution of time-to-first-violation
- Max force ever observed (not mean)
- Computational overhead: time per step for policy forward pass + filter evaluation + fallback

### Stress tests (reframed for COWORKER parameter axes)

**E5.1 — Tail-risk evaluation:** Report CVaR and max-over-distribution metrics alongside means, on the COWORKER eval ParameterSpace.

**E5.2 — Out-of-distribution robustness:** Two axis groups, both available "for free" because the eval ParameterSpace already contains samples outside the train ranges:

- **Perception OOD:** σ_test ∈ {0.05, 0.10, 0.15, 0.20} m beyond the training-time σ_train = 0.05 m.
- **Scenario OOD:** five coworker parameter axes already span eval-only regions (closer approach, faster reach period, more EE-targeted reaching, longer / shorter NEAR loiter, faster walk). Report safety metrics on in-distribution samples vs eval-only samples to quantify the OOD degradation per axis.

The strict-superset eval ParameterSpace makes this comparison test-enforced rather than an after-the-fact carve-out.

**E5.3 — Real BodySLAM++ comparison:** If feasible, render MuJoCo camera images and run real BodySLAM++ on them. Compare error distributions to the Mock BodySLAM++ noise model. Quantifies the sim-to-real transfer gap.

**E5.4 — Computational feasibility:** Confirm real-time feasibility at 50 Hz control rate on target hardware. With CQN-AS the relevant cost is the coarse-to-fine zoom (multiple Q evaluations per action selection), not actor inference — verify this lands inside the 20 ms control budget.

### Deliverable

Final evaluation report with tables, Pareto curves, and stress-test results. Ready for writeup.

---

## Critical Path and Dependencies (Updated)

```
Phase 0 (DONE) ──► Phase 1 wrapper (DONE) ──► E1.1 BC ablation (DONE, negative)
                                              │
                                              ├──► E1.4 RL reward-on (GPU PENDING)
                                              │         │
                                              │         ▼
                                              │    Decision: does obs channel help under RL?
                                              │         │
                   Phase 2 code (DONE) ───────┤         ├── yes → Phase 3 uses bodyslam=noisy
                        │                     │         └── no  → Phase 3 uses bodyslam=off
                        ▼                     │                   (channel feeds filter only)
                   Phase 2 GPU work ──────────┤
                        │                     │
                        ▼                     ▼
                   Pareto curve          Phase 3 (constrained RL)
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                              Phase 4 (hybrid)
                                   │
                                   ▼
                              Phase 5 (eval)
```

**E1.4 and Phase 2 GPU work can run in parallel** — they are independent. Phase 3 depends on both: E1.4 determines the obs configuration, Phase 2 provides the training-time safety filter.

---

## Open Bugs

### PFL contact-detection bug (HIGH PRIORITY)

`ep_pfl_violation_rate`, `ep_max_pfl_force_ratio`, and `ep_max_contact_force` are identically zero across every experimental cell — including cases where `ep_min_ssm_margin = -13m` (the human pelvis is geometrically inside the robot). Root cause is in BiGym/MuJoCo's runtime robot attachment suppressing `data.ncon` for human↔robot pairs despite collision eligibility.

**Impact:** All safety labels are SSM-only. The `use_pfl` flag is wired but inert in the Phase 2 labeller, the Phase 2 critic, and the Phase 3 cost signal. PFL gets retrofitted when the bug is fixed — flip the flag and re-collect/retrain.

**Non-blocking for:** Phases 2–4. SSM-only labels are sufficient for the safety filter and constrained RL. PFL is the higher-fidelity signal (force-based rather than distance-based) but not gating.

### SSM margin outliers

Occasional `ep_min_ssm_margin` values of -16m or -25m appear in evaluation data. These are single-episode artifacts where the SSM formula's velocity terms produce physically unreasonable required separation distances. The v_h_max cap (Phase 0 fix) handles the human velocity; the robot velocity term can still produce large values when the robot is tumbling or moving fast. Consider clamping or investigating further.

---

## Risk Register (Updated)

### Materialised risks

- **Phase 1 shows no benefit from human state (original risk):** Partially materialised. E1.1 was negative, but the experiment was confounded by the absence of a safety penalty. E1.4 disambiguates. Risk level: **reduced but not eliminated**.

### Resolved-by-construction (no longer active)

- **~~RoboBase ActorCritic shared-encoder surgery proves too invasive.~~** Eliminated by the switch to CQN-AS, which is critic-only. The architectural decision around the cost critic is now justified on stationarity/weight-transfer grounds, not as a workaround.

### Active high-severity risks

- **CQN-AS integration does not compose with BodySLAMWrapper or COWORKER sampler.** Mitigation: pre-port smoke gates (1–4 above) before any full training run. Fallback path: keep DrQ-V2+ E1.4 code as a regression option.
- **CQN-AS action-sequence aggregation hides per-step violations.** If the cost Bellman backup operates at K-step chunk resolution, the policy will satisfy mean-over-chunk cost while spiking within chunks. Mitigation: enforce per-step Bellman target on `Q_c`; verified in pre-port smoke.
- **Workspace shaping β too low — evacuation persists.** Mitigation: E3.X.workspace sweep characterises the β–evacuation-rate relationship; choose at the knee.
- **Workspace shaping β too high — policy holds position into violations.** Mitigation: same sweep; the β–violation-rate relationship reveals the upper bound.
- **Distributional cost critic (B-value-CVaR) doesn't learn well on available data.** Quantile regression needs density across the return distribution; if the safety dataset is too sparse in the high-cost tail, upper-quantile estimates will be poorly calibrated. Mitigation: oversample violating transitions (already in Phase 2 sampler), use Gaussian parametrisation (WCSAC-style) as less data-hungry fallback, treat E3.5 as the diagnostic.
- **Lagrangian training oscillates or collapses.** Mitigation: clamp λ_max, use PID updates from day one, start with loose cost budget and tighten gradually. Option B-value's decoupled critics reduce oscillation relative to Option A-value.
- **Phase 2 critic doesn't learn the safety landscape on ~310k transitions (now collected under COWORKER train space).** Mitigation: check violation rate in dataset (target 5–10%), verify Bellman loss decreases, inspect Q-value distributions.
- **COWORKER train ParameterSpace produces too few violations for filter training.** Mitigation: increase random-policy fraction in dataset sources; add a violation-oversampling pass at training time (already in `WeightedRandomSampler`).

### Active medium-severity risks

- **CQL conservatism too aggressive, filter freezes the robot.** Mitigation: sweep α, upgrade fallback to proportional damping.
- **PFL contact bug remains unresolved through Phase 5.** Mitigation: SSM-only evaluation is defensible but weakens the ISO 15066 compliance claim.
- **Zero-velocity fallback destabilises the humanoid mid-task.** Mitigation: accepted for Phase 2 v1; proportional damping in Phase 4.
- **WCSAC external baseline (E3.7) doesn't reproduce reported performance.** Reimplementing safe-RL methods is notoriously fragile. Mitigation: use authors' code where available; if it fails, fall back to comparing against the Safety-Gym Lagrangian baseline that ships with most safe-RL codebases. Even an "honest reimplementation that we couldn't match" is reportable if the methodology is documented.
- **CQN-AS K-step horizon and filter granularity mismatch.** Filtering only on the first action of a chunk may miss in-chunk violations. Mitigation: v1 plan filters first-action-only; sequence-aware filter is a Phase 4 upgrade path if needed.

---

## Compute Budget Estimate

| Work item | GPU-hours (approx) |
|-----------|-------------------|
| E1.4 (3 training cells) | 9–12 |
| Phase 2 dataset collection | 2–3 |
| Phase 2 CQL training (α=5.0) | 0.5–1 |
| Phase 2 eval + sweep | 1–2 |
| Phase 3 E3.1 (3 cost formulations) | 15–20 |
| Phase 3 E3.2 (4 cost budgets) | 15–20 |
| Phase 3 E3.3 + E3.4 ablations | 10–15 |
| Phase 4 E4.1 (4 configurations × tasks) | 5–10 |
| Phase 5 evaluation | 5–10 |
| **Total remaining** | **~60–95** |

---

## Summary

The project has cleared the infrastructure phase (Phase 0 + Phase 0.5) and now has a single sustained-coworker disruption with verified train/eval parameter splits, a working ACT baseline, and a code-complete safety-filter pipeline. Three architectural changes are now baked into the plan: COWORKER as the only disruption, CQN-AS as the RL backbone, and workspace reward shaping as an addition to the task reward.

The immediate critical path:

1. **CQN-AS pre-port smoke (this week).** Pull the reference implementation, run it on stock `reach_target_single`, then confirm composition with `BodySLAMWrapper` + COWORKER sampler. Gate every Phase 1.4 / Phase 3 commitment on this passing.
2. **Phase 2 dataset regen on COWORKER train space.** ~310k transitions × 2 tasks × 3 sources. Replaces the stale 5-disruption dataset.
3. **In parallel:** E1.4 CQN-AS observation ablation (3 cells × `reach_target_single`) once the smoke gate is clear.

Phase 3 begins after both. The Phase 3 structure is unchanged at the level of experimental questions (A vs B-mean vs B-CVaR; budget sweep; PID vs gradient; filter on/off) but re-expressed in value-based form, with a new workspace-β sweep (E3.X.workspace) added to defend the reward-shaping choice.

The key architectural insight from the original plan remains: the hybrid (constrained RL training + runtime safety filter) is the right design regardless of any single phase's outcome. What CQN-AS changes is *how* constrained RL is implemented, not whether the hybrid story holds. Each phase continues to produce a standalone deliverable.
