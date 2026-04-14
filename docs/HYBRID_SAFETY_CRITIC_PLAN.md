# Implementation Plan: Hybrid Safety Critic for safety_bigym

## Overview

The hybrid approach combines two mechanisms: a constrained-RL-trained policy that internalises safety through training, and a decoupled SVF safety filter that provides a runtime backup. The policy handles smooth safe behaviour in normal operation; the filter catches edge cases and provides the hard guarantees needed for ISO 15066 compliance.

This plan is structured in five phases over roughly 10 weeks. Each phase produces a standalone deliverable that works even if later phases are delayed.

---

## Phase 0: Preparation and Baselines (Week 1)

**Goal:** Establish the current state of the system and measure baseline safety metrics before any changes.

### Tasks

- **Fix the unused continuous cost signals.** `info["safety"]["ssm_margin"]` and `info["safety"]["pfl_force_ratio"]` are already computed by the wrapper but not consumed anywhere. Verify they are correct, logged, and accessible during training. This is a prerequisite for everything else
- **Establish baseline metrics.** Run the current Diffusion Policy setup on all three configured tasks (`reach_target_single`, `dishwasher_load_plates`, `dishwasher_close`) across all five disruption types. Record SSM violation rate, PFL violation rate, task success rate, mean episode reward, time-to-first-violation, and max contact force
- **Fix the hardcoded paths.** The `/Users/ayushpatel/...` paths in `safety_bigym_factory.py` and `safety_bigym.yaml` must be parametrised via Hydra config to enable multi-machine experimentation
- **Clean up the dual SSMConfig classes.** Consolidate `config.py` and `iso15066_wrapper.py` versions to avoid the bug where edits to one don't affect behaviour
- **Set up W&B project structure** with consistent naming for the ~40+ experimental runs this plan will generate

### Deliverable

A baseline table showing current safety metrics and task performance across all tasks and disruption types. This is the reference everything else will be compared against.

---

## Phase 1: Mock BodySLAM++ Observation Wrapper (Week 2)

**Goal:** Validate that human state information helps at all, before investing in complex safety mechanisms.

### Tasks

- **Build `BodySLAMWrapper`** as a `gym.ObservationWrapper` that:
  - Reads `info["safety"]["human_pos"]` from the previous step
  - Adds temporally-correlated noise via an Ornstein-Uhlenbeck process (α ≈ 0.9, σ = 0.05m on position, calibrated to BodySLAM++'s ~3cm ATE)
  - Applies a 2-3 step latency buffer (simulates 15 FPS perception vs 50 Hz control)
  - Appends the noisy estimate to the observation space under a new key `human_state_estimate`
- **Add occlusion modelling** using ray-casting from the robot's head camera. When the human pelvis is not visible (blocked by fixtures or robot links), increase noise by 3× and flag an `occluded` boolean in the observation
- **Implement a "tracking lost" fallback** — with probability 0.02 per step, drop the estimate entirely and return the last known position with increasing staleness

### Experiments

1. **E1.1 — Observation ablation:** Train Diffusion Policy with three observation conditions: baseline (no human state), oracle (ground truth), noisy (Mock BodySLAM++). Compare safety metrics. Expected: oracle > noisy > baseline
2. **E1.2 — Noise sensitivity sweep:** Fix the policy and sweep σ ∈ {0.02, 0.05, 0.10, 0.15, 0.20}m to find the breakpoint where the policy can no longer effectively use human state information
3. **E1.3 — Temporal structure ablation:** Compare i.i.d. Gaussian vs OU noise vs OU + occlusion + dropout. Measures whether temporal correlation matters for the policy

### Success criteria

Oracle condition shows ≥20% reduction in SSM violation rate vs baseline. If it does not, the current reward structure cannot use human state information regardless of its accuracy, and you must address the cost signal (Phase 2) before doing anything else.

### Deliverable

`BodySLAMWrapper` module + three sets of training runs + a clear answer to "does human state help?"

---

## Phase 2: Offline SVF Safety Filter (Weeks 3-4)

**Goal:** Build the runtime safety filter as a standalone module that works with any task policy.

### Tasks

- **Collect the safety training dataset.** Run rollouts from three sources and save transitions with full safety info:
  - BiGym demonstrations (already loaded during pretraining)
  - A random policy (ensures coverage of unsafe regions)
  - The Phase 1 Diffusion Policy (ensures coverage of realistic robot behaviour)
  - Target: ~500k transitions with roughly 5-10% containing violations
- **Label transitions with binary safety reward:** `r_safe = 0 if ssm_violation or pfl_violation else 1`, with terminal flag set on violation
- **Implement the safety critic network** as a standalone PyTorch module independent of RoboBase:
  - Input: robot proprioception + Mock BodySLAM++ estimate + action (concatenated MLP input)
  - Architecture: MLP [256, 256, 256] with ReLU activations
  - Output: scaled sigmoid bounded to [0, 1/(1-γ)] — mathematically prevents overestimation
  - Do not share weights with any RoboBase network
- **Train with CQL:**
  - Standard CQL regulariser with α ∈ {1.0, 5.0, 10.0} (sweep to find appropriate conservatism)
  - Target policy for CQL evaluation is the random policy (provides OOD action coverage)
  - Add supervised auxiliary loss on known collision states: `L_aux = E[||V(x)||²] for x ∈ X_unsafe` using the privileged AMASS ground truth you already have
  - Train until the Q-values on violating states have converged to near-zero
- **Calibrate the runtime threshold R.** Start at `R = 1/(2(1-γ))`. Evaluate on held-out rollouts and measure:
  - Intervention rate (fraction of steps where filter triggers)
  - Residual violation rate (violations that slip through despite the filter)
  - Sweep R to trace the conservatism-violation Pareto frontier
- **Implement the runtime wrapper.** A `gym.Wrapper` that intercepts the action, evaluates `Q_safe(s, a)`, and either passes through or substitutes a fallback action. Start with zero-velocity braking as the fallback; upgrade in Phase 4

### Experiments

1. **E2.1 — CQL α sweep:** Train with α ∈ {1.0, 5.0, 10.0} and measure how conservative each is on held-out data
2. **E2.2 — Filter effectiveness:** Apply the filter to the Phase 1 Diffusion Policy (without retraining) and measure reduction in violation rate. Compare against the unfiltered baseline
3. **E2.3 — Threshold Pareto curve:** For the best CQL model, sweep R and plot violation rate vs intervention rate. Identify the knee of the curve as the operating point
4. **E2.4 — Robustness to perception noise:** Test the filter with σ_test > σ_train (e.g., σ_test = 0.1m when trained on σ_train = 0.05m). The filter should gracefully degrade (more interventions) rather than silently fail

### Deliverable

A frozen safety filter module + wrapper + measured Pareto frontier. At this point you have a working safety mechanism, even without constrained RL.

---

## Phase 3: Constrained RL Integration (Weeks 5-8)

**Goal:** Train a task policy that internalises safety via the Lagrangian cost constraint, using the Phase 2 filter to prevent unsafe exploration during training.

### Tasks

- **Choose the integration strategy.** Two options, in increasing invasiveness:

  **Option A — Wrapper-level (start here):** Implement the Lagrangian constraint entirely outside RoboBase via a reward wrapper that maintains λ and modifies the reward: `r_modified = r_task - λ · c_t`. The cost signal uses the continuous formulation from Phase 0. This requires zero RoboBase changes.

  **Option B — Dual-critic (upgrade path):** Fork `ActorCritic` in RoboBase to add a second cost critic head C(s,a) alongside the task critic Q(s,a). Implement Lagrangian update in the training loop. This provides better gradients but requires ~500 lines of careful RoboBase modification.

  Start with Option A. Only upgrade to Option B if Option A proves insufficient for meeting ISO 15066 compliance targets.

- **Design the continuous cost signal:**
  ```python
  d_buffer = 0.3  # metres — activate cost before violation
  c_ssm = max(0, 1.0 - ssm_margin / d_buffer)
  c_pfl = max(0, pfl_force_ratio - 0.8)  # activate before threshold 1.0
  c_t = max(c_ssm, c_pfl)  # worst-case across SSM and PFL
  ```

- **Implement PID-controlled λ update:**
  ```python
  cost_violation = rolling_mean_cost - d
  λ = max(0, λ + K_I · cost_violation + K_P · cost_violation + K_D · Δcost_violation)
  λ = min(λ, λ_max)  # clamp to prevent collapse to frozen policy
  ```
  Start with `K_I = 1e-3, K_P = 1e-2, K_D = 0, λ_max = 100, d = 0.01` (1% expected violation rate). Tune on one task before scaling to all three

- **Integrate the Phase 2 filter during training.** The filter sits between the actor and the environment during training rollouts, vetoing catastrophically unsafe actions. This prevents the policy from ever experiencing states where the safety critic would need to extrapolate dangerously. CBF-RL (2025) shows this combination accelerates convergence to safe policies

- **Handle the asymmetric observation problem.** Since RoboBase's shared encoder makes true asymmetry difficult, use the pragmatic workaround: give the actor the noisy Mock BodySLAM++ estimate, and apply a supervised auxiliary loss on the safety critic using privileged ground-truth collision labels. The asymmetry is in the loss function, not the architecture

### Experiments

1. **E3.1 — Cost signal comparison:** Train with three cost formulations: fixed -1.0 penalty (baseline), binary 0/1 cost, continuous smooth cost. Measure training stability, final task reward, and final violation rate
2. **E3.2 — Lagrangian tuning:** Sweep the cost budget `d ∈ {0.001, 0.01, 0.05, 0.1}` to trace the task-safety Pareto frontier. Identify the tightest budget at which tasks are still completable
3. **E3.3 — λ update method:** Compare gradient-ascent λ updates vs PID-controlled updates. Measure training oscillation magnitude and final performance
4. **E3.4 — Filter during training:** Ablate whether the Phase 2 filter active during training improves final policy safety. Expected: yes, because it prevents catastrophic exploration

### Deliverable

A Lagrangian-trained policy that achieves the baseline task reward from Phase 0 with significantly reduced violation rate, without the runtime filter active at evaluation time.

---

## Phase 4: Full Hybrid Deployment and Fallback Upgrade (Weeks 8-9)

**Goal:** Combine the Lagrangian-trained policy with the frozen Phase 2 safety filter, and improve the fallback action for smooth recovery.

### Tasks

- **Deploy the full stack:**
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

- **Upgrade the fallback action** from zero-velocity braking to one of:
  - **Proportional damping:** `u_safe = (Q_safe / R) · u_nom` — preserves direction, scales magnitude with predicted safety
  - **Trajectory replay:** cache the last N = 10 safe actions; on filter trigger, replay the most recent. Momentum-preserving, avoids jerky freezes
  - **Retreat controller:** add a repulsive velocity component pointing away from the nearest human body part. Requires closest-point computation (may be expensive for 76 DOF)

  Benchmark all three fallbacks under filter intervention and pick based on task success rate during interventions

- **Re-calibrate the filter threshold R with the Lagrangian policy.** Because the policy is already mostly safe, the filter triggers rarely. This means you can use a more aggressive (less conservative) R than Phase 2's calibration suggested. Re-run E2.3 with the Lagrangian policy and find a new operating point

### Experiments

1. **E4.1 — Full hybrid comparison:** Compare four configurations across all tasks and disruption types:
   - Baseline (Diffusion Policy + fixed penalty)
   - Lagrangian policy alone (no filter)
   - Safety filter alone (with Diffusion Policy)
   - Full hybrid (Lagrangian policy + filter)

2. **E4.2 — Fallback action ablation:** Hold the policy fixed; sweep across the three fallback options. Measure task success, smoothness (jerk), and recovery time after intervention

3. **E4.3 — Intervention rate analysis:** Measure how often the filter triggers with the hybrid vs filter-alone. Confirm that the Lagrangian policy's internalised safety reduces filter dependence

### Deliverable

The full hybrid system, operating smoothly on all three tasks with measurably better safety than any single approach.

---

## Phase 5: Evaluation and Stress Testing (Week 10)

**Goal:** Rigorous final evaluation including tail-risk metrics and robustness testing.

### Tasks

- **Compute tail-risk metrics** beyond means:
  - CVaR(0.95) of max contact force across episodes
  - 95th and 99th percentile of SSM margin at closest approach
  - Distribution of time-to-first-violation
  - Max force ever observed (not mean)
  - These capture worst-case behaviour, which is what ISO 15066 compliance actually requires

- **Stress test under increased perception noise.** Evaluate at σ_test ∈ {0.05, 0.10, 0.15, 0.20}m — going beyond what the system was trained on. Both the policy and the filter should degrade gracefully

- **Stress test under adversarial human behaviour:** increase walk speeds in scenario sampler, enable sudden direction changes, add previously-unseen disruption parameter combinations

- **Measure computational overhead:** time per step for policy forward pass, filter evaluation, and fallback action computation. Confirm real-time feasibility at 50 Hz control rate

- **Sim-to-real gap estimation.** If possible, render MuJoCo camera images and run real BodySLAM++ on them (rather than Mock BodySLAM++). Compare error statistics to the noise model used in training. This quantifies the residual transfer gap

### Experiments

1. **E5.1 — Tail-risk evaluation:** Report CVaR and max-over-distribution metrics alongside means
2. **E5.2 — Out-of-distribution robustness:** Safety degradation curves as σ and human speed exceed training distribution
3. **E5.3 — Real BodySLAM++ comparison:** Error distribution comparison between Mock and real perception

### Deliverable

Final evaluation report with tables, Pareto curves, and stress-test results. Ready for writeup.

---

## Critical Path and Dependencies

```
Phase 0 (baselines) ──┐
                      ├──► Phase 1 (observation wrapper) ──┐
                      │                                     │
                      └──► Phase 2 (safety filter) ────────┤
                                          │                 │
                                          └─► Phase 3 (constrained RL) ──► Phase 4 (hybrid) ──► Phase 5 (eval)
```

**Phase 2 and Phase 3 can run in parallel** if you have compute. Phase 2 is standalone offline training; Phase 3 depends only on Phase 1 outputs.

---

## Risk Register

**High-severity risks:**

- **RoboBase surgery proves too invasive (Phase 3).** Mitigation: start with wrapper-level Option A; only upgrade to Option B if necessary. Option A alone can deliver ~80% of the benefit
- **Lagrangian training oscillates or collapses.** Mitigation: clamp λ_max, use PID updates from day one, start with loose cost budget (d = 0.05) and tighten gradually
- **Phase 1 shows no benefit from human state.** Mitigation: this is actually informative — it means the fixed penalty is the bottleneck, and Phase 2/3 become higher priority than expected
- **Filter triggers too frequently with Lagrangian policy.** Mitigation: this is Phase 4's re-calibration step; expect to re-tune R once the policy is trained

**Medium-severity risks:**

- **CQL conservatism too aggressive, filter freezes the robot.** Mitigation: sweep α during training, upgrade fallback action to proportional damping
- **AMASS-derived cost labels don't generalise to Mock BodySLAM++ at runtime.** Mitigation: train filter on Mock estimates, not ground truth (already in plan)
- **Contact force predictions don't transfer from MuJoCo to reality.** Mitigation: this is fundamental to the sim-to-real gap; Phase 5 measures it but can't fix it

---

## Summary

Ten weeks, five phases, each producing a standalone deliverable. The key insight is that every phase produces something useful even if the next phase fails: Phase 1 gives you a baseline for human-state observation, Phase 2 gives you a working safety filter for any policy, Phase 3 gives you a constrained policy that works without a filter, and Phase 4 combines them into the full hybrid system. Phase 5 validates everything rigorously.

Start with Phase 0 this week — fix the unused cost signals, measure baselines, clean up the hardcoded paths. Without that foundation, nothing else is reproducible.
