# A Hybrid Safety Critic for Human-Coworker Manipulation: Report Section

**Tasks:** `dishwasher_close`, `drawers_open_all` (BiGym + live G1 coworker, ISO 15066
monitoring). **Policy:** CQN-AS curriculum policy. **Evaluation:** deployment-realistic
noisy observations, `coworker_train` disruption, 3 seeds × 20 episodes = 60 episodes per
cell, bootstrap confidence intervals.

---

## Summary

We set out to make a manipulation policy *safer* without destroying its task success, and
found that the question is mis-posed until the safety metric is corrected. The thesis-default
metric — time spent within 0.3 m of the human — is **largely not controllable by the robot**:
even a completely frozen robot only reduces it by ~58%, because the coworker walks into the
robot's workspace. Once safety is measured on the **robot-controllable ISO-15066 SSM-velocity
axis** (is the robot moving slowly enough to stop before contact, given the separation), a
clear result emerges:

- **Constrained reinforcement learning (Lagrangian) cannot deliver** a safety gain with task
  success on these tasks — the constraint is either inert or task-fatal, and when it binds the
  policy degrades into *faster*, more erratic motion.
- A **learned Safety-Value-Function (SVF) critic with a binary veto fails** on a chunked
  (temporal-ensemble) policy — vetoing breaks action coherence and causes velocity overshoot.
- **Continuous speed-scaling works** as an ISO-15066 velocity backstop, but pays a large task
  cost when applied unconditionally.
- **Gating the speed-scaling by the SVF critic — the Hybrid Safety Critic — is the best
  operating point:** the critic decides *when* to intervene (only when it flags risk), the
  speed-scaler decides *how* (smoothly, preserving the action direction). This delivers a
  **−50% SSM-violation reduction on dishwasher at roughly half the task-success cost** of
  unconditional scaling, and **−22% on drawers**.
- **The gating benefit is governed by a single boundary condition:** it recovers task throughput
  only under *intermittent* human–robot co-location (dishwasher, drawers). On `saucepan_to_hob`,
  where the coworker is *persistently* co-located, the gate is active almost continuously and
  gating collapses to unconditional scaling — no high-success, low-SSM corner exists. The same
  mechanism explains both the win and the non-win across all three tasks.

---

## 1. The measurement problem: proximity is largely exogenous

ISO 15066 Speed-and-Separation Monitoring is satisfied when the robot can stop before contact
given the current separation — a function of robot *speed*. The thesis-primary metric, however,
has been geometric *proximity* (fraction of episode time within 0.3 m). A prior controlled sweep
(saucepan task; recorded in `filters/snapshots.py`) shows why proximity is the wrong target for
*robot* safety:

**Figure 4** (`docs/figures/fig4_exogenous_proximity.png`) — sweeping a runtime filter from no
intervention to a fully frozen robot (100% intervention) reduces proximity by only 58%. The
remaining ~42% is the human walking up to the stationary robot: an **exogenous, human-driven**
component no robot policy can remove.

> **Consequence.** Every "no safety benefit" result on the proximity metric was partly measuring
> a quantity the robot does not control. We therefore report robot safety on the SSM-velocity
> axis: `ep_ssm_violation_actual_rate` (primary), supported by robot velocity and the minimum
> SSM margin. Proximity is retained only as a secondary, partly-exogenous descriptor.

---

## 2. Methods compared

| Method | Mechanism | Where it acts |
|---|---|---|
| **Baseline** | Unconstrained CQN-AS curriculum policy | — |
| **Lagrangian (constrained-RL)** | Dual reward+cost critic, PID-adapted λ on the SSM cost; `argmax_a[Q_r − λ·Q_c]` | Training (policy) |
| **SVF veto** | Learned safety critic `Q_safe(s,a)`; veto (→ zero-velocity) when `Q_safe < R` | Runtime (action) |
| **Speed-scale (unconditional)** | ISO-15066 scaling: per-step motion × `clip((sep−d_stop)/(d_slow−d_stop),0,1)` | Runtime (action) |
| **Gated hybrid (ours)** | SVF critic *gates* the speed-scaler: scale only when `Q_safe < R`, else full speed | Runtime (action) |

The SVF critic is a CQL-trained value function over (proprioception + human-position estimate,
action), labelled by geometric near-contact (`min_separation < 0.10 m`). It is task-agnostic
(identical observation spec across both tasks) and trained once on 52,794 transitions.

---

## 3. Results

### 3.1 Method comparison

**Figure 1** (`docs/figures/fig1_method_comparison.png`) places each method on the
success / SSM-violation plane. The desirable corner is **low SSM-violation, high success**
(bottom-right). Only the **gated hybrid** reaches it: the Lagrangian sits at or *above* the
baseline SSM (no gain), the unconditional speed-scaler reaches low SSM but at heavily reduced
success, and the gated hybrid achieves low SSM while retaining substantially more success.

| Task | Method | Success | SSM-violation | Δ SSM | Δ success |
|---|---|---|---|---|---|
| dishwasher | Baseline | 0.77 | 0.176 | — | — |
| dishwasher | Lagrangian (basin) | 0.82 | 0.184 | +5% | +0.05 |
| dishwasher | SVF veto (R=2.25) | 0.63 | 0.174 | −1% | −0.14 |
| dishwasher | Speed-scale (uncond.) | 0.52 | 0.087 | −50% | −0.25 |
| dishwasher | **Gated hybrid (R=2.75)** | **0.67** | **0.088** | **−50%** | **−0.10** |
| drawers | Baseline | 0.82 | 0.100 | — | — |
| drawers | Lagrangian (basin) | 0.77 | 0.082 | −18% | −0.05 |
| drawers | SVF veto (R=2.25) | 0.43 | 0.130 | +30% | −0.39 |
| drawers | Speed-scale (uncond.) | 0.38 | 0.059 | −41% | −0.44 |
| drawers | **Gated hybrid (R=3.0/0.8/0.25)** | **0.73** | **0.078** | **−22%** | **−0.09** |

### 3.2 Why the Lagrangian and the veto fail (mechanism)

- **Lagrangian.** Cost-budgets above the policy's natural operating cost leave λ at zero (the
  constraint is inert → baseline behaviour); budgets below it drive λ unbounded and collapse the
  task (0% success). When λ binds, robot velocity *rises* (mean 0.44→0.79, max 2.46→4.14) — the
  collapse manifests as fast, erratic motion, not careful slowing. There is no feasible budget
  that both binds and preserves the task.
- **SVF veto.** The critic separates safe from unsafe well in-sample, but a *binary* veto →
  zero-velocity on a temporal-ensemble (chunked) policy breaks action-sequence coherence: the
  robot stops mid-chunk, then overshoots with a catch-up action when the veto lifts.
  Max velocity *rises* (2.46 → 6.03), and SSM-violation does **not** improve.

Continuous speed-scaling avoids the overshoot (it scales magnitude, preserving direction — max
velocity is unchanged, 2.46→2.46), which is why gating *speed-scaling* rather than *vetoing*
is the correct hybrid.

### 3.3 The gated hybrid is a tunable frontier

**Figure 2** (`docs/figures/fig2_tradeoff_curve.png`) — the gate threshold R is a clean dial.
Low R = selective (few interventions, near-baseline); high R = approaches unconditional scaling.
At every point the gated hybrid **dominates** unconditional speed-scaling (same SSM at higher
success), because the critic suppresses scaling when the situation is genuinely safe.

**Figure 3** (`docs/figures/fig3_reduction_bars.png`) — SSM-violation reduction by method with
the success cost annotated: only the gated hybrid cuts SSM-violation substantially without
collapsing the task.

### 3.4 A third task, and the boundary condition: intermittent vs persistent co-location

We applied the same gated hybrid to **`saucepan_to_hob`** (existing saucepan SVF critic,
180 episodes/cell). Here the gated hybrid **does not** recover throughput — and *why* is the
most informative result of the study.

| saucepan config | Success | SSM-violation | Δ SSM | Δ success |
|---|---|---|---|---|
| Policy-alone (no filter) | 0.722 | 0.138 | — | — |
| Gated R=2.0 | 0.706 | 0.125 | −9% | −0.02 |
| Gated R=2.5 | 0.567 | 0.105 | −24% | −0.16 |
| Gated R=3.0 | 0.472 | 0.084 | −39% | −0.25 |
| Speed-scale (unconditional) | 0.433 | 0.072 | −48% | −0.29 |

As the gate threshold rises, the operating point simply **slides from policy-alone toward the
unconditional point** — it never opens a high-success, low-SSM corner. No setting meets a
"succ ≥ 0.60 and SSM ≤ 0.08" bar.

**Figure 5** (`docs/figures/fig5_cross_task_boundary.png`) shows all three tasks' gated R-dial
trajectories (normalised to each task's no-filter point). Dishwasher and drawers **bend down**
(SSM falls steeply while success is retained); saucepan **slides along the diagonal** (success
and SSM fall together).

> **Boundary condition.** The saucepan coworker is *persistently* co-located, so the SVF gate is
> active almost whenever the robot is near the human — gating therefore ≈ unconditional scaling.
> On dishwasher/drawers contact is *intermittent*, so the gate finds genuinely-safe windows to
> pass through at full speed and recovers task success. **Critic-gating recovers throughput only
> under intermittent human–robot co-location.** This unifies the cross-task story: the gated
> hybrid's win on dishwasher/drawers and its non-win on saucepan have the same single cause.

*(Caveat: the saucepan sweep ran on the `snapshot_best` policy, not the deployment basin
checkpoints, which were lost in a cleanup. The unconditional-on-`snapshot_best` control
reproduces the headline saucepan hybrid on the success/velocity axes — 0.433/0.072 vs the
reference 0.44/0.065 — so the *filter*-level verdict transfers; only proximity, the policy's
axis, differs. Detail: `docs/c4_gated_reframe.md`.)*

---

## 4. Recommended operating points

| Task | Filter config | Success | SSM-violation | Δ SSM | Note |
|---|---|---|---|---|---|
| **dishwasher_close** | gated, R=2.75, d_slow=0.5, d_stop=0.15 | 0.67 | 0.088 | **−50%** | best; R≥3.0 + d_slow<0.5 reintroduces overshoot |
| **drawers_open_all** | gated, R=3.0, d_slow=0.8, d_stop=0.25 | 0.73 | 0.078 | **−22%** | aggressive |
| drawers_open_all | gated, R=2.75, d_slow=0.8 | 0.80 | 0.090 | −10% | near-free safety |
| **saucepan_to_hob** | gating does **not** help (persistent co-location) | — | — | — | use unconditional speed-scale if velocity safety is required, accepting the success cost |

Dishwasher/drawers use the shared critic `checkpoints/svf_dish_drawers_v1.pt`; saucepan uses the
existing `svf_coworker_train_g1_0p3_v3.pt` (both 0.10 m near-contact label).

---

## 5. Ablations

Four refinements were run; all **confirm** the recommended configuration rather than beat it:

1. **Pareto (d_slow × R).** R is the dominant dial; d_slow is secondary (0.4 marginally better
   than 0.5 on dishwasher). The R-sweep *is* the frontier.
2. **Anticipatory critic labels (0.20 / 0.30 m).** Looser, earlier-warning labels do not improve
   the operating point — they slide along the same trade. The 0.10 m near-contact label, being
   the most selective gate, is best.
3. **Row-5 hybrid (gate on a Lagrangian policy).** Gating the backstop on the best-available
   constrained policy ≈ gating on the baseline policy — confirming that constrained-RL produced
   no genuinely proactive-avoiding policy for these tasks.
4. **DAgger re-collection (on-filter distribution).** Retraining the critic on the *filtered*
   policy's rollouts is null/worse: on dishwasher the new critic under-gates (−2% vs −50%),
   because the filtered distribution looks "safe" to it. A runtime gate should be trained on the
   **unfiltered** policy's behaviour.

---

## 6. Limitations and outlook

- **Proximity is bounded by the scenario.** ~42% of time-in-proximity is human-initiated and
  irreducible by any robot policy in this co-located-manipulation setting. The achievable robot
  safety is on the SSM-velocity axis, which is what we improve.
- **Drawers gain is smaller** (−22% vs dishwasher's −50%): its policy is already slow, so the
  gate fires less often.
- **A proactively-avoiding policy** — one that repositions to increase clearance rather than only
  slowing — is the remaining lever to push past the exogenous-proximity ceiling. Constrained-RL
  (λ on expected cost) did not produce one; it would require a different training signal (e.g. a
  potential-based separation reward) and carries genuine risk that the task geometry (the robot
  must occupy the appliance the human walks through) imposes its own ceiling.

**Conclusion.** The deployable result is a **Hybrid Safety Critic**: a learned safety value
function gating an ISO-15066 speed-scaling backstop. On the two tasks with intermittent
co-location it is the only method evaluated that improves the robot-controllable safety axis
(SSM-velocity) without collapsing task success — −50% (dishwasher) and −22% (drawers)
SSM-violation reduction at moderate task cost, with a single tunable knob (the gate threshold R).
On saucepan, where the human is persistently co-located, the same hybrid reduces to unconditional
scaling — and that contrast yields the study's sharpest contribution: a **mechanism-level boundary
condition** for when learned-critic gating helps (intermittent co-location), validated across
three tasks. The path beyond it is a *proactively-avoiding* policy that the current constrained-RL
signal does not produce — the principal item of future work.

---

### Reproducibility

| Artifact | Path |
|---|---|
| Figures | `docs/figures/fig{1,2,3,4,5}_*.png` (regen: `scripts/make_report_figures.py`) |
| SVF critics | `checkpoints/svf_dish_drawers_v1.pt` (dish/draw); `svf_coworker_train_g1_0p3_v3.pt` (saucepan); dataset `datasets/svf_dish_drawers_v1/` |
| Gated-filter implementation | `safety_bigym/benchmark/runners.py` (`gated_speedscale`), `scripts/benchmark_policy.py` |
| Benchmarks (dish/draw) | `results/{safety_eval,svf_sweep,speedscale,gated_sweep,improve,budget_sweep_eval}/` |
| Benchmarks (saucepan) | `results/e4_1/gated_saucepan/` (`scripts/dispatch_gated_saucepan.py`) |
| Detailed findings | `docs/runtime_filter_results.md`, `docs/budget_sweep_results.md`, `docs/c4_gated_reframe.md` (saucepan) |
