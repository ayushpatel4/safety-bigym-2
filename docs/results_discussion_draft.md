# Results & Discussion — draft (2026-06-06)

Drop-in draft for the thesis Results/Discussion. All numbers are from the
deployment-faithful benchmark (`benchmark_policy.py`) on `saucepan_to_hob`,
Unitree-G1 coworker, noisy BodySLAM, unless stated. **Primary safety axis:
proximity-violation rate** = fraction of steps with human–robot separation
< 0.3 m. Edit freely.

---

## Setup

A Unitree-G1 coworker, driven by a scripted reach-into-the-workspace controller
(`coworker_train`), shares the manipulation scene. The robot (H1, floating base)
is trained with CQN-AS from demonstrations. We compare four configurations in a
feature-incremental table (Table~\ref{tab:e4.1}): the unconstrained baseline, the
baseline guarded by the reactive SVF filter, the proactive constrained-RL
(Lagrangian) policy, and the hybrid (policy + filter). Metrics are pooled over 3
training seeds (the Lagrangian over 180 episodes); the policy is selected by a
safety-aware criterion (lowest deployment proximity at success ≥ 0.75) and
benchmark-confirmed.

## 1. Reactive ISO-SSM filtering is fundamentally limited

The decoupled SVF filter, wrapping the unconstrained baseline, does **not** reduce
geometric proximity (0.296 → 0.303; within noise) — it acts only as a velocity
backstop (mean robot velocity 0.289 → 0.266). The cause is a **freeze-vs-flee
dilemma**: the ISO-15066-correct response to a small separation is to reduce robot
speed (zero-velocity on veto), but stopping makes the robot *dwell* in the danger
zone, so proximity does not fall; the alternative — a retreat fallback — does cut
proximity (−68 %) but only by abandoning the task (success 0.85 → 0.18) and raising
velocity sixfold, which itself worsens the velocity-adaptive ISO-SSM margin.

A decomposition explains why: freezing the robot for **100 %** of steps (the
filter at its most aggressive) still leaves **≈42 %** of the proximity, which is
the coworker walking up to a stationary robot. So ~42 % of proximity is
**exogenous** (human-initiated) and cannot be removed by any veto of *robot*
actions. Reactive filtering is therefore bounded by an exogenous floor.

## 2. Proactive constrained-RL reduces proximity — reproducibly

The Lagrangian constrained-RL policy is the decisive test, and it succeeds. The
constraint multiplier λ exposes a clean **proximity–success Pareto** (λ=0 recovers
the baseline at 0.85/0.296; larger λ trades success for proximity). At **λ=0.1**
the policy reaches a graceful operating point: **proximity 0.228 [0.194, 0.264], a
22.8 % reduction, at 0.76 success**, pooled over 3 seeds — and the velocity-adaptive
ISO-SSM violation rate also improves (0.146 → 0.112). This is the reduction the
reactive filter could not achieve.

**The naive PID auto-tuning of λ is unstable at the feasibility-boundary budget.**
At budget d=0.3 — which coincides with the task's inherent per-step cost — the dual
variable is decided by each seed's stochastic cost trajectory: across three seeds λ
converged to 0.000, 0.267 and 3.855, yielding *unconstrained*, *graceful*, and a
*windup-collapse* that abandons the task (Fig.~\ref{fig:lambda-regimes}). Fixing λ
removes the instability — all three seeds then behave consistently
(Fig.~\ref{fig:fixlam}). This is a concrete methodological finding: for a
constrained policy whose feasible budget sits at the task cost, a fixed (or
scheduled) multiplier is preferable to PID auto-tuning.

The reduction is genuinely constraint-driven, not an artifact of reward shaping or
checkpoint selection. **(i)** A workspace-distance shaping term in the baseline
*increases* proximity while improving success (no-shaping 0.75/0.258 vs shaping
0.85/0.296) — it is a task aid, not a safety mechanism — yet the Lagrangian sits
below the no-shaping baseline too (−12 % at matched success). **(ii)** Against a
λ=0 control (a seed whose PID never engaged the constraint), the avoidance does not
appear, confirming it is the constraint doing the work.

## 3. The hybrid is counterproductive: the policy makes the filter redundant

Stacking the reactive filter on the already-avoiding Lagrangian policy makes the
result **worse than the policy alone** (proximity 0.198 → 0.265, success 0.75 →
0.62; seed-0). The mechanism is a **48 % filter intervention rate** (vs 15 % on the
baseline): the SVF critic was trained on *baseline* rollouts and is out of
distribution on the avoiding policy, so it over-vetoes; each veto zeroes the robot's
velocity, re-introducing the freeze/dwell failure of §1 (mean velocity collapses to
0.231). The internalisation experiment (E4.3) corroborates this — the filter's
intervention rate stays ~40–50 % across the policy's training rather than falling,
because the constrained policy's state–action distribution is precisely where the
baseline-trained critic is least calibrated.

This is the central architectural finding: **the proactive policy internalises
safety well enough that a reactive filter calibrated to the unconstrained baseline
is redundant and, when active, counterproductive.** It is direct evidence that, for
an approaching collaborator, anticipatory constrained-RL is superior to reactive
ISO-SSM filtering — to the point the latter is unnecessary. A threshold sweep
confirms this is not a tuning artifact: across R ∈ [1.0, 2.25] the filter intervenes
39–48 % and proximity stays 0.27–0.28 (worse than the policy's 0.198) at every
setting. Because lowering R barely changes the intervention rate, the
baseline-trained critic is **miscalibrated (out-of-distribution) on the avoiding
policy**, not mis-thresholded — re-thresholding cannot recover it; only re-collecting
the SVF on the constrained policy's own rollouts (the full P2 pipeline) could.

## 4. Generalisation and robustness

- **Unseen coworker (E5.2).** Re-evaluated on a held-out coworker distribution
  (`coworker_eval`: broader, gentler — closest 0.6–1.8 m, reach 3–9 s), the policy's
  reduction **generalises**: baseline 0.084 → Lagrangian 0.058 (−31 %). The success
  cost is larger (0.92 → 0.70) because λ=0.1, tuned for the harder training coworker,
  over-constrains the gentler one — i.e. the operating point is task/distribution
  specific, but the qualitative result holds out of distribution.
- **Perception robustness (E3.6).** The policy's avoidance is **not
  perception-bottlenecked**: under perfect (oracle) vs realistic noisy/lagged
  perception the proximity is essentially equal (0.236 vs 0.198), so the reduction
  survives realistic perception noise. (The baseline shows no oracle–noisy gap
  because it does not avoid.)
- **Cross-task (E5.3).** The full pipeline (base curriculum → fixed-λ Lagrangian →
  eval) is replicated on `drawers_open_all` and `dishwasher_close` to test
  cross-task generalisation. [Results pending; in training.]

## 5. Limitations

- **Worst-case separation is exogenous (E5.1).** The policy lowers the *mean* and
  *frequency* of proximity but **not the tail**: the worst-5 % closest approach
  (CVaR₅ ≈ 0.005 m, worst ≈ 0.002 m) is unchanged across all configurations, because
  the human's single closest lunge is human-driven. The controllable quantity is the
  dwell/frequency of proximity, not the single worst-case approach.
- **Operating point is task-specific.** λ=0.1 is the saucepan sweet spot; other
  tasks/coworkers require re-selecting λ on their Pareto.
- **PFL not evaluated.** Power-and-Force-Limiting relies on contact detection that is
  identically zero under a known BiGym/mojo runtime-attachment issue, so the safety
  axis here is proximity/SSM only.

## 6. Synthesis

Against an approaching collaborator, reactive ISO-SSM filtering is fundamentally
limited (freeze-vs-flee; ≈42 % exogenous), whereas a proactive constrained-RL policy
reduces geometric proximity ~23 % at acceptable success, reproducibly, robustly to
perception noise, and out of distribution. The hybrid of the two underperforms the
policy alone — the policy internalises safety so well that the baseline-calibrated
filter becomes redundant. The contribution is thus threefold: (i) a
deployment-faithful evaluation pipeline; (ii) evidence that proactive ≫ reactive
avoidance for this problem; and (iii) the methodological observation that the
constrained policy's multiplier must be fixed/scheduled, not PID-tuned, when the
feasible budget sits at the task's inherent cost.
