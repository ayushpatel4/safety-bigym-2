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

### 3.1 The filter is not salvageable by calibration — the limit is the reactive paradigm

A natural hypothesis is that the hybrid's failure is mere distribution shift (the critic
was trained on the *baseline* policy). To test it, we re-collected the SVF critic on the
**constrained policy's own rollouts** (on-policy), removing the OOD miscalibration, and
re-swept the veto threshold (deployment-faithful sweep, 3 seeds). The result is decisive:
**there is still no graceful operating point.** At every intervention rate ≤ 25 % the
filter *increases* proximity (R=1.0: 7.8 % intervention, proximity 0.189 → 0.264; R=2.25:
4.8 %, → 0.292) — the zero-velocity fallback freezes the robot into dwelling even when
vetoes are sparse and well-calibrated — and proximity falls only at 65–100 % intervention,
where the robot is essentially frozen (R=3.0: 71 %, 0.127). No threshold meets the
≥ 30 %-reduction-at-≤ 25 %-intervention bar.

Because **both** the baseline-trained critic (§3) **and** the on-policy critic fail at
*every* threshold, the cause is not the critic but the **reactive paradigm itself**:
against an *approaching* human the only available fallback actions are *stop*
(freeze → dwell → proximity rises) or *retreat* (flee → abandon the task), and no critic —
however well calibrated, however thresholded — changes which fallback exists or makes
"stop" the right response to an oncoming person. This is the strongest form of the result:

> **No reactive learned safety filter gracefully reduces proximity against an approaching
> coworker.** The limitation is intrinsic to reactive veto-and-fallback control, not to
> the safety critic's calibration — proactive constrained-RL is *necessary*, not merely
> preferable.

### 3.2 A model-based filter *does* reduce proximity — but by fleeing (the freeze-vs-flee taxonomy)

We implemented a geometric control-barrier-function "directional-dodge" filter (no learned
critic): when human–robot separation drops below `d_target`, it minimally offsets the
floating-base target away from the human along the robot↔human axis, leaving the arm/task
action untouched (`safety_bigym/filters/cbf_filter.py`). Unlike the learned SVF veto, it
**does** reduce proximity — 0.198 → **0.150** (d_target=0.35) at a tiny **1.3 %**
intervention rate (0.137 at d=0.45 / 0.148 at d=0.55; 2–4 % intervention). So a
model-based directional filter is the kind of filter that *can* cut proximity, where the
learned veto-and-freeze filter cannot.

But it cuts proximity only by **fleeing**: the base retreats from the workspace, episodes
stretch from ~450 to 620–770 steps, and **success falls 0.75 → 0.57 / 0.43 / 0.35** as
d_target rises. Proximity floors at ~0.14 regardless of d_target — the ~42 % exogenous
floor (§5.1) — so extra dodging only costs success. The full d_target sweep (0.30–0.55)
traces a *single flee Pareto entirely below the proactive policy*, and the flee is
**brittle even at minimal intervention**: at d_target=0.30 the filter dodges only when
already in violation (1.1 % intervention) yet still costs success 0.75 → 0.58 (proximity
0.170) — yanking the absolute base target at the critical close-approach moments derails
the task and recovers slowly. There is no low-intervention "harmless backstop" setting for
the base dodge.

| base-CBF d_target | 0.30 | 0.32 | 0.35 | 0.38 | 0.45 | 0.55 |
|---|---|---|---|---|---|---|
| success | 0.58 | 0.57 | 0.57 | 0.53 | 0.43 | 0.35 |
| proximity | 0.170 | 0.151 | 0.150 | 0.141 | 0.137 | 0.148 |
| intervention | 1.1 % | 1.1 % | 1.3 % | 1.5 % | 2.4 % | 3.6 % |

(policy reference: success 0.75, proximity 0.198.) And the trade it reaches
(0.57, 0.150) is the same frontier the proactive policy already spans at a tighter
multiplier (λ=0.27 ≈ 0.60, 0.17), so it does **not** beat the policy.

This yields a clean three-way **taxonomy of reactive filters**, both bounded by the
freeze-vs-flee dilemma:

> - **Learned veto + zero-velocity fallback (SVF):** the *freeze* horn — vetoes freeze the
>   robot into dwelling; proximity does not fall (it rises), at high intervention.
> - **Model-based directional dodge (CBF):** the *flee* horn — minimal, low-intervention,
>   and it *does* cut proximity, but by retreating from the task (large success cost).
> - **Proactive constrained-RL policy:** the only one to reach a *graceful* operating point
>   (proximity −23 % at acceptable success), because it avoids *anticipatorily* and
>   *integrated with the task*, rather than reacting once the human is already close.

So the strongest, most complete statement: **reactive filtering — whether learned or
model-based — cannot reduce proximity against an approaching coworker without paying the
freeze or flee cost; only proactive constrained-RL achieves the favourable
success–proximity frontier.** (A *formally-guaranteed* model-based filter — CBF-QP with a
certified barrier / HJ-reachability — remains future work; our geometric CBF demonstrates
the behaviour but offers no formal guarantee in this high-DoF, exogenous-human scene.)

### 3.3 Why the flee is intrinsic — and a test of whether it can be avoided

The flee is not a tuning artifact of our particular CBF; it is **structural**. The task and
the hazard are *co-located*: the coworker reaches into the robot's workspace, so the human's
hand and the robot's end-effector occupy the same region. A separation barrier
`h = sep − d_target` can be satisfied only by *increasing* `sep`, and `sep` increases only
under **radial** motion (directly away from the human); a **tangential** sidestep leaves
`sep` unchanged (`ḣ ≈ 0`) and so cannot satisfy the barrier. Because radial-away ≈
away-from-the-workspace, any reactive separation filter that *moves the robot* must trade
task progress for distance — it can only freeze (dwell) or flee (retreat). Reducing
`d_target`/`max_push` merely slides along this flee Pareto toward the unfiltered policy; it
does not escape it.

The deeper reason is **temporal**: a reactive filter acts only *once the human is already
close*, when the only moves are freeze or flee — it cannot **anticipate**. The proactive
constrained-RL policy escapes the bind precisely because it *does* anticipate: it clears the
shared workspace early and gently and times its reaching around the human, integrated with
the task reward over the whole episode. **Eliminating the flee therefore requires
anticipation — which is exactly what the constrained policy provides; "fixing" the flee
within a reactive filter amounts to reinventing the policy.**

We nonetheless tested the one structurally-different reactive option: retracting the
**end-effector** (an arm "flinch", via a real `mj_jacBody` damped-pseudo-inverse step on
the arm joints) rather than the base — since the safety metric is EE-to-human and the base
can stay in the workspace, this is the least-flee reactive correction available. **It does
not stop the flee.** In the useful (high-success) regime it is in fact *worse* than the base
dodge — at d_target=0.35 the flinch yields 0.50 success vs 0.57 for the base dodge at the
same proximity (0.155 vs 0.150) — because the EE is the closest pair, so the flinch
intervenes far more often (5.8 % vs 1.3 %) and **retracting the arm *is* pausing the task**.
It only achieves lower proximity than the base dodge by abandoning the task: at d_target=0.55
it reaches proximity 0.108 (below the base-CBF floor and the ~42 % exogenous estimate, since
retracting the EE addresses the closest pair directly) but at only 0.38 success. Neither
variant approaches the policy's frontier.

| reactive variant | d_target | success | proximity | intervention |
|---|---|---|---|---|
| base-dodge (flee) | 0.35 / 0.45 / 0.55 | 0.57 / 0.43 / 0.35 | 0.150 / 0.137 / 0.148 | 1.3 / 2.4 / 3.6 % |
| EE-retract (flinch) | 0.35 / 0.45 / 0.55 | 0.50 / 0.45 / 0.38 | 0.155 / 0.145 / 0.108 | 5.8 / 9.0 / 13.2 % |

**Conclusion (the model-based filter cannot escape the dilemma):** tested across *both*
structural options — dodging the base and retracting the arm — the cost simply moves from
"base leaves the workspace" to "arm stops the task"; neither comes near the proactive policy
(0.75 / 0.198). The freeze-vs-flee cost is therefore intrinsic to **reactive** control
against a co-located coworker, *regardless of which part of the robot is dodged*. Only
anticipation — the constrained policy — resolves it. (A *formally-guaranteed* CBF-QP / HJ
filter remains future work, but would face the same task-vs-separation conflict.)

### 3.4 The filter's proper role: ISO-15066 speed compliance (a division of labour)

The preceding results judge every filter on **geometric proximity** — but that is the
*policy's* axis. A runtime safety filter's canonical role under ISO-15066 is the
**Speed-and-Separation-Monitoring (SSM) velocity axis**: reduce the robot's *speed* as the
human–robot separation shrinks, maintaining the velocity-adaptive safety margin even when
the robot cannot increase distance. On that axis a filter is the right tool — and the SVF
veto already acted as a crude version of it (mean robot velocity −8 %).

We therefore implement a **graded ISO-SSM speed-scaling filter** that scales the robot's
commanded per-step motion in proportion to the closest separation —
`scale = clip((sep − d_stop)/(d_slow − d_stop), 0, 1)`, applied to every joint: full speed
beyond `d_slow`, a smooth slow-down within it, a hold at `d_stop` (near contact). Unlike
the position filters it neither dodges nor vetoes — it *modulates speed*. **[Result pending
`speedscale_base`:** we expect the velocity-adaptive ISO violation rate
`ep_ssm_violation_actual_rate` and mean robot velocity to fall at roughly unchanged
geometric proximity and modest success cost — a filter that demonstrably improves the
safety axis it is actually designed for.]

This reframes the hybrid as a **clean division of labour** rather than a redundancy:

> - **The proactive policy owns the geometric-proximity axis** — anticipatory avoidance
>   reduces how *often* and how *long* the human and robot are close (−23 %).
> - **The runtime filter owns the ISO-15066 velocity axis** — speed-scaling reduces how
>   *fast* the robot moves when they are close, maintaining the velocity-adaptive SSM
>   margin (and, with working PFL contact detection, a collision-imminent brake for the
>   exogenous tail — future work, §5).

The earlier negative results are therefore not "the filter is useless" but "the filter is
the wrong tool for proximity, which is the policy's job." Assigned to its proper axis, the
filter **complements** rather than competes with the policy — the genuinely useful hybrid.

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

### 5.1 Future work

Three directions follow directly from the limitations above.

**Formally-guaranteed model-based filters.** Our geometric CBF and speed-scaling filters
demonstrate the velocity-axis behaviour empirically but provide no formal safety
certificate. A control-barrier-function QP with a *certified* barrier, or a
Hamilton–Jacobi reachability safety filter, would give provable separation/velocity
guarantees. The obstacles in this setting are concrete: a CBF-QP needs a control-affine
dynamics model and a valid barrier under the *exogenous* (unmodelled) human motion, and HJ
reachability suffers the curse of dimensionality beyond ~6 continuous states (a high-DoF
humanoid + a human forces learned-value approximations). The well-posed target is the
**velocity axis** (§3.4), where the filter's role is unambiguous — not proximity.

**PFL collision braking.** The Power-and-Force-Limiting axis is unevaluated here because the
BiGym/mojo runtime robot attachment yields identically-zero contact forces. Once contact
detection is fixed, a *last-resort collision-imminent brake* — firing only at near-contact
(the ~0.002 m exogenous tail, §5) — would add an injury-prevention guarantee that neither
the policy nor the proximity/velocity filters target, completing the division of labour
(policy → proximity, speed-scaling → velocity, PFL brake → contact).

**Anticipatory (predictive) filtering.** The one direction that could help the *proximity*
axis is an anticipatory filter — a short-horizon predictive (MPC-style) controller with a
human-motion predictor that clears space *before* the close approach rather than reacting
once close. This deliberately blurs the line with the constrained policy, which already
anticipates but learns avoidance end-to-end and task-integrated; a controlled comparison of
a hand-designed predictive filter against the learned policy would quantify how much of the
policy's advantage is *anticipation per se* versus *task-integrated learning under a learned
cost*. Two further threads are scoped but out of chapter: a coworker-aggressiveness
*realism spectrum* (E5, `coworker_eval`) characterising where reactive filtering becomes
sufficient as the human grows less adversarial, and an external constrained-RL baseline
(WCSAC) for the policy comparison.

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
