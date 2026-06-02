# SVF Runtime Filter — Fallback Study, Exogenous-Proximity Result, and Coworker-Aggressiveness Framing

**Status: 2026-06-02.** Report-feeding analysis. All numbers below are on the
**valid v3 critic** (`checkpoints/svf_coworker_train_g1_0p3_v3.pt`), evaluated via
the deployment-faithful benchmark harness, on the `coworker_train` disruption
(noisy BodySLAM, G1 coworker, proximity threshold τ = 0.3 m). The hybrid result
(rows 3/5 — Lagrangian policy + filter) is pending E3.1/E3.2 and is **not** in this
doc; this doc is the *filter component* analysis on the unconstrained baseline policy.

---

## 0. Precondition: the pipeline is valid (sweep predicts the benchmark)

This analysis is only meaningful because the v3 SVF pipeline was first made
deployment-faithful. Four collection-vs-deployment bugs were found and fixed in
`svf_collect_dataset.py` (each a place where the hand-built collection env/policy
drifted from the RoboBase factory `_create_env` the agent actually deploys through):

1. **action de-normalisation** — demo-derived stats, not `env.action_space`.
2. **action-execution mode** — open-loop chunks + temporal-ensemble blend, not
   receding-horizon `chunk[0]`.
3. **control_frequency** — `CONTROL_FREQUENCY_MAX // demo_down_sample_rate` (20 Hz),
   not the full 500 Hz (the 25× mismatch made the policy never complete the task).
4. **coworker scenario** — read `coworker_train.yaml` (closest 0.60–0.95 m, fast
   reach 1.3–2.2 s), not the drifted `_COWORKER_TRAIN_RANGES` Python preset
   (closest 0.9–1.4 m, slow reach 4.5–6.5 s).

**Validation:** the v3 sweep's R=0 (filterless) proximity = **0.286** matches the
benchmark row-1 proximity **0.296**, and the filtered `mean_q` at deployment (≈3.0)
matches the sweep (≈2.9–3.2). The earlier v1/v2 "31.7 % @ 21.6 % knee" was an
artifact of the broken collection (the policy never got close, so the critic looked
good but was meaningless at deployment).

---

## 1. The filter is a reactive backstop with a freeze-vs-flee dilemma

Two fallback strategies were evaluated (the action substituted when the filter
vetoes, i.e. when `Q_safe(s, a) < R`): **`zero_velocity`** (stop the robot) and
**`retreat`** (drive the floating base away from the estimated human). E4.1 rows
1 (baseline, no filter) vs 4 (baseline + filter), 60 episodes (3 seeds × 20):

| config | interv. | proximity (τ=0.3) | success | mean robot vel | ssm-actual | episode len | verdict |
|---|---|---|---|---|---|---|---|
| baseline (no filter) | — | 0.296 | 0.85 | 0.289 | 0.147 | 449 | reference |
| **zero_velocity**, R=2.25 | 15 % | 0.303 (≈0) | 0.78 | 0.266 (↓8 %) | 0.148 | 493 | **dwells** — no proximity reduction; mild velocity backstop |
| **retreat**, R=2.5, step 0.10 | 14 % | **0.095 (−68 %)** | **0.18** | **1.70 (6×)** | **0.180 (worse)** | 182 | **flees** — buys separation by abandoning the task and speeding up |
| **retreat**, R=2.25, step 0.04 | 2 % | 0.290 (≈0) | 0.75 | 0.451 | 0.158 | 358 | does nothing useful, still costs a little |

(Success/proximity changes for the two near-baseline rows are within overlapping
95 % CIs, i.e. "no real effect," not "small effect.")

**The dilemma.** A reactive filter that triggers when the human is already close
has only two responses, and neither wins on both safety axes *and* the task:

- **Freeze (`zero_velocity`)** is the ISO-15066-correct response (reduce robot
  speed when separation is small) — but stopping makes the robot **dwell** in the
  danger zone (episodes 449 → 493), so geometric proximity does not fall.
- **Retreat** *does* cut geometric proximity, but only by the robot **moving away
  continuously**, which (a) abandons the task (success 0.85 → 0.18, episodes 449 →
  182) and (b) **raises robot velocity 6×** (mean 0.29 → 1.70), which itself
  *worsens* the velocity-adaptive ISO-SSM margin (`ssm-actual` 0.147 → 0.180).
- A **gentle** retreat (rare + small step) avoids the flee cost but then reduces
  no proximity (0.296 → 0.290).

There is **no reactive operating point** (across R and `SVF_RETREAT_STEP`) that
cuts proximity at acceptable success and velocity. This is a property of reactive
control against a *moving* obstacle, not a tuning failure.

The sweep (geometric-proximity proxy) made `retreat` look excellent — R=2.5 gives
82 % proximity reduction at 16 % intervention — precisely because the sweep does
not measure success or velocity. The deployment benchmark exposes that the 82 % is
the robot running away. **Lesson for the write-up: the sweep is a necessary proxy
for tuning R, but task/velocity cost must be read from the full benchmark.**

Artifacts: `RetreatFallback` + `FALLBACK` / `SVF_RETREAT_STEP` knobs
(`safety_bigym/filters/fallback.py`), `tests/test_retreat_fallback.py`,
sweeps `results/svf_sweep_g1_0p3_v3/` (zero_velocity) and
`results/svf_sweep_g1_0p3_v3_retreat/` (retreat).

---

## 2. Why: proximity is largely exogenous (~42 % human-driven)

The cleanest single number: at R=4.0 the `zero_velocity` filter intervenes **100 %**
of the time — the robot is frozen every step — yet proximity only falls **58 %**
(0.286 → 0.12). The residual **42 %** is the coworker walking up to the *stationary*
robot. So baseline proximity decomposes into:

- **~58 % robot-driven** — the robot reaching into the shared workspace toward the
  human. A filter can suppress this (by freezing/retreating).
- **~42 % human-driven (exogenous)** — the human approaching the robot. **No veto
  of robot actions can prevent this**, because the filter only controls the robot.

This exogenous floor is what caps any reactive filter and motivates *proactive*
(anticipatory) avoidance by the policy.

---

## 3. Framing: is `coworker_train` a fair test?

A legitimate critique of the result above: the `coworker_train` coworker has **no
collision-awareness** — it executes a scripted trajectory toward the robot's
workspace regardless of the robot, and (`coworker_target_mix_p_ee ≈ 0.45–0.72`)
often reaches toward the robot's *end-effector*. A real human collaborator has
self-preservation and would not walk into a robot. So a chunk of the exogenous
42 % is, by construction, the human's "fault," and no robot-side safety system can
remove it.

This is a real point, but it cuts a specific way:

- **It is a *realism* argument, not a *flattery* argument.** Any change to the
  coworker must be justified as "this is a more realistic collaborator"
  (e.g. lower `target_mix_p_ee` so it reaches for the task object, not the robot's
  hand; keep `closest_approach` plausible), never as "this makes the filter pass."
  Choosing the hazard level that makes a method look good is the classic confound a
  reviewer will catch.

- **The freeze-vs-flee dilemma is robust to aggressiveness.** Lowering
  aggressiveness shrinks the *exogenous* component (so the filter can do more), but
  the dilemma itself — a reactive trigger can only freeze or flee — holds at every
  level. Aggressiveness sets the *magnitude* of the exogenous floor, not the
  existence of the dilemma.

### Which conclusion is stronger for the report?

**Keep the current (adversarial) finding as the centerpiece; do *not* replace it
with a softened-human "filter works" result.** Ranking:

1. **Strongest:** the adversarial case (reactive filter fails, freeze-vs-flee
   dilemma, ~42 % exogenous) **+** the proactive policy overcoming it (rows 3/5)
   **+** a spectrum across coworker aggressiveness (`coworker_eval` / E5). Reads as:
   *"reactive ISO-SSM filtering is fundamentally limited against an approaching
   human; proactive constrained-RL avoidance resolves it; here is the regime
   structure."* Novel, honest, cherry-pick-proof.
2. **Good but incomplete:** the adversarial finding alone — interesting, but needs
   the policy to avoid reading as "nothing works."
3. **Weakest:** a realistic-coworker-only "filter reduces proximity by X %"
   result — confirmatory (expected of any safety system), and still exposed to the
   "you picked the regime" critique even under a realism label.

**Rationale:** a well-explained *limitation* of one's own method teaches a
principle and motivates the architecture; a positive result on a benign,
self-selected scenario confirms the obvious and invites the cherry-pick question.
The realism point is best used to build a *spectrum* (characterising where reactive
filtering is sufficient vs where only proactive avoidance helps), which makes the
freeze-vs-flee result a deliberate part of the analysis rather than a weakness.

---

## 4. Recommendation / next steps

1. **Decide with the policy, not the human.** Run the hybrid (E4.1 rows 3/5) on the
   *current* `coworker_train` once E3.1/E3.2 finish. The decisive question is
   whether the Lagrangian policy reduces proximity **proactively** (anticipating the
   approach, clearing space early/gently) **without** the flee cost — i.e. whether it
   escapes the dilemma the reactive filter cannot.
   - Policy wins → keep the adversarial scenario; report filter-weak / policy-strong
     / hybrid-best. Strongest outcome.
   - Policy also fails → the adversarial scenario is unwinnable for everyone; *then*
     adopt a realism-justified, task-focused coworker as the primary eval, **and**
     report across the aggressiveness spectrum so it is unambiguously a robustness
     study, not a cherry-pick.
2. **Report the spectrum regardless** (E5, `coworker_eval`): filter / policy / hybrid
   vs coworker aggressiveness. This is the most defensible framing and turns the
   freeze-vs-flee dilemma into a characterised regime boundary.
3. **Operating point:** `snapshots.py` stays `_v3` + R=2.25 + `zero_velocity` (the
   honest baseline velocity backstop — it does not improve geometric proximity, but
   it does not abandon the task either). Re-confirm the filter R on the row-3
   Lagrangian policy in the hybrid, where the filter sees far fewer close approaches.

---

## 5. Policy-side preliminary (E3, training-eval — to be benchmark-confirmed)

⚠ Training-eval numbers, some cells in-flight; confirm with `benchmark_policy`
before locking. But the pattern is informative and complements §1–2:

**E3.1 cost form (at the default tight budget d=0.01):**

| cell | success | proximity | note |
|---|---|---|---|
| binary Lagrangian | 0.00 | 0.024 | safe by *abandoning the task* |
| continuous Lagrangian | 0.00 | 0.013 | safe by *abandoning the task* |
| **fixed** (CQN-AS + reward penalty β=0.05) | **0.80** | **0.242** | **graceful: −18% proximity @ −6% success** |

**E3.2 budget sweep (continuous Lagrangian):** d=0.001→0.05→0.1 → success
0.00→0.00→0.13. Every tested budget ≤0.1 collapses to ~0 success.

**Reading:**
- **Proactive avoidance beats reactive filtering (the thesis result).** The
  `fixed` policy cuts proximity 18% at ~6% success cost — a graceful net-positive
  the reactive filter never achieved (0% at acceptable cost). Modest, because it
  is still bounded by the ~42% exogenous floor (§2).
- **The hard Lagrangian over-constrains at every tested budget because the budgets
  are too tight.** Per-step cost ∈ [0,1] and the *task itself* incurs ~0.2–0.3
  average cost (baseline proximity ≈0.30), so budgets d≤0.1 are *below the task's
  inherent cost* → the only feasible policy is to not do the task. The graceful
  Lagrangian regime (d≈0.2–0.3, near baseline cost) was not swept; extend it
  (`COST_BUDGETS="0.2 0.3 0.5"`).
- **ROW3 candidate:** `fixed` (the one graceful policy so far); a loose-budget
  Lagrangian may match or beat it once trained. Benchmark-confirm before locking.

---

## 6. First full headline (ROW3 = fixed-penalty policy, noisy eval) — NULL on proximity; perception confound

⚠ 1-seed ROW3 (fixed policy), noisy eval. `results/e4_1/..._134241/`:

| metric | row1 base | row3 fixed-pol | row4 base+filter | row5 hybrid |
|---|---|---|---|---|
| success | 0.85 | 0.78 | 0.78 | 0.68 |
| proximity (τ=0.3) | 0.296 | 0.302 | 0.303 | 0.302 |
| ssm-actual | 0.147 | 0.149 | 0.148 | 0.143 |
| mean robot vel | 0.289 | 0.290 | 0.266 | 0.265 |

- **No row reduces geometric proximity** (all ~0.30). The fixed policy's
  *training-eval* −18% (proximity 0.242) **did not transfer** to noisy deployment
  (0.302). Only effect: the filter trims mean velocity ~8% (rows 4/5), at a success
  cost (hybrid worst, 0.68).
- **Likely cause — perception confound (testable):** policies *train on `oracle`*
  (clean human-pos, per the plan's footnote ²) but the headline *evals on `noisy`*.
  The training-eval 0.242 was oracle; the benchmark 0.302 is noisy. So noisy
  human-tracking probably degrades the proactive avoidance. **Diagnostic:** re-run
  the headline with `OBS_MODE=oracle` and compare row3-vs-row1 proximity.
  - Policy reduces proximity on oracle but not noisy → *proactive avoidance is
    perception-bottlenecked* (interesting, on-theme with BodySLAM).
  - No reduction even on oracle → aggressive scenario unwinnable for all methods →
    realism-spectrum pivot (well-evidenced).
- **Caveats:** ROW3 = fixed-penalty policy at 1 seed; the **Lagrangian** (the thesis
  method) is not deployment-tested yet (budget scan in flight). Re-run the headline
  with the scan-chosen Lagrangian budget, on both `noisy` and `oracle`.

---

## TL;DR

- The v3 SVF pipeline is **valid** (sweep predicts benchmark); the four-bug fix is a
  standalone methodological contribution.
- On the unconstrained baseline + adversarial coworker, the reactive filter has a
  **freeze-vs-flee dilemma** — neither fallback cuts proximity at acceptable
  success/velocity — because **~42 % of proximity is human-driven (exogenous)**.
- This **limitation finding is the stronger, more interesting, more defensible
  result** than a softened-human "filter works" result; the realism point is best
  used to build a **spectrum**, with the **proactive policy (rows 3/5)** as the
  decisive test of whether the hard scenario stays the headline.
