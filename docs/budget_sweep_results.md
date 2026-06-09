# Cost-budget retarget sweep — does re-targeting λ fix the null safety result?

**Question.** The main safety sweep (budgets 0.1/0.3/0.5) gave no deployment safety
because λ→0: budgets 0.3/0.5 sat *above* the policy's natural rolling cost (~0.25),
so the constraint was inert, and budget 0.1 was task-fatal (λ→21.5, 0% success). This
sweep re-targets the budget into the unsampled window **below** the natural cost to
test whether a binding-but-feasible operating point exists, or whether the cost/success
frontier is a hard cliff.

**Setup.** Same adaptive-λ recipe as `dispatch_safety.py` (cqn_as_lagrangian, workspace
OFF, widened critic support, warm from curriculum stage-2), only `cost_budget` varies:
{0.16, 0.19, 0.22} × {dishwasher_close, drawers_open_all}, seed 0, 40k frames.
(0.24 cells were skipped — redundant with the non-binding upper end.) Cost is the
SSM-margin signal `c_t = max(0, 1 − ssm_margin/0.3)`; budget is on its rolling mean.

## The constraint now binds — and binding collapses the task

Final (40k) λ read from each cell's snapshot PID state, and final in-training success:

| task | budget | final λ | rolling cost | final succ (in-train) | outcome |
|---|---|---|---|---|---|
| dishwasher | 0.16 | 13.8 | 0.225 | 0.30 | binds hard, **collapsed** |
| dishwasher | 0.19 | 9.3 | 0.235 | 0.10 | binds hard, **collapsed** |
| dishwasher | 0.22 | 1.5 | 0.252 | 0.30 | binds late, **collapsed** |
| drawers | 0.16 | 10.4 | 0.209 | 0.00 | binds hard, **collapsed** |
| drawers | 0.19 | 5.2 | 0.231 | 0.30 | binds hard, **collapsed** |
| drawers | 0.22 | 1.3 | 0.236 | 0.60 | binds late, partial |

Unlike the main sweep (λ=0 everywhere), λ is now positive — re-targeting *did* activate
the constraint. But the rolling cost never reaches the budget for 0.16/0.19 (stuck at
~0.21–0.24 > budget), so λ integrates upward unbounded and destroys the task. The
**minimum achievable cost is ~0.21–0.23 ≈ the natural operating cost**: there is no
budget that both binds and preserves the task. `dish_b22` was succ 1.00 at 37k (λ=0.69)
but collapsed to 0.30 by 40k (λ=1.5) — even the floor budget has no *stable* feasible
point.

## Deployment benchmark of the budget-0.22 cells (noisy, 3 seeds × 20 ep)

For each 0.22 cell we benchmarked its best λ-active "basin" checkpoint and its final
checkpoint, vs the unconstrained baseline:

| cell | success | prox (τ=0.3) | Δprox vs base | min-sep | robot vel |
|---|---|---|---|---|---|
| **dishwasher** baseline | 0.767 | 0.246 | — | 0.403 | 0.444 |
| dish b22 basin (snap 35026) | 0.817 [0.72,0.92] | 0.243 [0.15,0.34] | **−1%** | 0.412 | 0.532 |
| dish b22 final (snap 37801) | 0.483 [0.35,0.62] | 0.230 [0.14,0.32] | −7% | 0.387 | 0.785 |
| **drawers** baseline | 0.817 | 0.211 | — | 0.099 | 0.192 |
| draw b22 basin (snap 32655) | 0.767 [0.65,0.87] | 0.184 [0.14,0.24] | **−13%** | 0.114 | 0.211 |
| draw b22 final (snap 38453) | 0.317 [0.20,0.43] | 0.149 [0.09,0.21] | −29% | 0.166 | 0.433 |

- The basin checkpoints' low *in-training* proximity (dish 0.16, draw 0.13–0.16) is mostly
  10-ep oracle-obs eval noise: dish_b22 basin regressed to prox 0.243 (≈ baseline) under
  the 60-ep noisy benchmark.
- The relationship is a **monotonic tradeoff**: proximity falls only as success falls
  (final checkpoints: −7%/−29% prox, but success 0.48/0.32).
- The single best feasible point — **drawers budget-0.22 basin: −13% proximity at −5pt
  success** — has a proximity CI [0.14, 0.24] that overlaps the baseline (0.211). It is
  marginal / within noise, not a robust win.

## Addendum — warm-started rolling-cost estimate (tried, ineffective)

Hypothesis: λ binds late (~17k frames) because the PID's rolling-cost estimate warms
up from 0 over ~20k frames (0.99 momentum), so binding it earlier (init the estimate to
the natural cost ~0.25 via the new `agent.rolling_cost_init` knob) would give the
constraint the full run to settle gently instead of bang-banging late.

Result: **ineffective, for an instructive reason.** With `rolling_cost_init=0.25`, the
saved estimate had already **decayed to 0.050 by frame 2564** (and λ was still 0 at
4.5k, identical to the un-warm-started trajectory). The late binding is therefore *not*
a from-zero estimation artifact — the early rolling cost genuinely *is* low, because the
replay is dominated by the 69 loaded demos (collected without the coworker → ~zero SSM
cost) until on-policy data accumulates. The PID correctly tracks that low demo-diluted
cost, so any scalar init washes out in ~2k frames and λ still binds only once the true
on-policy cost climbs past budget (~17k). Run aborted after this diagnosis rather than
burning ~24 GPU-h re-confirming the plain-sweep collapse. (The knob is kept — default
0.0 preserves original behaviour. The mechanistically-correct version of "bind early"
would be `lambda_init>0`, but that applies sustained pressure that the cliff analysis
says collapses the task; the only lever with a real shot at *breaking* the cliff is
relaxing the episode time limit so the policy can slow near the human instead of
abandoning the task — a task-setup change, not a λ tweak.)

## Verdict

**The cliff is real.** Re-targeting the budget (the Tier-1 fix) makes the constraint bind,
but there is no budget at which scalar-λ constrained RL on this SSM-margin cost buys a
clear deployment-safety gain while preserving task success. The best case is a marginal,
within-noise −13% proximity. This confirms — now on the *binding* side of the budget
range, not just the inert side — that the bottleneck is the method, not the budget:
a single multiplier on *expected* cost cannot separate task-necessary proximity from
gratuitous risk.

**Implication.** The path to deployable safety is the decoupled runtime **Safety Value
Function filter** (Hybrid Safety Critic) — vetoing only the high-risk actions at
deployment — rather than more constrained-RL budget/λ tuning. Secondary levers that
could still help constrained RL but were not the bottleneck here: warm-starting the
rolling-cost estimate (it spent ~20k frames warming from 0, delaying binding), a
CVaR/tail cost instead of mean cost, relaxing episode time pressure so the policy can
slow near the human instead of abandoning the task, and fixing the identically-zero PFL
force term. Provenance: `exp_local/budget_sweep/`, `results/budget_sweep_eval/`,
`scripts/dispatch_budget_sweep.py`, `scripts/budget_sweep_status.py`.
