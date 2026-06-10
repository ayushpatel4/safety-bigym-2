# Runtime safety filters on dishwasher_close & drawers_open_all

Follow-up to `docs/budget_sweep_results.md` (constrained-RL failed to deliver
deployable safety). This investigates **runtime filters** — wrapping the curriculum
policy at deployment — and, critically, re-frames *which safety axis is even
controllable by the robot*.

## The reframing: proximity is largely exogenous

The prior saucepan SVF work (note in `filters/snapshots.py`) established, and this
re-analysis confirms on these two tasks, that **time-in-proximity is ~42% not
robot-controllable**: even a frozen robot only cuts it ~58%; the rest is the
coworker walking into the robot's space. So no robot policy/filter can move it much,
and every "no safety benefit" result on the proximity metric was partly measuring an
exogenous quantity.

The **robot-controllable** safety axis is ISO-15066 SSM: is the robot moving slow
enough to stop before contact given the separation. Metrics: `ep_ssm_violation_actual_rate`,
robot velocity, `ep_min_ssm_margin_actual`. All results below are reported on this axis.

## SVF learned critic + binary veto — backfires

Trained a Safety Value Function (CQL, 200k steps, 52,794 transitions collected
random+on-policy under coworker_train; `checkpoints/svf_dish_drawers_v1.pt`). The
critic **separates** safe vs unsafe in-sample (safe q̄=3.14, unsafe q̄=2.26; at R=2.75
it catches 92% of near-contact transitions vetoing 22% of safe). But the **runtime
veto → zero-velocity backfires**:

| task | R | interv | succ | ssm_viol | meanVel | maxVel | minSSM |
|---|---|---|---|---|---|---|---|
| dish baseline | 0 | 0.00 | 0.77 | 0.176 | 0.444 | 2.46 | 0.160 |
| dish svf | 2.25 | 0.12 | 0.63 | 0.174 | 0.469 | 3.77 | −1.69 |
| dish svf | 3.0 | 0.84 | 0.12 | 0.138 | 0.300 | 6.03 | −6.84 |
| draw baseline | 0 | 0.00 | 0.82 | 0.100 | 0.192 | 2.87 | −0.84 |
| draw svf | 2.25 | 0.17 | 0.43 | 0.130 | 0.311 | 3.43 | −1.28 |

**Diagnosis:** the policy is CQN-AS with **TemporalEnsembleControl** (chunked action
sequences). A binary veto → zero-velocity mid-chunk breaks the action-sequence
coherence: the robot stops, then **overshoots with a large catch-up action** when the
veto lifts. Max velocity *rises* (2.46→6.03), SSM margins go sharply negative, the
SSM-violation rate does NOT improve, and success collapses. A binary action veto is
incompatible with chunked control. (The critic is fine; the intervention is wrong.)

## Speed-scaling continuous backstop — works

`--safety-filter speedscale` scales motion-bearing action dims *smoothly* by
separation (ISO-15066: full speed beyond d_slow, linearly to 0 at d_stop), preserving
the policy's action *direction* — no veto, no overshoot.

| task | config | succ | **ssm_viol** | Δssm | meanVel | maxVel | minSSM |
|---|---|---|---|---|---|---|---|
| dish | baseline | 0.77 | 0.176 | — | 0.444 | 2.46 | 0.160 |
| dish | speedscale_mod (0.5/0.15) | 0.52 | **0.087** | **−50%** | 0.340 | 2.46 | 0.206 |
| dish | speedscale_agg (0.8/0.25) | 0.28 | **0.086** | **−51%** | 0.257 | 2.46 | 0.216 |
| draw | baseline | 0.82 | 0.100 | — | 0.192 | 2.87 | −0.836 |
| draw | speedscale_mod | 0.38 | **0.059** | **−41%** | 0.135 | 2.89 | −0.657 |
| draw | speedscale_agg | 0.20 | **0.053** | **−47%** | 0.107 | 2.85 | −0.451 |

Speed-scaling **halves the SSM-violation rate**, slows the robot near the human, and
*improves* the min stopping margin — and **maxVel is unchanged** (2.46→2.46), confirming
no overshoot (vs the veto's 6.03). The cost is task success (dish 0.77→0.52, draw
0.82→0.38 at the moderate config). This is the **first method in the whole
investigation** to improve the robot-controllable safety axis: a real, monotonic
safety↔success trade, deployable as an ISO-15066 velocity backstop.

## Gated hybrid (SVF critic gates speed-scaling) — the best operating point

The unconditional speed-scaler slows the robot whenever it is within d_slow of the
human — including when the situation is actually safe — paying a large success cost.
The **gated hybrid** (`--safety-filter gated_speedscale`, code in `runners.py` +
`benchmark_policy.py`) applies the *same* speed-scaling but **only when the SVF critic
flags risk** (q < R); when q ≥ R the action passes at full speed. Critic decides
*when*, speed-scaling decides *how* (overshoot-free). R is the selectivity dial.

| task | method | succ | Δsucc | ssm_viol | Δssm | minSSM |
|---|---|---|---|---|---|---|
| dish | baseline | 0.77 | — | 0.176 | — | 0.160 |
| dish | speedscale (uncond) | 0.52 | −0.25 | 0.087 | −50% | 0.206 |
| dish | gated R=2.0 | 0.77 | 0.00 | 0.176 | 0% | 0.110 |
| dish | gated R=2.5 | 0.68 | −0.08 | 0.130 | −26% | 0.102 |
| dish | **gated R=2.75** | **0.67** | **−0.10** | **0.088** | **−50%** | 0.121 |
| draw | baseline | 0.82 | — | 0.100 | — | −0.836 |
| draw | speedscale (uncond) | 0.38 | −0.43 | 0.059 | −41% | −0.657 |
| draw | gated R=2.25 | 0.85 | +0.03 | 0.101 | +1% | −0.834 |
| draw | **gated R=2.5** | **0.80** | **−0.02** | **0.091** | **−9%** | −0.836 |
| draw | gated R=2.75 | 0.80 | −0.02 | 0.092 | −8% | −0.832 |

**Result:** gating recovers the success the unconditional scaler gave up while keeping
the safety gain. On dishwasher, **R=2.75 matches the full −50% SSM-violation reduction
at less than half the success cost** (0.67 vs 0.52). On drawers the trade is gentler
(−9% SSM at −0.02 success vs unconditional's −41% at −0.43). R dials the operating
point: low R = selective/no-op, high R → unconditional. The critic correctly suppresses
scaling when the state is safe, so the robot only slows at genuinely risky moments.

## Follow-up experiments

**(1) Pareto d_slow × R (dishwasher).** Across d_slow ∈ {0.4,0.5,0.6,0.8} at d_stop=0.15,
the **gate threshold R is the dominant dial**; d_slow is secondary. R=2.75 gives −49…−51%
SSM at succ 0.63–0.67 regardless of d_slow; R=2.5 gives −18…−26% at succ ~0.68. Smaller
d_slow (0.4) is marginally better on success at fixed SSM. So the R-sweep *is* the Pareto
frontier; **d_slow=0.4, R=2.75** is a hair better than 0.5/2.75. (`results/hybrid_extra/`)

**(2) Row-5 hybrid — gate on a Lagrangian policy.** Gating the backstop on the budget-0.22
basin (the best-available "proactively-avoiding" Lagrangian policy) vs the plain curriculum
policy is a **wash**: dish 0.70/−36% (Lagrangian) vs 0.67/−50% (curriculum); draw 0.77/−25%
vs 0.80/−8%. Neither dominates. This **confirms the cliff** — constrained-RL produced no
genuinely proactive-avoider for these tasks, so the full row-5 hybrid ≈ gate-on-baseline.

**(4) Velocity-aware critic.** The naive SSM-margin label is a documented dead end (69%
of transitions have ssm_margin<0 at kitchen velocities → degenerate "everything unsafe").
Pivoted to **anticipatory proximity thresholds** (relabel at 0.20/0.30 m so the critic warns
earlier). Result: the anticipatory critics (p20/p30) are **no better** — they trade along the
same success/safety curve (dish R=2.75: p20 0.63/−35%, p30 0.63/−44%, vs orig-0.10m
0.67/−50%; drawers p30 0.52/−26% — more braking, less success). **The 0.10 m near-contact
label is the best gate** (most selective). Key robustness finding: the hybrid's operating
point is set by the speed-scale mechanism + the exogenous-proximity ceiling, **not** by critic
calibration. (`results/anticipatory_sweep/`, `checkpoints/svf_dish_drawers_p{20,30}.pt`)

**(3) DAgger re-collection on filtered rollouts.** Added a gated-filter wrapper to the
collector (`svf_collect_dataset.py --filter-critic`, `_GatedFilterCollectPolicy`) so the
snapshot source records the ON-FILTER distribution; re-collected (43k transitions),
retrained the critic (`svf_dish_drawers_dagger.pt`), re-benchmarked. Result: **null /
slightly worse** — dish R=2.75 0.68/−2% (vs orig 0.67/−50%); draw 0.70/−16% (vs 0.80/−8%).
On dishwasher the DAgger critic **under-gates** because it was trained on the *filtered*
distribution (where the filter already slowed the policy → those states look safe → it
rarely fires). Lesson: for a runtime gate, train the critic on the **unfiltered** policy's
behaviour — it must recognise when the *raw* policy would be unsafe. (Also re-confirmed the
exogenous-proximity ceiling: the filtered dishwasher snapshot still had 29% near-contact —
the filter slows the robot but can't stop the human walking in.)

**All four refinements point back to the original config** for dishwasher; **drawers was
improved** by pushing the gate harder (it gates rarely on the slow drawers policy, so it
had headroom). Recommended operating points (original 0.10 m critic, gated_speedscale):
- **dishwasher: R=2.75, d_slow=0.5, d_stop=0.15 → 0.67 succ, −50% SSM.** (R=3.0 doesn't help;
  R=3.0 + d_slow=0.4 reintroduces veto-like *overshoot* maxVel 6.10 — keep d_slow ≥ 0.5.)
- **drawers: R=3.0, d_slow=0.8, d_stop=0.25 → 0.73 succ, −22% SSM** (vs the earlier
  R=2.75/0.5 → 0.80/−8%). Gentler point R=2.75/d_slow=0.8 → 0.80/−10% (≈free). The drawers
  gate fires rarely at R=2.75 (slow policy, few fast-near-human moments), so R=3.0 + earlier
  scaling (d_slow=0.8) ~tripled the SSM reduction. (`results/improve/`)

## Verdict

- **Geometric proximity is not the right target** — it is ~42% human-driven. Report
  robot safety on the SSM-velocity axis.
- **Constrained-RL (Lagrangian)** can't deliver: budgets are inert above natural cost,
  task-fatal below it, and binding λ degrades into *faster* motion (`budget_sweep_results.md`).
- **Learned-SVF binary veto** can't deliver on a chunked policy: the veto breaks action
  coherence and causes overshoot.
- **Continuous speed-scaling IS the working velocity backstop** — ~50% SSM-violation
  reduction at a success cost (unconditional: dish 0.77→0.52, draw 0.82→0.38).
- **The critic-gated hybrid is the best operating point** — same ~50% SSM reduction on
  dishwasher (R=2.75) at *half* the success cost of unconditional scaling, by slowing
  only when the SVF critic flags risk. R dials selectivity. This is the deployable
  Hybrid Safety Critic for these tasks: a learned safety critic gating an ISO-15066
  velocity backstop. **Recommended config: gated_speedscale, d_slow=0.5/d_stop=0.15,
  R≈2.75 (dish) / 2.5 (draw).**

**Next** (not run): (1) sweep d_slow/d_stop × R for a full Pareto surface; (2) the proper
hybrid eval — gated backstop on top of a Lagrangian (proactively-avoiding) policy rather
than the plain curriculum policy; (3) re-collect the SVF dataset on-policy from the
*filtered* rollouts (the critic was trained on unfiltered behaviour); (4) a velocity-aware
critic (the current SVF labels geometric near-contact, not stop-distance).

Provenance: `results/svf_sweep/` (veto), `results/speedscale/` (uncond speed-scale),
`results/gated_sweep/` (gated hybrid), `checkpoints/svf_dish_drawers_v1.pt`,
`datasets/svf_dish_drawers_v1/`, `scripts/dispatch_{svf,gated}_sweep.py`. Gated-filter
code: `safety_bigym/benchmark/runners.py` (`gated_speedscale`), `scripts/benchmark_policy.py`.
