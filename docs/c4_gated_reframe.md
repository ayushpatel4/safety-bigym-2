# C4 and the gated hybrid: result + reframe (RESOLVED 2026-06-10)

**Outcome in one line:** critic-gating the speed-scaler — the mechanism that recovers
throughput on dishwasher/drawers — **does not recover it on saucepan_to_hob**. C4 stays
the unconditional point; the experiment yields a clean *boundary condition* for when the
gated hybrid helps. This is the pre-registered **"else" branch** of the decision rule.

Provenance: `scripts/dispatch_gated_saucepan.py` (30 cells, gated_speedscale + controls,
3 seeds × 60 ep = 180 ep/row, noisy/G1/coworker_train); `results/e4_1/gated_saucepan/`
(+`summary.json`); figure `results/figs/gated_pareto.png` (`scripts/plot_gated_pareto.py`).

---

## What C4 is

C4 = ROW3 Lagrangian policy + **unconditional** speed-scaler (`d_slow=0.40`): success
**0.85 → 0.44**, prox 0.250, ssm-actual 0.065 (180 ep; reproduced from
`hybrid_speedscale_d0p40_seed{0,1,2}.csv`). Structural for that filter: it slows the robot
*every* time the coworker is within `d_slow`, so the policy cost (×0.89) and reactive cost
(×0.58) stack multiplicatively. The fix we tested: gate the scaler on the SVF critic
(`gated_speedscale`, q<R), which on dishwasher/drawers recovered most of the lost success at
the same velocity gain (dish uncond 0.52 → gated 0.67; draw 0.38 → 0.80,
`docs/runtime_filter_results.md`). It had never been run on saucepan.

## ⚠️ Snapshot caveat (read first)

The exact C3/C4 **basin** checkpoints (`fixlam_0p1/lam0p1_seed{0,1,2}/snapshot_30546,8225,
32696.pt`, picked by *deployment proximity*) were **deleted in a cleanup**. The only
survivors are `snapshot_best.pt` per seed, which `train_cqn_as` selected by **peak eval
success** (steps 25058/32706/20854) — the Lagrangian mis-selection the gotcha in
`docs/CLAUDE.md` warns about (peak-success → less-avoiding checkpoint). So this sweep ran on
a *less-avoiding* policy than the headline. The built-in R=0 control caught it: it gives
deployment prox **0.271** vs C3's 0.228.

**Why the conclusion still holds.** The unconditional-on-`snapshot_best` control reproduces
C4 on the two axes the gating question is about: **succ 0.433 ≈ C4 0.44, ssm 0.072 ≈ C4
0.065** (only proximity differs, 0.276 vs 0.250 — the policy's axis, not the filter's). The
gating verdict is a statement about the *filter* on the velocity/success axes, where
`snapshot_best` is a faithful stand-in. A fully apples-to-apples rerun needs the basin
checkpoints (retrain the 3 fixed-λ policies → re-pick by `analyze_row3.py pick`; multi-hour
hand-off) — see "Should we retrain?" below.

## Results (all on `snapshot_best`, 180 ep/row)

| config | R | success | proximity | **ssm-actual** | reads as |
|---|---|---|---|---|---|
| v3 gated | 0.0 | 0.722 | 0.271 | 0.138 | policy alone (gate never fires) |
| v3 gated | 1.5 | 0.717 | 0.299 | 0.127 | gate barely active |
| v3 gated | 2.0 | 0.706 | 0.295 | 0.125 | |
| v3 gated | 2.25 | 0.700 | 0.297 | 0.133 | |
| v3 gated | 2.5 | 0.567 | 0.282 | 0.105 | success starts to fall |
| v3 gated | 2.75 | 0.472 | 0.275 | 0.102 | |
| v3 gated | 3.0 | 0.472 | 0.283 | 0.084 | approaching unconditional |
| **uncond** | d=0.40 | **0.433** | 0.276 | **0.072** | gate's R→large limit (≈ C4) |
| v3op gated | 2.0 | 0.733 | 0.297 | 0.149 | on-policy critic: **no help** |
| v3op gated | 2.5 | 0.644 | 0.279 | 0.130 | on-policy critic: no help |

*Refs (basin ckpts): C1 baseline 0.85/0.296/0.146 · C3 policy 0.76/0.228/0.112 · C4 hybrid
0.44/0.250/0.065.*

**Decision rule (pre-registered): a gated row must reach succ ≥ 0.60 AND ssm-actual ≤ 0.08.
No row does.** The closest is v3 R=3.0 (0.472/0.084) — below the success floor — and v3 R=2.5
(0.567/0.105) — below both. → **C4 stays the unconditional point.**

## Why gating fails here (the actual finding)

1. **Gating just slides the operating point from policy-alone to unconditional.** As R rises,
   (succ, ssm) moves monotonically from (0.722, 0.138) → (0.433, 0.072); the gated curve
   *tracks* the unconditional point rather than dominating it. It never opens a
   high-success + low-velocity corner. See `results/figs/gated_pareto.png`.
2. **Because the coworker is persistently co-located.** On saucepan the human reaches into
   the workspace almost continuously (the §3.4 "freeze-lite, fires almost constantly"
   observation), so the SVF gate is *on* whenever the robot is near the human — which is most
   of the task. Gating therefore degenerates to unconditional scaling. On dishwasher/drawers
   contact is *intermittent*, so the gate finds genuine safe windows and recovers success.
   **This is the boundary condition: critic-gating recovers throughput only under
   intermittent co-location.**
3. **The on-policy critic does not rescue it — it is worse** (v3op R=2.0: ssm 0.149 >
   policy-alone 0.138; no velocity help at all). Re-collecting the critic on the avoiding
   policy's own rollouts (the expensive P2 fix for the §3.1 OOD problem) does not help,
   confirming the limit is the **task structure**, not critic calibration — the same message
   as §3.1 ("the limit is the reactive paradigm, not the critic").

## Thesis framing (honest, and stronger than a bare 0.44)

This is **not** "the hybrid is broken." It completes the §3.4 story:

> The critic-gated hybrid recovers throughput when the coworker is **intermittently**
> co-located (dishwasher, drawers) but **collapses to the unconditional scaler when the
> coworker is persistently co-located** (saucepan), because the safety gate is then active
> almost continuously. On saucepan, comprehensive both-axis ISO safety (proximity *and*
> velocity) is therefore reachable only at the unconditional scaler's explicit
> safety–throughput cost (C4: 0.85→0.44); gating cannot buy it back, and neither can an
> on-policy critic. The hybrid's value is thus **task-conditional**, set by the human's
> co-location pattern — a boundary condition, not a free lunch.

Keep C4 as the headline both-axis point; add this as the "we tried to rescue it" paragraph +
the `gated_pareto.png` figure. It ties §3.1 (critic isn't the lever), §3.4 (persistent
co-location), and the dishwasher/drawers gated success into one cross-task account.

## This is operating-point selection, not cherry-picking

The whole sweep is reported (table + figure); the headline pick follows the **stated rule**
decided before the runs; every safety number carries its success cost; the protocol matches
the rest of the table. We did **not** hide the success cost, quote a favourable subset, or
swap which hybrid the headline means. The honest answer turned out to be a negative result —
and we report it as one.

## Should we retrain for exact apples-to-apples numbers?

**Probably not necessary.** The unconditional control on `snapshot_best` already reproduces
C4 on the success+velocity axes (0.433/0.072 ≈ 0.44/0.065), and the gating conclusion is a
velocity/success-axis statement, so it transfers. Retraining the 3 fixed-λ=0.1 policies to
recover the basin checkpoints (then re-pick by `analyze_row3.py pick --by safety`) is a
multi-hour GPU hand-off that would, at most, shift the proximity column and sharpen the
exact numbers — it is **unlikely to flip the verdict** (a *more*-avoiding basin policy has
*less* velocity for the gate to cut, so gating would help no more than here). Recommend:
report the result with this caveat; retrain only if a reviewer demands the basin policy
specifically. **Action item if retraining:** also re-save basin checkpoints with
`pick_best_snapshot.py --by safety` so they survive the next cleanup (and add them to the
snapshot-registry allowlist).
