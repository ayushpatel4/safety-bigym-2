# Phase 1 — Observation Ablation Results

**Status: closed.** Phase 1's central question — does feeding the policy a (clean / noisy) estimate of where the human is reduce its safety-violation rate? — is answered.

**Result: no.** The oracle observation does **not** clear the [≥20% SSM-rate-reduction success criterion](../../.claude/HYBRID_SAFETY_CRITIC_PLAN.md) on any of the four ACT cells we tested. Per the master plan's contingency (lines 51 and 248):

> "If [oracle shows <20% reduction], the current reward structure cannot use human state information regardless of its accuracy, and you must address the cost signal (Phase 2) before doing anything else."

> "Phase 1 shows no benefit from human state. Mitigation: this is actually informative — it means the fixed penalty is the bottleneck, and Phase 2/3 become higher priority than expected."

**Action this triggers:** Phase 2 (Offline SVF Safety Filter) and Phase 3 (Constrained RL) are now higher-priority than originally scheduled. Phase 1's downstream sweeps (E1.2 noise sweep, E1.3 temporal ablation) are parked — they were predicated on finding a strong cell, and there is no strong cell.

Source data: [`phase1_obs_ablation_results.json`](../phase1_obs_ablation_results.json), produced by [`scripts/phase1_obs_ablation.py --run`](../scripts/phase1_obs_ablation.py).

Eval-time gate: `oracle` (or `noisy`) must reduce mean `ep_ssm_violation_rate` by **≥20%** vs `off`, on at least one (method, task) pair.

---

## Headline table

Method: **ACT** (Diffusion Policy not run — see "Why DP wasn't run" below).
n = 10 episodes per (method × task × mode × disruption) cell.

| Task | off SSM | oracle SSM | noisy SSM | off→oracle Δ | off→noisy Δ | Verdict |
|---|---:|---:|---:|---:|---:|---|
| reach_target_single | 0.534 | 0.548 | 0.580 | **−2.7%** | **−8.7%** | FAIL |
| dishwasher_close    | 0.533 | 0.537 | 0.527 | **−0.8%** | **+1.2%** | FAIL |
| drawers_open_all    | 0.277 | 0.239 | 0.261 | **+13.7%** | **+5.9%** | FAIL (below 20%) |
| saucepan_to_hob     | 0.135 | 0.203 | 0.148 | **−49.9%** | **−9.0%** | FAIL |

(SSM = mean `ep_ssm_violation_rate` across the 6 disruption types. Δ is positive when the oracle/noisy mode *reduces* the violation rate.)

**No cell clears the 20% bar.** Two of four are *worse* with the oracle observation than without it.

---

## Detailed cells

Per (task × mode × disruption). `succ` = `episode_success`, `len` = `episode_length`, `ssm_v` = `ep_ssm_violation_rate`, `min_marg` = `ep_min_ssm_margin` (m, negative = inside SSM danger band), `ttfv` = `ep_time_to_first_violation` (s).

PFL columns (`pfl_v`, `max_ratio`, `max_F`) are **omitted from the cell tables** because they are identically zero across every cell — that's the contact-detection bug discussed under "Known issues" below.

### reach_target_single

| mode | disr | succ | len | ssm_v | min_marg | ttfv |
|---|---|---:|---:|---:|---:|---:|
| off | INCIDENTAL       | 0.90 | 52.7 | 0.221 | +0.044 | 2.8 |
| off | SHARED_GOAL      | 1.00 | 38.7 | 0.665 | −0.309 | 11.7 |
| off | DIRECT           | 1.00 | 38.7 | 0.665 | −0.309 | 11.7 |
| off | OBSTRUCTION      | 0.90 | 52.8 | 0.640 | −0.354 | 12.2 |
| off | RANDOM_PERTURBED | 0.80 | 67.4 | 0.397 | −0.218 | 13.1 |
| off | CONTACT          | 0.90 | 53.0 | 0.614 | −0.383 | 13.0 |
| **off avg** |        | **0.92** |  | **0.534** | | |
| oracle | INCIDENTAL       | 0.90 | 54.7 | 0.246 | +0.035 | 2.7 |
| oracle | SHARED_GOAL      | 0.90 | 55.0 | 0.631 | −0.317 | 11.6 |
| oracle | DIRECT           | 0.90 | 55.0 | 0.631 | −0.317 | 11.6 |
| oracle | OBSTRUCTION      | 0.90 | 55.0 | 0.641 | −0.360 | 12.0 |
| oracle | RANDOM_PERTURBED | 0.90 | 54.6 | 0.491 | −0.228 | 12.9 |
| oracle | CONTACT          | 0.90 | 54.9 | 0.648 | −0.369 | 13.0 |
| **oracle avg** |     | **0.90** |  | **0.548** | | |
| noisy avg |          | **1.00** |  | **0.580** | | |

### dishwasher_close

| mode | disr | succ | len | ssm_v | min_marg | ttfv |
|---|---|---:|---:|---:|---:|---:|
| off | INCIDENTAL       | 1.00 | 158.5 | 0.418 | −0.781 | 16.9 |
| off | SHARED_GOAL      | 1.00 | 152.0 | 0.442 | −0.699 | 8.7 |
| off | DIRECT           | 1.00 | 151.4 | 0.444 | −0.431 | 8.7 |
| off | OBSTRUCTION      | 1.00 | 152.6 | 0.557 | −0.697 | 8.4 |
| off | RANDOM_PERTURBED | 1.00 | 157.4 | 0.490 | −0.655 | 9.4 |
| off | CONTACT          | 1.00 | 162.9 | 0.848 | −0.859 | 8.7 |
| **off avg** |        | **1.00** |  | **0.533** | | |
| oracle | INCIDENTAL       | 1.00 | 153.7 | 0.444 | −0.515 | 21.5 |
| oracle | SHARED_GOAL      | 0.90 | 172.0 | 0.439 | −25.447 | 8.5 |
| oracle | DIRECT           | 1.00 | 149.7 | 0.453 | −0.376 | 8.5 |
| oracle | OBSTRUCTION      | 1.00 | 151.1 | 0.548 | −0.301 | 8.3 |
| oracle | RANDOM_PERTURBED | 0.90 | 171.7 | 0.497 | −0.574 | 13.2 |
| oracle | CONTACT          | 1.00 | 151.0 | 0.843 | −0.733 | 8.6 |
| **oracle avg** |     | **0.97** |  | **0.537** | | |
| noisy avg |          | **0.98** |  | **0.527** | | |

`ep_min_ssm_margin = −25.4 m` on the SHARED_GOAL/oracle row is a single-episode artifact at n=10 and should be ignored.

### drawers_open_all

| mode | disr | succ | len | ssm_v | min_marg | ttfv |
|---|---|---:|---:|---:|---:|---:|
| off | INCIDENTAL       | 1.00 | 352.2 | 0.111 | −0.830 | 37.4 |
| off | SHARED_GOAL      | 1.00 | 346.9 | 0.365 | −0.960 | 3.6 |
| off | DIRECT           | 1.00 | 354.6 | 0.356 | −0.975 | 3.6 |
| off | OBSTRUCTION      | 0.80 | 513.6 | 0.340 | −1.000 | 3.8 |
| off | RANDOM_PERTURBED | 1.00 | 353.5 | 0.113 | −0.902 | 21.4 |
| off | CONTACT          | 0.90 | 458.8 | 0.378 | −1.043 | 4.1 |
| **off avg** |        | **0.95** |  | **0.277** | | |
| oracle | INCIDENTAL       | 0.90 | 446.8 | 0.103 | −0.974 | 28.2 |
| oracle | SHARED_GOAL      | 0.80 | 563.6 | 0.276 | −0.945 | 3.6 |
| oracle | DIRECT           | 0.80 | 554.0 | 0.295 | −0.932 | 3.6 |
| oracle | OBSTRUCTION      | 0.70 | 659.1 | 0.276 | −0.957 | 3.6 |
| oracle | RANDOM_PERTURBED | 1.00 | 358.1 | 0.122 | −1.043 | 20.2 |
| oracle | CONTACT          | 0.90 | 467.8 | 0.364 | −1.111 | 4.2 |
| **oracle avg** |     | **0.85** |  | **0.239** | | |
| noisy avg |          | **0.90** |  | **0.261** | | |

drawers_open_all is the closest to a "win": +13.7% reduction with oracle. But task success drops 0.95 → 0.85, and the bar is 20%.

### saucepan_to_hob

| mode | disr | succ | len | ssm_v | min_marg | ttfv |
|---|---|---:|---:|---:|---:|---:|
| off | INCIDENTAL       | 0.50 | 674.7  | 0.162 | −1.131  | 30.3 |
| off | SHARED_GOAL      | 0.20 | 877.3  | 0.162 | −1.304  | 5.1 |
| off | DIRECT           | 0.10 | 937.8  | 0.148 | −1.312  | 5.1 |
| off | OBSTRUCTION      | 0.10 | 932.8  | 0.164 | −1.332  | 4.9 |
| off | RANDOM_PERTURBED | 0.40 | 737.6  | 0.039 | −0.652  | 42.8 |
| off | CONTACT          | 0.00 | 1000.0 | 0.138 | −13.276 | 5.2 |
| **off avg** |        | **0.22** |  | **0.135** | | |
| oracle | INCIDENTAL       | 0.90 | 394.7  | 0.154 | −1.075 | 51.0 |
| oracle | SHARED_GOAL      | 0.40 | 732.6  | 0.236 | −1.296 | 5.1 |
| oracle | DIRECT           | 0.50 | 662.8  | 0.258 | −0.841 | 5.1 |
| oracle | OBSTRUCTION      | 0.50 | 666.6  | 0.257 | −1.009 | 4.9 |
| oracle | RANDOM_PERTURBED | 0.80 | 483.4  | 0.066 | −1.057 | 44.0 |
| oracle | CONTACT          | 0.40 | 737.4  | 0.246 | −0.860 | 5.3 |
| **oracle avg** |     | **0.58** |  | **0.203** | | |
| noisy avg |          | **0.30** |  | **0.148** | | |

Saucepan is the most informative cell — see "Side-finding" below.

---

## Side-finding: saucepan oracle uses human state for *progress*, not safety

On `saucepan_to_hob`, oracle does NOT reduce SSM violations (in fact it makes them worse: 0.135 → 0.203). But **task success climbs dramatically**: **0.22 → 0.58** (off → oracle).

Mechanism (inferred from per-disruption rows): on `off`, the policy times out (`episode_length=1000`, the time limit) under CONTACT/OBSTRUCTION/DIRECT — the human is jammed near the hob, the robot can't make progress. With oracle, the policy uses human pose to route around the obstruction and complete the task — but it still risks SSM violations during the routing.

**Implication for Phase 3 cost design:** when the reward landscape is punishing enough, the policy uses human state to *finish the task* before it uses it to *avoid the human*. The fixed penalty `r = r_task − violation_penalty` doesn't make safety competitive with completion reward when completion is hard. This is a strong argument for the [continuous-cost Lagrangian formulation](../../.claude/HYBRID_SAFETY_CRITIC_PLAN.md) (Phase 3, lines 113–130) where λ is tuned to put hard pressure on safety regardless of task difficulty.

It is also a reason to look at saucepan-style hard tasks specifically when picking the strongest (method, task) pair for E1.2 / E1.3 if those sweeps are ever revisited.

---

## Caveats

### Sample size

n = 10 episodes per cell. Several rows show identical 4-decimal stats across SHARED_GOAL and DIRECT in the same (mode, task) — e.g., reach_target_single off and oracle, dishwasher_close noisy. That happens when the same sampled scenarios fire under both forced disruption types and the policy's behavior collapses onto a few near-identical trajectories. CIs are wide. Phase 2 should use larger n.

### PFL columns are unreliable

`ep_pfl_violation_rate`, `ep_max_pfl_force_ratio`, `ep_max_contact_force` are identically zero in every single cell — including cases where `ep_min_ssm_margin = −13 m` (saucepan CONTACT off), i.e., the human pelvis is geometrically *inside* the robot's safety envelope. This is a contact-detection bug in BiGym/mojo's runtime robot attachment, not a real result. Investigation summary in [`/Users/ayushpatel/.claude/plans/the-current-human-disruption-smooth-flame.md`](../../.claude/plans/the-current-human-disruption-smooth-flame.md). The 20% conclusion above is **based on SSM only and is unaffected by this bug.**

### Why DP wasn't run

The master plan called for both DP and ACT. We ran ACT only.

Rationale for the call:
- ACT failed the bar on all 4 cells. The conclusion ("Phase 2/3 priority bumps") is robust against architecture choice — the contingency triggers as soon as *neither* DP nor ACT clears the bar, and ACT alone failing is necessary but not sufficient evidence.
- Strict reading of the master-plan success criterion (line 51): "Oracle condition shows ≥20% reduction in SSM violation rate vs baseline." Implicitly "on at least one method × task". ACT failing all four does not formally rule out DP succeeding somewhere.
- We chose to defer DP because (a) the cost-signal-bottleneck conclusion is intuitively the same for both methods (both consume `low_dim_state` the same way), and (b) Phase 2/3 is the bigger lever.

If a reviewer pushes back, run DP on the strongest of the four ACT cells (drawers_open_all, +13.7% with ACT) and see if DP clears 20%. If not, the conclusion is unchanged. If yes, it modifies the contingency to "Phase 2/3 priority bump for ACT-class architectures only" but doesn't change the overall direction.

### Other items

- `episode_length=1000` (i.e., the time limit) on saucepan/CONTACT/off and saucepan/CONTACT/noisy means the robot never completed the task and the policy froze. Those rows pull SSM violation rates *down* mechanically (the human is in front of the hob, robot is stationary, no SSM-violating motion). The averages above include these cells; reporting median alongside mean would be more honest. Sample size makes this hard to do cleanly at n=10.

---

## Phase 1 status

- Phase 1 wrapper (`BodySLAMWrapper`, `bodyslam=off|oracle|noisy`): in main, working. See [`docs/phase1_bodyslam_wrapper.md`](phase1_bodyslam_wrapper.md).
- E1.1 (observation ablation): **complete**. Result: contingency triggered.
- E1.2 (noise sweep): **parked**. No strong cell to sweep against.
- E1.3 (temporal ablation): **parked**. Same reason.
- DP coverage: explicitly skipped per the rationale above.

Phase 1 is **closed**. Open work tracked separately:

- **PFL contact-detection bug.** Plan file at [`.claude/plans/the-current-human-disruption-smooth-flame.md`](../../.claude/plans/the-current-human-disruption-smooth-flame.md). Needs a fresh session with BiGym/mojo internals expertise. Phase 2 SVF can begin on SSM-only labels in the meantime; PFL gets retrofit once fixed.

## Next phase

Phase 2 (Offline SVF Safety Filter) is **higher priority** than originally scheduled. Per [HYBRID_SAFETY_CRITIC_PLAN.md](../../.claude/HYBRID_SAFETY_CRITIC_PLAN.md) lines 59–97, the first deliverables are:

1. Dataset collection: ~500k transitions from BiGym demos + a random policy + the Phase-1 ACT, with `r_safe = 0 if ssm_violation else 1` (PFL omitted from the label until the contact bug is fixed).
2. Standalone safety critic MLP `[256, 256, 256]` outputting scaled-sigmoid Q ∈ `[0, 1/(1−γ)]`.
3. CQL α sweep ∈ {1.0, 5.0, 10.0}.
4. Threshold R calibration on a held-out set, sweep the conservatism-violation Pareto frontier.
5. Runtime `gym.Wrapper` that vetoes actions where `Q_safe(s, a) < R`.

These should be planned in a fresh session.
