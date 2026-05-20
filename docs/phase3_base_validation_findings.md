# Phase 3 base-policy validation — findings, diagnosis, and fix

*Record for the project report. Date: 2026-05-20. Task: `saucepan_to_hob` (BiGym), CQN-AS
agent, live SMPL-H coworker disruption. Documents why the pre-P3.1 base-policy validation
failed, the root-cause analysis, and the fix applied before constrained RL (P3.1) is trained.*

---

## 1. Run configuration

The validation was the gate that must pass before training the Lagrangian constrained-RL
policy (P3.1): demonstrations + workspace reward shaping must first produce a *non-degenerate*
base policy (one that attempts the task and whose episodes don't collapse). A Lagrangian
multiplier can only *trade away* task reward for safety — it cannot create task competence the
base policy lacks — so the base must be sound first.

| Setting | Value |
|---|---|
| Task | `saucepan_to_hob` (long-horizon bimanual manipulation) |
| Agent | CQN-AS (coarse-to-fine distributional critic, C51), demo-driven |
| Disruption | `coworker_train` (SMPL-H human approaches 0.9–1.4 m, reaches for the EE 40–60 % of visits) |
| BodySLAM | `oracle` (ground-truth human-position channel) |
| Demos | `num_demos=10` |
| Reward shaping | `add_workspace_penalty=true` (β=0.2, r_ws=0.4 m) |
| Frames | ~50 000 |

## 2. Observed results

- **Episode length:** ~1000 (the full step budget) on nearly every episode; dipped to ~890 a
  couple of times. The **31-step "evacuation collapse"** seen in the earlier demo-less run (C2)
  is **gone** — demos + shaping fixed that failure mode.
- **Episode reward:** fell **monotonically** over training, from **−78** early to **−775** late.
- **Behaviour (eval videos):**
  - *Early training:* the robot attempts the task (reaches for the cabinet) but struggles to open
    it; when the human approaches it moves to the far side of the cabinet, then retreats to a fixed
    distance and stays there.
  - *Late training:* task attempts get *worse*, not better — it parks away from the workspace.

**Verdict: FAILED.** Episodes no longer collapse, but the policy is degenerate in a new way — it
learns avoidance, task competence decays, and the logged return diverges.

## 3. Root-cause analysis

### Issue A (primary, a real bug): the dense shaping reward is incompatible with the critic's value support

The workspace penalty is a dense per-step term, added on top of the sparse task reward
(`{0, +1}`, +1 only at success):

```
r_ws(t) = -β · max(0, ‖p_ee − p_task‖ − r_ws),   β = 0.2,  r_ws = 0.4 m
```

(`SafetyBiGymEnv._compute_workspace_penalty`, `safety_env.py`.)

The CQN-AS critic is **C51 distributional with a fixed value support `[v_min, v_max] = [−2, +2]`**
(`cfgs/agent/cqn_as.yaml`), `γ = 0.99`, `nstep = 1`. Its Bellman target is **clamped to that
support** (`agent.py`):

```
Tz = reward + γ · support
Tz = Tz.clamp(min=v_min, max=v_max)
```

The discounted value of holding the EE at distance `d` is a geometric series:

```
V(d) = r_ws / (1 − γ) = −β·(d − r_ws) / (1 − γ) = −0.2·(d − 0.4) / 0.01 = −20·(d − 0.4)
```

So `V(d) = −2` already at **d = 0.5 m**, and *every* larger distance clamps to the −2 floor. The
critic **saturates** across the entire penalty region: it cannot distinguish 0.5 m from 4 m, the
per-bin Q-values go flat, the coarse-to-fine `argmax` becomes arbitrary, and **the gradient that
should pull the EE back toward the task is clipped away**. Before shaping, returns lived in
`[0, ~1]` (sparse reward), comfortably inside `[−2, +2]`; adding the dense penalty silently broke
the support assumption the critic was tuned for.

This matches the symptoms exactly: the *logged* return worsens to −775 (≈ a 1000-step undiscounted
sum of the penalty several metres out), while the *learning signal* is dead, so the policy drifts
to wherever physics + behaviour-cloning push it — away from the blocking human — and stays.

### Issue B (secondary): train/demo distribution mismatch

The 36 cached BiGym demos contain **no human** — they replay with `cost=0` and no coworker, and the
human-position channel is synthesised from AMASS only for observation-width matching
(`env_adapter.py::get_demos`). The live training env *does* have an approaching coworker. Behaviour
cloning thus teaches "go to the task, ignore the human," but at runtime the human physically blocks
that path, producing off-distribution states where the (already saturated) critic gives no guidance
→ retreat. The run also used only 10 of the 36 available demos.

## 4. The fix (4 levers; user decision 2026-05-20)

Strategy: **de-saturate the critic** so the shaping gradient survives, and **reduce the
distribution gap** so the policy learns the task before facing the full human disruption.

1. **Bound the penalty (rescale).** New config `workspace_excess_cap = 1.0 m` caps the excess
   distance, and `workspace_beta` is lowered `0.2 → 0.05`, so the per-step penalty ∈ [−0.05, 0].
   Discounted floor = `−0.05 / (1 − 0.99) = −5`. The penalty is now **flat** beyond the cap, which
   is exactly the property the critic needs (it never has to represent an unbounded return).
2. **Widen + skew the value support.** `v_min = −6, v_max = +2, atoms = 101` (≈0.08 per atom —
   unchanged resolution; `v_min=−6` gives ~20 % headroom over the −5 floor; `v_max=+2` still covers
   the discounted sparse success). **Design invariant:** `β · excess_cap / (1 − γ) ≤ |v_min|`.
3. **Bump demos 10 → 36** (all cached `saucepan_to_hob` demos).
4. **Human curriculum (staged).** The env is stateless w.r.t. training step (no within-run ramp
   without new plumbing), but snapshot-resume works with zero new code (`train_cqn_as.py::load_snapshot`),
   so the curriculum is three sequential stages, each resuming the prior snapshot:
   **stage 0** idle/distant human (`disruption=coworker_idle` — present but ~3 m off and never
   reaching, so it doesn't interfere) → **stage 1** gentle coworker (`disruption=coworker_easy`)
   → **stage 2** full `disruption=coworker_train`. Stage 0 doubles as a "can it learn the task at
   all?" sanity check. *The human is kept present (not removed) so the obs width / BodySLAMWrapper
   / `human_pos_estimate` channel — and therefore the model architecture — match across all stages
   and snapshots resume cleanly; `inject_human=false` would change the obs width under
   `bodyslam=oracle` and break resume.*

### Code changes (landed 2026-05-20)
- `workspace_excess_cap` threaded through 4 sites: `SafetyConfig` (`config.py`), env yaml
  (`cfgs/env/safety_bigym.yaml`), factory (`safety_bigym_factory.py`), and the penalty formula
  (`safety_env.py::_compute_workspace_penalty`). `SafetyConfig` defaults updated: `workspace_beta
  0.2 → 0.05`, `workspace_excess_cap = 1.0` (set `None` to recover the old unbounded behaviour).
- New `cfgs/disruption/coworker_easy.yaml` — gentle stage-1 coworker (farther, less frequent reach).
- New `scripts/run_base_curriculum.sh` — 3 staged launches with snapshot-resume; carries levers
  1–3 as CLI overrides. Human runs it on the GPU box (`SMOKE=1` does a ≤2000-frame stage-0 smoke).
- `tests/test_workspace_shaping.py` extended with cap-bounds + support-invariant tests (14 pass).
- `scripts/phase3_p30_smoke.py` reference penalty made cap-aware.

## 5. Verification
- **Local (done):** `pytest tests/test_workspace_shaping.py` → 14 pass (incl. cap saturation, cap
  inactive below cap, `cap=None` reproduces unbounded, support invariant). Full suite: 321 pass,
  34 skipped, 0 fail. Invariant check: `0.05 · 1.0 / 0.01 = 5 ≤ |v_min| = 6`. ✅
- **GPU box (staged, human launches `run_base_curriculum.sh`):**
  - **Stage 0 (idle/distant human)** is the gate — episode_reward must climb **>0 on some episodes**
    with returns inside [−6, 2] (no saturation). If stage 0 fails, the problem is demos/CQN-AS, not
    safety: stop and reassess.
  - Stages 1→2: task attempts persist as the human ramps in; episodes stay long; `safety/ssm_violation`
    stable/down; reward does not diverge.
- **Then** un-park P3.1: `agent=cqn_as_lagrangian` 2000-frame smoke on the fixed base, then hand the
  E3.* sweeps to the human.

## 6. P3.1 status — parked (no change needed)
P3.1 (the Lagrangian glue) is code-complete + unit-tested on branch
`safety-critic/phase-3-constrained-rl` (`lagrangian.py`, `lagrangian_agent.py`,
`cfgs/agent/cqn_as_lagrangian.yaml`, 3 test files; 10 local tests pass, agent test gated on
`tensordict`). It correctly remains behind this validation gate and needs no edits.

## 7b. Bring-up bug found during re-validation: C51 projection index overshoot (fixed 2026-05-21)

The first stage-0 smoke crashed with an opaque CUDA device-side assert
(`indexFuncLargeIndex ... dstIndex < dstAddDimSize`) inside the vendored
`compute_target_q_dist` `index_add_`. Isolation: `num_demos=36` crashed at ~step 871,
`num_demos=10` ran clean — and a finite-batch guard added to `train_cqn_as` did **not**
fire, proving the reward was finite (not a NaN).

Root cause: the C51 target projection computes `b = (Tz − v_min)/delta_z` then
`upper = ceil(b)`. `delta_z` (here 0.08) is not exactly representable in float32, so
when a transition's target reaches the support ceiling `Tz = v_max` — which happens for
**reward=1 success transitions** (`1 + 0.99·v_max` clamps to `v_max`) — the division
rounds up (`b = 100.0000022` for `atoms=101`) and `ceil(b) = atoms`, one past the valid
range `[0, atoms-1]`, so `index_add_` addresses an out-of-bounds atom. It is stochastic:
more successful demos (29/36 vs 6/10) → more reward=1 transitions sampled → it surfaces
within a few hundred steps instead of never. The widened support and demo bump only
*exposed* this latent vendored bug; they didn't cause it.

Fix: clamp `b` to `[0, atoms-1]` before floor/ceil in `compute_target_q_dist`
(`agent.py`) — a one-line numerical bugfix matching standard Rainbow implementations;
projection math is unchanged for interior values. Also fixes the same path used by the
P3.1 cost critic. Regression test: `tests/test_c51_projection_bounds.py` (forces the
`Tz=v_max` boundary; tensordict-gated). A finite-batch guard in `train_cqn_as` now turns
any *future* non-finite reward/discount/action/cost into a legible error instead of a
device-side assert.

## 7. Durable lesson
**Any shaped/dense reward must keep its discounted return inside the critic's value support
`[v_min, v_max]`, or the C51 Bellman-target clamp silently saturates value learning** — the
agent trains without error and produces a degenerate policy. When adding reward shaping to a
distributional critic, always check `|shaped per-step reward| / (1 − γ) ≤ support half-range`,
or bound the shaping term as we did here.

**And: a C51 target projection must clamp its atom index** — float rounding at the support
ceiling can push `ceil(b)` one atom past the end and crash `index_add_` with an opaque
device-side assert. Diagnose opaque CUDA index asserts by re-running with
`CUDA_LAUNCH_BLOCKING=1` (pins the real op) and guarding batch tensors with `torch.isfinite`
to rule out NaN first.
