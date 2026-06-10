# Phase 3 — Constrained RL Integration: scope & status

Created 2026-05-20. A standalone read to understand what Phase 3 is, what's done, what's left,
and the decision points. Authoritative design lives in [UPDATED_PROJECT_PLAN.md](UPDATED_PROJECT_PLAN.md)
(Phase 3 section); live task status in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). This doc
is the orientation layer over both.

---

## 1. Where Phase 3 sits

The project builds a **Hybrid Safety Critic** for a manipulation robot sharing space with a live human:

| Phase | What | Status |
|---|---|---|
| 0 / 0.5 | Env + ISO 15066 safety signals (SSM/PFL), human injection, per-step + per-episode metrics | ✅ done |
| 1 | Observation ablation under BC (ACT) — does a human-state channel help? | ✅ done (channel alone didn't clear the bar) |
| 1.4 (C2) | Same question under RL (CQN-AS) | ⚠️ degenerate w/ `num_demos=0`; **folded into Phase 3 E3.6** |
| 2 | Offline **Safety Value Function** (SVF) filter — a runtime gate trained on collected transitions | ✅ done (`checkpoints/svf_coworker_train_v1.pt`, operating point R≈4.0) |
| Workstream D / BASE-FIX | CQN-AS demos + bounded workspace curriculum — unblocks non-degenerate training | ✅ SMPL-H base curriculum finished; G1/tightened-stage verification pending |
| **3** | **Constrained RL** — a policy that *internalises* safety via a Lagrangian cost constraint | 🔵 **P3.0 done; P3.1 code complete/unit-tested; GPU smoke pending** |
| 4 | Hybrid deployment — Phase 3 policy + frozen Phase 2 filter, improved fallback | ⬜ not started |
| 5 | Evaluation — tail-risk metrics, baselines (incl. WCSAC) | ⬜ not started — **WCSAC baseline ✅ implemented** (`agent=wcsac`), GPU runs pending |

**The Phase 2 filter and the Phase 3 policy are two different safety mechanisms** that converge in Phase 4:
- **Phase 2 SVF** = a *runtime filter*. It sits outside the policy and vetoes unsafe actions at deployment. Trained offline, frozen.
- **Phase 3 policy** = a policy that has *learned to be safe itself* during training, so it rarely needs the filter.

They share the same network shape on purpose (weight transfer), but play different roles.

---

## 2. Goal of Phase 3

> Train a task policy that internalises safety via a Lagrangian cost constraint, using a **continuous**
> cost signal, so that it achieves baseline task performance with a **>50% reduction in SSM violations**
> and **<10% task-reward loss**, *without the runtime filter active at eval time*.

The motivating finding from Phases 1/1.4: a *binary* violation penalty (or no reward gradient at all)
doesn't redirect the policy toward safety — it either ignores the human or, worse, learns to **evacuate
the workspace** (the C2 collapse). Phase 3's thesis is that a **continuous, anticipatory cost** plus a
**Lagrangian multiplier** gives a rich enough gradient to trade a little task efficiency for a lot less
risk — *while staying engaged with the task* (enforced by workspace shaping).

---

## 3. The architecture (value-based Lagrangian on CQN-AS)

CQN-AS is **critic-only** — no actor network. The policy is implicit: `argmax_a Q(s, a_bin)` over
coarse-to-fine discretised action bins. So the classic actor-critic Lagrangian is re-expressed in
value-based form. Three options, in increasing sophistication:

| Option | Cost handling | Action selection | Role |
|---|---|---|---|
| **A-value** | single Q on shaped reward `r' = r_task + r_workspace − λ·c_t` | `argmax_a Q(s,a)` | prototype only (non-stationary target) |
| **B-value-mean** | dual Q: `Q_r` (reward) + `Q_c` (mean cost) | `argmax_a [Q_r − λ·Q_c]` | **recommended starting headline** |
| **B-value-CVaR** | `Q_c` distributional (Gaussian/quantile head) | `argmax_a [Q_r − λ·CVaR_α(Z_c)]`, α∈{0.95,0.99} | **recommended final headline** |

**Why decoupled `Q_c`** (vs one shaped reward): each Q-network regresses toward its *own stationary
scalar* — `Q_r` toward task reward, `Q_c` toward cost. λ enters **only at action-selection time**, never
in a regression target. That removes non-stationarity by construction. Plus, `Q_c` is architecturally
identical to the Phase 2 SVF, so it can warm-start from it.

### The pieces of the objective

```
# 1. Continuous per-step cost (anticipatory — fires BEFORE a violation)
c_ssm = max(0, 1 − ssm_margin / d_buffer)      # 0 far, 1 at the violation boundary; d_buffer = 0.3 m
c_pfl = max(0, pfl_force_ratio − 0.8)          # 0 until 80% of the force limit
c_t   = min(1, max(c_ssm, c_pfl))              # worst-case across both ISO 15066 criteria

# 2. Workspace shaping (keeps the policy engaged — counters evacuation)
r_workspace = −β · min(max(0, ‖p_ee − p_task‖ − r_ws), cap)
r_ws=0.4 m, β=0.05, cap=1.0                         # bounded to fit C51 support
r_task'     = r_task + r_workspace

# 3. PID-controlled Lagrangian multiplier λ (the "how hard to enforce safety" knob)
cost_violation = rolling_mean_cost − d              # d = cost budget; rolling CVaR_α for B-CVaR
λ = clip(λ + K_I·cost_violation + K_P·cost_violation + K_D·Δcost_violation, 0, λ_max)
# starting gains: K_I=1e-3, K_P=1e-2, K_D=0, λ_max=100, d=0.01
```

**Critical detail — per-step cost backup.** CQN-AS executes K-step action *sequences*. `Q_c`'s Bellman
target must be computed **per executed env-step**, not per K-step chunk — otherwise the policy can
satisfy the mean-over-chunk budget while spiking violations *inside* a chunk. The pipeline already
carries `c_t` at per-env-step granularity (P3.0c); P3.1 must keep it that way in the backup.

---

## 4. What is COMPLETED (P3.0 scaffolding + supporting work)

All bodyslam-agnostic scaffolding is merged (PR #9 + follow-ups on `main`). Smoke: `python scripts/phase3_p30_smoke.py` (~10 s).

| Component | File(s) | State |
|---|---|---|
| **Workspace reward shaping** (P3.0a) | [`SafetyConfig`](../safety_bigym/config.py) (`add_workspace_penalty`/`workspace_radius=0.4`/`workspace_beta=0.05`/`workspace_excess_cap=1.0`); `SafetyBiGymEnv._compute_workspace_penalty()` in [`safety_env.py`](../safety_bigym/envs/safety_env.py); threaded via [`safety_bigym_factory._create_env`](../safety_bigym/envs/safety_bigym_factory.py); CLI-overridable via `cfgs/env/safety_bigym.yaml` | ✅ off by default; tested (`tests/test_workspace_shaping.py`) |
| **Per-step continuous cost `c_t`** (P3.0c) | [`filters/cost_signal.py`](../safety_bigym/filters/cost_signal.py) `compute_cost()`; attached to TimeStep in [`agents/cqn_as/env_adapter.py`](../safety_bigym/agents/cqn_as/env_adapter.py); n-step discounted `cost` + `max_cost` in [`replay_buffer.py`](../safety_bigym/agents/cqn_as/replay_buffer.py); `cost` data_spec in [`train_cqn_as.py`](../train_cqn_as.py) | ✅ flows end-to-end into the batch dict |
| **Cost critic scaffolding** (P3.0b) | [`filters/cost_critic.py`](../safety_bigym/filters/cost_critic.py) — MLP twin of the Phase 2 `SafetyCritic`; `warm_start_from_svf()` refuses without `force_sign_flip=True` | ✅ module + tests (`tests/test_cost_critic.py`); retained for future warm-start variants |
| **Lagrangian CQN-AS agent** (P3.1) | [`agents/cqn_as/lagrangian_agent.py`](../safety_bigym/agents/cqn_as/lagrangian_agent.py), [`agents/cqn_as/lagrangian.py`](../safety_bigym/agents/cqn_as/lagrangian.py), `cfgs/agent/cqn_as_lagrangian.yaml` | ✅ code complete + unit-tested; GPU smoke pending |
| **Demo pipeline** (Workstream D) | `SafetyBiGymCQNAdapter.get_demos()` in [`env_adapter.py`](../safety_bigym/agents/cqn_as/env_adapter.py) | ✅ tests green; 2k smoke passed; demos carry safe-side `cost=0.0` |
| **CQN-AS vendor + train entrypoint** | [`agents/cqn_as/`](../safety_bigym/agents/cqn_as/), [`train_cqn_as.py`](../train_cqn_as.py) | ✅ snapshot/eval-video cadence fixed; demo replay buffer wired |
| **Phase 2 SVF (warm-start source)** | `checkpoints/svf_coworker_train_v1.pt` | ✅ trained/eval'd/swept |

**Current P3.1 status:** the per-step cost `c_t` is in the training batch
(`batch["cost"]`, `batch["max_cost"]`) and the Lagrangian agent consumes it
through a CQN-AS-shaped cost critic. The remaining gate is a 2000-frame GPU
smoke on a usable base snapshot, then E3.* sweeps.

> ⚠️ **`c_pfl` is identically zero** under the open PFL contact-detection bug (see CLAUDE.md). So in
> practice `c_t == c_ssm` right now. Phase 3 proceeds on the SSM-driven cost; PFL is a later retrofit.

---

## 5. What is TO BE DONE

### P3.1 — the Lagrangian glue (code complete; GPU smoke pending)

The constrained-RL machinery is implemented in a sibling Lagrangian agent rather
than by editing the vendored `agent.py`: CQN-AS-shaped `Q_c` + target network,
λ PID on rolling cost, dual-Q selection `argmax_a [Q_r − λ·Q_c]` at each
coarse-to-fine level, and logging for λ / rolling cost / `q_c_loss`.

Remaining: run the 2000-frame GPU smoke with `agent=cqn_as_lagrangian` on the
usable base snapshot, confirm λ moves and `q_c_loss` logs, then hand full E3.*
sweeps to the GPU box.

### P3.2+ — experiments (GPU sweeps, after P3.1 lands and smokes)

| Exp | Question | Sweep |
|---|---|---|
| E3.1 | cost signal form | fixed −0.05 penalty vs binary 0/1+λ vs continuous+λ |
| E3.2 | cost budget Pareto | d ∈ {0.001, 0.01, 0.05, 0.1} |
| E3.3 | λ update method | gradient ascent vs PID |
| E3.4 | filter during training | Phase 2 SVF on/off as a training-time veto |
| E3.5 | architecture | A-value vs B-value-mean vs B-value-CVaR (headline) |
| E3.6 | observation channel | `bodyslam=off` vs `noisy` (this is where the old E1.4 question lands) |
| E3.7 | external baseline | WCSAC (distributional safe-RL reference) on the humanoid — ✅ DONE (`agent=wcsac`); trained+evaluated, results in `docs/wcsac_results.md` |
| E3.X.workspace | defend β | β ∈ {0.0, 0.05, 0.2, 0.5, 1.0} — pick the knee that prevents evacuation w/o task-success loss |

### Pre-sweep smoke gates (must pass before any full GPU sweep)
1. CQN-AS runs end-to-end on stock `reach_target_single`. ✅ (A1)
2. CQN-AS + BodySLAMWrapper + COWORKER composes, channel consumed. ✅ (A6)
3. **Per-step cost backup confirmed for `Q_c` (not chunk-level).** ⬜ P3.1 must verify.
4. **Workspace penalty lands in `r_task` before Q-learning, right sign.** ✅ wiring done (P3.0a/d); confirm under live training.

---

## 6. Gating & dependencies (what unblocks what)

- **Obs config (old E1.4 / C3 decision): folded into Phase 3.** We are **not** re-running E1.4 as a
  standalone 3-cell gate (user decision 2026-05-20). Rationale: Phase 1 already showed the channel
  isn't the dominant lever, the cost-signal Lagrangian is the thesis lever, and `Q_c`/the filter
  consume the channel regardless. → Start the actor with `bodyslam=oracle` (or noisy) and run the
  off/oracle/noisy ablation inside **E3.6** on the constrained policy.
- **Demo pipeline and BASE-FIX curriculum: unblocks non-degenerate training.** Without demos, CQN-AS never
  discovers the sparse task reward (the C2 collapse). The bounded SMPL-H base curriculum has passed;
  the current next gate is the G1/tightened-stage curriculum run before trusting G1-constrained sweeps.
- **Phase 2 SVF: optional warm-start + training-time filter.** `Q_c` can warm-start from the SVF
  (`force_sign_flip=True`); the SVF can also act as a training veto (E3.4).

---

## 7. Contingencies & known risks

### Outcome decision table (from the plan)
| Outcome | Interpretation | Next step |
|---|---|---|
| SSM violations ↓>50% with <10% task-reward loss | constrained RL works | → Phase 4 hybrid |
| Violations ↓ but task reward collapses (>30%) | λ too aggressive / budget too tight | loosen `d`; lower `λ_max`; check reward scaling |
| Oscillation (safe-frozen ↔ unsafe-productive) | λ update unstable | increase `K_D`, reduce `K_P` |
| No improvement over fixed-penalty baseline | Option A too crude | move to B-value-mean/CVaR (dual critic) |
| Continuous helps but binary doesn't | confirms the gradient-richness hypothesis | validates the continuous formulation |

### Standing risks specific to this codebase
- **PFL bug → `c_pfl ≡ 0`.** Cost is SSM-only today. If PFL contact detection gets fixed, a *new*
  dataset collection + relabel is needed (the schema is forward-compatible; the values aren't). Phase 3
  is defensible on the proximity/SSM cost alone (anticipatory, fires before contact).
- **Evacuation local optimum.** The whole reason bounded workspace shaping exists. Keep the support invariant
  `β·workspace_excess_cap/(1−γ) ≤ |v_min|`; the unbounded β=0.2 setting already failed by saturating the critic.
- **Per-chunk vs per-step cost.** The seductive bug: averaging cost over the K-step action sequence
  hides intra-chunk spikes. Backup must stay per-env-step. The pipeline already preserves this; P3.1
  must not regress it.
- **Warm-start sign error.** `Q_c` (high = dangerous) is the *opposite* of the SVF (high = safe).
  `warm_start_from_svf` guards this with `force_sign_flip=True` (copies the body, reinits the head).
- **Don't launch multi-hour training from inside the agent.** Write + smoke (≤100 steps / ≤2k frames),
  then hand the full sweep to the GPU box (project rule).

---

## 8. Definition of done (Phase 3)

A Lagrangian-trained CQN-AS policy on the COWORKER eval ParameterSpace that:
- reduces SSM violation rate **>50%** vs the fixed-penalty baseline,
- loses **<10%** task reward,
- stays engaged (no evacuation — bounded distance-to-task, low EE-left-workspace fraction),
- achieves this **with the runtime filter OFF at eval** (the filter is Phase 4's job),
- with the architecture/cost/budget/β choices each defended by their own sub-experiment (E3.2/3.5/X).

Then → **Phase 4** (compose with the frozen SVF filter; the filter should trigger rarely, allowing a
less conservative R than Phase 2's calibration).

---

## 9. Quick reference

- **Run a constrained-RL train (once P3.1 lands):** `python train_cqn_as.py env=safety_bigym/<task> disruption=coworker_train bodyslam=oracle num_demos=10 env.safety.add_workspace_penalty=true save_video=true wandb.use=true ...` (+ P3.1 flags for λ/d/option).
- **P3.0 smoke:** `python scripts/phase3_p30_smoke.py` (`+phase3_p30_smoke.dry_run=true` skips MuJoCo).
- **Headless GPU box:** `export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`; **AMASS:** `export AMASS_DATA_DIR=.../CMU/CMU`.
- **Key constants:** `d_buffer=0.3 m`, `r_ws=0.4 m`, `β=0.05`, `workspace_excess_cap=1.0`, PID `K_I=1e-3/K_P=1e-2/K_D=0`, `λ_max=100`, `d=0.01`.
- **Branch:** `safety-critic/phase-3-constrained-rl` (per branch strategy); recent urgent fixes went direct to `main` by explicit override.
