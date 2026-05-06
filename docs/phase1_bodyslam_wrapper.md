# Phase 1 — Mock BodySLAM++ Observation Wrapper

Branch: `safety-critic/phase-1-bodyslam-wrapper` (off `main` once the
Phase-0 retrain merges).

Phase 1 of the [Hybrid Safety Critic plan](../../.claude/HYBRID_SAFETY_CRITIC_PLAN.md)
asks one question: **does giving the policy a noisy estimate of where the
human is help safety?** This phase builds the obs-side plumbing and the
sweep harness. It deliberately does *not* answer the question — that's the
job of the GPU sweep + the follow-up `phase1_observation_results.md`.

The deliverable is:

- A `BodySLAMWrapper` that injects `human_pos_estimate ∈ ℝ⁶` into the env's
  observation dict, with a three-way switch (`off` / `oracle` / `noisy`).
- An AMASS-driven demo-replay path so BC pretraining sees realistic human
  trajectories on this channel — not a constant sentinel.
- A `cfgs/bodyslam/` Hydra config group plumbed through the factory.
- Three sweep scripts (E1.1 / E1.2 / E1.3) for GPU hand-off.

This phase touches both Diffusion Policy and ACT — the wrapper writes a
`(6,)` obs key that RoboBase's `ConcatDim` folds into `low_dim_state`, so
both policy classes see the new dims via the same code path.

---

## Architecture

The factory composes the train-env wrapper chain as:

```
SafetyBiGymEnv                         emits info["safety"]["human_pos"]
└─ BodySLAMWrapper      (NEW)          reads info, writes obs["human_pos_estimate"]
   └─ EpisodeSafetyMetrics             reads info["safety"], emits info["episode_safety"]
      └─ RescaleFromTanh / ConcatDim   (RoboBase) folds (6,) into low_dim_state
         └─ FrameStack / TimeLimit / ActionSequence ...
```

Order matters: the wrapper has to read `info["safety"]["human_pos"]`
populated by `ISO15066Wrapper.build_safety_info` at
[iso15066_wrapper.py:73-96](../safety_bigym/safety/iso15066_wrapper.py#L73),
and it has to emit obs *before* `ConcatDim` runs.

The factory also overrides `_wrap_env` so the **demo env** gets its own
BodySLAMWrapper (in `demo_replay=True` mode, with an AMASS position
provider). Without this, the demo replay buffer rejects demos with
`ValueError: arg low_dim_state has shape (63,), expected (69,)` — the
train env has the new key but the loaded BiGym demos don't.

```
DemoEnv                                replays recorded BiGym timesteps
└─ BodySLAMWrapper (demo_replay=True)  reads from AMASSDemoPositionProvider
   └─ RescaleFromTanh / ConcatDim ...
```

---

## Observation contract

New observation key: **`human_pos_estimate`** with shape `(6,)`.

| Index | Field          | Notes                                                                |
|-------|----------------|----------------------------------------------------------------------|
| 0–2   | x, y, z        | Pelvis position estimate (m, world frame).                           |
| 3     | occluded       | 1.0 if line-of-sight is blocked this step (default 0.0).             |
| 4     | staleness      | Steps since last fresh estimate. Rises during dropout, 0 otherwise.  |
| 5     | confidence     | `(1 − 0.5·occluded) · max(0, 1 − staleness/10)` ∈ [0, 1].            |

`confidence` is a derived continuous handle; redundant given
`occluded` and `staleness` but cheap and gives downstream Phase-2/3
critics one graded "trust" signal without rev'ing the obs shape later.

The Box space has `low = [-inf, -inf, -inf, 0, 0, 0]` and `high = [inf,
inf, inf, 1, inf, 1]`, dtype `float32`. The position channels are
unbounded; the flag channels stay in [0, 1] so RoboBase's
`RescaleFromTanhWithMinMax` is a no-op for them.

---

## Operating modes

Hydra config group `bodyslam` selects the mode:

| Preset            | Effect                                                                            |
|-------------------|-----------------------------------------------------------------------------------|
| `bodyslam=off`    | Wrapper is **not inserted**. `human_pos_estimate` is absent. Baseline arm.        |
| `bodyslam=oracle` | Wrapper inserts the key with the **clean** ground-truth pelvis position; flags 0. |
| `bodyslam=noisy`  | Full OU + latency + dropout pipeline (and optional ray-cast occlusion).           |

`bodyslam=off` is the default — runs that don't opt in are byte-identical
to pre-Phase-1 behaviour.

### Noise pipeline (`bodyslam=noisy`)

1. **OU process** on position. `x_{t+1} = α·x_t + (1−α)·μ_t + σ·ε_t`,
   default `α=0.9`, `σ=0.05` m, `ε ~ N(0, I_3)`. State held across steps;
   reset on `env.reset()`. Stationary std ≈ `σ/√(1−α²)` ≈ `0.115` m.
2. **Latency buffer.** A `deque(maxlen=L+1)` of recent OU samples; the
   wrapper emits `buf[0]` (oldest), giving an `L`-step lag — default `L=3`,
   ≈60 ms at 50 Hz.
3. **Occlusion.** Optional ray-cast from the H1 head camera to the human
   pelvis collision geom (`MujocoRayOcclusion`). When occluded, σ inflates
   by `occlusion_noise_mult` (default 3×). Off by default — opt in via
   `bodyslam.use_occlusion=true`. The factory falls back to no-occlusion
   if camera/geom name lookup fails.
4. **Dropout.** With probability `p=0.02` per step, the emitted estimate
   is the **last fresh sample** (frozen) and `staleness` increments. The
   OU **internal state continues to update** every step, so when dropout
   ends the resumed emit is smooth (not a jump). The recovery-no-jump
   property is regression-guarded by
   `test_dropout_recovery_no_discontinuity`.

All randomness goes through one `np.random.Generator` seeded in `reset()`
from `env.unwrapped._current_scenario.seed XOR cfg.bodyslam.seed`. Same
scenario seed → same noise trace, deterministic across reruns.

### Demo replay (`AMASSDemoPositionProvider`)

BC pretrain on raw BiGym demos has no live human in the scene, so
`info["safety"]` is absent. Two bad options:

- **Sentinel** (constant zero or "no human"): the policy learns the channel
  is a constant during pretraining. At deployment the channel suddenly
  carries a moving person — exactly the distribution shift we want to
  avoid on the channel we most want the policy to attend to.
- **Crash**: not viable.

**Resolution.** When `demo_env=True`, the factory inserts
`BodySLAMWrapper(demo_replay=True, position_provider=AMASSDemoPositionProvider(...))`.
The provider:

1. On `reset()`, samples a clip from `cfg.env.motion_clip_paths` and a
   random root transform (distance ∈ [1.5, 3.0] m, yaw ∈ [-π, π]).
2. On each step, returns the world-frame pelvis position from the clip's
   `root_translation` (already SMPL-Y-up → MuJoCo-Z-up converted by
   `AMASSLoader`).

The OU + latency + dropout pipeline still runs over this trajectory, so
the demo and live distributions on `human_pos_estimate` overlap.
Trajectory/disruption logic is intentionally *not* reproduced — there's
no robot to react to during demo replay; an AMASS playback in the world
frame is enough to keep the channel non-degenerate.

Occlusion in demo mode is forced off (`NoOcclusion`); the live
distribution will cover it. The recovery-no-jump and OU statistics tests
also cover demo replay.

---

## Files

**Created:**

- [safety_bigym/perception/__init__.py](../safety_bigym/perception/__init__.py)
- [safety_bigym/perception/bodyslam_wrapper.py](../safety_bigym/perception/bodyslam_wrapper.py)
- [safety_bigym/perception/demo_position_provider.py](../safety_bigym/perception/demo_position_provider.py)
- [tests/test_bodyslam_wrapper.py](../tests/test_bodyslam_wrapper.py) — 19 tests
- [cfgs/bodyslam/{off,oracle,noisy}.yaml](../cfgs/bodyslam/)
- [scripts/phase1_obs_ablation.py](../scripts/phase1_obs_ablation.py) — E1.1
- [scripts/phase1_noise_sweep.py](../scripts/phase1_noise_sweep.py) — E1.2
- [scripts/phase1_temporal_ablation.py](../scripts/phase1_temporal_ablation.py) — E1.3

**Modified:**

- [safety_bigym/envs/safety_bigym_factory.py](../safety_bigym/envs/safety_bigym_factory.py) —
  `_maybe_wrap_bodyslam` (train env), `_wrap_env` override (demo env),
  `_build_ray_occlusion`.
- [safety_bigym/envs/safety_env.py](../safety_bigym/envs/safety_env.py) —
  emits `info["safety"]["human_pos"]` at reset (was step-only). Required
  so the wrapper can initialise its OU state from the first observation
  rather than after the first step.
- [cfgs/safety_config.yaml](../cfgs/safety_config.yaml) — adds `bodyslam:
  "off"` to the defaults list.

**Deliberately not modified:**

- RoboBase. `ConcatDim` already folds the `(6,)` key into `low_dim_state`;
  no upstream change needed.
- `iso15066_wrapper.py`, `episode_metrics_wrapper.py`. The wrapper
  consumes their existing outputs.
- DP / ACT method configs. The new dims flow through `low_dim_state`,
  which both methods already consume.

---

## Usage

### Hydra

```bash
# Baseline (no perception layer)
python train_safety.py launch=dp_pixel_safety_bigym \
  env=safety_bigym/reach_target_single bodyslam=off

# Oracle (clean human_pos)
python train_safety.py launch=dp_pixel_safety_bigym \
  env=safety_bigym/reach_target_single bodyslam=oracle

# Realistic (OU + latency + dropout)
python train_safety.py launch=dp_pixel_safety_bigym \
  env=safety_bigym/reach_target_single bodyslam=noisy

# Override individual knobs (e.g. for E1.2 noise sweep)
python train_safety.py launch=dp_pixel_safety_bigym \
  env=safety_bigym/reach_target_single bodyslam=noisy \
  ++env.bodyslam.noise_std=0.10
```

ACT works the same way — swap `dp_pixel_safety_bigym` for
`act_pixel_safety_bigym`.

### From Python

```python
from safety_bigym.perception import BodySLAMWrapper, AMASSDemoPositionProvider

# Wrap any env that emits info["safety"]["human_pos"]
env = BodySLAMWrapper(env, mode="noisy", noise_std=0.05, latency_steps=3)

# For demo replay (no live info["safety"])
provider = AMASSDemoPositionProvider(
    clip_paths=["74/74_01_poses.npz"],
    motion_dir=os.environ["AMASS_DATA_DIR"],
)
env = BodySLAMWrapper(env, mode="noisy", demo_replay=True, position_provider=provider)
```

### Sweep scripts

Each script has four modes:

- `--train` prints training commands (one per cell).
- `--eval` prints eval commands once `SNAPSHOTS` is filled (one per cell ×
  disruption).
- `--run` executes every eval cell sequentially, captures per-cell metrics
  via `+eval_output_path=<tmp.json>`, dumps the aggregate JSON to the repo
  root, and prints a summary table.
- `--smoke` runs ≤100 train frames locally for a plumbing check.

```bash
# E1.1 — does the obs help? (3 tasks × 2 methods × 3 modes = 18 train cells)
python scripts/phase1_obs_ablation.py --train | tee phase1_e11_train.sh
# … train all cells on GPU, fill SNAPSHOTS with peak-by-W&B-curve checkpoints …
python scripts/phase1_obs_ablation.py --run     # 90 eval cells + summary table

# E1.2 — at what σ does it stop helping? (5 σ values × 5 disruptions)
python scripts/phase1_noise_sweep.py --method <m> --task <t> --train
python scripts/phase1_noise_sweep.py --method <m> --task <t> --run

# E1.3 — does temporal structure matter, or just marginal σ? (3 variants × 5 disruptions)
python scripts/phase1_temporal_ablation.py --method <m> --task <t> --train
python scripts/phase1_temporal_ablation.py --method <m> --task <t> --run
```

After training, pick **peak-by-W&B-curve** checkpoints (NOT the final
snapshot — DP over-fits past its eval-success peak on small pixel demo
sets, see [PHASE_0_HUMAN_FIX.md](PHASE_0_HUMAN_FIX.md)), fill the script's
`SNAPSHOTS` dict, then `--run`.

**Output of `--run` (E1.1):**

- `phase1_obs_ablation_results.json` — full per-cell `eval_metrics` keyed
  by `[method][task][mode][disruption]`.
- A summary table grouped by `(method, task)` with `ssm_viol`, `pfl_viol`,
  `episode_success` (averaged across the 5 disruption types), the
  off→mode SSM-rate reduction, and a **PASS** flag when reduction ≥ 20%.
- Per-method and overall criterion check
  (`PASS` / `FAIL — Phase 2/3 contingency triggers`).

E1.2 and E1.3 produce analogous JSON dumps and trend tables (E1.2 by σ,
E1.3 by variant with reduction relative to `iid`).

---

## Testing

Run the wrapper suite locally:

```bash
cd safety_bigym && pytest tests/test_bodyslam_wrapper.py -v
```

19 tests; one (`test_factory_inserts_wrapper_when_enabled`) is
AMASS-skipped when `AMASS_DATA_DIR` is unset.

Test layout:

| # | Test                                                | Stub or live |
|---|-----------------------------------------------------|--------------|
| 1 | `test_obs_space_extended`                           | stub         |
| 2 | `test_oracle_mode_returns_clean_pos`                | stub         |
| 3 | `test_off_mode_raises`                              | stub         |
| 4 | `test_noise_seeded_deterministic`                   | stub         |
| 5 | `test_ou_temporal_correlation`                      | stub         |
| 6 | `test_iid_baseline_for_comparison`                  | stub         |
| 7 | `test_noise_std_calibration`                        | stub         |
| 8 | `test_latency_buffer_lag`                           | stub         |
| 9 | `test_dropout_repeats_last_known_and_increments_staleness` | stub  |
| 10| `test_dropout_prob_zero_means_zero_staleness`       | stub         |
| 11| `test_reset_resets_state`                           | stub         |
| 12| `test_seeding_propagates_from_scenario`             | stub         |
| 13| `test_dropout_recovery_no_discontinuity`            | stub         |
| 14| `test_confidence_derivation`                        | stub         |
| 15| `test_demo_replay_mode_drives_from_amass`           | stub         |
| 16| `test_long_episode_no_drift`                        | stub         |
| 17| `test_factory_inserts_wrapper_when_enabled`         | live (AMASS) |
| 18| `test_occlusion_flag_set_when_geom_blocks`          | minimal MJCF |
| 19| `test_noisy_mode_never_nans_or_infs`                | stub         |

Note on test 13: the regression guard for the OU-keeps-updating-during-dropout
design choice. If someone "fixes" it later by freezing OU during dropout,
this test catches the resulting recovery jump.

---

## Smoke verification (local, pre-handoff)

What was checked on Mac before the GPU sweep is queued:

1. **Hydra composition** — all three presets compose cleanly into
   `cfg.env.bodyslam.{mode,…}`.
2. **Factory env construction + 30-step rollout** — `mode={off, oracle,
   noisy}` × `reach_target_single`. Obs shape is `(6,)` for oracle/noisy,
   absent for off; values finite; OU + dropout pipeline visibly active.
3. **Demo loading into replay buffer** — all three modes load 79 demo
   timesteps without `low_dim_state` shape errors. The
   AMASSDemoPositionProvider activates on the demo path
   (`Inserting BodySLAMWrapper(mode=oracle, demo_replay=True) around DemoEnv`).
4. **Pretrain forward pass** reaches the network forward, then trips the
   Mac-MPS `float64` limitation in DiffusionPolicy. **This is unrelated
   to Phase-1**; it's a known Mac-only DP issue and the GPU box doesn't
   hit it.

The wrapper integration is verified end-to-end. Everything beyond the
forward pass is GPU territory.

---

## Known limitations / deferred

- **Live MuJoCo ray-cast occlusion is opt-in.** Default for
  `bodyslam=noisy` is `use_occlusion: false`. The lookup of `head` camera
  + `Pelvis_col` geom may fail on environments with renamed assets. When
  it fails the factory logs a warning and falls back to `NoOcclusion`.
  If the GPU runs need occlusion, set
  `++env.bodyslam.use_occlusion=true` and confirm the geom names match.
- **Demo-replay occlusion is hard-coded off.** The plan accepts this as
  "the right side to err on" — live coverage will dominate the obs
  distribution in deployment.
- **Real BodySLAM++ on rendered MuJoCo frames is out of scope.** That's
  Phase 5 (E5.3).
- **Filter-side use of `human_pos_estimate` is out of scope.** Phase 2's
  Safety Value Function will consume this channel.
- **Asymmetric actor/critic obs is out of scope.** Phase 3's constrained
  RL may use it.

---

## Hand-off checklist

Before kicking off the GPU sweep:

- [ ] Branch `safety-critic/phase-1-bodyslam-wrapper` pushed.
- [ ] Robobase drift patch (`phase0_workspace_drift.patch`) applied on the
      GPU clone if not already done.
- [ ] `pytest tests/` green locally (1 pre-existing failure on
      `test_no_episode_safety_until_done` is unrelated and predates
      Phase 1).
- [ ] `AMASS_DATA_DIR` exported on the GPU box.
- [ ] `phase1_obs_ablation.py --train` output reviewed — 18 commands.

After the sweep:

- [ ] Pick peak-by-W&B-curve snapshots per (method, task, mode); update
      `SNAPSHOTS` in `phase1_obs_ablation.py`.
- [ ] Run `--eval` to produce the Phase-1 results table (each cell × 5
      disruption types).
- [ ] Pick the strongest (method, task) pair; run `phase1_noise_sweep.py`
      and `phase1_temporal_ablation.py` against it.
- [ ] Author `docs/phase1_observation_results.md` with the success-criterion
      check: **oracle ≥ 20% reduction in SSM violation rate vs baseline,
      on at least one of DP / ACT.** If neither hits the bar, the master
      plan's contingency triggers — escalate Phase 2/3 priority and
      capture that decision in the doc.
