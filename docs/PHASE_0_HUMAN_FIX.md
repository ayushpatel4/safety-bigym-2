# Phase 0 — Human Collision & SSM Velocity Fix

Branch: `safety-critic/phase-0-human-fix` (off `main`).

Fixes for two defects that made `safety_bigym` unusable for diffusion-policy
training on `DishwasherClose`: (1) the injected SMPL-H human physically
penetrated task geometry and crashed the simulator ~1 s into every eval;
(2) ISO 15066 SSM reported nonsense violations driven by a phantom
human velocity of ~120 m/s.

## 1. The 1-second truncation bug

### Symptom
DP eval videos were only ~1 s long. Root-causing via `scripts/diagnose_truncation.py`
(new) showed 10 % of 30 zero-action episodes truncated inside 50 control steps
at the training 25 Hz control rate (20 physics sub-steps per `env.step`).
`scripts/diagnose_contacts.py` (new) showed hundreds of human↔dishwasher
contacts per episode with `max_penetration ≈ 0.99 m` and `max_force ≈ 2.3 e21 N`
— NaN in `QACC`, `PhysicsError`, `EnvHealth.is_healthy = False`, BiGym's
`truncate` flag flips, episode ends.

### Cause
The SMPL-H human and the scene (dishwasher, cabinets, walls) were on the same
MuJoCo collision channel (`contype=1, conaffinity=1`), and the human's PD
actuators were 10× too stiff (`kp=2000 kv=50` in the body XML vs. the
`HumanConfig` default of `200/20`). Under a 20-substep physics budget, the
actuators drove SMPL body parts straight through the dishwasher.

### Fix (2 surgical edits)

- **[`safety_bigym/assets/smplh_human_body.xml`](../safety_bigym/assets/smplh_human_body.xml)**
  - `human_collision` default now uses `contype=2 conaffinity=2` — the human
    sits on collision **bit 1 only**; default scene geoms (bit 0) no longer see it.
  - `position_actuator` gains reduced to `kp=200 kv=20` to match the
    `HumanConfig` contract.

- **[`safety_bigym/envs/safety_env.py`](../safety_bigym/safety_bigym/envs/safety_env.py)**
  — new `_configure_collision_bits()` called after `super().__init__` when a
  human is injected. It OR's bit 1 into `model.geom_contype` / `geom_conaffinity`
  for every robot collision geom (via `_is_robot_geom`) and the floor. Scene
  geoms stay on bit 0 only.
  - Collision matrix after the fix:
    - human ↔ scene: **disabled** (no bit overlap) → human passes through
    - human ↔ robot: **enabled** (both carry bit 1) → PFL still sees contacts
    - human ↔ floor: **enabled** (floor promoted to bit 1) → human still stands
    - robot ↔ scene: **enabled** (both carry bit 0) → unchanged

Safety semantics preserved: SSM is geometric distance between body centers
([`compute_ssm`](../safety_bigym/safety/iso15066_wrapper.py#L379)) — independent
of contact bits. PFL still sees human↔robot contact forces because that pair
shares bit 1.

### Verification
After the fix, re-running the Phase A scripts shows:
- 0 % truncation over 30 zero-action episodes, 0 physics errors.
- 0 human↔scene contacts in `diagnose_contacts.py`.
- Median `end_step = 150/150`.

## 2. The SSM 18.35 m phantom-violation bug

### Symptom
```
SSM Violation! Distance: 3.06m, Required: 18.35m, Margin: -15.289m
```
The "required" separation distance was two orders of magnitude too large —
ISO 15066 stopping distances for a 1.6 m/s walking human are ~0.3–1.5 m.

### Cause
`SafetyBiGymEnv._human_ssm_state` read human linear velocity from
`data.cvel[bid, 3:6]`. But `HumanController.step` ([line 328](../safety_bigym/human/human_controller.py#L328))
teleports `data.qpos` directly every sub-step to play back the AMASS clip. MuJoCo
computes an implicit velocity `(qpos_new − qpos_old) / PHYSICS_DT` from those
teleports — a 2 cm frame hop at `dt = 0.002 s` becomes 10 m/s; at the extremes,
~120 m/s. Plugging that into `S_h = v_h · (T_r + T_s) = 120 · 0.15 ≈ 18 m`
reproduced the bogus number exactly.

The violation math was correct; the velocity it was given was not.

### Fix (1 line)
[`safety_env.py` `_human_ssm_state`](../safety_bigym/envs/safety_env.py#L647-L651):
cap `max_vel` at `SSMConfig.v_h_max` (1.6 m/s). This is the ISO 15066-prescribed
conservative bound — the standard assumes a bounded walking human, not the
instantaneous velocity of a motion-capture teleport.

```python
max_vel = min(max_vel, float(self.safety_config.ssm.v_h_max))
```

### Verification
- Required separation: 18.35 m → 0.34 m at the same sim state.
- Margin: −15.3 m → −0.17 m when the human is 0.17 m from the robot — a real,
  physically meaningful violation (the human is genuinely inside the safe
  stopping distance of the H1 arm).

## Diagnostic scripts (new)

All under [`safety_bigym/scripts/`](../scripts/); all require `AMASS_DATA_DIR`.

| Script | Purpose |
|--|--|
| `diagnose_truncation.py` | N×M zero/random/small-random rollouts; captures per-episode end step, `UnstableSimulationWarning` count, scenario params. |
| `diagnose_contacts.py` | Enumerates every `data.contact` each sub-step; classifies into `{human↔robot, human↔scene, robot↔scene}`; reports top offending pairs + max penetration/force. |
| `diagnose_no_human.py` | `diagnose_truncation.py` with `inject_human=False` — baseline reference. |
| `diagnose_spawn_geometry.py` | `mj_forward`-only check for AABB overlap between human and scene at reset (frame 0). |

## Regression tests (new)

- [`tests/test_collision_groups.py`](../tests/test_collision_groups.py) — 5
  tests that lock the MuJoCo collision-bit invariant:
  - no `(human, scene)` pair is collision-enabled
  - every `(human, robot)` pair is collision-enabled
  - every `(human, floor)` pair is collision-enabled
  - human geoms carry `contype = conaffinity = 2` exactly

- [`tests/test_safety_preserved.py`](../tests/test_safety_preserved.py) — 4
  tests that force a head-on approach via a monkey-patched `ScenarioParams`:
  - `min_separation` drops ≥ 0.2 m during approach
  - `ssm_margin` stays finite and drops below 0 (violation fires)
  - `pfl_force_ratio` stays finite and non-negative
  - episode does not truncate early (≥ 100 of 150 control steps)

All 9 pass. Pre-existing failure in `test_no_episode_safety_until_done` is
unrelated.

## Files changed

```
safety_bigym/assets/smplh_human_body.xml     | contype/conaffinity, PD gains
safety_bigym/envs/safety_env.py              | _configure_collision_bits +
                                             | human-velocity cap in _human_ssm_state
scripts/diagnose_truncation.py               | new
scripts/diagnose_contacts.py                 | new
scripts/diagnose_no_human.py                 | new
scripts/diagnose_spawn_geometry.py           | new
tests/test_collision_groups.py               | new
tests/test_safety_preserved.py               | new
```

## How to verify end-to-end

```bash
cd safety_bigym
export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU

# Quantitative
./venv/bin/python scripts/diagnose_truncation.py --episodes 30 --max-steps 150 \
  --out experiments/diagnose/truncation_post_fix.csv
./venv/bin/python scripts/diagnose_contacts.py --episodes 10 --max-steps 60 \
  --out experiments/diagnose/contacts_post_fix.csv
./venv/bin/python -m pytest tests/test_collision_groups.py tests/test_safety_preserved.py

# Visual
mjpython scripts/demo_safety_env.py
```

The real gate is a DP smoke run on the GPU box:

```bash
python train_safety.py launch=dp_pixel_safety_bigym \
  env=safety_bigym/dishwasher_close \
  num_train_frames=100 eval_every_steps=50 num_eval_episodes=10 \
  wandb.use=true wandb.name=phase0-human-fix-smoke
```

Target: `eval/episode_length` mean over 10 eps ≥ 100 steps (was ~25 before).
