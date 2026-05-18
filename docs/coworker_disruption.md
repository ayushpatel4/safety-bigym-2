# COWORKER disruption type

## What it is

`DisruptionType.COWORKER` models a **sustained co-working scenario** rather
than a one-shot intrusion. The human enters the robot's workspace, stays
nearby for the whole episode, and periodically reaches toward the robot's
end-effector or the task object — then pulls the arm back to its side and
waits before reaching again. This is the deployment pattern the safety
filter will face most often, and it's the dominant failure mode the
hybrid safety critic (`.claude/HYBRID_SAFETY_CRITIC_PLAN.md` Phase 2/3)
needs robust training coverage on.

Contrast with the six other disruption types — `INCIDENTAL`,
`SHARED_GOAL`, `DIRECT`, `OBSTRUCTION`, `RANDOM_PERTURBED`, `CONTACT` —
which all encode a single intrusion event per episode.

## Three spawn / trajectory modes

For COWORKER scenarios the sampler picks uniformly among three trajectory
types ([scenario_sampler.py:_select_trajectory_type](../safety_bigym/scenarios/scenario_sampler.py)):

| Trajectory | What the human does | When the arm reaches |
| --- | --- | --- |
| `STATIONARY` | Spawns already at NEAR distance, stays put. | Whole episode. |
| `APPROACH_LOITER_DEPART` | Spawns far, walks in to NEAR, stays put. | After arrival. |
| `COWORKER_PATROL` | Walks in to NEAR, then on a cycle: depart to an AWAY position (90°–270° offset from NEAR, ~2–3 m), loiter, return to a *new* NEAR (different angle, slightly different distance), repeat. | Only while at NEAR — the reach gate suppresses motion when the shoulder is too far from the target. |

Both NEAR and AWAY phases use the trajectory phase string `"loiter"` so
`HumanController.step` blends body qpos toward the IK callback's output.
The reach gate handles the AWAY case by returning the rest pose, so the
human stands at AWAY with arms hanging instead of waving toward the
robot.

## Arm state machine

The arm cycle runs on a fixed-period state machine independent of the
trajectory phase ([coworker_behavior.py:CoworkerArmController.compute_qpos](../safety_bigym/scenarios/coworker_behavior.py)):

```
[0,                                  t_extend)        -> EXTEND   (blend rest -> IK target)
[t_extend,                  t_extend + t_hold)        -> HOLD     (arm at IK target)
[..,        t_extend + t_hold + t_retract)            -> RETRACT  (blend IK -> rest)
[..,                                          T)       -> IDLE     (rest pose)
```

Default fractions (out of period `T`): `extend=0.15`, `hold=0.20`,
`retract=0.15`, `idle=0.50`. Sampler picks `T ∈ [4.5, 6.5] s`, so each
cycle has ~2.5 s of idle between reaches — clear gap, not constant
movement.

At cycle start the controller samples whether to reach for the robot EE
or the task object via `coworker_target_mix` (defaults to `(0.5, 0.5)`).
The chosen target is re-resolved each step within the cycle so it tracks
moving robots / task objects during HOLD.

## Reach gate

`CoworkerArmController._max_reach_dist = 0.75 m`. Each `compute_qpos`
call measures the active shoulder's distance to the current target; if
it exceeds the gate, the controller returns the cached rest qpos instead
of solving IK. This is what makes the AWAY phase of `COWORKER_PATROL`
look natural — the human walks off, the state machine still cycles
in the background, but the arm stays at the side.

## Arms-down rest pose

The rest pose is built once at controller init by IK-solving each arm
toward a point 0.7 m below the corresponding shoulder
([`_build_arms_down_rest_pose`](../safety_bigym/scenarios/coworker_behavior.py)).
Verified geometry: wrist is 0.53 m directly below the shoulder with
0.00 m horizontal offset — both arms hang straight down. This dictates
the *direction* the wrist travels during RETRACT: from "in front of
body" downward to the side, never laterally through the robot's
workspace.

**Critical implementation detail.** `HumanIK.solve` copies `data.qpos`
into its working data but **not** `data.mocap_pos` / `mocap_quat`. Since
SMPL-H Pelvis is a mocap body, `compute_qpos` and
`_build_arms_down_rest_pose` both sync `mocap_pos` / `mocap_quat`
manually before each solve. Without that sync the IK works as if the
Pelvis were at the world origin, and the arm "reaches" toward the wrong
absolute coordinates regardless of where the human is actually standing.

## Patrol NEAR distance sampling

Each NEAR visit (the initial walk-in and every return after an AWAY
excursion) samples its distance to the robot as
`closest_approach + N(0, patrol_near_distance_std)` clipped to ±0.25 m.
With the scenario's `closest_approach` mean of ~0.99 m and a stdev of
0.12 m, NEAR distances across one episode look like:

```
visit 1: 1.003 m       visit 2: 0.924 m
visit 3: 1.101 m       visit 4: 0.993 m
```

So the human stands at a *slightly* different proximity each time it
returns — not a snap to exactly the same spot.

## Continuous knobs (train vs. eval distribution)

Five COWORKER axes are exposed as continuous `ParameterSpace` ranges,
so train and eval distributions can be configured independently for
generalisation experiments:

| Knob | ParameterSpace field | Train (moderate) | Eval (wider) |
| --- | --- | --- | --- |
| Closest-approach distance | `coworker_closest_approach_range` | `0.9 – 1.4 m` | `0.6 – 1.8 m` |
| Reach period *(= 1 / frequency)* | `coworker_reach_period_range` | `4.5 – 6.5 s` | `3.0 – 9.0 s` |
| P(reach EE) *(vs task obj)* | `coworker_target_mix_p_ee_range` | `0.4 – 0.6` | `0.1 – 0.9` |
| Dwell time at NEAR | `coworker_near_loiter_range` | `7 – 11 s` | `4 – 16 s` |
| Walk speed | `coworker_walk_speed_range` | `1.0 – 1.6 m/s` | `0.6 – 2.2 m/s` |

The eval range is a strict superset of the train range on every axis,
so eval rollouts probe both in-distribution and out-of-distribution
conditions. The test suite (`test_coworker_eval_is_strict_superset_of_train`,
`test_eval_space_samples_outside_train_ranges`) enforces this invariant
and verifies ≥10% of eval samples per axis land OOD.

### Python API

```python
from safety_bigym.scenarios import (
    ScenarioSampler,
    make_coworker_train_space,
    make_coworker_eval_space,
)

# Training run — sampler stays in the moderate band.
train_sampler = ScenarioSampler(
    parameter_space=make_coworker_train_space(clip_paths=[...]),
    motion_dir=...,
)

# Held-out eval — same code, wider ranges.
eval_sampler = ScenarioSampler(
    parameter_space=make_coworker_eval_space(clip_paths=[...]),
    motion_dir=...,
)
```

Both factories force `disruption_weights = {COWORKER: 1.0}` so other
disruption types don't dilute the experiment. Pass any override as a
keyword: `make_coworker_train_space(coworker_walk_speed_range=(0.5, 2.0))`.

### Hydra / YAML API

Two presets ship under `cfgs/disruption/` (registered as a Hydra group
in `cfgs/safety_config.yaml`):
[`coworker_train.yaml`](../cfgs/disruption/coworker_train.yaml),
[`coworker_eval.yaml`](../cfgs/disruption/coworker_eval.yaml). Activate
the group with `disruption=coworker_train` on the CLI, or override
per-axis:

```
python train_safety.py \\
    env.disruptions.coworker_closest_approach_range='[0.9, 1.4]' \\
    env.disruptions.coworker_reach_period_range='[4.5, 6.5]' \\
    env.disruptions.coworker_target_mix_p_ee_range='[0.4, 0.6]' \\
    env.disruptions.coworker_near_loiter_range='[7.0, 11.0]' \\
    env.disruptions.coworker_walk_speed_range='[1.0, 1.6]'
```

The factory ([safety_bigym_factory.py](../safety_bigym/envs/safety_bigym_factory.py))
reads each `coworker_*_range` from `cfg.env.disruptions` and forwards
it to `ParameterSpace`. Unspecified axes fall back to the
ParameterSpace defaults (which match the train preset).

### Other (fixed-for-now) geometry knobs

| Knob | Range | Notes |
| --- | --- | --- |
| `spawn_distance` | `closest_approach + 1.5 – 2.5 m` | Forced per-episode so the walk-in is visibly a walk. |
| `patrol_away_distance` | `2.0 – 3.0 m` | How far the AWAY position sits from the robot. |
| `patrol_excursions` | `1 or 2` | Number of depart→return cycles inside an episode. |
| `patrol_away_loiter` | `3 – 5 s` | Time at AWAY per visit. |
| `approach_angle` | `0 – 360°` | Walk-in / spawn-in-place direction. |
| `coworker_active_arm` | `left_arm` / `right_arm` | Sampler picks per episode. |

## Visual verification

```
export AMASS_DATA_DIR=/Users/ayushpatel/Documents/FYP3/CMU/CMU
cd safety_bigym

# Live viewer, single trajectory mode
mjpython scripts/demo_coworker.py --spawn walk_in --reach-target alternate
mjpython scripts/demo_coworker.py --spawn in_place --arm left
mjpython scripts/demo_coworker.py --spawn patrol --reach-target task

# Mix of all three (default --spawn alternate)
mjpython scripts/demo_coworker.py --task dishwasher_close

# Record MP4s across a task × spawn-mode grid
python scripts/record_coworker_videos.py \\
    --tasks reach saucepan dishwasher_close drawers_open_all \\
    --spawns walk_in in_place patrol \\
    --out-dir vids/coworker \\
    --sim-seconds 35
```

`demo_coworker.py` logs each phase transition to stdout:

```
[t= 5.20s] traj=loiter     reach_phase=hold     target_kind=task_object in_reach     pos=(0.51, 0.04, 1.02)
[t=12.30s] traj=depart     reach_phase=idle     target_kind=ee          in_reach     pos=...
[t=14.85s] traj=loiter     reach_phase=extend   target_kind=ee          OUT-OF-REACH pos=...
```

Task list (run `--list-tasks` to discover all 18 registered keys, or
pass any `module.path:ClassName` directly):

| key | class |
| --- | --- |
| `reach` | `bigym.envs.reach_target:ReachTargetSingle` |
| `saucepan` | `bigym.envs.pick_and_place:SaucepanToHob` |
| `dishwasher_close` | `bigym.envs.dishwasher:DishwasherClose` |
| `drawers_open_all` | `bigym.envs.cupboards:DrawersAllOpen` |
| `flip_cup` | `bigym.envs.manipulation:FlipCup` |
| `pick_box` | `bigym.envs.pick_and_place:PickBox` |
| ... | (see `TASK_MAP` in `scripts/demo_coworker.py`) |

## Task-object detection

`SafetyBiGymEnv._get_robot_state` populates `task_object_pos` from
whatever the loaded BiGym task exposes ([safety_env.py:_lookup_task_object_pos](../safety_bigym/envs/safety_env.py)):

- Reach tasks: `self.targets[0].body.get_position()`
- Manipulables: `self.box`, `self.saucepan`, `self.cup`, `self.cube`, `self.plate`, `self.mug`, `self.kettle`, `self.pan`
- List manipulables: `self.cups[0]`, `self.plates[0]`, `self.cubes[0]`, `self.boxes[0]`, `self.mugs[0]`, `self.cutlery[0]`, `self.items[0]`
- Scene props: `self.dishwasher`, `self.cabinet_*`, `self.shelf`, `self.tray`

If none of these match, the COWORKER callback falls back to always
reaching for the EE.

## Implementation files

| Concern | File |
| --- | --- |
| Disruption enum + per-type config | [`safety_bigym/scenarios/disruption_types.py`](../safety_bigym/scenarios/disruption_types.py) |
| Arm state machine + rest pose + reach gate | [`safety_bigym/scenarios/coworker_behavior.py`](../safety_bigym/scenarios/coworker_behavior.py) |
| Scenario sampler (weights, trajectory choice, parameter ranges) | [`safety_bigym/scenarios/scenario_sampler.py`](../safety_bigym/scenarios/scenario_sampler.py) |
| Trajectory builders (`STATIONARY`, `COWORKER_PATROL`) | [`safety_bigym/human/trajectory_planner.py`](../safety_bigym/human/trajectory_planner.py) |
| Env wiring (IK callback, task-object detection) | [`safety_bigym/envs/safety_env.py`](../safety_bigym/envs/safety_env.py) |
| Time injection into IK callback | [`safety_bigym/human/human_controller.py`](../safety_bigym/human/human_controller.py) `_get_ik_targets` |
| Visual demo | [`safety_bigym/scripts/demo_coworker.py`](../scripts/demo_coworker.py) |
| Video recorder | [`safety_bigym/scripts/record_coworker_videos.py`](../scripts/record_coworker_videos.py) |
| Tests | [`safety_bigym/tests/test_coworker_disruption.py`](../tests/test_coworker_disruption.py) |
