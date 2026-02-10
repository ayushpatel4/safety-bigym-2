# Safety BiGym — User Guide

## Overview

**Safety BiGym** wraps [BiGym](https://github.com/chernyadev/bigym) environments with ISO 15066 safety monitoring. It adds:

- A **simulated human** driven by AMASS motion capture clips
- **Speed & Separation Monitoring (SSM)** — tracks robot-human distance vs required protective distance
- **Power & Force Limiting (PFL)** — monitors contact forces against body-region limits from ISO 15066
- **Parameterised scenario sampling** — diverse human disruption behaviours for evaluation

---

## 1. Quick Start

### Installation

```bash
cd safety_bigym
pip install -e .
```

### Minimal Example

```python
from bigym.action_modes import JointPositionActionMode
from safety_bigym import make_safety_env, SafetyConfig, HumanConfig

env = make_safety_env(
    task_cls=__import__('bigym.bigym_env', fromlist=['BiGymEnv']).BiGymEnv,
    action_mode=JointPositionActionMode(floating_base=True, absolute=True),
    safety_config=SafetyConfig(),
    human_config=HumanConfig(
        motion_clip_dir="/path/to/CMU/CMU",
        motion_clip_paths=["74/74_01_poses.npz"],
    ),
)

obs, info = env.reset()
for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    safety = info["safety"]
    if safety["ssm_violation"]:
        print(f"SSM violation! Separation: {safety['min_separation']:.2f}m")
    if safety["pfl_violation"]:
        print(f"PFL violation! Force: {safety['max_contact_force']:.1f}N")
env.close()
```

---

## 2. Wrapping a BiGym Task

Use `make_safety_env()` to combine any BiGym task class with safety monitoring:

```python
from bigym.envs.reach_target import ReachTargetSingle
from bigym.envs.pick_and_place import PickBox
from bigym.envs.dishwasher import DishwasherOpen
from bigym.action_modes import JointPositionActionMode
from safety_bigym import make_safety_env, SafetyConfig, HumanConfig

action_mode = JointPositionActionMode(floating_base=True, absolute=True)
human_config = HumanConfig(
    motion_clip_dir="/path/to/CMU/CMU",
    motion_clip_paths=["74/74_01_poses.npz"],
)

# Reach task with safety
env = make_safety_env(ReachTargetSingle, action_mode=action_mode, human_config=human_config)

# Pick & place with safety
env = make_safety_env(PickBox, action_mode=action_mode, human_config=human_config)

# Dishwasher with safety
env = make_safety_env(DishwasherOpen, action_mode=action_mode, human_config=human_config)
```

### How it works

`make_safety_env()` dynamically creates a class inheriting from both `SafetyBiGymEnv` and the task class:

```
SafetyReachTargetSingle (dynamic)
  ├── SafetyBiGymEnv  →  step(), reset(), safety monitoring
  └── ReachTargetSingle  →  scene setup, reward function
```

### Available tasks

| Key             | Task Class            | Description                    |
|-----------------|-----------------------|--------------------------------|
| `default`       | `BiGymEnv`            | Empty scene                    |
| `reach`         | `ReachTargetSingle`   | Reach a target with one arm    |
| `reach_dual`    | `ReachTargetDual`     | Reach targets with both arms   |
| `pick_box`      | `PickBox`             | Pick up a box                  |
| `saucepan`      | `SaucepanToHob`       | Move saucepan to hob           |
| `flip_cup`      | `FlipCup`             | Flip a cup                     |
| `stack_blocks`  | `StackBlocks`         | Stack blocks                   |
| `dishwasher_open` | `DishwasherOpen`    | Open dishwasher door           |
| `dishwasher_close`| `DishwasherClose`   | Close dishwasher door          |
| `cupboard_open` | `CupboardsOpenAll`    | Open all cupboards             |
| `drawer_open`   | `DrawerTopOpen`       | Open top drawer                |
| `move_plate`    | `MovePlate`           | Move a plate                   |
| `groceries`     | `GroceriesStoreLower` | Store groceries                |
| `take_cups`     | `TakeCups`            | Take cups from shelf           |
| `put_cups`      | `PutCups`             | Put cups on shelf              |

---

## 3. Configuring Safety Monitoring

### SafetyConfig

Controls how violations are detected and handled:

```python
from safety_bigym import SafetyConfig, SSMConfig

safety_config = SafetyConfig(
    # SSM parameters (ISO 15066)
    ssm=SSMConfig(
        T_r=0.2,       # Robot reaction time (s)
        T_s=0.1,       # System response time (s)
        a_max=5.0,     # Max braking deceleration (m/s²)
        C=0.1,         # Intrusion distance / uncertainty (m)
        v_h_max=1.6,   # Max assumed human velocity (m/s)
    ),
    
    # PFL settings
    use_pfl=True,                    # Enable force monitoring
    
    # Violation behaviour
    terminate_on_violation=False,    # End episode on violation
    add_violation_penalty=True,      # Add penalty to reward
    violation_penalty=-1.0,          # Penalty magnitude
    
    # Logging
    log_violations=True,             # Log violations to console
    log_all_contacts=False,          # Log all contacts (verbose)
)
```

### SSM formula

The protective separation distance S_p is computed as:

```
S_p = S_h + S_r + C

where:
  S_h = v_human × (T_r + T_s)           # Human contribution
  S_r = v_robot × T_r + v_robot² / 2a   # Robot stopping distance
  C   = intrusion distance               # Safety margin
```

A **violation** occurs when the actual robot-human distance < S_p.

### PFL body region limits

Contact forces are checked against ISO 15066 Table A.2 limits per body region:

| Region      | Quasi-static limit (N) | Transient limit (N) |
|-------------|------------------------|---------------------|
| Head/skull  | 130                    | 130                 |
| Face        | 65                     | 65                  |
| Chest       | 140                    | 140                 |
| Pelvis      | 210                    | 210                 |
| Upper arm   | 150                    | 150                 |
| Forearm     | 160                    | 160                 |
| Hand/finger | 140                    | 140                 |
| Thigh       | 220                    | 220                 |
| Lower leg   | 210                    | 210                 |

---

## 4. Reading Safety Info from `info["safety"]`

Every `env.step()` returns safety data in the info dict:

```python
obs, reward, terminated, truncated, info = env.step(action)
safety = info["safety"]
```

### Available fields

| Field                | Type    | Description                                              |
|----------------------|---------|----------------------------------------------------------|
| `ssm_violation`      | `bool`  | True if distance < required separation                   |
| `pfl_violation`      | `bool`  | True if contact force exceeds body-region limit          |
| `ssm_margin`         | `float` | Distance − S_p (negative = violation)                    |
| `pfl_force_ratio`    | `float` | max(F_actual / F_limit) across regions (>1 = violation)  |
| `min_separation`     | `float` | Actual robot-human distance (meters)                     |
| `max_contact_force`  | `float` | Peak contact force this step (Newtons)                   |
| `contact_region`     | `str`   | Body region of peak force contact                        |
| `contact_type`       | `str`   | `"quasi_static"` or `"transient"`                        |
| `violations_by_region` | `dict`| Count of violations per body region                      |

### Example: reward shaping with safety signals

```python
def safe_reward(obs, reward, info):
    """Shape reward with continuous safety signals."""
    safety = info["safety"]
    
    # Proportional SSM penalty (stronger when closer)
    if safety["ssm_margin"] < 0:
        reward += 0.5 * safety["ssm_margin"]  # Negative margin → penalty
    
    # Force penalty (proportional to how much limit is exceeded)
    if safety["pfl_force_ratio"] > 1.0:
        reward -= 2.0 * (safety["pfl_force_ratio"] - 1.0)
    
    return reward
```

### Example: episode-level safety statistics

```python
episode_ssm_violations = 0
episode_pfl_violations = 0
episode_min_separation = float('inf')

obs, info = env.reset()
done = False
while not done:
    obs, reward, term, trunc, info = env.step(policy(obs))
    safety = info["safety"]
    
    episode_ssm_violations += int(safety["ssm_violation"])
    episode_pfl_violations += int(safety["pfl_violation"])
    episode_min_separation = min(episode_min_separation, safety["min_separation"])
    done = term or trunc

print(f"SSM violations: {episode_ssm_violations}")
print(f"PFL violations: {episode_pfl_violations}")
print(f"Min separation: {episode_min_separation:.2f}m")
```

---

## 5. Parameterised Scenario Sampling

Each `env.reset()` automatically samples a new human behaviour scenario.

### Disruption types

The system defines 5 types of human disruption:

| Type               | Weight | IK?  | Description                                        |
|--------------------|--------|------|----------------------------------------------------|
| `INCIDENTAL`       | 30%    | No   | AMASS motion clips that cross robot workspace      |
| `SHARED_GOAL`      | 20%    | Yes  | Human reaches toward object near task goal          |
| `DIRECT`           | 20%    | Yes  | Human reaches toward robot end-effector             |
| `OBSTRUCTION`      | 15%    | Yes  | Human moves into robot's path and holds position   |
| `RANDOM_PERTURBED` | 15%    | No   | AMASS motion with Gaussian noise on trajectory     |

### Sampled parameters per scenario

| Parameter            | Range          | Description                           |
|----------------------|----------------|---------------------------------------|
| `clip_path`          | from clip dir  | AMASS motion capture clip             |
| `disruption_type`    | weighted       | Type of human disruption              |
| `trigger_time`       | 0.5–5.0s       | When disruption starts                |
| `blend_duration`     | 0.2–0.6s       | AMASS → IK transition time            |
| `speed_multiplier`   | 0.5–2.0×       | Motion playback speed                 |
| `height_percentile`  | 0.05–0.95      | Human anthropometry variation         |
| `approach_angle`     | 0–360°         | Direction human approaches from       |
| `spawn_distance`     | 1.0–2.0m       | Distance from robot at spawn          |
| `reaching_arm`       | left/right     | Based on approach angle               |

### Directional spawning

The human is automatically:
1. **Positioned** on a circle around the robot at `(distance × cos(angle), distance × sin(angle))`
2. **Oriented** to face toward the robot
3. **Motion-rotated** — the AMASS clip's natural forward direction is rotated to point at the robot

This ensures the human always approaches the robot regardless of which clip is selected.

### Using ScenarioSampler directly

```python
from safety_bigym import ScenarioSampler, ParameterSpace
from pathlib import Path

# Auto-discover motion clips
sampler = ScenarioSampler(motion_dir=Path("/path/to/CMU/CMU"))

# Sample a single scenario (reproducible with seed)
scenario = sampler.sample_scenario(seed=42)
print(f"Type: {scenario.disruption_type.name}")
print(f"Speed: {scenario.speed_multiplier:.1f}x")
print(f"Angle: {scenario.approach_angle:.0f}°")

# Sample a batch for evaluation
batch = sampler.sample_batch(n=100, base_seed=0)

# Stratified sampling (equal coverage per disruption type)
stratified = sampler.get_stratified_sample(n_per_type=20, base_seed=0)
for dtype, scenarios in stratified.items():
    print(f"{dtype.name}: {len(scenarios)} scenarios")
```

### Custom parameter space

```python
from safety_bigym import ParameterSpace, DisruptionType

# Only close-range, fast, direct interactions
custom_space = ParameterSpace(
    disruption_weights={
        DisruptionType.DIRECT: 0.5,
        DisruptionType.OBSTRUCTION: 0.3,
        DisruptionType.INCIDENTAL: 0.2,
        DisruptionType.SHARED_GOAL: 0.0,
        DisruptionType.RANDOM_PERTURBED: 0.0,
    },
    trigger_time_range=(0.5, 1.5),
    speed_range=(1.5, 2.0),
    spawn_distance_range=(0.8, 1.5),
)

sampler = ScenarioSampler(
    parameter_space=custom_space,
    motion_dir=Path("/path/to/CMU/CMU"),
)
```

### Scenario info in reset output

```python
obs, info = env.reset(seed=42)
print(info["scenario"])
# {'disruption_type': 'DIRECT', 'trigger_time': 2.3, 'clip_path': '74_01_poses.npz'}
```

---

## 6. Running the Demos

### Safety environment demo (with viewer)

```bash
# Default task
mjpython scripts/demo_safety_env.py

# Specific task
mjpython scripts/demo_safety_env.py --task reach
mjpython scripts/demo_safety_env.py --task pick_box
mjpython scripts/demo_safety_env.py --task dishwasher_open

# List all tasks
mjpython scripts/demo_safety_env.py --help
```

### Scenario sampling demo

```bash
# Print sampling statistics (no viewer needed)
python scripts/demo_scenario_sampling.py

# Run a specific scenario visually
mjpython scripts/demo_scenario_sampling.py --run --seed 42

# Cycle through all 5 disruption types
mjpython scripts/demo_scenario_sampling.py --run --stratified
```

---

## 7. Architecture Summary

```
safety_bigym/
├── envs/
│   └── safety_env.py          # SafetyBiGymEnv + make_safety_env()
├── safety/
│   ├── iso15066_wrapper.py    # ISO15066Wrapper (SSM + PFL)
│   └── pfl_limits.py          # Body-region force limits from ISO 15066
├── human/
│   └── human_controller.py    # HumanController (AMASS + IK + PD control)
├── motion/
│   └── amass_loader.py        # AMASS .npz clip loading
├── scenarios/
│   ├── scenario_sampler.py    # ScenarioSampler + ParameterSpace
│   └── disruption_types.py    # DisruptionType + DisruptionConfig
└── config.py                  # SafetyConfig, HumanConfig, SSMConfig
```

### Data flow per step

```
env.step(action)
  ├── Human controller updates pose (AMASS → IK → PD control)
  ├── MuJoCo physics sub-steps
  │   └── Contact forces captured per sub-step
  ├── SSM check: distance vs required separation
  ├── PFL check: peak force vs body-region limits
  └── Returns: obs, reward, terminated, truncated, info["safety"]
```
