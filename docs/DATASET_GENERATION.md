# Dataset Generation

Generate diverse human-robot interaction datasets using `generate_dataset.py`.

## Quick Start

```bash
# View a few episodes in MuJoCo viewer
mjpython scripts/generate_dataset.py --view --tasks reach --n-per-type 1

# Generate a full dataset
mjpython scripts/generate_dataset.py --output datasets/my_dataset
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `datasets/safety_dataset` | Output directory |
| `--tasks` | all tasks | Space-separated list of tasks |
| `--sampling` | `stratified` | `stratified` or `random` |
| `--n-per-type` | `5` | Episodes per disruption type (stratified) |
| `--total-episodes` | `100` | Total episodes per task (random) |
| `--max-steps` | `500` | Max steps per episode |
| `--seed` | `0` | Base random seed |
| `--view` | off | Enable MuJoCo viewer |
| `--save-obs` | off | Save full observations |
| `--motion-dir` | CMU path | Path to AMASS clips |

## Examples

### Watch episodes in viewer
```bash
mjpython scripts/generate_dataset.py --view --tasks reach pick_box --n-per-type 1
```

### Generate stratified dataset (5 episodes per disruption type)
```bash
mjpython scripts/generate_dataset.py \
    --tasks reach pick_box dishwasher_open \
    --n-per-type 5 \
    --output datasets/stratified
```
This creates 25 episodes per task (5 types × 5 episodes).

### Generate random dataset
```bash
mjpython scripts/generate_dataset.py \
    --sampling random \
    --total-episodes 100 \
    --tasks reach \
    --output datasets/random
```

### Save full observations for imitation learning
```bash
mjpython scripts/generate_dataset.py \
    --save-obs \
    --tasks reach \
    --n-per-type 10 \
    --output datasets/with_obs
```

## Available Tasks

`default`, `reach`, `reach_dual`, `pick_box`, `saucepan`, `flip_cup`, `stack_blocks`, `dishwasher_open`, `dishwasher_close`, `cupboard_open`, `drawer_open`, `move_plate`, `groceries`, `take_cups`, `put_cups`

## Disruption Types (Human Behaviours)

| Type | Description |
|------|-------------|
| `INCIDENTAL` | Human walks through workspace |
| `SHARED_GOAL` | Human reaches toward task object |
| `DIRECT` | Human reaches toward robot end-effector |
| `OBSTRUCTION` | Human blocks robot path |
| `RANDOM_PERTURBED` | Motion with random perturbations |

## Output Structure

```
datasets/my_dataset/
├── metadata.json       # Dataset configuration
├── summary.csv         # Episode overview (task, violations, etc.)
└── episodes/
    ├── reach_0000.npz
    ├── reach_0001.npz
    └── ...
```

### Episode File Contents

Each `.npz` file contains:

```python
data = np.load('episodes/reach_0000.npz', allow_pickle=True)

# Scenario info
data['task']                      # 'reach'
data['scenario_disruption_type']  # 'DIRECT'
data['scenario_seed']             # 42
data['scenario_trigger_time']     # 2.3
data['scenario_speed_multiplier'] # 1.2

# Episode summary
data['episode_length']            # 423
data['ssm_violations']            # 15
data['pfl_violations']            # 2
data['min_separation']            # 0.12
data['max_contact_force']         # 85.3

# Per-step arrays
data['ssm_violation']             # bool array (T,)
data['pfl_violation']             # bool array (T,)
data['ssm_margin']                # float array (T,)
data['human_positions']           # float array (T, 3)

# Optional (--save-obs)
data['observations']              # array (T, obs_dim)
data['actions']                   # array (T, act_dim)
```

## Loading Data

```python
import numpy as np
import pandas as pd

# Load summary
df = pd.read_csv('datasets/my_dataset/summary.csv')
print(df.groupby('disruption_type')['ssm_violations'].mean())

# Load single episode
data = np.load('datasets/my_dataset/episodes/reach_0000.npz', allow_pickle=True)
print(f"SSM violations: {data['ssm_violations']}")
print(f"Min separation: {data['min_separation']:.3f}m")
```
