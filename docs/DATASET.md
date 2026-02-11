# Dataset Collection Guide

This guide explains how to use `scripts/collect_dataset.py` to generate diverse human-robot interaction datasets using Safety BiGym.

## script: `collect_dataset.py`

Generate HDF5 datasets containing robot observations, actions, rewards, and detailed safety information across various tasks and random human scenarios.

### CLI Usage

**Basic collection:**
Collect 10 episodes for the `reach` task.
```bash
python scripts/collect_dataset.py --tasks reach --episodes-per-task 10 --output reach_data.h5
```

**Multiple tasks:**
Collect data for both `reach` and `pick_box`.
```bash
python scripts/collect_dataset.py --tasks reach pick_box --episodes-per-task 50 --output multi_task.h5
```

**All available tasks:**
Run episodes for every registered BiGym task.
```bash
python scripts/collect_dataset.py --tasks all --episodes-per-task 5 --output full_dataset.h5
```

**Visual debugging:**
Launch the MuJoCo viewer during collection to see what's happening (requires `mjpython` on macOS).
```bash
mjpython scripts/collect_dataset.py --tasks reach --episodes-per-task 2 --render
```

**Random actions:**
By default, the robot holds its position (zero actions). Use `--random-actions` to apply random control inputs.
```bash
python scripts/collect_dataset.py --tasks reach --random-actions
```

---

## Dataset Structure (HDF5)

The output is a hierarchical HDF5 file.

### file format

```
dataset.h5
├── [task_name]                  # Group per task (e.g., "reach")
│   ├── episode_0000             # Group per episode
│   │   ├── obs                  # Dataset: [T, obs_dim] (proprioception)
│   │   ├── actions              # Dataset: [T, action_dim]
│   │   ├── rewards              # Dataset: [T]
│   │   ├── human_state          # Dataset: [T, 7] (pos[3], quat[4])
│   │   └── safety               # Group: Safety signals
│   │       ├── ssm_violation    # Dataset: [T] (bool)
│   │       ├── pfl_violation    # Dataset: [T] (bool)
│   │       ├── min_separation   # Dataset: [T] (meters)
│   │       └── max_contact_force# Dataset: [T] (Newtons)
│   │
│   ├── episode_0001
│   └── ...
├── [task_name]
└── ...
```

### Metadata (Attributes)

Each **episode group** (e.g., `reach/episode_0000`) stores scenario configuration as HDF5 attributes:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `task` | Name of the task | `"reach"` |
| `seed` | Random seed for this episode | `1739294812` |
| `scenario_disruption_type` | Type of human behavior | `"DIRECT"`, `"SHARED_GOAL"`, `"INCIDENTAL"` |
| `scenario_trigger_time` | Time when disruption starts (s) | `2.45` |
| `scenario_clip_path` | AMASS motion clip used | `"74/74_01_poses.npz"` |
| `scenario_approach_angle` | Angle of human approach (deg) | `135.0` |
| `scenario_spawn_distance` | Starting distance from robot (m) | `1.5` |
| `scenario_speed_multiplier`| Playback speed of motion | `1.2` |

### Loading Data (Python Example)

```python
import h5py
import matplotlib.pyplot as plt

# Open dataset
f = h5py.File("dataset.h5", "r")

# Access first episode of 'reach'
ep = f["reach/episode_0000"]

# Read metadata
print(f"Disruption Type: {ep.attrs['scenario_disruption_type']}")

# Plot separation distance
# Note: Use [:] to cast HDF5 dataset to numpy array
min_sep = ep["safety/min_separation"][:]
plt.plot(min_sep)
plt.title("Minimum Separation Distance")
plt.xlabel("Step")
plt.ylabel("Distance (m)")
plt.show()
```
