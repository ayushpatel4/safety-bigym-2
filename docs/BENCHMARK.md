# Safety Benchmark Guide

This guide explains how to use the Safety Benchmark system for evaluating visuomotor policies in `safety-bigym`.

## 1. Overview

The benchmark evaluates policies on ISO 15066 safety metrics:
- **SSM (Speed and Separation Monitoring)**: Maintains safe distance based on relative velocity.
- **PFL (Power and Force Limiting)**: Checks if contact forces exceed bio-mechanical limits.
- **Collisions**: Tracks collision events and contact forces.

## 2. Running the Benchmark

Use the `benchmark_policies.py` script.

### 2.1 Basic Usage
Evaluate the `RandomPolicy` (baseline) on the `reach` task:

```bash
mjpython scripts/benchmark_policies.py --tasks reach --episodes 10
```

### 2.2 Comprehensive Mode
Get detailed metrics (time-to-violation, event counts, breakdown by scenario) and visualize with the viewer:

```bash
mjpython scripts/benchmark_policies.py --tasks reach --episodes 5 --comprehensive --render --max-steps 1000
```

### 2.3 Saving Reports
Save the full, non-truncated tables to a text file:

```bash
mjpython scripts/benchmark_policies.py ... --report-file benchmark_report.txt
```

JSON results are always saved to `benchmark_results.json` (configurable via `--output`).

## 3. Metrics Explained

| Metric | Description |
|--------|-------------|
| **SSM Rate** | % of episodes with at least one SSM violation. |
| **SSM Step %** | % of total episode duration spent in SSM violation. |
| **PFL Rate** | % of episodes with at least one PFL violation. |
| **Avg Force** | Average contact force (N) during collision frames. |
| **Max Force** | Peak contact force (N) observed in the episode. |
| **1st SSM (s)** | Time in seconds until the first SSM violation occurs. |
| **SSM Events** | Number of distinct violation events (transitions from safe to unsafe). |
| **Success** | Task success rate. |

## 4. Evaluating Your Policy

To benchmark your own policy, you need to:

1.  **Implement the `Policy` Interface**:
    Create a class that inherits from `Policy` in `safety_bigym/benchmark/policy.py`.

    ```python
    from safety_bigym.benchmark.policy import Policy
    
    class MyLearnedPolicy(Policy):
        def __init__(self, action_space, model_path):
            super().__init__(action_space)
            self.model = load_model(model_path)
            
        def act(self, obs):
            # Process observation
            action = self.model.predict(obs)
            return action
    ```

2.  **Update `benchmark_policies.py`**:
    Import your policy and instantiate it instead of `RandomPolicy`.

    ```python
    # ... inside main() ...
    if args.policy == "my_policy":
        policy = MyLearnedPolicy(action_space, args.model_path)
    ```

## 5. Configuration

The benchmark uses:
- **`SafetyConfig`**: Defined in `safety_bigym/config.py` (SSM/PFL thresholds).
- **`HumanConfig`**: Controls human motion and spawning.
- **`ScenarioSampler`**: Randomly selects human behaviors (Walking, Standing, diverse clips).

Metrics are broken down by **Scenario Type** (e.g., `DIRECT` approach vs `INCIDENTAL`) and **Motion Clip** when using `--comprehensive`.
