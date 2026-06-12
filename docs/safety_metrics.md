# Thesis-grade safety metrics

Last updated: 2026-05-27
Code: [safety/iso15066_wrapper.py](../safety_bigym/safety/iso15066_wrapper.py), [safety/episode_metrics_wrapper.py](../safety_bigym/safety/episode_metrics_wrapper.py)

This page is the load-bearing reference for which safety number to report
where in the thesis. The live training pipeline emits three flavours of
"safety violation" per env-step plus a set of per-episode aggregates; the
table below maps each metric to its reporting role.

## The three SSM/proximity flavours

| `info["safety"]` key      | Definition                                                                                                                                                                         | Thesis role                                                                                              |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `ssm_violation`           | ISO 15066 SSM under conservative caps. `S_p = v_h(T_r+T_s) + v_r·T_r + v_r²/(2·a_max) + C`, computed with `v_h = v_h_max` (1.6 m/s) and observed robot speed.                  | The conservative ISO worst-case bound — kept for ISO traceability and historical comparability.        |
| `ssm_violation_actual`    | Same formula as `ssm_violation` but with the **observed** human velocity instead of `v_h_max`. Margin = `min_separation - S_p_actual`; violation = margin < 0.                  | Formal ISO compliance under realized motion. Headline ISO number for the thesis Pareto plot.            |
| `proximity_violation`     | Pure geometric: `min_separation < SSMConfig.proximity_threshold` (**default 0.3 m as of 2026-05-30; was 0.5 m**, matches the G1 SVF production label — collect/relabel with `--proximity-threshold 0.3`). | The canonical "robot was too close to the human" rate. **Use this as the thesis's primary safety metric.** |

### Why three?

ISO 15066's SSM formula computes a velocity-dependent required separation
`S_p`. Under the conservative human-velocity cap `v_h_max = 1.6 m/s` and
typical kitchen-scale robot speeds, `S_p` lands around 0.5–5 m — far larger
than the geometric separation in a domestic-manipulation scene. The result
is an `ssm_violation` rate that over-fires (Phase 2 B2.3 measured 93 % on
random rollouts) and is more a function of robot-arm capability than actual
human proximity. Phase 2 SVF dataset labelling already worked around this
geometrically; promoting the geometric check to a first-class live metric
gives the thesis a defensible "actually too close" number alongside the
formal ISO-compliance figure.

### Failure modes the three flavours flag differently

* **Robot fast, human distant.** Both `ssm_violation` and `ssm_violation_actual` fire (robot's stopping distance exceeds the gap); `proximity_violation` doesn't.
* **Robot still, human inside 0.4 m.** `proximity_violation` fires; `ssm_violation_actual` only fires if `S_p_actual > 0.4 m` (depends on human's observed speed); `ssm_violation` typically fires because `S_h = v_h_max·(T_r+T_s) ≈ 0.24 m` plus `C` plus robot-side terms.
* **Robot still, human still, separation 0.6 m.** None fire — the safe regime.

## Per-step `info["safety"]` schema (post 2026-05-26)

Populated every env-step by [`ISO15066Wrapper.build_safety_info`](../safety_bigym/safety/iso15066_wrapper.py) inside [`SafetyBiGymEnv._aggregate_safety_info`](../safety_bigym/envs/safety_env.py). Keys:

| key                       | type   | meaning                                                                                  |
| ------------------------- | ------ | ---------------------------------------------------------------------------------------- |
| `ssm_violation`           | bool   | worst-case ISO 15066 SSM violation                                                       |
| `ssm_violation_actual`    | bool   | velocity-adaptive ISO 15066 SSM violation                                                |
| `proximity_violation`     | bool   | `min_separation < proximity_threshold`                                                   |
| `pfl_violation`           | bool   | any contact ratio ≥ 1.0 (PFL — currently always False under the open contact-detect bug) |
| `ssm_margin`              | float  | `min_separation − S_p_worst` (negative = violation)                                      |
| `ssm_margin_actual`       | float  | `min_separation − S_p_actual`                                                            |
| `pfl_force_ratio`         | float  | `max_c contact.force / contact.force_limit` (zero under the open bug)                    |
| `min_separation`          | float  | closest human-joint / robot-link pair, meters                                            |
| `max_contact_force`       | float  | peak Newton across sub-steps                                                             |
| `contact_region`          | str    | ISO body region (e.g. `forearm`) for the worst contact this step                         |
| `contact_type`            | str    | `quasi_static` or `transient`                                                            |
| `proximity_threshold`     | float  | echoes `SSMConfig.proximity_threshold` for downstream analysis                           |
| `robot_vel`               | float  | observed max robot link speed (m/s) used in `ssm_violation_actual`                       |
| `human_vel`               | float  | observed max human body speed (m/s, capped at `v_h_max` for AMASS-noise safety)          |
| `violations_by_region`    | dict   | counts of PFL violations by body region                                                  |
| `robot_pos` / `human_pos` | list   | closest-pair geom positions                                                              |
| `closest_human_joint`     | str    | name of the human body that drove `min_separation`                                       |
| `closest_robot_link`      | str    | name of the robot link that drove `min_separation`                                       |

## Per-episode aggregates `info["episode_safety"]`

Emitted by [`EpisodeSafetyMetrics`](../safety_bigym/safety/episode_metrics_wrapper.py) every step (running summary) and at `terminated/truncated` (final aggregate). RoboBase's `Workspace` and `train_cqn_as.py._safety_payload` forward these to W&B under the `episode_safety/*` prefix.

| key                                  | meaning                                                                          | thesis use                                                          |
| ------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `ep_steps`                           | episode length in env-steps                                                      |                                                                     |
| `ep_ssm_violation_rate`              | fraction of steps with worst-case SSM violation                                  | conservative ISO traceability                                       |
| `ep_ssm_violation_actual_rate`       | fraction of steps with velocity-adaptive SSM violation                           | formal ISO compliance (Pareto y-axis option)                        |
| `ep_proximity_violation_rate`        | fraction of steps with `min_separation < τ`                                      | **canonical safety axis** (Pareto x-axis)                           |
| `ep_pfl_violation_rate`              | fraction of steps with any PFL violation                                         | currently always 0 (open contact-detect bug)                        |
| `ep_time_in_proximity_0p3m`          | fraction of steps within 0.3 m                                                   | acute-risk-integral; near-impact dwell                              |
| `ep_time_in_proximity_0p5m`          | fraction of steps within 0.5 m                                                   | matches default proximity threshold                                 |
| `ep_time_in_proximity_1p0m`          | fraction of steps within 1.0 m                                                   | "in the same kitchen cell" dwell                                    |
| `ep_min_separation`                  | min over the episode                                                             | dangerous-tail snapshot                                             |
| `ep_mean_separation`                 | mean over the episode                                                            | sustained closeness                                                 |
| `ep_p5_separation`                   | 5th-percentile (closest 5 % of the episode)                                      | dangerous-tail quantile                                             |
| `ep_p25_separation`                  | 25th-percentile (closest quarter of the episode)                                 | sustained-closeness quantile                                        |
| `ep_min_ssm_margin`                  | min of `ssm_margin` over the episode                                             | how badly the worst-case SSM bound was violated                     |
| `ep_min_ssm_margin_actual`           | min of `ssm_margin_actual` over the episode                                      | how badly the realistic SSM bound was violated                      |
| `ep_max_pfl_force_ratio`             | max contact-force ratio over the episode                                         | currently always 0                                                  |
| `ep_max_contact_force`               | peak Newton over the episode                                                     | currently always 0                                                  |
| `ep_max_robot_vel`, `ep_mean_robot_vel` | robot-link speed extremes/means (m/s)                                          | explains why worst-vs-actual SSM diverge                            |
| `ep_time_to_first_violation`         | step idx of first `ssm_violation OR pfl_violation` (−1 if clean)                 | reaction-time diagnostic                                            |
| `ep_region_<region>`                 | per-body-region PFL violation count                                              | currently empty                                                     |

## Lagrangian-specific episode logging

[`train_cqn_as.py._lagrangian_payload`](../train_cqn_as.py) emits two extra
keys at episode-end under `episode/*` whenever the active agent is the
Lagrangian subclass (P3.1):

| key                       | meaning                                                                                                                                  |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `episode_lambda`          | running Lagrange multiplier λ at episode close (PID-driven over `rolling_cost − cost_budget`)                                            |
| `episode_cost_integral`   | Σ_t `c_t` summed across the episode (`c_t = max(c_ssm, c_pfl)` from [`filters/cost_signal.py`](../safety_bigym/filters/cost_signal.py))  |

`episode_cost_integral` is emitted on the unconstrained baseline too (it
just won't have `episode_lambda` alongside) — useful for telling what the
cost *would* be if λ were active.

Per-update Lagrangian metrics (`lambda`, `rolling_cost`, `cost_violation`,
`batch_cost`, `q_c_loss`) continue to flow through `agent.update()` →
`_log(metrics, ty="train")`. No change needed to surface them.

## W&B run tagging

[`scripts/run_base_curriculum.sh`](../scripts/run_base_curriculum.sh) emits
`+wandb.tags=[stage{0,1,2},method:unconstrained,task:${TASK},human:${HUMAN_MODEL}]` per stage.
[`train_cqn_as.py._setup_wandb`](../train_cqn_as.py) forwards the list to
`wandb.init(tags=...)`. The thesis Pareto / convergence plots filter on
these tags in the W&B UI.

Hydra reserves `=` and `,` inside override values, so use `:` as the
key/value separator inside tag strings. P3.1 launchers should append
`method:lagrangian` (or `method:hybrid` for
the eventual SVF + Lagrangian combination) instead of `method:unconstrained`.

## Local JSON dumps (resilient to W&B downtime)

Every `_log()` call in [`train_cqn_as.py`](../train_cqn_as.py) also appends
to two files under the per-run `work_dir` (i.e. each Hydra `hydra.run.dir`):

- **`metrics.jsonl`** — streaming, one JSON object per `_log` call:
  ```jsonc
  {"step": 40, "ty": "eval", "eval/success_rate": 0.4, "eval/ep_proximity_violation_rate": 0.28, ...}
  ```
  Load with `pandas.read_json("metrics.jsonl", lines=True)` for offline
  analysis. All four ty's (`train`, `episode`, `safety`, `eval`) interleave
  in step order — filter on `ty` to isolate one stream.
- **`final_metrics.json`** — written at end of `train()`. Headline numbers
  for the thesis writeup:
  - `config` — task / disruption / num_train_frames / num_demos / agent
    v_min / v_max / wandb name + tags.
  - `last_train_episode` — last training-side `train/episode_*` row.
  - `last_episode_safety` — last `episode/episode_safety/ep_*` row.
  - `last_eval` — last eval cycle's row.
  - `best_eval` — best (max-prefer for reward/success, min-prefer for
    safety axes) over all eval cycles: `success_rate`, `episode_reward`,
    `ep_proximity_violation_rate`, `ep_ssm_violation_actual_rate`,
    `ep_min_separation_lowest`.

The `eval()` loop also aggregates `info["episode_safety"]` across the
`num_eval_episodes` rollouts, so `eval/ep_proximity_violation_rate`,
`eval/ep_ssm_violation_actual_rate`, `eval/ep_min_separation`, etc. land
in both W&B and `metrics.jsonl` paired with `success_rate` /
`episode_reward` — the per-eval row of the thesis Pareto plot.

## PFL limitation (documented)

PFL contact-force monitoring is implemented and ISO Annex A force limits
are tabulated, but a known BiGym/MuJoCo contact-detection bug causes
`data.ncon = 0` for every human↔robot pair at runtime regardless of
geometric overlap. Safety metrics in this report are therefore
SSM/proximity-based; contact-force-based safety is a documented future
extension. The current diagnostic entry point is
[`scripts/diagnose_contact_forces.py`](../scripts/diagnose_contact_forces.py).

## Selecting τ (proximity_threshold)

Default is 0.3 m (as of 2026-05-30; was 0.5 m), matching the G1 SVF
production labelling bar. The SVF critic is trained at the same τ via
`svf_train_critic.py --proximity-threshold 0.3`, which relabels `r_safe`
on the fly from the stored per-step `min_separation` (no re-collection).
Override via Hydra: `env.safety.proximity_threshold=0.4` etc. The shard
schema (Phase 2) records raw `min_separation` per-step, so threshold
sweeps over historical data are free.

### How 0.3 m was calibrated (2026-05-30)

τ is set by triangulating two speed-independent analyses (full write-up for
the report: [proximity_threshold_calibration.tex](proximity_threshold_calibration.tex)):

1. **Contact geometry** — [`scripts/visualize_separation_distances.py`](../scripts/visualize_separation_distances.py)
   renders the coworker at controlled separations (no physics stepping; robot
   frozen in its reset pose, coworker pelvis mocap slid along the approach
   axis, `min_separation` recomputed via `_aggregate_safety_info`). Collision
   geoms **touch at `min_separation ≈ 0.01–0.10 m`**; a clearly visible gap has
   opened by **≈ 0.30 m**. So any usable τ must sit above the ~0.1 m contact
   band — a violation should *precede* contact, not coincide with it.

2. **Empirical distribution** — [`scripts/calibrate_proximity_threshold.py`](../scripts/calibrate_proximity_threshold.py)
   rolls out a policy under `coworker_train` and collects per-step
   `min_separation`. Since the proximity-violation rate at threshold τ is
   exactly the empirical CDF `P(min_separation < τ)`, τ reads straight off the
   curve. The distribution is **bimodal** (a near-contact mode at 0–0.2 m plus
   a far mid-/patrol tail), and the CDF has a **knee at τ ≈ 0.2–0.3 m**. On the
   headline G1 stage-2 baseline (3 seeds × 20 ep, 30 000 steps,
   `results/prox_calib_row5.*`):

   | τ (m)          | 0.20 | **0.30** | 0.40 | 0.50 | 0.75 | 1.00 |
   | -------------- | ---- | -------- | ---- | ---- | ---- | ---- |
   | violation rate | 14.0 % | **18.9 %** | 24.4 % | 30.3 % | 48.4 % | 64.4 % |

   The knee location is consistent across two independent trained policies, so
   it reflects scene contact geometry rather than one controller.

**Decision:** τ = 0.3 m sits at the knee, ~0.2 m above the contact band, and
captures the genuine near-contact population without absorbing the benign
mid-range proximity that 0.5 m (30 %) or 1.0 m (64 %) would. It is
speed-independent, so the proximity-violation rate is comparable across
policies, tasks, and disruption bands; it is held **fixed across all
experiments**. The high baseline rate (18.9 % of steps below τ) is precisely
the unsafe behaviour the constrained-RL policy and SVF filter are meant to
reduce. `scripts/calibrate_proximity_threshold.py` reads the live
`SSMConfig.proximity_threshold` so its "current threshold" marker tracks any
future change.
