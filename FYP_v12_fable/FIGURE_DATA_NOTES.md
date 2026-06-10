# FIGURE_DATA_NOTES — F7 gate activity + vector fig1–fig5

Generated 2026-06-10 by `scripts/make_f7_gate_activity.py` (F7) and
`scripts/make_report_figures_pdf.py` (fig1–fig5 PDFs). All paths relative to the
repo root `/home/ap2322/Documents/safety_bigym`.

## What F7 measures (and what "intervention" means)

For the **critic-gated speed-scaling** filter the benchmark runner
(`safety_bigym/benchmark/runners.py`, `CQNASRunner.step`, `gated_speedscale`
branch) counts an intervention on **every env step where the gate fires**
(`q_safe(s,a) < R`), *independent of the resulting speed scale* (even if
separation ≥ d_slow and the action passes through at scale 1.0). So

```
gate-active fraction = Σ n_interventions / Σ filter_steps
```

over episodes is the **exact per-step fraction of steps the gate is active** —
not a proxy. `n_interventions` and `filter_steps` are stored per episode in the
`.raw_episodes.parquet` / `.episodes.jsonl` sidecars; F7 recomputes the pooled
rate from those (the same statistic the harness writes as
`filter_intervention_rate` in `safety_bigym/benchmark/aggregate.py`).

For the **unconditional** speed-scale rows, `intervened=True` only when the
filter actually scales, i.e. when separation < d_slow
(`safety_bigym/filters/cbf_filter.py`, `SpeedScaleFilter.apply`). So the uncond
"intervention rate" is the proximity-trigger rate at radius d_slow.

**Proxy status: no proxy was needed.** The preferred refinement "gate-active
among near-human steps only" is NOT computable — the sidecars are per-episode
scalars; no per-step (min_separation, intervened) traces are stored anywhere
under `results/`. The figure therefore shows gate-active fraction over **all**
steps, which is the literal quantity named in the figure.

**CI method:** 95% nonparametric bootstrap of the pooled (step-weighted) rate,
resampling episodes (10,000 resamples, numpy seed 0).

## F7 figure numbers — matched gate threshold R=2.75 (all three tasks)

| task | critic (checkpoint) | R | d_slow / d_stop | gate-active rate | 95% CI | counts (interv/steps) | n_ep | mean q | source files |
|---|---|---|---|---|---|---|---|---|---|
| saucepan_to_hob (persistent) | v3 = `checkpoints/svf_coworker_train_g1_0p3_v3.pt` | 2.75 | 0.40 / 0.15 | **0.6151** (61.5%) | [0.5722, 0.6540] | 75184 / 122235 | 180 | 1.667 | `results/e4_1/gated_saucepan/v3_R2p75_s{0,1,2}.raw_episodes.parquet` |
| dishwasher_close (intermittent) | v1 = `checkpoints/svf_dish_drawers_v1.pt` | 2.75 | 0.50 / 0.15 | **0.2654** (26.5%) | [0.1633, 0.3671] | 3700 / 13940 | 60 | 3.095 | `results/gated_sweep/dish_R2p75.raw_episodes.parquet` |
| drawers_open_all (intermittent) | v1 = `checkpoints/svf_dish_drawers_v1.pt` | 2.75 | 0.50 / 0.15 | **0.1919** (19.2%) | [0.1267, 0.2528] | 6844 / 35673 | 60 | 3.073 | `results/gated_sweep/draw_R2p75.raw_episodes.parquet` |

Reference line on the saucepan bar (same ROW3 policy snapshots, same harness):

| config | "intervention" meaning | rate | 95% CI | counts | n_ep | source |
|---|---|---|---|---|---|---|
| saucepan unconditional speed-scale, d_slow=0.40 / d_stop=0.15 | scaled because sep < 0.40 m | **0.4415** (44.2%) | [0.3981, 0.4838] | 56615 / 128222 | 180 | `results/e4_1/gated_saucepan/uncond_R0p4_s{0,1,2}.raw_episodes.parquet` |

Reading: at the matched threshold the saucepan gate is open on 61.5% of steps —
**more often than the unconditional scaler's own proximity trigger fires
(44.2%)** — so on this task the gate has no selectivity left to exploit and
gating degenerates toward unconditional scaling. Dish/drawers gate on 26.5% /
19.2% of steps. (As R→∞ the gate fires on 100% of steps and the gated filter
becomes exactly the unconditional scaler; 44.2% is how often that limit filter
actually modifies actions.)

## Operating-point set (recommended configs; quoted in text, not in the figure)

| task | config | gate-active rate | 95% CI | counts | n_ep | mean q | source |
|---|---|---|---|---|---|---|---|
| dishwasher_close | R=2.75, d_slow=0.50, d_stop=0.15 (recommended; same cell as matched-R) | 0.2654 | [0.1633, 0.3671] | 3700 / 13940 | 60 | 3.095 | `results/gated_sweep/dish_R2p75.raw_episodes.parquet` |
| drawers_open_all | R=3.0, d_slow=0.80, d_stop=0.25 (recommended) | 0.2660 | [0.1929, 0.3328] | 10649 / 40031 | 60 | 3.039 | `results/improve/draw_R3p0_ds0p8_dt0p25.raw_episodes.parquet` |
| saucepan_to_hob | R=2.5, d_slow=0.40 | 0.5067 | [0.4564, 0.5521] | 55887 / 110286 | 180 | 1.821 | `results/e4_1/gated_saucepan/v3_R2p5_s*.raw_episodes.parquet` |
| saucepan_to_hob | R=2.75, d_slow=0.40 | 0.6151 | [0.5722, 0.6540] | 75184 / 122235 | 180 | 1.667 | `results/e4_1/gated_saucepan/v3_R2p75_s*.raw_episodes.parquet` |
| saucepan_to_hob | R=3.0, d_slow=0.40 | 0.6328 | [0.5900, 0.6710] | 77800 / 122940 | 180 | 1.663 | `results/e4_1/gated_saucepan/v3_R3p0_s*.raw_episodes.parquet` |

At the operating points the contrast survives: saucepan 50.7–63.3% gate-active
vs dish 26.5% / drawers 26.6%.

## Supplementary: full gate-activity dials (pooled rate per cell)

- saucepan v3 (d_slow=0.40, 180 ep/cell): R=0.0 → 0.0000 (control, gate never
  fires); R=1.5 → 0.3777; R=2.0 → 0.3941; R=2.25 → 0.3912; R=2.5 → 0.5067;
  R=2.75 → 0.6151; R=3.0 → 0.6328
- saucepan v3op (on-policy critic OOD-hedge, d_slow=0.40, 180 ep/cell):
  R=2.0 → 0.1540; R=2.5 → 0.2787
- dish v1 (d_slow=0.50 unless noted, 60 ep/cell): R=2.0 → 0.0102;
  R=2.25 → 0.0123; R=2.5 → 0.1311; R=2.75 → 0.2654;
  R=3.0 ds=0.40 → 0.3535; R=3.0 ds=0.50 → 0.3551
- drawers v1 (60 ep/cell): R=2.0 → 0.1086; R=2.25 → 0.1146; R=2.5 → 0.1530;
  R=2.75 → 0.1919; R=2.75 ds=0.80 → 0.1849; R=3.0 ds=0.50 → 0.2789;
  R=3.0 ds=0.80 → 0.2933; R=3.0 ds=0.80 dt=0.25 → 0.2660

Note dish R=3.0 at d_slow 0.40 vs 0.50 gives near-identical gate activity
(0.3535 vs 0.3551) — direct evidence that d_slow does not drive the gate-active
metric (the gate condition is q < R only; d_slow shapes the response).

## Success / SSM context for the F7 cells (for cross-quoting)

- saucepan v3 R=2.75: success 0.4722, SSM-actual 0.1023 (`results/e4_1/gated_saucepan/summary.json`)
- saucepan v3 R=3.0: success 0.4722, SSM-actual 0.0843; v3 R=2.5: success 0.5667, SSM-actual 0.1054
- saucepan uncond d_slow=0.40: success 0.4333, SSM-actual 0.0724
- saucepan v3 R=0.0 (policy-alone control): success 0.7222, SSM-actual 0.1383
- dish R=2.75: success 0.6667, SSM-actual 0.0884 (`results/gated_sweep/dish_R2p75.csv`)
- drawers R=2.75: success 0.8000, SSM-actual 0.0919 (`results/gated_sweep/draw_R2p75.csv`)
- drawers R=3.0/0.8/0.25: success 0.7333, SSM-actual 0.0780 (`results/improve/draw_R3p0_ds0p8_dt0p25.csv`)

## Caveats (read before quoting)

1. **Why summary.json had NaN:** the per-run CSVs for the gated saucepan rows
   (`v3_*`, `v3op_*`) were written with empty `filter_intervention_rate` /
   `n_interventions` columns (the CSV aggregation omitted the filter-mechanics
   block for gated runs at the time; the uncond row has them), so
   `scripts/aggregate_e4_1.py` averaged blanks into NaN. The episode-level
   sidecars always carried the counts; F7 recomputes from those.
2. **Different critics per task.** Saucepan uses its own SVF (v3,
   baseline-policy-trained, same recipe family as dish/drawers v1). R=2.75 is
   numerically matched but the critics' q-scales differ: mean q over gated eval
   steps ≈ 1.67 on saucepan vs ≈ 3.09/3.07 on dish/drawers. Lower saucepan q is
   substantially the phenomenon itself (the critic sees persistent risk), but
   critic calibration and true risk exposure cannot be fully separated here.
   The matched-R comparison is defensible because R was swept per task on each
   task's own critic and the recommended operating knees all fall in 2.75–3.0.
3. **d_slow / d_stop differ across tasks** (0.40 saucepan, 0.50 dish/drawers
   sweep, 0.80/0.25 drawers operating point). This does NOT affect the
   gate-active fraction (gate fires on q < R only); it only changes how strongly
   the action is scaled once fired. See the dish ds=0.40 vs 0.50 note above.
4. **Pooled vs per-episode-mean rates differ**, most on saucepan: pooled 0.6151
   vs mean-of-per-episode-rates 0.4392 (dish 0.2654 vs 0.1846; drawers 0.1919
   vs 0.1137). Heavily-gated episodes run longer (slowed robot), so they weigh
   more in the step-pooled statistic. F7 quotes the pooled per-step fraction —
   the literal "fraction of steps the gate is active" and the harness's own
   `filter_intervention_rate` convention.
5. **Uncond rate cross-check:** global step-pooled 0.4415 vs 0.4412 in
   `summary.json` (which averages the three per-run CSV rates). Negligible;
   figure/notes use 0.4415, the report may cite 0.441 either way.
6. **v3op (on-policy critic) hedge rows** gate far less (R=2.0 → 0.1540,
   R=2.5 → 0.2787) because the on-policy critic assigns higher q (mean ≈ 2.5–2.7)
   on this policy's distribution. The figure uses v3 — the apples-to-apples
   analog of the dish/drawers v1 critic (both trained on baseline-policy data).
7. **Episode pooling:** each saucepan cell pools 3 policy-snapshot seeds
   (`_s0/_s1/_s2` files; ROW3 fixed-λ=0.1 per-seed `snapshot_best.pt`) × 60 ep =
   180 ep; dish/drawers cells are single runs with env seeds 0,1,2 × 20 ep = 60 ep.
   All cells: G1 human, coworker_train disruption, noisy obs, CQN-AS policy.

## fig1–fig5 vector versions (Task A)

All five regenerated as PDFs by `scripts/make_report_figures_pdf.py` — an
output-format/destination-only copy of `scripts/make_report_figures.py` (same
data files, same plotting code; no content/number changes). All input CSVs were
present; nothing failed:

- `fig1_method_comparison.pdf`, `fig2_tradeoff_curve.pdf`,
  `fig3_reduction_bars.pdf`, `fig4_exogenous_proximity.pdf`,
  `fig5_cross_task_boundary.pdf` — in this directory.
- The `.png` copies already in this directory were verified byte-identical to
  `docs/figures/*.png` (cmp), so they were left untouched.
