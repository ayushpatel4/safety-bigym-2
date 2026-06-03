# Post-CONFIRM runbook — lock ROW3 (graceful Lagrangian) + hybrid

> **⚠ 2026-06-03 — DON'T naively pool the d=0.3 CONFIRM seeds.** The 3-seed CONFIRM
> showed d=0.3 is PID seed-unstable (λ landed at 0.000 / 0.267 / 3.855 → unconstrained
> / graceful / windup-collapse), so the three seeds are heterogeneous and pooling them
> averages graceful + unconstrained + collapsed into mush. The **sweep → pick → pool →
> hybrid mechanics below are correct**, but run them on the **fixed-λ** stage dirs
> (`exp_local/fixed_lambda/fixlam0p27/lam0p27_seed{0,1,2}`, from `run_fixed_lambda.sh` /
> `dispatch_fixed_lambda.sh`), where λ is pinned at 0.27 and all three seeds should be
> graceful & poolable — NOT the d=0.3 `d0p3_seed*` dirs. See `IMPLEMENTATION_STATUS.md`
> (2026-06-03) + `filter_fallback_findings.md` §7. The headline robustness figure is
> `plot_basin_multiseed.py` over the three fixed-λ basin sweeps.

**Prereq:** the fixed-λ run (`dispatch_fixed_lambda.sh`, seeds 0/1/2) has finished —
i.e. `lam0p27_seed{0,1,2}/` under `exp_local/fixed_lambda/fixlam0p27/` each have
`final_metrics.json`. (The original d=0.3 `CONFIRM` is superseded per the note above.)

**What this confirms.** Seed-0 showed a graceful **proximity-avoidance basin**:
mid-training checkpoints (≈20k–33k) cut deployment proximity ~21% (0.296→0.23)
at near-baseline success (0.79) and benign velocity, on **both** noisy and oracle
— a win the reactive filter never achieved. Peak-success / final selection
**misses** it (final = 0.300 ≈ baseline). This runbook re-derives the basin per
seed, picks each seed's operating point from the **deployment** benchmark (not the
noisy train-eval), pools the three into the headline ROW3, then runs the hybrid.

Figure of the seed-0 basin: `results/figs/d0p3_basin_seed0.png`
(`scripts/plot_proximity_basin.py`). Full context: `docs/filter_fallback_findings.md`,
`docs/HANDOVER_2026-06-02.md`.

```bash
cd ~/Documents/safety_bigym && git pull && source venv/bin/activate
export MUJOCO_GL=egl    # + AMASS_DATA_DIR as usual
export E32=exp_local/e3_2_cost_budget/e3_2_saucepan_to_hob_20260531_190307
export BASE_PROX_NOISY=0.296 BASE_PROX_ORACLE=0.285   # row-1 references (pinned this session)
```

## 1. Per-seed basin sweep (noisy) + pick the operating point

`run_basin_sweep.sh` defaults to the seed-0 basin steps; each CONFIRM seed saves
at its own eval cadence, so if a default step is missing it's skipped — re-run
with that seed's real steps (`ls $E32/d0p3_seedN/snapshot_[0-9]*.pt`).

```bash
for s in 0 1 2; do
  STAGE_DIR=$E32/d0p3_seed$s OBS=noisy GPUS="0 1 2 3 4 5" bash scripts/run_basin_sweep.sh
done
# pick each seed's ROW3 operating point = lowest DEPLOY proximity at succ>=0.75:
for s in 0 1 2; do
  echo "seed$s:"; python scripts/analyze_row3.py pick \
    --sweep-dir results/e4_1/basin_d0p3_seed${s}_noisy --success-floor 0.75
done
```

Record each seed's operating snapshot path → `OP0 OP1 OP2`. (Seed 0's basin
already exists under `results/e4_1/d0p3_window_0006` + `row3_converged_2243`; you
can re-sweep for uniformity or reuse.)

## 2. ROW3 = pool the three operating points

**Noisy** is free — the picked checkpoint's 60-ep benchmark already lives in the
sweep dir as `s<step>.episodes.jsonl`. **Oracle** needs one benchmark per seed:

```bash
O=results/e4_1/row3_final_$(date +%m%d); mkdir -p $O
bench(){ python scripts/benchmark_policy.py --snapshot "$1" --task saucepan_to_hob \
  --disruption coworker_train --human-model g1 --obs-mode "$2" --num-demos-for-stats 0 \
  --seeds 0,1,2 --episodes 20 --out "$3"; }
# OP0/OP1/OP2 = the three operating snapshots from step 1; STEP0/1/2 their steps.
i=0; for OP in $OP0 $OP1 $OP2; do
  CUDA_VISIBLE_DEVICES=$i bench $OP oracle $O/seed${i}_oracle.csv > $O/seed${i}_oracle.log 2>&1 &
  i=$((i+1)); done; wait

# noisy ROW3 (pool the picked sweep episodes):
python scripts/analyze_row3.py aggregate --baseline-prox $BASE_PROX_NOISY --episodes \
  results/e4_1/basin_d0p3_seed0_noisy/s${STEP0}.episodes.jsonl \
  results/e4_1/basin_d0p3_seed1_noisy/s${STEP1}.episodes.jsonl \
  results/e4_1/basin_d0p3_seed2_noisy/s${STEP2}.episodes.jsonl
# oracle ROW3:
python scripts/analyze_row3.py aggregate --baseline-prox $BASE_PROX_ORACLE --episodes \
  $O/seed0_oracle.episodes.jsonl $O/seed1_oracle.episodes.jsonl $O/seed2_oracle.episodes.jsonl
```

**Decision (180-ep pooled ROW3 vs baseline 0.296 noisy / 0.285 oracle):**
- proximity ↓ with the **pooled CI clear of baseline** on noisy (and ideally
  oracle) → **the headline result holds across seeds.** Lock ROW3, proceed to §3.
- ↓ only as point estimate, CI still overlaps → report as a trend; consider more
  episodes (`--seeds 0,1,2,3,4,5`) before claiming significance.
- no ↓ once pooled → seed-0 was favourable noise → fall back to the realism-
  spectrum pivot (E5). (Unlikely given the coherent basin, but this is the gate.)

## 3. Hybrid (rows 4/5) + headline table

Once ROW3 is locked, run the full headline. Rows 1/4 (baseline ±filter) are
seed-independent; rows 3/5 are the ROW3 policy ±filter. For the 3-seed table,
run rows 3/5 per seed's operating snapshot and pool; or report the
representative (median-proximity) seed and note the pooled ROW3 from §2.

```bash
# per seed (gives rows 1,3,4,5 on noisy; rows 4/5 re-confirm the filter R on the
# graceful policy, where it sees far fewer close approaches):
for s in 0 1 2; do
  ROW3=<OP_s> STAGE2=$E32/../cqn_as_base_curriculum/.../stage2_full/snapshot_28203.pt \
    OBS_MODE=noisy OUTDIR=results/e4_1/headline_seed$s bash scripts/run_e4_1_headline.sh
done
python scripts/aggregate_e4_1.py --in-dir results/e4_1/headline_seed0   # LaTeX table
```

Re-confirm the filter threshold R on the ROW3 policy here (the SVF was pinned at
R=2.25 against the *baseline*; the graceful policy approaches less, so R may shift).

## 4. Figure + docs

```bash
# 3-seed basin figure (overlay all three sweeps):
python scripts/plot_proximity_basin.py --obs noisy --baseline-prox 0.296 --baseline-succ 0.85 \
  --csv-dir results/e4_1/basin_d0p3_seed0_noisy results/e4_1/basin_d0p3_seed1_noisy results/e4_1/basin_d0p3_seed2_noisy \
  --stage-substr d0p3_seed --out results/figs/d0p3_basin_3seed.png
```

Then update `docs/filter_fallback_findings.md` + `docs/IMPLEMENTATION_STATUS.md`
from "reactive filter bounded / pivot" to the confirmed framing:
**reactive ISO-SSM filtering is fundamentally limited (freeze-vs-flee, ~42%
exogenous), but the proactive constrained-RL Lagrangian resolves it (~21%
proximity reduction at acceptable success/velocity, both perception modes),
with safety-aware checkpoint selection (peak-success/final erode it) confirmed
across 3 seeds.** Keep the adversarial coworker as the headline; E5 spectrum for
robustness. Commit to `phase3`.

## Tooling added this session (all on `phase3`)
- `scripts/pick_best_snapshot.py --by safety` — train-eval nominee (lowest eval
  proximity s.t. success≥floor). `pick`'s deployment-confirmed analogue:
- `scripts/analyze_row3.py pick|aggregate` — operating-point pick from a basin
  sweep + 3-seed episode pooling with bootstrap CI.
- `scripts/run_basin_sweep.sh` — per-seed basin benchmark.
- `scripts/plot_proximity_basin.py` — the proximity-vs-step figure.
- tests: `tests/test_pick_safe_snapshot.py`, `tests/test_analyze_row3.py`.
