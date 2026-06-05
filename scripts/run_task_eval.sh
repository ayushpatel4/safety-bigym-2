#!/usr/bin/env bash
# run_task_eval.sh — policy-only E4.1 eval for a task's fixed-λ Lagrangian run.
# Per-seed basin sweep (noisy) → pick operating point → pool 3-seed ROW3 → headline
# figure, plus the unconstrained baseline (row 1). Reusable across tasks
# (saucepan_to_hob, drawers_open_all, ...). PURE EVAL — no training.
#
# Inputs (env):
#   TASK          saucepan_to_hob | drawers_open_all | ...        (default saucepan_to_hob)
#   FIXLAM_DIR    fixed-λ run dir holding <LAM_TAG>_seed{0,1,2}    (REQUIRED)
#                 e.g. exp_local/fixed_lambda/drawers_fixlam0p1
#   LAM_TAG       lambda subdir tag                                (default lam0p1)
#   BASELINE      unconstrained stage-2 snapshot for row 1         (optional; else BASE_PROX used)
#   BASE_PROX     baseline proximity for the Δ print               (default 0.296; overwritten if BASELINE given)
#   GPUS          GPU pool for the sweeps                          (default "0 1 2 3 4 5")
#   SUCCESS_FLOOR operating-point success floor                    (default 0.75)
#   MIN_STEP      sweep checkpoints with step ≥ this               (default 7000)
#
# Usage:
#   TASK=drawers_open_all FIXLAM_DIR=exp_local/fixed_lambda/drawers_fixlam0p1 \
#     BASELINE=exp_local/cqn_as_base_curriculum/<drawers>/stage2_full/snapshot_<N>.pt \
#     bash scripts/run_task_eval.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

TASK="${TASK:-saucepan_to_hob}"
: "${FIXLAM_DIR:?set FIXLAM_DIR to the fixed-λ run dir (holds <LAM_TAG>_seed{0,1,2})}"
LAM_TAG="${LAM_TAG:-lam0p1}"
GPUS="${GPUS:-0 1 2 3 4 5}"
SUCCESS_FLOOR="${SUCCESS_FLOOR:-0.75}"
MIN_STEP="${MIN_STEP:-7000}"
BASE_PROX="${BASE_PROX:-0.296}"
G0="${GPUS%% *}"   # first GPU for the single-shot baseline bench

echo "== run_task_eval | TASK=$TASK FIXLAM_DIR=$FIXLAM_DIR LAM_TAG=$LAM_TAG =="

# 1) baseline (row 1) — benchmark if a snapshot is given; else fall back to BASE_PROX.
if [[ -n "${BASELINE:-}" ]]; then
  echo "== row1 baseline ($BASELINE) =="
  CUDA_VISIBLE_DEVICES="$G0" python scripts/benchmark_policy.py --snapshot "$BASELINE" \
    --task "$TASK" --disruption coworker_train --human-model g1 --obs-mode noisy \
    --num-demos-for-stats 0 --seeds 0,1,2 --episodes 20 --out "results/e4_1/row1_${TASK}.csv" || true
  if [[ -f "results/e4_1/row1_${TASK}.csv" ]]; then
    BASE_PROX="$(python -c "import pandas as pd;print(round(float(pd.read_csv('results/e4_1/row1_${TASK}.csv').iloc[-1].ep_proximity_violation_rate),3))" 2>/dev/null || echo "$BASE_PROX")"
  fi
fi
echo "   baseline proximity = $BASE_PROX"

# 2) per-seed basin sweep (noisy)
for s in 0 1 2; do
  sd="$FIXLAM_DIR/${LAM_TAG}_seed$s"
  if [[ ! -d "$sd" ]]; then echo "WARN: missing $sd (skip seed$s)"; continue; fi
  STEPS="$(ls "$sd"/snapshot_[0-9]*.pt 2>/dev/null | sed 's/.*snapshot_//;s/\.pt//' | awk -v m="$MIN_STEP" '$1>=m' | sort -n | tr '\n' ' ')"
  [[ -z "$STEPS" ]] && { echo "WARN: no snapshots ≥$MIN_STEP in $sd"; continue; }
  STAGE_DIR="$sd" OBS=noisy TASK="$TASK" STEPS="$STEPS" GPUS="$GPUS" bash scripts/run_basin_sweep.sh
done

# 3) pick each seed's operating point (lowest deploy prox at succ≥floor)
EP=()
for s in 0 1 2; do
  swp="results/e4_1/basin_${LAM_TAG}_seed${s}_noisy"
  [[ -d "$swp" ]] || continue
  echo "== seed$s operating point =="
  line="$(python scripts/analyze_row3.py pick --sweep-dir "$swp" --success-floor "$SUCCESS_FLOOR" 2>&1 || true)"
  echo "$line"
  step="$(printf '%s\n' "$line" | grep -oE 'step=[0-9]+' | head -1 | cut -d= -f2 || true)"
  [[ -n "$step" && -f "$swp/s${step}.episodes.jsonl" ]] && EP+=("$swp/s${step}.episodes.jsonl")
done

# 4) pooled ROW3 + headline figure
if (( ${#EP[@]} )); then
  echo "== ROW3 (pooled $((${#EP[@]}*60)) ep) =="
  python scripts/analyze_row3.py aggregate --baseline-prox "$BASE_PROX" --episodes "${EP[@]}"
else
  echo "WARN: no operating points cleared succ≥${SUCCESS_FLOOR}; loosen SUCCESS_FLOOR and re-pick."
fi
python scripts/plot_basin_multiseed.py \
  --sweep-dir "results/e4_1/basin_${LAM_TAG}_seed0_noisy" "results/e4_1/basin_${LAM_TAG}_seed1_noisy" "results/e4_1/basin_${LAM_TAG}_seed2_noisy" \
  --baseline-prox "$BASE_PROX" --baseline-succ 0.85 --ci \
  --out "results/figs/${TASK}_${LAM_TAG}_3seed.png" \
  --title "${TASK}  fixed-${LAM_TAG}: graceful avoidance across 3 seeds" 2>/dev/null || echo "(figure step skipped)"
echo "== eval done. figure: results/figs/${TASK}_${LAM_TAG}_3seed.png =="
