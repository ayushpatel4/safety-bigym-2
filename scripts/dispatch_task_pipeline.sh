#!/usr/bin/env bash
# dispatch_task_pipeline.sh — end-to-end policy pipeline for ONE task, inside a GPU
# "lane", with the phases overlapped:
#   Phase 1  base curriculum (serial stage0→stage1→stage2, 1 GPU)   [+ optional no-shaping]
#   Phase 2  when stage1_easy lands → Lagrangian fixed-λ ×3 seeds (pools the lane; the
#            gpu-busy guard skips the GPU still running stage2, so it overlaps)
#   Phase 3  when base stage2 + all λ cells finish → run_task_eval.sh (sweep→pick→pool→figure)
#
# Run TWO tasks at once by giving disjoint lanes:
#   TASK=drawers_open_all GPUS="0 1 2" nohup bash scripts/dispatch_task_pipeline.sh >logs/pipe_drawers.out 2>&1 &
#   TASK=dishwasher_close GPUS="3 4 5" nohup bash scripts/dispatch_task_pipeline.sh >logs/pipe_dish.out   2>&1 &
#
# GPU budget per lane: 1 (base) for most of ~1 day, peaking at min(lane,4) during the
# stage2⊕Lagrangian overlap. A 3-GPU lane works (the 3rd λ-seed waits for stage2 to
# free its GPU); a 4-GPU lane runs all 3 λ-seeds during the overlap. NO_SHAPING=1 adds
# a 2nd base curriculum (needs a ≥5-GPU lane or it starves the Lagrangian).
#
# Inputs (env): TASK, GPUS, LAMBDA(0.1), NO_SHAPING(0), POLL(120s), TIMEOUT_H(48),
#               FRAMES(40000 λ-cells), SUCCESS_FLOOR(0.75).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

TASK="${TASK:-drawers_open_all}"
GPUS="${GPUS:-0 1 2 3 4 5}"; read -r -a GPU <<<"$GPUS"
LAMBDA="${LAMBDA:-0.1}"; LAM_TAG="lam${LAMBDA//./p}"
NO_SHAPING="${NO_SHAPING:-0}"
POLL="${POLL:-120}"; TIMEOUT_H="${TIMEOUT_H:-48}"
FRAMES="${FRAMES:-40000}"; SUCCESS_FLOOR="${SUCCESS_FLOOR:-0.75}"
LOGDIR="$REPO_ROOT/logs/pipe_${TASK}"; mkdir -p "$LOGDIR"
BASE_OUT="$REPO_ROOT/exp_local/cqn_as_base_curriculum/pipe_base_${TASK}"
NOWS_OUT="$REPO_ROOT/exp_local/cqn_as_base_curriculum/pipe_base_${TASK}_nows"
LAG_OUT="$REPO_ROOT/exp_local/fixed_lambda/${TASK}_fixlam${LAMBDA//./p}"
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not on PATH" >&2; exit 1; }

wait_for() {  # wait_for <file> <label> ; honours TIMEOUT_H
  local f="$1" label="$2" waited=0 max=$((TIMEOUT_H*3600))
  echo "[$(date +%H:%M)] waiting for $label : $f"
  until [[ -f "$f" ]]; do
    sleep "$POLL"; waited=$((waited+POLL))
    if (( waited > max )); then echo "FATAL: timed out (${TIMEOUT_H}h) waiting for $label" >&2; exit 1; fi
  done
  echo "[$(date +%H:%M)] ready: $label"
}
newest_snap() { ls -t "$1"/snapshot_[0-9]*.pt 2>/dev/null | head -1; }

echo "== pipeline | TASK=$TASK lane='$GPUS' λ=$LAMBDA no_shaping=$NO_SHAPING =="
echo "   base=$BASE_OUT"; echo "   lag =$LAG_OUT"; echo "   logs=$LOGDIR"

# ---- Phase 1: base curriculum (serial) on the lane's first GPU ----------------
echo "[$(date +%H:%M)] Phase 1: base curriculum on gpu ${GPU[0]}"
TASK="$TASK" OUTDIR="$BASE_OUT" CUDA_VISIBLE_DEVICES="${GPU[0]}" \
  nohup bash scripts/run_base_curriculum.sh >"$LOGDIR/base.log" 2>&1 &
if [[ "$NO_SHAPING" == "1" && ${#GPU[@]} -ge 2 ]]; then
  echo "[$(date +%H:%M)] + no-shaping base on gpu ${GPU[1]}"
  TASK="$TASK" WORKSPACE_PENALTY=0 OUTDIR="$NOWS_OUT" CUDA_VISIBLE_DEVICES="${GPU[1]}" \
    nohup bash scripts/run_base_curriculum.sh >"$LOGDIR/base_nows.log" 2>&1 &
fi

# ---- Phase 2: when stage1 lands, launch the Lagrangian (overlaps stage2) -------
wait_for "$BASE_OUT/stage1_easy/final_metrics.json" "base stage1_easy"
STAGE1="$(newest_snap "$BASE_OUT/stage1_easy")"
[[ -n "$STAGE1" ]] || { echo "FATAL: no stage1 snapshot under $BASE_OUT/stage1_easy" >&2; exit 1; }
echo "[$(date +%H:%M)] Phase 2: Lagrangian λ=$LAMBDA ×3 (warmstart $STAGE1)"
TASK="$TASK" LAMBDAS="$LAMBDA" SEEDS="0 1 2" FRAMES="$FRAMES" WARMSTART="$STAGE1" \
  OUTDIR="$LAG_OUT" RUN_TAG="$(basename "$LAG_OUT")" GPUS="$GPUS" POLL="$POLL" \
  nohup bash scripts/dispatch_fixed_lambda.sh >"$LOGDIR/lagrangian.log" 2>&1 &
LAG_PID=$!

# ---- Phase 3: when base stage2 + all λ cells are done, run the eval ------------
wait_for "$BASE_OUT/stage2_full/final_metrics.json" "base stage2_full (row-1 baseline)"
for s in 0 1 2; do wait_for "$LAG_OUT/${LAM_TAG}_seed${s}/final_metrics.json" "λ seed$s"; done
wait "$LAG_PID" 2>/dev/null || true
BASELINE="$(newest_snap "$BASE_OUT/stage2_full")"
echo "[$(date +%H:%M)] Phase 3: eval (baseline=$BASELINE)"
TASK="$TASK" FIXLAM_DIR="$LAG_OUT" LAM_TAG="$LAM_TAG" BASELINE="$BASELINE" \
  GPUS="$GPUS" SUCCESS_FLOOR="$SUCCESS_FLOOR" bash scripts/run_task_eval.sh

echo "== pipeline DONE for $TASK. figure: results/figs/${TASK}_${LAM_TAG}_3seed.png =="
