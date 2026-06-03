#!/usr/bin/env bash
# scripts/dispatch_fixed_lambda.sh
# GPU-pool dispatcher for the fixed-lambda Lagrangian cells (the d=0.3 PID
# instability fix). Launches one cell (lambda x seed) per FREE GPU, only onto
# GPUs with no running compute, and picks GPUs up as they free. Idempotent:
# skips cells with final_metrics.json, skips in-flight cells, re-queues a crashed
# cell up to MAX_RETRY times. Mirrors dispatch_p3p4_pool.sh.
#
# Run under tmux/nohup; cells are nohup'd so they survive the dispatcher.
#
# Usage:
#   WARMSTART=<stage1.pt> bash scripts/dispatch_fixed_lambda.sh              # lam 0.27 x seeds 0,1,2
#   LAMBDAS="0.2 0.27 0.35" SEEDS="0 1 2" GPUS="2 3 4 5" WARMSTART=... bash scripts/dispatch_fixed_lambda.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"

GPUS="${GPUS:-0 1 2 3 4 5}"
POLL="${POLL:-30}"
FRAMES="${FRAMES:-40000}"
MAX_RETRY="${MAX_RETRY:-1}"
LAMBDAS="${LAMBDAS:-0.27}"
SEEDS="${SEEDS:-0 1 2}"
RUN_TAG="${RUN_TAG:-fixlam}"
: "${WARMSTART:?Set WARMSTART to the P1 stage-1 snapshot (.pt)}"
[[ -f "$WARMSTART" ]] || { echo "FATAL: WARMSTART=$WARMSTART not found (cwd $(pwd))" >&2; exit 1; }
WARMSTART="$(cd "$(dirname "$WARMSTART")" && pwd)/$(basename "$WARMSTART")"   # absolute (Hydra chdir-safe)
OUTDIR="${OUTDIR:-$REPO_ROOT/exp_local/fixed_lambda/$RUN_TAG}"; mkdir -p "$OUTDIR"
LOGDIR="$REPO_ROOT/logs/dispatch_fixed_lambda"; mkdir -p "$LOGDIR"
command -v nvidia-smi >/dev/null || { echo "FATAL: nvidia-smi not on PATH" >&2; exit 1; }

# Cells = "lambda:seed".
QUEUE=(); for L in $LAMBDAS; do for s in $SEEDS; do QUEUE+=("$L:$s"); done; done
declare -A ATTEMPTS PID_ON CELL_ON

cell_dir()     { local L="${1%%:*}" s="${1##*:}"; echo "$OUTDIR/lam${L//./p}_seed${s}"; }
cell_done()    { [[ -f "$(cell_dir "$1")/final_metrics.json" ]]; }
cell_running() { pgrep -f "hydra.run.dir=$(cell_dir "$1") " >/dev/null 2>&1; }
gpu_busy()     { local o; o="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$1" 2>/dev/null)"; [[ -n "${o//[$' \t\n']/}" ]]; }

launch_cell() { local g="$1" c="$2"; local L="${c%%:*}" s="${c##*:}"
  CUDA_VISIBLE_DEVICES="$g" LAMBDAS="$L" SEEDS="$s" FRAMES="$FRAMES" \
    WARMSTART="$WARMSTART" OUTDIR="$OUTDIR" RUN_TAG="$RUN_TAG" \
    nohup bash scripts/run_fixed_lambda.sh >"$LOGDIR/lam${L//./p}_seed${s}.log" 2>&1 &
  echo "$!"; }

echo "== fixed-lambda pool | GPUS='$GPUS' POLL=${POLL}s frames=$FRAMES =="
echo "   lambdas='$LAMBDAS' seeds='$SEEDS' -> $OUTDIR  (${#QUEUE[@]} cells; logs $LOGDIR)"
echo "   warmstart=$WARMSTART"

while :; do
  # 1) reap finished assignments (verify completion; retry crashes)
  for g in $GPUS; do
    pid="${PID_ON[$g]:-}"; [[ -z "$pid" ]] && continue
    kill -0 "$pid" 2>/dev/null && continue
    c="${CELL_ON[$g]}"
    if cell_done "$c"; then echo "[$(date +%H:%M)] gpu$g  DONE  $c"
    else a="${ATTEMPTS[$c]:-0}"
      if (( a < MAX_RETRY )); then ATTEMPTS[$c]=$((a+1)); QUEUE+=("$c")
        echo "[$(date +%H:%M)] gpu$g  FAIL  $c -> requeued (retry $((a+1))/$MAX_RETRY)"
      else echo "[$(date +%H:%M)] gpu$g  FAIL  $c -> giving up (see $LOGDIR)"; fi
    fi
    unset 'PID_ON[$g]' 'CELL_ON[$g]'
  done
  # 2) prune already-finished cells
  if (( ${#QUEUE[@]} )); then nq=(); for c in "${QUEUE[@]}"; do cell_done "$c" || nq+=("$c"); done; QUEUE=("${nq[@]}"); fi
  # 3) assign queued cells to free GPUs
  for g in $GPUS; do
    [[ -n "${PID_ON[$g]:-}" ]] && continue   # we already run a cell here
    gpu_busy "$g" && continue                 # external proc -> wait for it to free
    (( ${#QUEUE[@]} )) || continue
    idx=-1; for i in "${!QUEUE[@]}"; do c="${QUEUE[$i]}"; cell_done "$c" && continue; cell_running "$c" && continue; idx="$i"; break; done
    [[ "$idx" -lt 0 ]] && continue
    c="${QUEUE[$idx]}"; np="$(launch_cell "$g" "$c")"; PID_ON[$g]="$np"; CELL_ON[$g]="$c"
    echo "[$(date +%H:%M)] gpu$g  <-    $c  (pid $np)"
    unset 'QUEUE[$idx]'; QUEUE=("${QUEUE[@]}"); sleep 5
  done
  # 4) done when queue empty AND nothing we launched is still running
  act=0; for g in $GPUS; do [[ -n "${PID_ON[$g]:-}" ]] && act=$((act+1)); done
  (( ${#QUEUE[@]} == 0 && act == 0 )) && { echo "== all fixed-lambda cells complete =="; break; }
  sleep "$POLL"
done
echo "== DONE. next: per-seed basin sweep -> pick -> pool =="
echo "   for s in $SEEDS; do STAGE_DIR=$OUTDIR/lam${LAMBDAS// /_}_seedN OBS=noisy bash scripts/run_basin_sweep.sh; done"
