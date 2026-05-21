#!/usr/bin/env bash
# Staged human-curriculum re-validation of the CQN-AS base policy on
# saucepan_to_hob, with the 2026-05-20 reward/critic-support fix applied.
#
# Why this script exists: the 50k single-stage validation produced a degenerate
# "retreat from human" policy. Root cause = the dense workspace penalty's
# discounted return blew past the C51 critic support [-2,2], saturating value
# learning (full writeup: docs/phase3_base_validation_findings.md). The fix is
# four levers, all applied here:
#   (1) bounded penalty:  workspace_beta=0.05, workspace_excess_cap=1.0
#   (2) widened support:  agent.v_min=-6 agent.v_max=2 agent.atoms=101
#   (3) more demos:       num_demos=36
#   (4) human curriculum: 3 stages via snapshot-resume (the env is stateless
#       w.r.t. training step, so no within-run ramp — we stage it instead):
#         stage 0  disruption=null          (no human; "can it learn the task?")
#         stage 1  disruption=coworker_easy  (gentle coworker)
#         stage 2  disruption=coworker_train (full coworker)
#
# Each stage resumes the previous stage's final snapshot via +snapshot_path=...
# (train_cqn_as.py load_snapshot). The env is rebuilt per stage, so the new
# disruption config takes effect with no code change.
#
# NOT launched from inside the agent — a human runs this on the GPU box.
#
# Prereqs:
#   export AMASS_DATA_DIR=/path/to/CMU/CMU
#   export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0   # headless GPU
#   cd safety_bigym && source venv/bin/activate    # needs tensordict
#
# Usage:
#   scripts/run_base_curriculum.sh                 # full run (stages 0->1->2)
#   SMOKE=1 scripts/run_base_curriculum.sh         # ≤2000-frame stage-0 smoke only
#   STAGE0_FRAMES=30000 STAGE1_FRAMES=30000 STAGE2_FRAMES=40000 \
#       scripts/run_base_curriculum.sh             # override per-stage budgets
#
#   # Resume ONLY stage 2 (e.g. after a machine crash) — skips stages 0/1 and
#   # restarts stage 2 from the newest snapshot it can find. Point RESUME_DIR at
#   # the prior run dir; the resumed run writes to <RESUME_DIR>/stage2_full_resume.
#   RESUME_STAGE2=1 RESUME_DIR=exp_local/cqn_as_base_curriculum/<run_tag> \
#       scripts/run_base_curriculum.sh
#   # ...or pass an explicit checkpoint:
#   RESUME_STAGE2=1 RESUME_DIR=<dir> RESUME_SNAPSHOT=<path.pt> \
#       scripts/run_base_curriculum.sh

set -euo pipefail

if [[ -z "${AMASS_DATA_DIR:-}" ]]; then
  echo "ERROR: export AMASS_DATA_DIR before running (see CLAUDE.md)." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-base_curriculum_$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-${REPO_ROOT}/exp_local/cqn_as_base_curriculum/${RUN_TAG}}"
mkdir -p "${OUTDIR}"

# Shared overrides — the reward/support fix (levers 1-3) + cadence/logging.
COMMON=(
  env=safety_bigym/saucepan_to_hob
  bodyslam=oracle
  num_demos=36
  env.safety.add_workspace_penalty=true
  env.safety.workspace_beta=0.05
  env.safety.workspace_excess_cap=1.0
  agent.v_min=-6.0
  agent.v_max=2.0
  agent.atoms=101
  save_snapshot=true
  save_video=true
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  STAGE0_FRAMES="${STAGE0_FRAMES:-2000}"
  STAGE1_FRAMES=0
  STAGE2_FRAMES=0
  WANDB=(wandb.use=false)
  # Synchronous CUDA so any device-side assert points at the real op/line
  # (smoke is short; the slowdown is irrelevant here).
  export CUDA_LAUNCH_BLOCKING=1
else
  STAGE0_FRAMES="${STAGE0_FRAMES:-30000}"
  STAGE1_FRAMES="${STAGE1_FRAMES:-30000}"
  STAGE2_FRAMES="${STAGE2_FRAMES:-40000}"
  WANDB=(wandb.use=true)
fi

# Latest snapshot (by mtime) in a stage dir; the final-state save at loop exit
# is the newest snapshot_<step>.pt.
latest_snapshot() {
  ls -t "$1"/snapshot_*.pt 2>/dev/null | head -1
}

run_stage() {
  local name="$1" disruption="$2" frames="$3" resume_from="${4:-}"
  local stage_dir="${OUTDIR}/${name}"
  if [[ "${frames}" -le 0 ]]; then
    echo "== skip ${name} (frames=${frames}) =="
    return
  fi
  echo "== ${name}: disruption=${disruption} frames=${frames} -> ${stage_dir} =="
  local extra=()
  [[ -n "${resume_from}" ]] && extra+=("+snapshot_path=${resume_from}")
  python train_cqn_as.py \
    "${COMMON[@]}" "${WANDB[@]}" \
    "disruption=${disruption}" \
    num_train_frames="${frames}" \
    "hydra.run.dir=${stage_dir}" \
    "wandb.name=${RUN_TAG}_${name}" \
    "${extra[@]}"
}

# --- Resume-only path: rerun stage 2 from the newest available snapshot ---
if [[ "${RESUME_STAGE2:-0}" == "1" ]]; then
  if [[ -z "${RESUME_DIR:-}" ]]; then
    echo "ERROR: RESUME_STAGE2=1 requires RESUME_DIR=<prior run dir>." >&2
    exit 1
  fi
  OUTDIR="${RESUME_DIR}"
  RUN_TAG="${RUN_TAG:-$(basename "${OUTDIR}")}"
  # Explicit snapshot wins; else newest from a prior stage-2 (resume or not),
  # else fall back to stage 1's final snapshot.
  SNAP="${RESUME_SNAPSHOT:-}"
  if [[ -z "${SNAP}" ]]; then
    SNAP="$(ls -t "${OUTDIR}"/stage2_full*/snapshot_*.pt 2>/dev/null | head -1)"
  fi
  [[ -z "${SNAP}" ]] && SNAP="$(latest_snapshot "${OUTDIR}/stage1_easy")"
  if [[ -z "${SNAP}" ]]; then
    echo "ERROR: no snapshot found under ${OUTDIR} (stage2_full*/ or stage1_easy/)." >&2
    exit 1
  fi
  echo "== resuming stage 2 from: ${SNAP} =="
  run_stage stage2_full_resume coworker_train "${STAGE2_FRAMES}" "${SNAP}"
  echo "== stage 2 resume complete. snapshots under ${OUTDIR}/stage2_full_resume =="
  exit 0
fi

# Stage 0: human present but idle/distant (no interference; keeps obs width and
# model architecture identical to stages 1-2 so snapshots resume cleanly). This
# is the GATE — episode_reward must climb >0 on some episodes with returns inside
# [-6, 2]. If it fails, the problem is demos/CQN-AS, not safety: stop and reassess.
run_stage stage0_idle coworker_idle "${STAGE0_FRAMES}"
SNAP0="$(latest_snapshot "${OUTDIR}/stage0_idle")"

run_stage stage1_easy coworker_easy "${STAGE1_FRAMES}" "${SNAP0}"
SNAP1="$(latest_snapshot "${OUTDIR}/stage1_easy")"

run_stage stage2_full coworker_train "${STAGE2_FRAMES}" "${SNAP1}"

echo "== curriculum complete. snapshots under ${OUTDIR} =="
