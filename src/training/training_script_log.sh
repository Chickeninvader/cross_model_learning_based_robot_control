#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_for_task.sh put_rubbish_in_bin
#   ./run_all_for_task.sh put_rubbish_in_bin --debug
#   ./run_all_for_task.sh put_rubbish_in_bin --dataset-base /path/to/datasets
#
# Debug mode runs jobs directly (no sbatch) and forwards --debug to model scripts.
# In debug mode:
#   - NUM_STEPS is forced to 2 inside each run_*_sol.sh script
#   - BATCH_SIZE is kept as normal default (64 unless you override env)
#   - Typical timing is startup-dominated (see per-model estimate printed below)
TASK_NAME=""
DEBUG_MODE="false"
DATASET_BASE="${DATASET_BASE:-datasets/lerobot_trial_2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug|-d)
      DEBUG_MODE="true"
      shift
      ;;
    --dataset-base|-p)
      DATASET_BASE="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 <task_name> [--debug] [--dataset-base <path>]"
      echo ""
      echo "Options:"
      echo "  --debug, -d    Run locally for sanity check (no sbatch) with tiny train config"
      echo "  --dataset-base, -p"
      echo "                 Base dataset directory that contains <task>_eef and <task>_joint"
      echo "                 (default: datasets/lerobot_trial_2, or \$DATASET_BASE if set)"
      exit 0
      ;;
    *)
      if [[ -z "${TASK_NAME}" ]]; then
        TASK_NAME="$1"
        shift
      else
        echo "Error: unexpected argument '$1'"
        echo "Usage: $0 <task_name> [--debug] [--dataset-base <path>]"
        exit 2
      fi
      ;;
  esac
done


if [[ -z "${TASK_NAME}" ]]; then
  echo "Error: please provide task name, e.g. lamp_on"
  echo "Usage: $0 <task_name> [--debug] [--dataset-base <path>]"
  exit 2
fi

echo "=== JOB SUBMISSION PLAN ==="
echo "TASK_NAME    : ${TASK_NAME}"
echo "DATASET_BASE : ${DATASET_BASE}"
echo "DEBUG_MODE   : ${DEBUG_MODE}"
echo "==========================="

for MODEL in smolvla groot; do
  for MODE in eef joint; do
    DATASET_ID="${TASK_NAME}_${MODE}"
    DATASET_ROOT="${DATASET_BASE}/${DATASET_ID}"
    SCRIPT="./src/training/run_${MODEL}_sol.sh"

    if [[ ! -d "$DATASET_ROOT" ]]; then
      echo "Skip: missing dataset folder: $DATASET_ROOT"
      continue
    fi

    JOB_ARGS=(
      "$MODE"
      --dataset-id "$DATASET_ID"
      --dataset-root "$DATASET_ROOT"
      --task-name "$TASK_NAME"
    )

    if [[ "${DEBUG_MODE}" == "true" ]]; then
      DEBUG_TS="$(date +%Y%m%d_%H%M%S)"
      DEBUG_OUTPUT_DIR="output/lerobot/debug/${MODEL}_${DATASET_ID}_${DEBUG_TS}"
      DEBUG_JOB_NAME="debug_${MODEL}_${DATASET_ID}"

      if [[ "${MODEL}" == "smolvla" ]]; then
        echo "[DEBUG] Estimated timing (${MODEL}): ~20-60s first step, ~2-8 min total (2 steps)"
      else
        echo "[DEBUG] Estimated timing (${MODEL}): ~30-90s first step, ~3-10 min total (2 steps)"
      fi
      printf '[DEBUG] Command     : bash %q' "$SCRIPT"
      printf ' %q' "${JOB_ARGS[@]}"
      printf ' %q\n' "--debug"
      echo "[DEBUG] OUTPUT_DIR=${DEBUG_OUTPUT_DIR}"
      BATCH_SIZE="${BATCH_SIZE:-64}" \
      SAVE_FREQ="${SAVE_FREQ:-1}" \
      LOG_FREQ="${LOG_FREQ:-1}" \
      NUM_WORKERS="${NUM_WORKERS:-1}" \
      WANDB_ENABLE="${WANDB_ENABLE:-false}" \
      OUTPUT_DIR="${OUTPUT_DIR:-${DEBUG_OUTPUT_DIR}}" \
      JOB_NAME="${JOB_NAME:-${DEBUG_JOB_NAME}}" \
      bash "$SCRIPT" "${JOB_ARGS[@]}" --debug
    else
      printf '[SUBMIT] Command    : sbatch %q' "$SCRIPT"
      printf ' %q' "${JOB_ARGS[@]}"
      printf '\n'
      sbatch "$SCRIPT" "${JOB_ARGS[@]}"
    fi
  done
done