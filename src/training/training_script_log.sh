#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all_for_task.sh put_rubbish_in_bin
#   ./run_all_for_task.sh put_rubbish_in_bin --debug
#
# Debug mode runs jobs directly (no sbatch) with very small settings:
#   NUM_STEPS=1, BATCH_SIZE=2, SAVE_FREQ=1, LOG_FREQ=1, NUM_WORKERS=1
# You can override these by exporting env vars before running this script.
TASK_NAME=""
DEBUG_MODE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug|-d)
      DEBUG_MODE="true"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 <task_name> [--debug]"
      echo ""
      echo "Options:"
      echo "  --debug, -d    Run locally for sanity check (no sbatch) with tiny train config"
      exit 0
      ;;
    *)
      if [[ -z "${TASK_NAME}" ]]; then
        TASK_NAME="$1"
        shift
      else
        echo "Error: unexpected argument '$1'"
        echo "Usage: $0 <task_name> [--debug]"
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${TASK_NAME}" ]]; then
  echo "Error: please provide task name, e.g. lamp_on"
  echo "Usage: $0 <task_name> [--debug]"
  exit 2
fi

DATASET_BASE="datasets/lerobot"

for MODEL in smolvla groot; do
  for MODE in eef joint; do
    DATASET_ID="${TASK_NAME}_${MODE}"
    DATASET_ROOT="${DATASET_BASE}/${DATASET_ID}"
    SCRIPT="./src/training/run_${MODEL}_sol.sh"

    if [[ ! -d "$DATASET_ROOT" ]]; then
      echo "Skip: missing dataset folder: $DATASET_ROOT"
      continue
    fi

    if [[ "${DEBUG_MODE}" == "true" ]]; then
      DEBUG_TS="$(date +%Y%m%d_%H%M%S)"
      DEBUG_OUTPUT_DIR="output/lerobot/debug/${MODEL}_${DATASET_ID}_${DEBUG_TS}"
      DEBUG_JOB_NAME="debug_${MODEL}_${DATASET_ID}"

      echo "[DEBUG] Running locally: $SCRIPT $MODE --dataset-id $DATASET_ID --dataset-root $DATASET_ROOT --task-name $TASK_NAME"
      echo "[DEBUG] OUTPUT_DIR=${DEBUG_OUTPUT_DIR}"
      NUM_STEPS="${NUM_STEPS:-1}" \
      BATCH_SIZE="${BATCH_SIZE:-2}" \
      SAVE_FREQ="${SAVE_FREQ:-1}" \
      LOG_FREQ="${LOG_FREQ:-1}" \
      NUM_WORKERS="${NUM_WORKERS:-1}" \
      WANDB_ENABLE="${WANDB_ENABLE:-false}" \
      OUTPUT_DIR="${OUTPUT_DIR:-${DEBUG_OUTPUT_DIR}}" \
      JOB_NAME="${JOB_NAME:-${DEBUG_JOB_NAME}}" \
      bash "$SCRIPT" "$MODE" \
        --dataset-id "$DATASET_ID" \
        --dataset-root "$DATASET_ROOT" \
        --task-name "$TASK_NAME"
    else
      echo "Submitting: $SCRIPT $MODE --dataset-id $DATASET_ID --dataset-root $DATASET_ROOT --task-name $TASK_NAME"
      sbatch "$SCRIPT" "$MODE" \
        --dataset-id "$DATASET_ID" \
        --dataset-root "$DATASET_ROOT" \
        --task-name "$TASK_NAME"
    fi
  done
done