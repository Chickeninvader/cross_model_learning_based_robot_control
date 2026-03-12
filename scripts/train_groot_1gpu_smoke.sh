#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/train_groot_1gpu_smoke.sh <DATASET_ID> [DATASET_ROOT]
#
# Examples:
#   # Single variation (dataset under datasets/lerobot/)
#   bash scripts/train_groot_1gpu_smoke.sh stack_cups_variation1
#
#   # Merged all-variations dataset
#   bash scripts/train_groot_1gpu_smoke.sh put_rubbish_in_bin_all datasets/lerobot_merged/put_rubbish_in_bin_all
#
#   # Absolute path to dataset on HPC
#   bash scripts/train_groot_1gpu_smoke.sh put_rubbish_in_bin_all /scratch/kpham34/cross_model_learning_based_robot_control/datasets/lerobot_merged/put_rubbish_in_bin_all
#
# Environment variable overrides (take precedence over positional args):
#   DATASET_ID, DATASET_ROOT, OUTPUT_DIR, JOB_NAME, BATCH_SIZE,
#   NUM_STEPS, SAVE_FREQ, LOG_FREQ, NUM_PROCESSES
# ---------------------------------------------------------------------------

# Positional args with env-var fallback
DATASET_ID="${DATASET_ID:-${1:-stack_cups_variation1}}"
DATASET_ROOT="${DATASET_ROOT:-${2:-datasets/lerobot/${DATASET_ID}}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/lerobot/groot_smoke_${DATASET_ID}_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-groot_smoke_${DATASET_ID}}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
LOG_FREQ="${LOG_FREQ:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

echo "=========================================="
echo " GR00T Training"
echo "  DATASET_ID   : ${DATASET_ID}"
echo "  DATASET_ROOT : ${DATASET_ROOT}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  BATCH_SIZE   : ${BATCH_SIZE}"
echo "  NUM_STEPS    : ${NUM_STEPS}"
echo "  NUM_PROCESSES: ${NUM_PROCESSES}"
echo "=========================================="

accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  "$(which lerobot-train)" \
  --output_dir="${OUTPUT_DIR}" \
  --save_checkpoint=true \
  --batch_size="${BATCH_SIZE}" \
  --steps="${NUM_STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --policy.type=groot \
  --policy.push_to_hub=false \
  --policy.tune_diffusion_model=false \
  --dataset.repo_id="${DATASET_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --wandb.enable=false \
  --job_name="${JOB_NAME}"

echo "Saved outputs to: ${OUTPUT_DIR}"
