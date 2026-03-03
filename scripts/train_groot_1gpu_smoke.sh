#!/usr/bin/env bash
set -euo pipefail

DATASET_ID="${DATASET_ID:-stack_cups_variation0}"
DATASET_ROOT="${DATASET_ROOT:-datasets/lerobot_groot/${DATASET_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-output/lerobot/groot_smoke_${DATASET_ID}_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-groot_smoke_1step}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-1}"
LOG_FREQ="${LOG_FREQ:-1}"
DATASET_EPISODES="${DATASET_EPISODES:-[0,1,2]}"

accelerate launch \
  --num_processes=1 \
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
  --dataset.episodes="${DATASET_EPISODES}" \
  --dataset.use_imagenet_stats=false \
  --wandb.enable=false \
  --job_name="${JOB_NAME}"

echo "Saved outputs to: ${OUTPUT_DIR}"
