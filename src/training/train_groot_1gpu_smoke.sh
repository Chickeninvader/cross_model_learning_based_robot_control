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
#   NUM_STEPS, SAVE_FREQ, LOG_FREQ, NUM_PROCESSES, NUM_WORKERS,
#   TUNE_DIFFUSION_MODEL, TUNE_PROJECTOR, TUNE_VISUAL, TUNE_LLM
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
NUM_WORKERS="${NUM_WORKERS:-4}"

# Configure Hugging Face cache locations (use project `models/hf_cache` by default).
# These can be overridden by exporting HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE
# / HUGGINGFACE_HUB_CACHE in the environment prior to running this script.
export HF_HOME="${HF_HOME:-models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

# Policy tuning toggles
TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-false}"
TUNE_PROJECTOR="${TUNE_PROJECTOR:-true}"
TUNE_VISUAL="${TUNE_VISUAL:-false}"
TUNE_LLM="${TUNE_LLM:-false}"

echo "=========================================="
echo " GR00T Training"
echo "  DATASET_ID   : ${DATASET_ID}"
echo "  DATASET_ROOT : ${DATASET_ROOT}"
echo "  OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "  BATCH_SIZE   : ${BATCH_SIZE}"
echo "  NUM_STEPS    : ${NUM_STEPS}"
echo "  NUM_PROCESSES: ${NUM_PROCESSES}"
echo "  NUM_WORKERS  : ${NUM_WORKERS}"
echo "  TUNE_DIFFUSION_MODEL: ${TUNE_DIFFUSION_MODEL}"
echo "  TUNE_PROJECTOR      : ${TUNE_PROJECTOR}"
echo "  TUNE_VISUAL         : ${TUNE_VISUAL}"
echo "  TUNE_LLM            : ${TUNE_LLM}"
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
  --num_workers="${NUM_WORKERS}" \
  --policy.type=groot \
  --policy.push_to_hub=false \
  --policy.tune_diffusion_model="${TUNE_DIFFUSION_MODEL}" \
  --policy.tune_projector="${TUNE_PROJECTOR}" \
  --policy.tune_visual="${TUNE_VISUAL}" \
  --policy.tune_llm="${TUNE_LLM}" \
  --dataset.repo_id="${DATASET_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.use_imagenet_stats=false \
  --dataset.video_backend=pyav \
  --wandb.enable=false \
  --job_name="${JOB_NAME}"

echo "Saved outputs to: ${OUTPUT_DIR}"
