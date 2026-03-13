#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/train_smolvla_1gpu.sh <DATASET_ID> [DATASET_ROOT] [-- <extra lerobot-train args...>]
#
# Examples:
#   # Local merged dataset
#   bash scripts/train_smolvla_1gpu.sh put_rubbish_in_bin_all datasets/lerobot_without_prompt/put_rubbish_in_bin_all
#
#   # Override defaults
#   BATCH_SIZE=64 NUM_STEPS=20000 bash scripts/train_smolvla_1gpu.sh put_rubbish_in_bin_all
#
# Environment variable overrides:
#   DATASET_ID, DATASET_ROOT, OUTPUT_DIR, POLICY_PATH, POLICY_DEVICE,
#   BATCH_SIZE, NUM_STEPS, SAVE_FREQ, LOG_FREQ, NUM_PROCESSES, NUM_WORKERS,
#   MIXED_PRECISION, VIDEO_BACKEND, WANDB_ENABLE, RENAME_MAP
# ---------------------------------------------------------------------------

DATASET_ID="${DATASET_ID:-${1:-}}"
if [[ -z "${DATASET_ID}" ]]; then
  echo "Usage: bash scripts/train_smolvla_1gpu.sh <DATASET_ID> [DATASET_ROOT] [-- <extra lerobot-train args...>]"
  exit 2
fi

DATASET_ROOT="${DATASET_ROOT:-${2:-datasets/lerobot/${DATASET_ID}}}"

# Forward remaining args (after the first 2 positional args). If you want to pass
# extra args, you can also do:  ... -- --wandb.enable=true
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 && "${1:-}" == "--" ]]; then shift; fi
EXTRA_TRAIN_ARGS=("$@")

OUTPUT_DIR="${OUTPUT_DIR:-output/lerobot/smolvla_${DATASET_ID}_$(date +%Y%m%d_%H%M%S)}"

POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_STEPS="${NUM_STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-50}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Good default on A100; set to "fp16" or "no" if needed.
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# If decoding is slow/broken on your node, try: VIDEO_BACKEND=torchvision_av
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# Configure Hugging Face cache locations (use project `models/hf_cache` by default).
# These can be overridden by exporting HF_HOME / TRANSFORMERS_CACHE / HF_DATASETS_CACHE
# / HUGGINGFACE_HUB_CACHE in the environment prior to running this script.
export HF_HOME="${HF_HOME:-models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"

# Map dataset visual keys to policy visual keys when they differ.
# Default maps RLBench-style cameras to SmolVLA camera slots.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.front_rgb\":\"observation.images.camera1\",\"observation.images.wrist_rgb\":\"observation.images.camera2\"}}"

echo "=========================================="
echo " SmolVLA Training"
echo "  DATASET_ID       : ${DATASET_ID}"
echo "  DATASET_ROOT     : ${DATASET_ROOT}"
echo "  POLICY_PATH      : ${POLICY_PATH}"
echo "  OUTPUT_DIR       : ${OUTPUT_DIR}"
echo "  BATCH_SIZE       : ${BATCH_SIZE}"
echo "  NUM_STEPS        : ${NUM_STEPS}"
echo "  NUM_PROCESSES    : ${NUM_PROCESSES}"
echo "  NUM_WORKERS      : ${NUM_WORKERS}"
echo "  MIXED_PRECISION  : ${MIXED_PRECISION}"
echo "  VIDEO_BACKEND    : ${VIDEO_BACKEND}"
echo "  WANDB_ENABLE     : ${WANDB_ENABLE}"
echo "  RENAME_MAP       : ${RENAME_MAP}"
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  echo "  EXTRA_TRAIN_ARGS : ${EXTRA_TRAIN_ARGS[*]}"
fi
echo "=========================================="

accelerate launch \
  --num_processes="${NUM_PROCESSES}" \
  --mixed_precision="${MIXED_PRECISION}" \
  "$(which lerobot-train)" \
  --output_dir="${OUTPUT_DIR}" \
  --save_checkpoint=true \
  --batch_size="${BATCH_SIZE}" \
  --steps="${NUM_STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --num_workers="${NUM_WORKERS}" \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${POLICY_DEVICE}" \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend="${VIDEO_BACKEND}" \
  --rename_map="${RENAME_MAP}" \
  --wandb.enable="${WANDB_ENABLE}" \
  "${EXTRA_TRAIN_ARGS[@]}"

echo "Saved outputs to: ${OUTPUT_DIR}"