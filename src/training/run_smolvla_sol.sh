#!/usr/bin/env bash
#SBATCH -A grp_hwei27
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -G a100:1
#SBATCH -p general
#SBATCH -q public
#SBATCH -J smolvla_lerobot
#SBATCH -o /scratch/kpham34/cross_model_learning_based_robot_control/output/slurm_log/slurm_smolvla_%j.out
#SBATCH -e /scratch/kpham34/cross_model_learning_based_robot_control/output/slurm_log/slurm_smolvla_%j.err
#SBATCH --export=NONE

set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   sbatch src/training/run_smolvla_sol.sh [-- <extra train args...>]
#   sbatch src/training/run_smolvla_sol.sh --dataset-kind eef
#   sbatch src/training/run_smolvla_sol.sh --dataset-kind joint
#   sbatch src/training/run_smolvla_sol.sh --dataset-id put_rubbish_in_bin_all
#
# Environment variable overrides:
#   TASK_NAME, DATASET_KIND, DATASET_ID, DATASET_ROOT
#   BATCH_SIZE, NUM_STEPS, SAVE_FREQ, LOG_FREQ, NUM_PROCESSES, NUM_WORKERS
#   MIXED_PRECISION, VIDEO_BACKEND, WANDB_ENABLE, POLICY_PATH, POLICY_DEVICE
# ---------------------------------------------------------------------------

REPO_ROOT="/scratch/kpham34/cross_model_learning_based_robot_control"

TASK_NAME="${TASK_NAME:-put_rubbish_in_bin}"
DATASET_KIND="${DATASET_KIND:-}"
DATASET_ID="${DATASET_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    eef|joint)
      if [[ -z "${DATASET_KIND}" ]]; then
        DATASET_KIND="$1"
      else
        EXTRA_TRAIN_ARGS+=("$1")
      fi
      shift
      ;;
    --dataset-kind|-k)
      DATASET_KIND="$2"
      shift 2
      ;;
    --dataset-id)
      DATASET_ID="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --task-name)
      TASK_NAME="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: sbatch src/training/run_smolvla_sol.sh [-- <extra train args...>]"
      echo ""
      echo "Options:"
      echo "  eef|joint              Dataset variant to train"
      echo "  --dataset-kind, -k     Explicitly set dataset variant (eef or joint)"
      echo "  --dataset-id           Override dataset id (default: put_rubbish_in_bin_all)"
      echo "  --dataset-root         Override dataset path"
      echo "  --task-name            Override task prefix (default: put_rubbish_in_bin)"
      echo "  --                     Forward remaining args to lerobot-train"
      echo ""
      echo "Examples:"
      echo "  sbatch src/training/run_smolvla_sol.sh"
      echo "  sbatch src/training/run_smolvla_sol.sh --dataset-kind eef"
      echo "  sbatch src/training/run_smolvla_sol.sh --dataset-kind joint"
      echo "  sbatch src/training/run_smolvla_sol.sh -- --steps 5000"
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${DATASET_KIND}" ]]; then
  case "${DATASET_KIND}" in
    eef|joint)
      ;;
    *)
      echo "Error: DATASET_KIND must be 'eef' or 'joint', got '${DATASET_KIND}'."
      exit 2
      ;;
  esac
fi

# Match default local command requested by user.
if [[ -z "${DATASET_ID}" ]]; then
  if [[ -n "${DATASET_KIND}" ]]; then
    DATASET_ID="${TASK_NAME}_${DATASET_KIND}"
  else
    DATASET_ID="put_rubbish_in_bin_all"
  fi
fi
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/lerobot/${DATASET_ID}}"

OUTPUT_DIR="${OUTPUT_DIR:-output/lerobot/smolvla_${DATASET_ID}_$(date +%Y%m%d_%H%M%S)}"
POLICY_PATH="${POLICY_PATH:-lerobot/smolvla_base}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

# Map dataset visual keys to policy visual keys when they differ.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.front_rgb\":\"observation.images.camera1\",\"observation.images.wrist_rgb\":\"observation.images.camera2\"}}"

# Default training parameters requested by user; caller can override via env vars.
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_STEPS="${NUM_STEPS:-20000}"
export SAVE_FREQ="${SAVE_FREQ:-4000}"
export LOG_FREQ="${LOG_FREQ:-50}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

# Configure Hugging Face cache locations (use project models/hf_cache by default).
export HF_HOME="${HF_HOME:-models/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Error: dataset directory not found: ${DATASET_ROOT}"
  exit 1
fi

echo "=== SOL JOB CONFIGURATION ==="
echo "REPO_ROOT        : ${REPO_ROOT}"
echo "TASK_NAME        : ${TASK_NAME}"
echo "DATASET_KIND     : ${DATASET_KIND:-<unset>}"
echo "DATASET_ID       : ${DATASET_ID}"
echo "DATASET_ROOT     : ${DATASET_ROOT}"
echo "OUTPUT_DIR       : ${OUTPUT_DIR}"
echo "POLICY_PATH      : ${POLICY_PATH}"
echo "POLICY_DEVICE    : ${POLICY_DEVICE}"
echo "BATCH_SIZE       : ${BATCH_SIZE}"
echo "NUM_STEPS        : ${NUM_STEPS}"
echo "SAVE_FREQ        : ${SAVE_FREQ}"
echo "LOG_FREQ         : ${LOG_FREQ}"
echo "NUM_PROCESSES    : ${NUM_PROCESSES}"
echo "NUM_WORKERS      : ${NUM_WORKERS}"
echo "MIXED_PRECISION  : ${MIXED_PRECISION}"
echo "VIDEO_BACKEND    : ${VIDEO_BACKEND}"
echo "WANDB_ENABLE     : ${WANDB_ENABLE}"
echo "RENAME_MAP       : ${RENAME_MAP}"
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  echo "EXTRA_TRAIN_ARGS : ${EXTRA_TRAIN_ARGS[*]}"
fi
echo "============================="

module load mamba/latest
source activate lerobot

cd "${REPO_ROOT}"

echo "Starting SmolVLA training on Sol..."
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
