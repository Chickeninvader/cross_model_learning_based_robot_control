#!/usr/bin/env bash
#SBATCH -A grp_yyang305
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -G a100:1
#SBATCH -p general
#SBATCH -q public
#SBATCH -J groot_lerobot
#SBATCH -o /scratch/kpham34/cross_model_learning_based_robot_control/output/slurm_log/slurm_groot_%j.out
#SBATCH -e /scratch/kpham34/cross_model_learning_based_robot_control/output/slurm_log/slurm_groot_%j.err
#SBATCH --export=NONE

set -euo pipefail

# ---------------------------------------------------------------------------
# Usage:
#   sbatch src/training/run_groot_sol.sh [-- <extra args ignored by wrapper>]
#   sbatch src/training/run_groot_sol.sh --dataset-kind eef
#   sbatch src/training/run_groot_sol.sh --dataset-kind joint
#   sbatch src/training/run_groot_sol.sh --dataset-id put_rubbish_in_bin_all
#
# Environment variable overrides:
#   TASK_NAME, DATASET_KIND, DATASET_ID, DATASET_ROOT
#   BATCH_SIZE, NUM_STEPS, SAVE_FREQ, LOG_FREQ, NUM_PROCESSES, NUM_WORKERS
#   TUNE_DIFFUSION_MODEL, TUNE_PROJECTOR, TUNE_VISUAL, TUNE_LLM
#   OUTPUT_DIR, JOB_NAME
# ---------------------------------------------------------------------------

REPO_ROOT="/scratch/kpham34/cross_model_learning_based_robot_control"

TASK_NAME="${TASK_NAME:-put_rubbish_in_bin}"
DATASET_KIND="${DATASET_KIND:-}"
DATASET_ID="${DATASET_ID:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
DEBUG_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    eef|joint)
      if [[ -z "${DATASET_KIND}" ]]; then
        DATASET_KIND="$1"
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
    --debug)
      DEBUG_MODE=1
      shift
      ;;
    --help|-h)
      echo "Usage: sbatch src/training/run_groot_sol.sh"
      echo ""
      echo "Options:"
      echo "  eef|joint              Dataset variant to train"
      echo "  --dataset-kind, -k     Explicitly set dataset variant (eef or joint)"
      echo "  --dataset-id           Override dataset id (default: put_rubbish_in_bin_all)"
      echo "  --dataset-root         Override dataset path"
      echo "  --task-name            Override task prefix (default: put_rubbish_in_bin)"
      echo "  --debug                Debug run (forces NUM_STEPS=2; keeps BATCH_SIZE unchanged)"
      echo ""
      echo "Examples:"
      echo "  sbatch src/training/run_groot_sol.sh"
      echo "  sbatch src/training/run_groot_sol.sh --dataset-kind eef"
      echo "  sbatch src/training/run_groot_sol.sh --dataset-kind joint"
      echo "  sbatch src/training/run_groot_sol.sh --dataset-id put_rubbish_in_bin_all"
      echo "  sbatch src/training/run_groot_sol.sh --debug"
      exit 0
      ;;
    --)
      # Positional extras are intentionally ignored for compatibility.
      break
      ;;
    *)
      # Ignore unknown args for compatibility with sbatch wrappers that pass-through args.
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

if [[ -z "${DATASET_ID}" ]]; then
  if [[ -n "${DATASET_KIND}" ]]; then
    DATASET_ID="${TASK_NAME}_${DATASET_KIND}"
  else
    DATASET_ID="put_rubbish_in_bin_all"
  fi
fi
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/lerobot/${DATASET_ID}}"

OUTPUT_DIR="${OUTPUT_DIR:-output/lerobot/groot_smoke_${DATASET_ID}_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-groot_smoke_${DATASET_ID}}"

# Practical defaults for non-smoke training. Caller can override via env vars.
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_STEPS="${NUM_STEPS:-20000}"
export SAVE_FREQ="${SAVE_FREQ:-4000}"
export LOG_FREQ="${LOG_FREQ:-50}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ "${DEBUG_MODE}" -eq 1 ]]; then
  # Debug mode only reduces step count; batch size remains unchanged.
  export NUM_STEPS=2
fi

# Policy tuning toggles.
export TUNE_DIFFUSION_MODEL="${TUNE_DIFFUSION_MODEL:-false}"
export TUNE_PROJECTOR="${TUNE_PROJECTOR:-true}"
export TUNE_VISUAL="${TUNE_VISUAL:-false}"
export TUNE_LLM="${TUNE_LLM:-false}"

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

echo "=== SOL JOB CONFIGURATION (GR00T) ==="
echo "REPO_ROOT            : ${REPO_ROOT}"
echo "TASK_NAME            : ${TASK_NAME}"
echo "DATASET_KIND         : ${DATASET_KIND:-<unset>}"
echo "DATASET_ID           : ${DATASET_ID}"
echo "DATASET_ROOT         : ${DATASET_ROOT}"
echo "OUTPUT_DIR           : ${OUTPUT_DIR}"
echo "JOB_NAME             : ${JOB_NAME}"
echo "BATCH_SIZE           : ${BATCH_SIZE}"
echo "NUM_STEPS            : ${NUM_STEPS}"
echo "SAVE_FREQ            : ${SAVE_FREQ}"
echo "LOG_FREQ             : ${LOG_FREQ}"
echo "NUM_PROCESSES        : ${NUM_PROCESSES}"
echo "NUM_WORKERS          : ${NUM_WORKERS}"
echo "TUNE_DIFFUSION_MODEL : ${TUNE_DIFFUSION_MODEL}"
echo "TUNE_PROJECTOR       : ${TUNE_PROJECTOR}"
echo "TUNE_VISUAL          : ${TUNE_VISUAL}"
echo "TUNE_LLM             : ${TUNE_LLM}"
echo "DEBUG_MODE           : ${DEBUG_MODE}"
if [[ "${DEBUG_MODE}" -eq 1 ]]; then
  echo "DEBUG_ESTIMATE       : ~30-90s for first step, ~3-10 min total to finish 2 steps"
fi
echo "======================================"

module load mamba/latest
source activate lerobot

cd "${REPO_ROOT}"

echo "Starting GR00T training on Sol..."
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
