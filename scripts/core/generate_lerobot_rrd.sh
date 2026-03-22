#!/usr/bin/env bash
set -euo pipefail

# Generate Rerun (.rrd) files for a local LeRobot v3 dataset.
#
# Usage:
#   bash scripts/generate_lerobot_rrd.sh <task> <eef|joint> [num_episodes]
#
# Examples:
#   bash scripts/generate_lerobot_rrd.sh put_rubbish_in_bin eef 4
#   bash scripts/generate_lerobot_rrd.sh put_rubbish_in_bin joint 4
#
# Output:
#   Writes one .rrd per episode into:
#     <dataset_root>/output/

TASK="${1:-}"
ACTION_SPACE="${2:-}"
NUM_EPISODES="${3:-4}"

if [[ -z "$TASK" || -z "$ACTION_SPACE" ]]; then
  echo "Usage: bash scripts/generate_lerobot_rrd.sh <task> <eef|joint> [num_episodes]" >&2
  exit 2
fi

if [[ "$ACTION_SPACE" != "eef" && "$ACTION_SPACE" != "joint" ]]; then
  echo "ERROR: action_space must be 'eef' or 'joint' (got: $ACTION_SPACE)" >&2
  exit 2
fi

if ! [[ "$NUM_EPISODES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: num_episodes must be a non-negative integer (got: $NUM_EPISODES)" >&2
  exit 2
fi

if [[ "$NUM_EPISODES" -lt 1 ]]; then
  echo "ERROR: num_episodes must be >= 1" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults (override via env vars if needed)
DATASETS_ROOT="${DATASETS_ROOT:-$REPO_ROOT/datasets/lerobot_without_prompt}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TOLERANCE_S="${TOLERANCE_S:-1e-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-0}"

DATASET_ID="${TASK}_${ACTION_SPACE}"
DATASET_ROOT="$DATASETS_ROOT/$DATASET_ID"
OUTPUT_DIR="$DATASET_ROOT/output"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset folder not found: $DATASET_ROOT" >&2
  if [[ -d "$DATASETS_ROOT" ]]; then
    echo "Available datasets matching '$TASK':" >&2
    ls -1 "$DATASETS_ROOT" 2>/dev/null | grep -E "^${TASK}_(eef|joint)$" >&2 || true
  fi
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# Prefer vendored lerobot sources.
export PYTHONPATH="$REPO_ROOT/external/lerobot/src:${PYTHONPATH:-}"

# Force CPU by default (rerun export doesn’t need GPU).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

VIZ_SCRIPT="$REPO_ROOT/external/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py"
if [[ ! -f "$VIZ_SCRIPT" ]]; then
  echo "ERROR: lerobot_dataset_viz.py not found: $VIZ_SCRIPT" >&2
  exit 2
fi

echo "Dataset:  $DATASET_ROOT"
echo "Output:   $OUTPUT_DIR"
echo "Episodes: 0..$((NUM_EPISODES-1))"

ep=0
while [[ "$ep" -lt "$NUM_EPISODES" ]]; do
  echo "[run] episode $ep"

  cmd=(
    python3 "$VIZ_SCRIPT"
    --repo-id "local/$DATASET_ID"
    --root "$DATASET_ROOT"
    --episode-index "$ep"
    --save 1
    --output-dir "$OUTPUT_DIR"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --tolerance-s "$TOLERANCE_S"
    --video-backend "$VIDEO_BACKEND"
  )

  if [[ "$DISPLAY_COMPRESSED_IMAGES" == "1" ]]; then
    cmd+=(--display-compressed-images)
  fi

  "${cmd[@]}"
  ep=$((ep + 1))
done

echo "Done. .rrd files are in: $OUTPUT_DIR"