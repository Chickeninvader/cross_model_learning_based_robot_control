#!/usr/bin/env bash
set -euo pipefail

# Generate Rerun (.rrd) files for a local LeRobot v3 dataset.
#
# Usage:
#   bash scripts/core/generate_lerobot_rrd.sh --dataset-path <path> [--start-episode N] [--num-episodes K]
#
# Examples:
#   bash scripts/core/generate_lerobot_rrd.sh \
#     --dataset-path datasets/lerobot_trial_2/push_button_eef \
#     --start-episode 0 \
#     --num-episodes 4
#
# Output:
#   Writes one .rrd per episode into:
#     <dataset_root>/output/

DATASET_PATH=""
START_EPISODE="0"
NUM_EPISODES="4"

usage() {
  cat <<'EOF'
Generate Rerun (.rrd) files for a local LeRobot v3 dataset.

Usage:
  bash scripts/core/generate_lerobot_rrd.sh --dataset-path <path> [options]

Required:
  --dataset-path <path>   Path to one dataset folder, e.g. datasets/lerobot_trial_2/push_button_eef

Optional:
  --start-episode <N>     First episode index to export (default: 0)
  --num-episodes <K>      Number of episodes to export (default: 4)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="${2:-}"
      shift 2
      ;;
    --start-episode)
      START_EPISODE="${2:-}"
      shift 2
      ;;
    --num-episodes)
      NUM_EPISODES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  usage >&2
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

if ! [[ "$START_EPISODE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: start_episode must be a non-negative integer (got: $START_EPISODE)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults (override via env vars if needed)
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TOLERANCE_S="${TOLERANCE_S:-1e-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
DISPLAY_COMPRESSED_IMAGES="${DISPLAY_COMPRESSED_IMAGES:-0}"

DATASET_ROOT="$DATASET_PATH"
DATASET_ID="$(basename "$DATASET_ROOT")"
OUTPUT_DIR="$DATASET_ROOT/output"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset folder not found: $DATASET_ROOT" >&2
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
echo "Episodes: $START_EPISODE..$((START_EPISODE + NUM_EPISODES - 1))"

ep="$START_EPISODE"
end_ep=$((START_EPISODE + NUM_EPISODES - 1))
while [[ "$ep" -le "$end_ep" ]]; do
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