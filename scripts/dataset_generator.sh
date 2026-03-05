RLBENCH_ROOT=/workspace/external/RLBench/rlbench
OUT_ROOT=/workspace/datasets/rlbench
PY=python

cd $RLBENCH_ROOT

# All available tasks
all_tasks=($(ls tasks | grep '\.py$' | grep -v __init__ | sed 's/\.py$//'))

mkdir -p "$OUT_ROOT"

missing=()

for t in "${all_tasks[@]}"; do
  episode_file="$OUT_ROOT/$t/variation0/episodes/episode_0/low_dim_obs.pkl"

  if [ ! -f "$episode_file" ]; then
    echo "Task $t is incomplete or missing."
    missing+=("$t")
  fi
done

echo "--------------------------------------"
echo "Total tasks: ${#all_tasks[@]}"
echo "Tasks needing collection: ${#missing[@]}"
printf '%s\n' "${missing[@]}"
echo "--------------------------------------"

if [ ${#missing[@]} -gt 0 ]; then
  echo "Running dataset generator with 8 processes..."

  $PY dataset_generator.py \
    --tasks "${missing[@]}" \
    --episodes_per_task 1 \
    --variations 1 \
    --start_variation 0 \
    --processes 8 \
    --image_size 256 256 \
    --renderer opengl3 \
    --save_path "$OUT_ROOT"

  echo "Resume run complete."
else
  echo "All tasks already complete."
fi