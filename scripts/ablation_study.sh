#!/bin/bash
#
# Ablation Study Script
# Runs comprehensive ablation experiments comparing different configurations
#

set -e  # Exit on error

# Configuration
POLICY_PATH="logs/cross_embodiment/best.pth"
OUTPUT_DIR="eval_results/ablation_study"
NUM_EPISODES=50

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Cross-Embodiment Ablation Study"
echo "=================================="
echo ""
echo "Policy: $POLICY_PATH"
echo "Episodes per variant: $NUM_EPISODES"
echo "Output: $OUTPUT_DIR"
echo ""

# 1. Full model (with scene graphs)
echo "[1/5] Evaluating: Full model (with scene graphs)"
python src/evaluation/evaluate_transfer.py \
    --policy-path "$POLICY_PATH" \
    --source-robot Franka \
    --target-robot Sawyer \
    --task Lift \
    --num-episodes "$NUM_EPISODES" \
    --output-dir "$OUTPUT_DIR/full_model" \
    --save-trajectories

# 2. Baseline (no scene graphs)
echo ""
echo "[2/5] Evaluating: Baseline (no scene graphs)"
python src/evaluation/evaluate_transfer.py \
    --policy-path "$POLICY_PATH" \
    --source-robot Franka \
    --target-robot Sawyer \
    --task Lift \
    --num-episodes "$NUM_EPISODES" \
    --no-scene-graph \
    --output-dir "$OUTPUT_DIR/no_scene_graph" \
    --save-trajectories

# 3. Within-robot baseline (Franka -> Franka)
echo ""
echo "[3/5] Evaluating: Within-robot baseline (Franka -> Franka)"
python src/evaluation/evaluate_transfer.py \
    --policy-path "$POLICY_PATH" \
    --source-robot Franka \
    --target-robot Franka \
    --task Lift \
    --num-episodes "$NUM_EPISODES" \
    --output-dir "$OUTPUT_DIR/within_robot_franka" \
    --save-trajectories

# 4. Reverse transfer (Sawyer -> Franka)
echo ""
echo "[4/5] Evaluating: Reverse transfer (Sawyer -> Franka)"
python src/evaluation/evaluate_transfer.py \
    --policy-path "$POLICY_PATH" \
    --source-robot Sawyer \
    --target-robot Franka \
    --task Lift \
    --num-episodes "$NUM_EPISODES" \
    --output-dir "$OUTPUT_DIR/reverse_transfer" \
    --save-trajectories

# 5. Different task (Stack)
echo ""
echo "[5/5] Evaluating: Different task (Stack)"
python src/evaluation/evaluate_transfer.py \
    --policy-path "$POLICY_PATH" \
    --source-robot Franka \
    --target-robot Sawyer \
    --task Stack \
    --num-episodes "$NUM_EPISODES" \
    --output-dir "$OUTPUT_DIR/stack_task" \
    --save-trajectories

# Aggregate results
echo ""
echo "=================================="
echo "Aggregating Results"
echo "=================================="

python << EOF
import json
import numpy as np
from pathlib import Path

results_dir = Path("$OUTPUT_DIR")
variants = ["full_model", "no_scene_graph", "within_robot_franka", "reverse_transfer", "stack_task"]

print("\nAblation Study Summary")
print("=" * 80)
print(f"{'Variant':<30} {'Success Rate':<15} {'Mean Reward':<15}")
print("=" * 80)

summary = {}
for variant in variants:
    metrics_file = results_dir / variant / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

        success_rate = metrics['success_rate'] * 100
        mean_reward = metrics['mean_reward']

        print(f"{variant:<30} {success_rate:>6.1f}%         {mean_reward:>7.2f}")
        summary[variant] = metrics

# Save summary
with open(results_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 80)
print(f"\nSummary saved to: {results_dir / 'summary.json'}")
EOF

echo ""
echo "=================================="
echo "Ablation Study Completed!"
echo "=================================="
echo "Results saved to: $OUTPUT_DIR"
