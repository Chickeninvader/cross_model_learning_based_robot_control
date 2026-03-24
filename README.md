# Cross-Model Learning-Based Robot Control

A pipeline for generating RLBench demonstrations, converting them to
GR00T-compatible LeRobot v3 datasets, and training/evaluating policies.

## Table of Contents

1. [RLBench Dataset Generation](#rlbench-dataset-generation)
2. [Scene Graph Generation](#scene-graph-generation)
3. [Dataset Conversion (RLBench -> LeRobot v3)](#dataset-conversion-rlbench---lerobot-v3)
4. [LeRobot Visualization (.rrd)](#lerobot-visualization-rrd)
5. [Training](#training)
6. [Inference & RLBench Evaluation](#inference--rlbench-evaluation)
7. [Troubleshooting](#troubleshooting)

---

## RLBench Dataset Generation

### Quick Start (Headless Docker + OSMesa)

```bash
cd /workspace/external/RLBench
docker-compose up -d
```

### Recommended dataset generation command (from repo root)

Use the core wrapper script (recommended):

```bash
cd /workspace
scripts/core/dataset_generator.sh 20 put_rubbish_in_bin meat_on_grill
```

This generates RLBench datasets under `datasets/rlbench_trial_2` by default.


---

## Scene Graph Generation

Detailed instructions are in:

- `src/data_collection/README.md`

Main tools:
- `src/data_collection/RLBench_scene_graph_collection.ipynb`
- `scripts/core/apply_rlbench_trial2_templates.sh`

Apply templates in batch:

```bash
cd /workspace
scripts/core/apply_rlbench_trial2_templates.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --episode 0 \
  --camera front
```

---

## Dataset Conversion (RLBench -> LeRobot v3)


```bash
cd /workspace
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --output-root /workspace/datasets/lerobot_trial_2 \
  --action-space both
```

This creates:

- Per-task datasets: `<task>_eef`, `<task>_joint`
- Final merged datasets:
  - `all_task_eef`
  - `all_task_joint`

### Common options

```bash
# selected tasks only
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --tasks "lamp_on,put_rubbish_in_bin"

# include context prompt generation
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --use-context-prompt
```

### LeRobot dataset layout

```text
datasets/lerobot_trial_2/<task_or_all_task>_<eef|joint>/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.front_rgb/chunk-000/file-000.mp4
    └── observation.images.wrist_rgb/chunk-000/file-000.mp4
```

---

## LeRobot Visualization (.rrd)

Use:

```bash
bash scripts/core/generate_lerobot_rrd.sh put_rubbish_in_bin eef 4
```

Outputs are written under:

- `datasets/.../<task>_<eef|joint>/output`

---

## Training


Training code is under:

- `src/training`

For remote checkpoint sync (ASU Sol HPC), use:

Use:

```bash
bash scripts/core/transfer_lerobot_last_checkpoints.sh
```

---

## Inference & RLBench Evaluation

Evaluation scripts are maintained under `scripts/core`:

- `scripts/core/eval.sh` canonical entrypoint.
- `scripts/core/eval_rlbench_single_task.sh` single-task evaluator.
- `scripts/core/eval_rlbench_all_tasks.sh` all-task evaluator.
- `src/inference/rlbench/eval.sh` backward-compatible wrapper.

### Single-task evaluation

```bash
cd /workspace
scripts/core/eval.sh \
  --task put_rubbish_in_bin \
  --runs 10 \
  --variation 0 \
  --dataset_parent /workspace/datasets/lerobot_trial_2 \
  --rlbench_root /workspace/datasets/rlbench_trial_2
```

### All-task evaluation (recommended for merged models)

```bash
cd /workspace
scripts/core/eval.sh --all_tasks \
  --tasks put_rubbish_in_bin,put_banana_in_bin,lamp_on,push_button,meat_on_grill \
  --runs 10 \
  --variation 0 \
  --dataset_mode all \
  --dataset_parent /workspace/datasets/lerobot_trial_2 \
  --rlbench_root /workspace/datasets/rlbench_trial_2
```

### Task description behavior

By default, you do not need to pass `--task_description`.

`src/inference/rlbench/eval.py` resolves text in this order:

1. explicit `--task_description` (if provided),
2. dataset parquet metadata (best effort),
3. RLBench reset-time task description.

### Summarize-only mode

```bash
scripts/core/eval.sh \
  --task lamp_on \
  --summarize_only

scripts/core/eval.sh --all_tasks \
  --tasks lamp_on,push_button \
  --summarize_only
```

### Outputs and metrics

Single-task outputs:

- `/workspace/output/rlbench_eval/<task>/...`

All-task outputs:

- `/workspace/output/rlbench_eval/all_tasks/<task>/<policy>_<state>/var<variation>_seed<seed>/...`

Each run contains rollout artifacts plus:

- `metrics.json`
- `summary.json`
- `per_run_metrics.csv`

Aggregate outputs include:

- `detailed_summary.json`
- `detailed_runs.csv`
- `overall_metrics.csv`

`per_run_metrics.csv` includes fields such as:

- `task_description`
- `instruction`
- `policy_success`
- `joint_l2`, `pos_l2`, `rot_deg`, `eef_l2`

---

## Troubleshooting

Troubleshooting and debugging are now maintained in a separate file:

- `TROUBLESHOOTING.md`

