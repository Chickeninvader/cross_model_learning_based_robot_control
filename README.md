# Cross-Model Learning-Based Robot Control

A pipeline for generating RLBench demonstrations, converting them to
GR00T-compatible LeRobot v3 datasets, and training/evaluating policies.

## Table of Contents

1. [RLBench Dataset Generation](#rlbench-dataset-generation)
2. [Scene Graph Generation](#scene-graph-generation)
3. [Dataset Conversion (RLBench -> LeRobot v3)](#dataset-conversion-rlbench---lerobot-v3)
4. [LeRobot Visualization (.rrd)](#lerobot-visualization-rrd)
5. [Training Notes (ASU Sol HPC)](#training-notes-asu-sol-hpc)
6. [RLBench Evaluation](#rlbench-evaluation)
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

## Training Notes (ASU Sol HPC)


### Transfer latest checkpoints

Use:

```bash
bash scripts/core/transfer_lerobot_last_checkpoints.sh
```

---

## RLBench Evaluation

Use:

- `src/inference/rlbench/eval.sh`

Batch summarize existing runs:

```bash
bash src/inference/rlbench/eval.sh --summarize_only
```

---

## Troubleshooting

Troubleshooting and debugging are now maintained in a separate file:

- `TROUBLESHOOTING.md`

# Cross-Model Learning-Based Robot Control

This repository provides a practical pipeline for:

1. generating RLBench demonstrations,
2. building scene graphs/templates,
3. converting to LeRobot v3 format,
4. training/evaluating policies.

Most runnable commands are in `scripts/core/`.

## Core Scripts

- `scripts/core/setup_external.sh` - setup external dependencies.
- `scripts/core/dataset_generator.sh` - generate RLBench task datasets.
- `scripts/core/apply_rlbench_trial2_templates.sh` - batch apply relationship templates.
- `scripts/core/convert_rlbench_to_lerobot_batch.sh` - batch convert RLBench -> LeRobot and merge final datasets.
- `scripts/core/generate_lerobot_rrd.sh` - export `.rrd` visualization files for LeRobot datasets.
- `scripts/core/transfer_lerobot_last_checkpoints.sh` - copy latest checkpoints from remote runs.

## End-to-End Quick Start

From repo root:

```bash
cd /workspace
```

### 1) Generate RLBench data

```bash
scripts/core/dataset_generator.sh 20 put_rubbish_in_bin meat_on_grill
```

### 2) Create scene graph metadata

Run notebooks:

- `src/data_collection/create_info_json.ipynb`
- `src/data_collection/RLBench_scene_graph_collection.ipynb`

### 3) Apply templates to all tasks/variations

```bash
scripts/core/apply_rlbench_trial2_templates.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --episode 0 \
  --camera front
```

### 4) Convert to LeRobot and create final merged datasets

```bash
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --output-root /workspace/datasets/lerobot_trial_2 \
  --action-space both
```

This creates per-task datasets plus:

- `/workspace/datasets/lerobot_trial_2/all_task_eef`
- `/workspace/datasets/lerobot_trial_2/all_task_joint`

## Common Commands

### Convert only selected tasks

```bash
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --tasks "lamp_on,put_rubbish_in_bin" \
  --output-root /workspace/datasets/lerobot_trial_2
```

### Use context prompt for language generation

```bash
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --use-context-prompt
```


## Notes

- Conversion is resilient by default: missing variation data is skipped.
- Scene graph template application skips tasks without template files.
- For data collection details, see `src/data_collection/README.md`.

