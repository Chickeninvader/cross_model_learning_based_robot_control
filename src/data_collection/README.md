# RLBench Scene Graph Data Collection

This folder contains the scene-graph annotation and conversion tooling used by this repository.

The recommended way to run the pipeline is through scripts in `scripts/core/`.

## Recommended Pipeline

1. Generate RLBench task data (raw episodes).
2. Create `info.json` for each task.
3. Annotate one reference variation and save relationship template.
4. Apply templates to all target tasks/variations.
5. Convert RLBench -> LeRobot datasets.
6. Merge all converted tasks into `all_task_eef` / `all_task_joint`.

## 1) Generate RLBench raw data

Use:

```bash
cd /workspace
scripts/core/dataset_generator.sh <num_variations> <task1> [task2 ...]
```

Example:

```bash
scripts/core/dataset_generator.sh 20 put_rubbish_in_bin meat_on_grill
```

Default output is under `datasets/rlbench_trial_2`.

## 2) Create `info.json` (once per task)

Open and run:

- `src/data_collection/create_info_json.ipynb`

Output:

- `datasets/<rlbench_root>/<task>/info.json`

`info.json` defines:

- objects
- relation types
- object handle mapping (`object_mapping`)

## 3) Annotate and save template

Open and run:

- `src/data_collection/RLBench_scene_graph_collection.ipynb`

Annotate one representative variation (usually `variation0`) and save:

- `<task>_relationship_template.json`

## 4) Apply templates in batch

Use:

```bash
cd /workspace
scripts/core/apply_rlbench_trial2_templates.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --episode 0 \
  --camera front
```

Optional subset:

```bash
scripts/core/apply_rlbench_trial2_templates.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --tasks "lamp_on,put_rubbish_in_bin" \
  --skip-videos
```

Per variation outputs:

- `<task>_scene_graph.json`
- `object_color_map.json`
- optional videos (`episode_overlay.mp4`, `episode_mask.mp4`, `episode_mask_encoded.mp4`)

## 5) Convert RLBench -> LeRobot

Use:

```bash
cd /workspace
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --output-root /workspace/datasets/lerobot_trial_2 \
  --action-space both
```

Optional:

- `--tasks "task_a,task_b"`
- `--use-context-prompt`
- `--strict` (fail on task/variation error)

Default behavior is fault-tolerant: missing/broken variations are skipped.

## 6) Final merged datasets

After batch conversion, final merged datasets are created automatically:

- `all_task_eef`
- `all_task_joint`

under your `--output-root` (for example `datasets/lerobot_trial_2`).

## Quick Troubleshooting

- Missing `low_dim_obs.pkl` in a variation: conversion skips that variation and continues.
- Missing relationship template in a task: template apply script skips that task.
- Need only scene graphs quickly: add `--skip-videos` in template apply step.

