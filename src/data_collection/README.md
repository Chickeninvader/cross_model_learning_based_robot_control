# Data Collection Quick Reference

The canonical project documentation now lives in the repository-root
`README.md`. This file keeps only the data-collection-specific quick reference.

## Main Files

- `src/data_collection/RLBench_scene_graph_collection.ipynb`
- `src/data_collection/apply_template_batch.py`
- `src/data_collection/convert_rlbench_to_lerobot.py`
- `src/utils/scene_graph_utils.py`
- `scripts/core/dataset_generator.sh`
- `scripts/core/apply_rlbench_scene_graph_templates.sh`
- `scripts/core/convert_rlbench_to_lerobot_batch.sh`

## Quick Workflow

### 1. Generate raw RLBench data

```bash
OUT_ROOT=datasets/rlbench_<run_name> bash scripts/core/dataset_generator.sh 20 put_rubbish_in_bin meat_on_grill
```

This writes to `OUT_ROOT` (required). Note that the current script default is
`START_VARIATION=20`; set `START_VARIATION=0` if you want to start from variation
0.

### 2. Create `info.json` and annotate a template

Open:

- `src/data_collection/RLBench_scene_graph_collection.ipynb`

This notebook now covers both:

- creating or updating `info.json`
- saving `<task>_relationship_template.json`

Expected task-level outputs:

- `$OUT_ROOT/<task>/info.json`
- `$OUT_ROOT/<task>/<task>_relationship_template.json`

### 3. Apply the template to all variations

```bash
bash scripts/core/apply_rlbench_scene_graph_templates.sh \
  --dataset-path "$OUT_ROOT" \
  --episode 0 \
  --camera front
```

Scene-graph-only run:

```bash
bash scripts/core/apply_rlbench_scene_graph_templates.sh \
  --dataset-path "$OUT_ROOT" \
  --tasks "lamp_on,put_rubbish_in_bin" \
  --skip-videos
```

Per-variation outputs:

- `<task>_scene_graph.json`
- `object_color_map.json`
- `episode_overlay.mp4`
- `episode_mask.mp4`
- `episode_mask_encoded.mp4`

### 4. Convert RLBench -> LeRobot

```bash
bash scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path "$OUT_ROOT" \
  --output-root datasets/lerobot_<run_name> \
  --action-space both
```

Useful options:

- `--tasks "task_a,task_b"`
- `--use-context-prompt`
- `--strict`
- `--no-merge-all-tasks`

## Cursor Guidance
This data-collection guide is designed to work with the repo's Cursor rules and skills:

- Project rules: `.cursor/rules/python-robotics-src.mdc` and `.cursor/rules/shell-pipeline-scripts.mdc` to keep Python/shell conventions consistent.
- Pipeline workflow skill: `.cursor/skills/robotics-pipeline-workflow/SKILL.md` for the correct step ordering (generate raw RLBench data -> create `info.json` and relationship templates -> apply templates across variations -> convert to LeRobot).

## Important Notes

- `info.json` stores a reference `object_mapping`, not a fixed mapping that can
  be reused blindly for every variation.
- `apply_template_batch.py` re-resolves the actual per-variation handle mapping
  through `discover_object_mapping()` before writing scene graphs.
- object color names come from heuristic estimation in
  `scene_graph_utils.py`; they are useful prompts, not guaranteed labels.
- notebooks and scripts should be launched from the repo root with
  `src` available on `PYTHONPATH`.

## Fast Troubleshooting

- Missing `low_dim_obs.pkl`: conversion skips that variation unless `--strict`
  is enabled.
- Missing `<task>_relationship_template.json`: batch template apply skips that
  task.
- Need only JSON outputs: use `--skip-videos`.
- Broader environment/debug notes live in `TROUBLESHOOTING.md`.

