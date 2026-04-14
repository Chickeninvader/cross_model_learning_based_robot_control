# Cross-Model Learning-Based Robot Control

This repository contains the active RLBench -> scene graph -> LeRobot ->
training -> RLBench evaluation pipeline used in this project.

`README.md` is now the canonical project guide. The local
`src/data_collection/README.md` is kept as a short quick reference, and
`TROUBLESHOOTING.md` collects operational/debug notes.

## Current Status

The codebase already reflects substantial recent work. The current pipeline
supports:

- RLBench dataset generation through `scripts/core/dataset_generator.sh`
- interactive `info.json` creation and relationship-template annotation inside
  `src/data_collection/RLBench_scene_graph_collection.ipynb`
- batch scene-graph/template application through
  `scripts/core/apply_rlbench_scene_graph_templates.sh`
- variation-aware object-handle remapping via
  `src/utils/scene_graph_utils.py::discover_object_mapping()`
- heuristic object color extraction used by scene-graph language generation
- RLBench -> LeRobot conversion with per-task and merged `all_task_*` datasets
- dataset upload to the training host via `scripts/core/transfer_dataset_to_server.sh`
- checkpoint download from the training host via
  `scripts/core/transfer_lerobot_last_checkpoints.sh`
- `.rrd` export through `scripts/core/generate_lerobot_rrd.sh`
- RLBench evaluation with relationship-template-aware segmentation and batch
  aggregation in `src/inference/rlbench/eval.py`
- Sol/SLURM training wrappers for GR00T and SmolVLA in `src/training/`

## Repository Map

```text
.
├── README.md
├── TROUBLESHOOTING.md
├── environment.yml
├── docker-compose.yml
├── scripts/core/                 # canonical shell entrypoints
├── scripts/dummy/                # ad hoc utilities / one-off helpers
├── src/data_collection/          # notebook + conversion/template tools
├── src/inference/rlbench/        # active RLBench evaluation flow
├── src/training/                 # Sol HPC training wrappers
└── src/utils/                    # shared RLBench / scene-graph helpers
```

Other inference stacks under `src/inference/` such as `OvSGTR`, `LASER`, and
`lang_sam` are present, but the main maintained end-to-end workflow in this
repo is the RLBench/LeRobot path described below.

## Setup

### 1. Conda environment

For local runs outside Docker, Conda is required. From the repository root:

```bash
conda env create -f environment.yml
conda activate rlbench_gui
```

### 2. External dependencies

From the repository root:

```bash
bash scripts/core/setup_external.sh
```

This prepares `external/RLBench` and `external/lerobot`.

### 3. Python path

For local shell runs and notebooks, start from the repository root and export:

```bash
export PYTHONPATH="$PWD:$PWD/src:$PWD/external/RLBench:$PWD/external/lerobot/src:${PYTHONPATH:-}"
```

This matters especially for notebooks and `src/data_collection/apply_template_batch.py`.

### 4. Optional Docker workflow

If you use Docker, you do not need the local Conda environment above.

If you use the provided container setup, start it from the repository root:

```bash
docker compose up -d
```

Do not run `docker compose` from `external/RLBench`; the tracked
`docker-compose.yml` lives at the repo root.

If you only want to run inference/evaluation, you can skip the full training
pipeline and download the shared weights from:

```text
https://drive.google.com/drive/folders/1kXG7FoKQNmraj1NjDM-65lYpO7_2kS4w?usp=sharing
```

Place the downloaded zip files under `output/lerobot/` and extract them so the
checkpoint folders land in the expected prompt-mode directories:

```text
output/lerobot/
├── with_context_prompt/
│   └── smolvla_all_task_eef_20260328_182423/
└── without_context_prompt/
    └── smolvla_all_task_eef_20260328_173446/
```

Each extracted run directory should contain checkpoint weights under
`checkpoints/last/pretrained_model/` and/or
`checkpoints/020000/pretrained_model/`, including `model.safetensors`.

Example:

```bash
mkdir -p output/lerobot/with_context_prompt
mkdir -p output/lerobot/without_context_prompt

unzip output/lerobot/with_context_prompt/smolvla_all_task_eef_20260328_182423.zip -d .
unzip output/lerobot/without_context_prompt/smolvla_all_task_eef_20260328_173446.zip -d .
```

After that, continue with the inference/evaluation steps below.

If you need to reproduce or extend training from scratch, continue with the
full end-to-end workflow below, including dataset generation, conversion,
transfer, training, checkpoint download, and evaluation.

## End-to-End Workflow

### 1. Generate RLBench raw data

Recommended entrypoint:

```bash
OUT_ROOT=datasets/rlbench_<run_name> bash scripts/core/dataset_generator.sh 20 put_rubbish_in_bin meat_on_grill
```

Required output root:

- output root: `$OUT_ROOT` (set `OUT_ROOT`; required)

Optional defaults in `scripts/core/dataset_generator.sh`:

- RLBench root: `external/RLBench` (override with `RLBENCH_ROOT`)
- episodes per variation: `1`
- start variation: `0`
- renderer: `opengl3`

To start from a non-zero variation, override the default:

```bash
START_VARIATION=20 bash scripts/core/dataset_generator.sh 5 lamp_on push_button
```

Useful environment overrides:

- `RLBENCH_ROOT`
- `OUT_ROOT`
- `EPISODES_PER_TASK`
- `PROCESSES`
- `IMAGE_WIDTH`, `IMAGE_HEIGHT`
- `RENDERER`

### 2. Create `info.json` and relationship templates

The single notebook `src/data_collection/RLBench_scene_graph_collection.ipynb`
now covers both:

- creating or updating task-level `info.json`
- annotating a reference variation and saving
  `<task>_relationship_template.json`

Expected outputs per task:

- `$OUT_ROOT/<task>/info.json`
- `$OUT_ROOT/<task>/<task>_relationship_template.json`

Important implementation detail:

- `info.json` stores a reference `object_mapping`
- later variations do not reuse the same raw RLBench mask handle IDs
- the batch pipeline re-resolves actual handle IDs per variation using
  `discover_object_mapping()` before applying the saved template

### 3. Apply templates across variations

```bash
bash scripts/core/apply_rlbench_scene_graph_templates.sh \
  --dataset-path "$OUT_ROOT" \
  --episode 0 \
  --camera front
```

Subset example:

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

Color labels are estimated heuristically from image/mask data in
`src/utils/scene_graph_utils.py`; treat them as approximate semantic labels, not
ground-truth physical color annotations.

### 4. Convert RLBench -> LeRobot

```bash
bash scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path "$OUT_ROOT" \
  --output-root datasets/lerobot_<run_name> \
  --action-space both
```

Common options:

```bash
# selected tasks only
bash scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path "$OUT_ROOT" \
  --output-root datasets/lerobot_<run_name> \
  --tasks "lamp_on,put_rubbish_in_bin"

# generate context-rich task descriptions
bash scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path "$OUT_ROOT" \
  --output-root datasets/lerobot_<run_name> \
  --use-context-prompt
```

Current behavior:

- creates per-task datasets such as `<task>_eef` and `<task>_joint`
- merges successful task conversions into `all_task_eef` and `all_task_joint`
  by default
- skips broken variations by default
- supports `--strict` to fail on task/variation errors

Typical layout:

```text
datasets/lerobot_<run_name>/<task_or_all_task>_<eef|joint>/
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

### 5. Transfer LeRobot data to the training server

After `datasets/lerobot_<run_name>/` exists (per-task folders and/or merged
`all_task_eef` / `all_task_joint` from step 4), sync it to the remote datasets
tree with rsync:

```bash
bash scripts/core/transfer_dataset_to_server.sh \
  --dataset-path datasets/lerobot_<run_name> \
  --task all_task
```

Other useful forms:

```bash
# entire local folder under the remote datasets root
bash scripts/core/transfer_dataset_to_server.sh \
  --dataset-path datasets/lerobot_<run_name>

# single task subtree only
bash scripts/core/transfer_dataset_to_server.sh \
  --dataset-path datasets/lerobot_<run_name> \
  --task put_rubbish_in_bin

# preview rsync without copying
bash scripts/core/transfer_dataset_to_server.sh \
  --dataset-path datasets/lerobot_<run_name> \
  --task all_task \
  --dry-run
```

Defaults are site-specific; override with environment variables documented at
the top of `scripts/core/transfer_dataset_to_server.sh` (for example
`REMOTE_HOST`, `REMOTE_DATASETS_DIR`). The transfer is resumable—if a run is
interrupted, rerun the same command to continue.

### Submit LeRobot training jobs (`training_script_log.sh`)

From the **repository root** on Sol, this script submits **four** training runs (SmolVLA + GR00T, each in **eef** and **joint** modes) via `sbatch`, using the per-task LeRobot folders produced by conversion (`<task>_eef` and `<task>_joint` under a shared base directory).

**Arguments (order matters):**

1. **`<task_name>`** (required) — e.g. `lamp_on`, `put_rubbish_in_bin`. Must match dataset folder names `<task_name>_eef` and `<task_name>_joint`.
2. **`--dataset-base <path>`** or **`-p <path>`** (optional) — parent directory that **contains** those folders, not the leaf dataset path. Default is `datasets/lerobot_trial_2`, or whatever you set in the environment variable **`DATASET_BASE`** before running the script.
3. **`--debug`** or **`-d`** (optional) — run the model scripts **locally** (no `sbatch`) with a tiny step count for a quick sanity check.

**Common mistake:** there is **no** `--dataset-root` flag. The script builds dataset roots itself as `<dataset-base>/<task_name>_eef` and `<dataset-base>/<task_name>_joint`. If you only pass `--dataset-root ...`, the shell treats the path as a second positional argument and you get `unexpected argument`.

Examples:

```bash
# Default base: datasets/lerobot_trial_2 (or $DATASET_BASE if set)
./src/training/training_script_log.sh put_rubbish_in_bin

# Explicit base directory (same layout as conversion output-root)
./src/training/training_script_log.sh put_rubbish_in_bin --dataset-base datasets/lerobot_trial_2

# Short local test instead of sbatch
./src/training/training_script_log.sh lamp_on --debug
```

If a folder like `datasets/lerobot_trial_2/<task>_eef` is missing, that model/mode pair is **skipped** with a message.

For per-model flags and environment variables (`OUTPUT_DIR`, `WANDB_ENABLE`, etc.), see `src/training/run_smolvla_sol.sh` and `src/training/run_groot_sol.sh`.

### 6. Export `.rrd` files for inspection

The current visualization script uses `--dataset-path`; the older positional
`task eef 4` form is stale.

```bash
bash scripts/core/generate_lerobot_rrd.sh \
  --dataset-path datasets/lerobot_<run_name>/push_button_eef \
  --start-episode 0 \
  --num-episodes 4
```

Outputs are written to:

- `datasets/lerobot_<run_name>/<dataset_name>/output`

This script uses the vendored
`external/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py`.

### 7. Train policies

Training wrappers live in `src/training/`:

- `src/training/run_groot_sol.sh`
- `src/training/run_smolvla_sol.sh`
- `src/training/training_script_log.sh`

Examples:

```bash
sbatch src/training/run_groot_sol.sh --dataset-kind eef
sbatch src/training/run_smolvla_sol.sh --dataset-kind joint
bash src/training/training_script_log.sh put_rubbish_in_bin
```

Important note:

- these scripts are currently Sol/SLURM oriented
- `REPO_ROOT`, scratch paths, and SBATCH account settings are site-specific
- adjust them before treating them as portable training entrypoints

Use step 5 to upload LeRobot datasets before running jobs on the remote
cluster.

### 8. Pull LeRobot checkpoints from the server

After training on the remote host, copy the **latest** checkpoint per run into
local `output/lerobot/` (paths and SSH target match the defaults in the
script, all overridable via env vars):

```bash
bash scripts/core/transfer_lerobot_last_checkpoints.sh
```

The script keeps a small state file so reruns skip checkpoints that are already
copied; if a download is partial, rerun the same command to continue.

Environment overrides are listed at the top of
`scripts/core/transfer_lerobot_last_checkpoints.sh` (for example
`REMOTE_HOST`, `REMOTE_OUTPUT_DIR`, `LOCAL_OUTPUT_DIR`, `STATE_FILE`).

## RLBench Evaluation

### Shell entrypoint (`scripts/core/eval.sh`)

One script covers **single-task** (default) and **all-task** (`--all_tasks`)
evaluation. Run `./scripts/core/eval.sh --help` for the full flag list.

**Single-task** (checkpoint folder names must match the **task** plus policy and
`eef` / `joint` under `--checkpoint_root`):

```bash
./scripts/core/eval.sh \
  --task put_rubbish_in_bin \
  --runs 10 \
  --variation 0 \
  --dataset_parent datasets/lerobot_<run_name> \
  --rlbench_root datasets/rlbench_<run_name>
```

**All-task** (checkpoint folder names must include a multi-task tag such as
`all_task`; override with `--checkpoint_tags`):

```bash
./scripts/core/eval.sh --all_tasks \
  --tasks put_rubbish_in_bin,put_banana_in_bin,lamp_on,push_button,meat_on_grill \
  --runs 10 \
  --variation 0 \
  --dataset_mode all \
  --dataset_parent datasets/lerobot_<run_name> \
  --rlbench_root datasets/rlbench_<run_name>
```

**Resume:** add `--skip_completed` to reuse per-task outputs when
`summary.json` matches the current run settings (see help text).

For debugging, you can call `src/inference/rlbench/eval.py` directly (same
arguments the shell builds).

### Prompt-mode behavior (important)

The eval pipeline now records and auto-aligns instruction style to match the
training dataset style for the loaded checkpoint:

- `lerobot_trial_2` style -> concise instruction text (no context block)
- `lerobot_trial_3` style -> full context prompt text

How it works:

- wrappers pass `--checkpoint` and `--dataset_root` to `eval.py`
- `eval.py` reads checkpoint `train_config.json` and dataset
  `meta/tasks.parquet` to infer expected style
- if requested prompt mode conflicts with expected training style, eval logs a
  mismatch and auto-switches to the expected mode

Practical implication:

- you can still pass `--with_context_prompt`, but eval will override it when it
  would mismatch the checkpoint's training-style dataset
- this prevents evaluating a trial-2 model with trial-3 style prompts (and
  vice-versa)

### Which checkpoint is selected automatically?

`scripts/core/eval.sh` discovers checkpoints from `--checkpoint_root`
(default: `output/lerobot`) by naming pattern:

- policy name (`groot` or `smolvla`)
- state (`eef` or `joint`)
- task/all-task tags
- newest timestamp-like suffix

If you keep your runs under `output/lerobot`, you usually do **not** need to
provide explicit checkpoint paths.

For strict debugging, point `--checkpoint_root` to an isolated folder containing
only the model(s) you want to compare.

### RLBench root selection note

If `--rlbench_root` does not contain `<task>_relationship_template.json`, the
wrappers attempt auto-fallback from dataset naming:

- `dataset_parent=.../lerobot_trial_2` -> try `datasets/rlbench_trial_2`
- `dataset_parent=.../lerobot_trial_3` -> try `datasets/rlbench_trial_3`

If your project convention is different (for example both trial2 and trial3
datasets use `rlbench_trial_2` templates), pass `--rlbench_root` explicitly.

### Summarize-only mode

```bash
bash scripts/core/eval.sh \
  --task lamp_on \
  --summarize_only

bash scripts/core/eval.sh --all_tasks \
  --tasks lamp_on,push_button \
  --summarize_only
```

### Advanced `eval.py` options

`src/inference/rlbench/eval.py` contains advanced evaluation features:

- `--robot_setup` with support for `panda`, `jaco`, `mico`, `sawyer`, `ur5`
- `--with_context_prompt` for context-rich per-segment instructions
- `--relationship_template` / `--no_relationship_template`
- `--binarize_gripper_action`
- `--policy_start_offset`
- `--debug_snapshot_restore`

Examples:

```bash
python src/inference/rlbench/eval.py \
  --task put_rubbish_in_bin \
  --checkpoint output/lerobot/example_checkpoint \
  --dataset_root datasets/lerobot_<run_name>/put_rubbish_in_bin_eef \
  --rlbench_root datasets/rlbench_<run_name> \
  --action_mode ee_planning \
  --robot_setup ur5 \
  --with_context_prompt \
  --save_path output/rlbench_eval/put_rubbish_in_bin/custom_run
```

Current evaluation behavior:

- if a relationship template exists, segmentation is aligned to template
  transitions and matching gripper changes
- otherwise, evaluation falls back to legacy instruction/gripper-change logic
- `joint_velocity` is restricted to `robot_setup=panda`
- non-Panda robots should use `ee_planning` or `ee_ik`
- prompt mode may be auto-aligned to training dataset style (recorded in output
  metadata fields `requested_with_context_prompt`, `expected_with_context_prompt`,
  and `with_context_prompt`)

Outputs typically include:

- `metrics.json`
- `summary.json`
- `per_run_metrics.csv`
- `detailed_summary.json`
- `detailed_runs.csv`
- `overall_metrics.csv`

### Output directory structure (`output/rlbench_eval`)

Single-task wrapper (default):

```text
output/rlbench_eval/<task>[/_with_context_prompt]/
└── <policy>_<state>[_<robot_setup>]/
    ├── run_000/
    │   ├── planner/episodes/episode0/...
    │   ├── policy/episodes/episode0/...
    │   └── metrics.json
    ├── summary.json
    ├── per_run_metrics.json
    └── per_run_metrics.csv
```

All-task wrapper (default):

```text
output/rlbench_eval/all_tasks[_with_context_prompt]/
├── <policy>_<state>[_<robot_setup>]/
│   ├── <task_1>/run_000/.../metrics.json
│   ├── <task_2>/run_000/.../metrics.json
│   └── all_tasks_summary.json
├── detailed_summary.json
├── detailed_runs.csv
└── overall_metrics.csv
```

For debugging comparisons, prefer explicit save roots, e.g.
`output/rlbench_eval_debug/trial2_auto_aligned_check` and
`output/rlbench_eval_debug/trial3_auto_aligned_check`.

Notebook for inspection:

- `src/inference/rlbench/inspect_eval_run.ipynb`

## Documentation Policy

To reduce drift, the documentation is now split intentionally:

- `README.md`: canonical project overview and active workflow
- `src/data_collection/README.md`: short data-collection quick reference
- `TROUBLESHOOTING.md`: debugging notes and environment-specific fixes

## Cursor Rules and Skills
This repo includes Cursor rules and skills under `.cursor/` to keep the agent consistent with how this pipeline is meant to be used:

- Project rules: `.cursor/rules/*.mdc` (short, always-on conventions like path/import norms and pipeline safety/repro guidelines)
- Project skills:
  - `.cursor/skills/robotics-pipeline-workflow/SKILL.md` for the step ordering from dataset generation -> scene graph templates -> LeRobot conversion -> upload to training host -> training -> checkpoint download -> evaluation
  - `.cursor/skills/robotics-eval-and-experiments/SKILL.md` for fair evaluation comparisons, metrics logging, and experiment result summaries

When working on data collection (`info.json` -> relationship templates), start from `src/data_collection/README.md` and follow the pipeline skill for the correct step ordering.

## Troubleshooting

See `TROUBLESHOOTING.md` for:

- RLBench rendering/display issues
- missing scene-graph/template artifacts
- LeRobot conversion failures
- `.rrd` export problems
- Sol/flash-attn and `torchcodec` issues
- evaluation path confusion
