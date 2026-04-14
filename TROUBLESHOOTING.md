# Troubleshooting and Debugging

This file centralizes common debugging issues for RLBench data generation,
scene-graph processing, LeRobot conversion, and HPC training.

## Dataset Generation

### RLBench headless rendering fails

Symptoms:

- simulator cannot start
- OpenGL context errors

Checks:

```bash
echo "$DISPLAY"
echo "$LD_PRELOAD"
```

Fix:

- Ensure `Xvfb` is running (`DISPLAY=:99`).
- Ensure OSMesa env vars are exported (see `README.md`).

## Scene Graph / Template Pipeline

### `info.json` missing for a task

Fix:

- Create it with `src/data_collection/create_info_json.ipynb`.

### Relationship template missing

Symptoms:

- apply-template batch script skips task

Fix:

- Create `<task>_relationship_template.json` in notebook.
- Re-run:

```bash
scripts/core/apply_rlbench_trial2_templates.sh --dataset-path /workspace/datasets/rlbench_trial_2
```

## RLBench -> LeRobot Conversion

### Missing variation files (for example `low_dim_obs.pkl`)

Symptoms:

- file not found errors in conversion logs

Current behavior:

- Converter skips broken variations by default and continues.

Strict mode:

```bash
scripts/core/convert_rlbench_to_lerobot_batch.sh \
  --dataset-path /workspace/datasets/rlbench_trial_2 \
  --strict
```

### Final merged `all_task_eef` / `all_task_joint` not created

Checks:

- confirm at least one task conversion succeeded
- confirm you did not pass `--no-merge-all-tasks`

Manual merge (advanced):

```bash
python src/data_collection/convert_rlbench_to_lerobot.py \
  --merge_all_tasks_root /workspace/datasets/lerobot_trial_2 \
  --output_root /workspace/datasets/lerobot_trial_2 \
  --action_space both
```

## LeRobot Visualization

### `.rrd` export fails

Checks:

- dataset path exists under `datasets/.../<task>_<eef|joint>`
- `external/lerobot` source exists

Run:

```bash
bash scripts/core/generate_lerobot_rrd.sh put_rubbish_in_bin eef 4
```

## Training (ASU Sol)

### flash-attn build issues (important)

Known issues:

- GLIBC mismatch with prebuilt wheels
- cross-device link errors in wheel move
- unsupported architecture flags

Recommended full setup on Sol:

```bash
module load mamba/latest
source activate lerobot
module load cuda-12.6.1-gcc-12.1.0
module load gcc-12.1.0-gcc-11.2.0
```

Build from source (do not use prebuilt wheel):

```bash
cd /scratch/$USER/tmp
git clone https://github.com/Dao-AILab/flash-attention.git flash-attn-src
cd flash-attn-src
```

Patch `setup.py` to avoid cross-device link error:

```python
# BEFORE
os.rename(wheel_filename, wheel_path)

# AFTER
import shutil
shutil.move(wheel_filename, wheel_path)
```

Build/install with safe flags:

```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=4
pip install . 2>&1 | tee /scratch/$USER/tmp/flashattn_build.log
```

Verify outside source dir:

```bash
cd /scratch/$USER
python -c "import flash_attn; print(flash_attn.__version__)"
```

Common pitfalls:

- `FLASH_ATTENTION_FORCE_BUILD` must be exactly `TRUE` (not `1`, `true`, or `True`).
- If you run Python from the flash-attn source dir, import can fail with module errors.
- `nvcc fatal: Unsupported gpu architecture 'compute_120'` means `FLASH_ATTN_CUDA_ARCHS` is wrong for your GPU.

### `torchcodec` symbol errors

Use `pyav` backend:

- install `av`
- configure training with `dataset.video_backend=pyav`

On Sol:

```bash
conda install -c conda-forge ffmpeg -y
pip install av
```

### Sol quick error map

| Error | Likely cause | Fix |
|---|---|---|
| `GLIBC_2.32 not found` when importing flash-attn | Prebuilt wheel incompatible with Rocky Linux 8 | Build flash-attn from source |
| `OSError: [Errno 18] Invalid cross-device link` during flash-attn build | `os.rename()` across `/tmp` and `/scratch` | Patch `setup.py` to `shutil.move(...)` |
| `FLASH_ATTENTION_FORCE_BUILD` ignored | Env var value not exactly expected | Set `FLASH_ATTENTION_FORCE_BUILD=TRUE` |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | Wrong CUDA arch target | Set `FLASH_ATTN_CUDA_ARCHS=80` for A100 |
| `ModuleNotFoundError: flash_attn_2_cuda` | Running from source dir or broken build | `cd` out of source dir and recheck install |
| `undefined symbol ... MessageLogger ...` from `torchcodec` | ABI mismatch with current torch build | Use `pyav` backend |

## Evaluation

### Evaluation script path confusion

Use current files:

- `src/inference/rlbench/eval.py`
- `src/inference/rlbench/eval.sh`

Do not use removed legacy paths like `infer_rlbench.py` or task-specific eval script names.


## Cursor Rules and Skills
This repo includes Cursor rules and skills under `.cursor/` to keep the agent consistent with how this pipeline is meant to be used:

- Project rules: `.cursor/rules/*.mdc` (short, always-on conventions like path/import norms and pipeline safety/repro guidelines)
- Project skills:
  - `.cursor/skills/robotics-pipeline-workflow/SKILL.md` for the step ordering from dataset generation -> scene graph templates -> LeRobot conversion -> upload to training host -> training -> checkpoint download -> evaluation
  - `.cursor/skills/robotics-eval-and-experiments/SKILL.md` for fair evaluation comparisons, metrics logging, and experiment result summaries

When working on data collection (`info.json` -> relationship templates), start from `src/data_collection/README.md` and follow the pipeline skill for the correct step ordering.