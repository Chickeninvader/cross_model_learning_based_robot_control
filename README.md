# Cross-Model Learning-Based Robot Control

A pipeline for generating RLBench demonstrations, converting them to
GR00T-compatible LeRobot v3 datasets, and training NVIDIA GR00T N1.5 policies.

## Table of Contents

1. [RLBench Dataset Generation](#rlbench-dataset-generation)
2. [Scene Graph Generation](#scene-graph-generation-environment)
3. [Dataset Conversion (RLBench → LeRobot v3)](#dataset-conversion-rlbench--lerobot-v3)
4. [GR00T N1.5 Training](#groot-n15-training-on-asu-sol-hpc)
5. [Troubleshooting](#troubleshooting-sol-specific)

---

## RLBench Dataset Generation

### Quick Start (Headless Docker + OSMesa)

```bash
cd external/RLBench
docker-compose up -d
```

Inside the Docker container:

```bash
python /workspace/external/RLBench/rlbench/dataset_generator.py \
    --tasks put_rubbish_in_bin \
    --variations 0 \
    --processes 1 \
    --episodes_per_task 1 \
    --save_path /workspace/datasets/rlbench \
    --image_size 256 256 \
    --renderer opengl3
```

### Quick Install (OSMesa + Xvfb)

Run as root or with `sudo` inside the container:

```bash
apt-get update && apt-get install -y \
    mesa-utils x11-utils libosmesa6 libosmesa6-dev xvfb
```

Add these environment variables:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_X11_NO_MITSHM=1
export QT_QPA_PLATFORM=xcb
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.6
export DISPLAY=:99
```

Start headless X server:

```bash
Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb-99.log 2>&1 &
export DISPLAY=:99
sleep 0.5
```

---

## Scene Graph Generation Environment

```bash
conda activate comp_robotics
conda install -c conda-forge opencv ipywidgets matplotlib jupyterlab gymnasium -y
```

Follow the notebook in `examples/scene_graph_analyzer.ipynb` to annotate datasets.

---

## Dataset Conversion: RLBench → LeRobot v3

### Prerequisites

- Python 3.10+ with conda env `lerobot` activated
- Packages: `pandas`, `pyarrow`, `numpy`, `Pillow`, `av` (PyAV)

```bash
source activate lerobot
```

### RLBench Dataset Structure (Input)

```
datasets/rlbench/<task_name>/variation<num>/
├── episodes/
│   └── episode0/
│       ├── front_rgb/        # 0.png, 1.png, …, N.png
│       ├── wrist_rgb/        # 0.png, 1.png, …, N.png
│       └── low_dim_obs.pkl   # RLBench Demo observations
├── <task_name>_scene_graph.json   # Per-frame scene graph
└── object_color_map.json          # Object name → mask color
```

### Convert a Single Variation

```bash
python src/data_collection/convert_rlbench_to_lerobot.py \
    --task_name stack_cups \
    --variation 0 \
    --rlbench_root datasets/rlbench \
    --output_root datasets/lerobot \
    --fps 20
```

### Convert All Variations + Merge

When `--variation` is omitted, the script converts **every** variation found
under `datasets/rlbench/<task_name>/` and then automatically **merges** them
into a single `<task_name>_all` dataset:

```bash
python src/data_collection/convert_rlbench_to_lerobot.py \
    --task_name put_rubbish_in_bin \
    --rlbench_root datasets/rlbench \
    --output_root datasets/lerobot \
    --fps 20
```

This produces:

```
datasets/lerobot/
├── put_rubbish_in_bin_variation0/   # per-variation dataset
├── put_rubbish_in_bin_variation1/
├── …
└── put_rubbish_in_bin_all/          # merged dataset (all variations)
```

Use `--no_merge` to skip the merge step and only produce per-variation datasets.

#### Arguments

| Argument | Default | Description |
|---|---|---|
| `--task_name` | *(required)* | RLBench task name, e.g. `stack_cups` |
| `--variation` | *(all)* | Variation number. Omit to process all and merge. |
| `--rlbench_root` | `datasets/rlbench` | Root directory of RLBench datasets |
| `--output_root` | `datasets/lerobot` | Root directory for output LeRobot datasets |
| `--fps` | `20` | Frames per second for output videos |
| `--episode` | `0` | Episode index inside the RLBench variation directory |
| `--use_context_prompt` | `false` | Prepend ConceptGraphs context to task descriptions |
| `--no_merge` | `false` | Skip auto-merge when processing all variations |

### LeRobot Dataset Structure (Output)

```
datasets/lerobot/<task_name>_variation<num>/
├── meta/
│   ├── info.json              # Dataset metadata, features, splits
│   ├── stats.json             # Global min/max/mean/std/q01/q99
│   ├── tasks.parquet          # Task descriptions
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # Per-episode metadata and stats
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # Per-frame data
└── videos/
    ├── observation.images.front_rgb/
    │   └── chunk-000/
    │       ├── file-000.mp4   # Episode 0 front camera
    │       └── file-001.mp4   # Episode 1 front camera
    └── observation.images.wrist_rgb/
        └── chunk-000/
            ├── file-000.mp4
            └── file-001.mp4
```

**Data columns:**

| Column | Type | Description |
|---|---|---|
| `observation.state` | float32[8] | EEF pose (x,y,z,qx,qy,qz,qw) + gripper_open |
| `action` | float32[8] | Delta EEF action + gripper_open |
| `episode_index` | int64 | Episode this frame belongs to |
| `timestamp` | float64 | Time in seconds from episode start |
| `next.done` | bool | True on the last frame of each episode |
| `next.reward` | float64 | 0.0 except 1.0 on last frame |
| `index` | int64 | Global frame index across all episodes |
| `task_index` | int64 | Index into tasks.parquet |
| `annotation.human.action.task_description` | int64 | Task description index (GR00T) |
| `annotation.human.action.task_name` | int64 | Short task name index (GR00T) |
| `annotation.human.validity` | int64 | Validity label index (GR00T) |

**Episode splitting:** Trajectories are split at scene-graph transitions.
For example, if the scene graph transitions from "no relationships" →
"robot holding cup" → "no relationships", that creates 2 episodes:
- Episode 0: approach + grasp
- Episode 1: place + release

### Visualize a Converted Dataset

```bash
CUDA_VISIBLE_DEVICES="" python3 external/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id local/<task_name>_variation<num> \
    --root datasets/lerobot/<task_name>_variation<num> \
    --episode-index 0 \
    --save 1 \
    --output-dir datasets/lerobot/<task_name>_variation<num>/output \
    --batch-size 16 --num-workers 0 --tolerance-s 1e-4
```

Download the `.rrd` files and open with [Rerun](https://rerun.io/):

```bash
pip install rerun-sdk
rerun local_stack_cups_variation1_episode_0.rrd
```

---

## GR00T N1.5 Training on ASU Sol HPC

### System Info

- **OS:** Rocky Linux 8 (GLIBC 2.28)
- **GPU:** NVIDIA A100 (sm_80 / compute capability 8.0)
- **CUDA toolkit:** 12.6.1 (via module)
- **GCC:** 12.1.0 (via module)
- **Python:** 3.10 (conda env `lerobot`)

### 1. Load Modules & Activate Environment

```bash
module load mamba/latest
source activate lerobot
module load cuda-12.6.1-gcc-12.1.0
module load gcc-12.1.0-gcc-11.2.0
```

### 2. Install Core Python Packages

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e external/lerobot
```

> **Note:** If `setuptools` version errors occur during lerobot install,
> pin it first: `pip install "setuptools<81.0.0"`

### 3. Build flash-attn from Source

Pre-built wheels require GLIBC ≥ 2.32 which Sol (Rocky Linux 8) does not have.
You must build from source.

```bash
# Clone flash-attn
cd /scratch/$USER/tmp
git clone https://github.com/Dao-AILab/flash-attention.git flash-attn-src
cd flash-attn-src
```

**Patch `setup.py`** — Replace `os.rename` with `shutil.move` on the wheel-copy
line (~line 563) to fix cross-device link errors between `/tmp` and `/scratch`:

```python
# In setup.py, near the end of the file:
# BEFORE:
#     os.rename(wheel_filename, wheel_path)
# AFTER:
import shutil
shutil.move(wheel_filename, wheel_path)
```

**Build and install:**

```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # must be exactly "TRUE", not "1"
export FLASH_ATTN_CUDA_ARCHS=80           # A100 only; avoids unsupported sm_120
export MAX_JOBS=4                          # prevent OOM during compilation

pip install . 2>&1 | tee /scratch/$USER/tmp/flashattn_build.log
```

> Build takes ~20 minutes on Sol. Verify with:
> ```bash
> cd /scratch/$USER   # do NOT run from the source directory
> python -c "import flash_attn; print(flash_attn.__version__)"
> # Expected: 2.8.3 (or newer)
> ```

### 4. Install FFmpeg & PyAV (Video Backend)

`torchcodec` (the default lerobot video backend) has a C++11 ABI mismatch with
the pip-installed PyTorch on Sol. Use `pyav` as the video backend instead.

```bash
conda install -c conda-forge ffmpeg -y
pip install av   # PyAV — should already be installed with lerobot
```

When running training, always pass `--dataset.video_backend=pyav` (already set
in `scripts/train_groot_1gpu_smoke.sh`).

### 5. Run a Smoke Test

```bash
# Single variation
bash scripts/train_groot_1gpu_smoke.sh stack_cups_variation1

# All variations (merged dataset)
bash scripts/train_groot_1gpu_smoke.sh put_rubbish_in_bin_all datasets/lerobot/put_rubbish_in_bin_all

# Absolute path on HPC
bash scripts/train_groot_1gpu_smoke.sh put_rubbish_in_bin_all \
    /scratch/kpham34/cross_model_learning_based_robot_control/datasets/lerobot/put_rubbish_in_bin_all
```

The first positional argument is the dataset ID, the second (optional) is the
dataset root path (defaults to `datasets/lerobot/<DATASET_ID>`).

All parameters can be overridden via environment variables:

```bash
BATCH_SIZE=4 NUM_STEPS=1000 SAVE_FREQ=100 LOG_FREQ=10 NUM_PROCESSES=2 \
    bash scripts/train_groot_1gpu_smoke.sh put_rubbish_in_bin_all
```

### Troubleshooting (Sol-specific)

| Problem | Cause | Fix |
|---|---|---|
| `GLIBC_2.32 not found` when importing flash-attn | Pre-built wheel needs newer GLIBC | Build from source (step 3) |
| `OSError: [Errno 18] Invalid cross-device link` during flash-attn build | `os.rename()` across `/tmp` ↔ `/scratch` | Patch setup.py: `os.rename` → `shutil.move` |
| `FLASH_ATTENTION_FORCE_BUILD` ignored | Env var checked with `== "TRUE"` | Set exactly `TRUE`, not `1` or `true` |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | CUDA 12.6 doesn't support sm_120+ | Set `FLASH_ATTN_CUDA_ARCHS=80` |
| `ModuleNotFoundError: No module named 'flash_attn_2_cuda'` | Running python from flash-attn source dir | `cd` out of the source directory first |
| `undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev` in torchcodec | C++11 ABI mismatch with PyTorch | Use `--dataset.video_backend=pyav` instead |
