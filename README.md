# Cross-Model Learning-Based Robot Control

## RLBench Dataset Generation — Headless Docker + OSMesa

Get started
--------
```bash
cd external/RLBench \
docker-compose up -d
```

Inside docker containter, run 

```bash
python dataset_generator.py \
    --tasks stack_cups \
    --variations 2 \
    --processes 1 \
    --episodes_per_task 1 \
    --save_path /workspace/datasets/rlbench \
    --image_size 256 256 \
    --renderer opengl3
```

to generate dataset 



Scene Graph Generation Environment
----------------------------------
Create/activate your conda environment and install the Python tooling used
for scene graph generation:

```bash
conda activate comp_robotics
conda install -c conda-forge opencv ipywidgets matplotlib jupyterlab gymnasium -y
```

and follow the jupyter notebook in examples/scene_graph_analyzer.ipynb to annotate dataset

Troubleshooting
--------


Quick install (OSMesa + Xvfb)
----------------------------
Run these as root or with sudo inside the container to install needed packages:

```bash
apt-get update && apt-get install -y \
    mesa-utils \
    x11-utils \
    libosmesa6 \
    libosmesa6-dev \
    xvfb
```

Add this env variable

```
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_X11_NO_MITSHM=1
export QT_QPA_PLATFORM=xcb
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libOSMesa.so.6   # change if different
export DISPLAY=:99
```

Start a headless X server (Xvfb)
--------------------------------
Start Xvfb and export DISPLAY before running any renderer-dependent code:

```bash
Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb-99.log 2>&1 &
export DISPLAY=:99
sleep 0.5
```


## Dataset Conversion: RLBench → LeRobot v3

### Prerequisites

- Python 3.10+ with conda env `lerobot_gpu` activated
- Packages: `pandas`, `pyarrow`, `numpy`, `Pillow`, `ffmpeg` (CLI)
- For visualization: `lerobot`, `rerun-sdk`

```bash
source activate lerobot_gpu
```

### RLBench Dataset Structure (Input)

Each task/variation must follow this layout:

```
datasets/rlbench/<task_name>/variation<num>/
├── episodes/
│   └── episode0/
│       ├── front_rgb/        # 0.png, 1.png, ..., N.png
│       ├── wrist_rgb/        # 0.png, 1.png, ..., N.png
│       ├── front_mask/       # 0.png, 1.png, ..., N.png
│       └── low_dim_obs.pkl   # RLBench Demo object with observations
├── episode_mask.mp4          # Segmentation mask video
├── <task_name>_scene_graph.json   # Per-frame scene graph with relationships
└── object_color_map.json     # Object name → mask color mapping
```

**Scene graph JSON** must contain `"objects"` and `"frames"` arrays. Each frame has `"frame_id"`, `"relationships"` (list of `{object1, object2, type}`). Transitions between different relationship states define episode boundaries.

### Convert a Dataset

```bash
python src/convert_rlbench_to_lerobot.py \
    --task_name stack_cups \
    --variation 1 \
    --rlbench_root datasets/rlbench \
    --output_root datasets/lerobot \
    --fps 10
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--task_name` | (required) | RLBench task name, e.g. `stack_cups` |
| `--variation` | (required) | Variation number, e.g. `1` |
| `--rlbench_root` | `datasets/rlbench` | Root directory of RLBench datasets |
| `--output_root` | `datasets/lerobot` | Root directory for output LeRobot datasets |
| `--fps` | `10` | Frames per second for output videos and timestamps |
| `--episode` | `0` | Episode index inside the RLBench variation directory |

### LeRobot Dataset Structure (Output)

The output at `datasets/lerobot/<task_name>_variation<num>/` follows LeRobot v3 format:

```
datasets/lerobot/<task_name>_variation<num>/
├── meta/
│   ├── info.json              # Dataset metadata, features, splits, object color map, scene graphs
│   ├── stats.json             # Global min/max/mean/std for all features
│   ├── tasks.parquet          # Task descriptions (one per transition-episode)
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet   # Per-episode metadata and stats
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # Per-frame data: observation.state, action, timestamps, etc.
└── videos/
    ├── observation.images.front_rgb/
    │   └── chunk-000/
    │       ├── file-000.mp4   # Episode 0 front camera
    │       └── file-001.mp4   # Episode 1 front camera
    ├── observation.images.wrist_rgb/
    │   └── chunk-000/
    │       ├── file-000.mp4
    │       └── file-001.mp4
    └── observation.images.mask/
        └── chunk-000/
            ├── file-000.mp4   # Episode 0 segmentation mask
            └── file-001.mp4
```

**Data columns** in the parquet file:

| Column | Type | Description |
|---|---|---|
| `observation.state` | float32[8] | EEF pose (x,y,z,qx,qy,qz,qw) + gripper_open |
| `action` | float32[8] | Next-step EEF target (same format as state) |
| `episode_index` | int64 | Which episode this frame belongs to |
| `frame_index` | int64 | Frame index within the episode (0-based) |
| `timestamp` | float32 | Time in seconds from episode start |
| `next.done` | bool | True on the last frame of each episode |
| `index` | int64 | Global frame index across all episodes |
| `task_index` | int64 | Task/transition index |

**Episode splitting:** The trajectory is split at scene-graph transitions. For example, if the scene graph goes from "no relationships" → "robot holding cup_1" → "no relationships", that creates 2 episodes:
- Episode 0: approach + grasp (begin: `[]`, end: `[robot holding cup_1]`)
- Episode 1: place + release (begin: `[robot holding cup_1]`, end: `[]`)

### Visualize a Converted Dataset

Save a `.rrd` file for offline viewing with [Rerun](https://rerun.io/):

```bash
# Visualize episode 0
CUDA_VISIBLE_DEVICES="" python3 external/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id local/<task_name>_variation<num> \
    --root datasets/lerobot/<task_name>_variation<num> \
    --episode-index 0 \
    --save 1 \
    --output-dir datasets/lerobot/<task_name>_variation<num>/output \
    --batch-size 16 \
    --num-workers 0 \
    --tolerance-s 1e-4
```

**Example** (stack_cups variation 1, both episodes):

```bash
# Episode 0
CUDA_VISIBLE_DEVICES="" python3 external/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id local/stack_cups_variation1 \
    --root datasets/lerobot/stack_cups_variation1 \
    --episode-index 0 \
    --save 1 \
    --output-dir datasets/lerobot/stack_cups_variation1/output \
    --batch-size 16 --num-workers 0 --tolerance-s 1e-4

# Episode 1
CUDA_VISIBLE_DEVICES="" python3 external/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
    --repo-id local/stack_cups_variation1 \
    --root datasets/lerobot/stack_cups_variation1 \
    --episode-index 1 \
    --save 1 \
    --output-dir datasets/lerobot/stack_cups_variation1/output \
    --batch-size 16 --num-workers 0 --tolerance-s 1e-4
```

The `.rrd` files are saved to `datasets/lerobot/<task_name>_variation<num>/output/`. Download them to your local machine and open with:

```bash
pip install rerun-sdk   # if not installed
rerun local_stack_cups_variation1_episode_0.rrd
```

**Notes:**
- `CUDA_VISIBLE_DEVICES=""` skips GPU initialization (faster startup on cluster nodes)
- `--repo-id` is just a label — it doesn't need to exist on HuggingFace
- `--num-workers 0` avoids multiprocessing issues on the cluster
- `--tolerance-s 1e-4` relaxes timestamp validation

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
bash scripts/train_groot_1gpu_smoke.sh
```

This runs 1 training step with batch_size=1 to verify the full pipeline works
(data loading, model forward pass, loss computation, checkpoint save).

### Troubleshooting (Sol-specific)

| Problem | Cause | Fix |
|---|---|---|
| `GLIBC_2.32 not found` when importing flash-attn | Pre-built wheel needs newer GLIBC | Build from source (step 3) |
| `OSError: [Errno 18] Invalid cross-device link` during flash-attn build | `os.rename()` across `/tmp` ↔ `/scratch` | Patch setup.py: `os.rename` → `shutil.move` |
| `FLASH_ATTENTION_FORCE_BUILD` ignored | Env var checked with `== "TRUE"` | Set exactly `TRUE`, not `1` or `true` |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | CUDA 12.6 doesn't support sm_120+ | Set `FLASH_ATTN_CUDA_ARCHS=80` |
| `ModuleNotFoundError: No module named 'flash_attn_2_cuda'` | Running python from flash-attn source dir | `cd` out of the source directory first |
| `undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev` in torchcodec | C++11 ABI mismatch with PyTorch | Use `--dataset.video_backend=pyav` instead |
