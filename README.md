# Cross-Model Learning-Based Robot Control

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