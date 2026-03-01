# RLBench Scene Graph Data Collection

This folder contains tools for creating **scene graph annotations** for RLBench episodes. The workflow has two steps:

1. **Create `info.json`** — define objects, relationships, and handle-to-object mappings
2. **Annotate scene graphs** — label per-frame object relationships in the episode

---

## Prerequisites

```bash
pip install imageio imageio-ffmpeg
# (numpy, opencv, matplotlib, ipywidgets should already be installed)
```

---

## Workflow

### Step 1: Create `info.json` (once per task)

Open **`create_info_json.ipynb`** and follow the cells:

1. **Configure** — set `TASK_NAME`, `VARIATION`, `EPISODE` in Cell 1
2. **Discover handles** — scan all masks to find unique handle IDs
3. **Visualize handles** — browse handles one by one. Each handle is highlighted on the RGB image so you can identify which object it belongs to. Use:
   - **Slider / dropdown** to browse individual handles
   - **"Show All Handles"** button to see all handles at once with a colour legend
   - **Frame selector** to try different frames (some objects may be occluded in certain frames)
4. **Define objects & relationships** — use the interactive form to:
   - Add object names (e.g. `robot`, `cup_1`, `cup_2`, `cup_3`)
   - Add relationship types (e.g. `holding`, `stacked`)
   - Map each handle ID to an object name
   - If `info.json` already exists, it will be pre-loaded so you can edit it
5. **Save** — click "Save info.json"
6. **Verify** — run the verification cell to see a coloured overlay confirming the mapping

The resulting file is saved at:
```
/workspace/datasets/rlbench/<task_name>/info.json
```

#### info.json format

```json
{
    "objects": {
        "0": { "name": "robot" },
        "1": { "name": "cup_1" },
        "2": { "name": "cup_2" }
    },
    "relations": {
        "0": { "name": "holding" },
        "1": { "name": "stacked" }
    },
    "object_mapping": {
        "31": "robot",
        "34": "robot",
        "35": "robot",
        "101": "cup_1",
        "106": "cup_2"
    }
}
```

- **objects**: logical entities in the scene (0-indexed)
- **relations**: types of relationships to annotate (0-indexed)
- **object_mapping**: maps RLBench mask handle IDs (integers, stored as strings) to object names. Multiple handles can map to the same object (e.g. multiple robot links).

---

### Step 2: Annotate Scene Graphs

Open **`RLBench_scene_graph_collection.ipynb`** and follow the cells:

1. **Configure** — set `TASK_NAME`, `VARIATION`, `EPISODE` in Cell 1. Objects and relationships are loaded automatically from `info.json`.
2. **Load episode** — loads RGB images, masks, and gripper states
3. **Extract masks** — scans masks and shows classified/unclassified handles
4. **Extract positions** — computes centre-of-mass positions for each object
5. **Annotate scene graph** — interactive editor with:

   | Control | Description |
   |---------|-------------|
   | **Slider** | Drag to browse frames |
   | **"Go to" box** | Type a frame number and press Enter to jump directly |
   | **← Prev / Next → Gripper Change** | Jump to the nearest gripper state change frame |
   | **Add Relationship** | Select two objects and a relationship type, then click to add. Applied to current frame and all subsequent frames. |
   | **Clear Selected Rel** | Remove a specific relationship from current frame onwards |
   | **Clear Frame Rels** | Remove all relationships from current frame only |
   | **Save Scene Graph** | Save to JSON |

   **Gripper state indicators:**
   - Frames where the gripper opens or closes are highlighted with a **red border** and bold text
   - The gripper state (OPEN/CLOSED) is shown for every frame
   - Use the gripper navigation buttons to quickly jump between state changes

6. **Generate videos** — creates three output files:
   - `episode_overlay.mp4` — RGB + coloured mask overlay + relationship arrows
   - `episode_mask.mp4` — vivid-colour mask video (easy to visualize)
   - `episode_mask_encoded.mp4` — encoded mask where pixel = (0, 0, object_index) for training
   - `object_color_map.json` — maps object indices to names and handle IDs

---

## Output Files

After completing both steps for a task/variation, you'll have:

```
/workspace/datasets/rlbench/<task_name>/
├── info.json                              # Task configuration (Step 1)
├── variation<N>/
│   ├── episodes/episode<M>/               # Raw RLBench data
│   │   ├── low_dim_obs.pkl
│   │   ├── front_rgb/
│   │   └── front_mask/
│   ├── <task>_scene_graph.json            # Scene graph annotations (Step 2)
│   ├── episode_overlay.mp4                # Overlay visualization video
│   ├── episode_mask.mp4                   # Vivid mask video
│   ├── episode_mask_encoded.mp4           # Encoded mask for training
│   └── object_color_map.json              # Object index mapping
```

---

## Mask Encoding

The **encoded** mask video (`episode_mask_encoded.mp4`) uses a simple encoding:
- Each pixel = `(0, 0, object_index)` in RGB
- Background = `(0, 0, 0)`
- Object indices are 1-based (matching `object_color_map.json`)

The **vivid** mask video (`episode_mask.mp4`) maps each index to a bright, distinguishable colour for easy visual inspection.

---

## Tips

- **Unmapped handles**: Not all handles need mapping. Background/table/irrelevant objects can be left unmapped — they'll be ignored in the scene graph.
- **Multiple handles per object**: The robot arm typically has 3+ handles (links, gripper fingers). Map all of them to `robot`.
- **Gripper changes**: The notebook auto-detects gripper open/close transitions — use the navigation buttons to jump to key moments for annotation.
- **Re-editing**: Both notebooks support re-running. `create_info_json.ipynb` loads existing `info.json` if present.
