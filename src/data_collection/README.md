# RLBench Scene Graph Data Collection

This folder contains tools for creating **scene graph annotations** for RLBench episodes. The workflow has three steps:

1. **Create `info.json`** — define objects, relationships, and handle-to-object mappings
2. **Annotate scene graphs** — label per-frame object relationships in the episode
3. **(Optional) Reuse relationship templates** — annotate once and apply relationships to other variations


## Target datasets for generate scenegraph:
1. stack_cups #
2. stack_blocks #
3. stack_chairs #
4. lamp_on #
5. lamp_off #
6. take umbrella out of umbrella_stand #
7. take_plate_off_colored_dish_rack #
8. take off weighting scale !?
9. take money out of safe #
10. remove cup !?
11. put umbrella in umbrella stand
12. put rubbish in bin
13. put knife on chopping board
14. put knife in knife block
15. push buttons
16. push button
17. pick up cup variation
18. meat on grill
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

---

## Relationship Template System (Optional)

The relationship template system lets you **annotate relationships once** and **apply them to other variations** of the same task.

### Key Concept: Gripper-Transition Templates

Templates are based on gripper transitions:
1. Identify gripper state changes (open→closed, closed→open)
2. Save relationships present at each transition
3. Apply those relationships to corresponding transitions in a new variation

Why this works well:
- Gripper transitions are semantically meaningful (grasp/release events)
- Relationship changes usually happen around those events
- Variations can differ in frame timing while preserving transition order

### Create a Template (One-Time Per Task)

1. Open `RLBench_scene_graph_collection.ipynb`
2. Set a reference variation (usually variation 0)
3. Annotate relationships in the scene graph editor
4. Click **Save as Template**

Template output path:
```
/workspace/datasets/rlbench/<task_name>/<task_name>_relationship_template.json
```

### Apply Template to Other Variations

#### Option A: Notebook (Interactive)

1. Switch `VARIATION` to a new variation
2. Run cells to the editor
3. Click **Load Template**
4. Review and adjust if needed
5. Click **Save Scene Graph**

#### Option B: Batch Script (Automated)

```bash
cd /workspace/src/data_collection

python apply_template_batch.py \
        --task stack_cups \
        --episode 0 \
        --camera front
```

This generates for each variation:
- `<task>_scene_graph.json`
- `episode_overlay.mp4`
- `episode_mask.mp4`
- `episode_mask_encoded.mp4`
- `object_color_map.json`

Faster mode (scene graphs only):

```bash
python apply_template_batch.py \
        --task stack_cups \
        --skip-videos
```

### Template JSON Format

```json
{
    "description": "Relationship template based on gripper transitions",
    "transitions": [
        {
            "transition_index": 0,
            "from_state": "open",
            "to_state": "closed",
            "relationships": [
                {
                    "object1": "robot",
                    "object2": "cup_1",
                    "type": "approaching"
                }
            ]
        }
    ]
}
```

### Programmatic API

Available in `src/utils/scene_graph_utils.py`:

- `save_relationship_template(scene_graph, gripper_states, output_path)`
- `apply_relationship_template(template_path, scene_graph, gripper_states)`

Example:

```python
import sys
import json
sys.path.insert(0, '/workspace/src')
import utils.scene_graph_utils as utils

demo, image_data = utils.load_episode_data('stack_cups', 1, 0)
gripper_states = utils.extract_gripper_states(demo)

scene_graph = {...}
template_path = '/workspace/datasets/rlbench/stack_cups/stack_cups_relationship_template.json'
utils.apply_relationship_template(template_path, scene_graph, gripper_states)

with open('/workspace/datasets/rlbench/stack_cups/variation1/stack_cups_scene_graph.json', 'w') as f:
        json.dump(scene_graph, f, indent=2)
```

### Template Troubleshooting

**Template not found**
- Ensure template exists at `/workspace/datasets/rlbench/<task>/<task>_relationship_template.json`
- Save a template once via **Save as Template**
- Or pass a custom path with `--template`

**Transition mismatch warnings**
- Check transition order/type between source and target variation
- Some variations may require a dedicated template

**Relationships look incorrect after apply**
- Verify gripper states are extracted correctly
- Inspect template JSON relationships
- Manually correct edge cases in the notebook

### Template Best Practices

- Use a representative reference variation
- Keep relationship naming consistent (`holding`, `on`, `above`, etc.)
- Validate 1–2 target variations before full batch processing
- Use `--skip-videos` when you only need scene graph JSONs
- See `batch_examples.sh` for ready-to-use command patterns
