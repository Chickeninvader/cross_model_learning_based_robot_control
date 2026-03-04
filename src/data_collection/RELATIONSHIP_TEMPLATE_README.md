# Scene Graph Relationship Template System

## Overview

The relationship template system allows you to **annotate relationships once** and **automatically apply them to other variations** of the same task. This is especially useful when you have multiple variations of a task (e.g., `stack_cups` with different object positions) where the relationship patterns are similar but occur at different frames due to gripper state changes.

## Key Concept: Gripper-Transition Based Templates

The system works by:
1. **Identifying gripper state transitions** (open→closed, closed→open)
2. **Saving relationships that exist at each transition**
3. **Applying those relationships to corresponding transitions** in other variations

This approach is robust because:
- ✅ Gripper transitions are **semantically meaningful** (e.g., grasp, release)
- ✅ Relationships typically change at these transitions
- ✅ Works across variations even if timing differs

## Workflow

### Step 1: Create a Template (One-Time Setup)

1. **Open the notebook**: `src/data_collection/RLBench_scene_graph_collection.ipynb`

2. **Configure for your reference variation** (e.g., variation 0):
   ```python
   TASK_NAME = 'stack_cups'
   VARIATION = 0  # Your reference variation
   EPISODE = 0
   CAMERA = 'front'
   ```

3. **Run all cells** up to the scene graph editor

4. **Manually annotate relationships** using the interactive editor:
   - Navigate to gripper transition frames
   - Add relationships (e.g., `robot` --on--> `cup_1`)
   - Relationships persist forward until changed

5. **Save as Template**: Click the "Save as Template" button
   - Template saved to: `{DATASET_PATH}/{TASK_NAME}/{TASK_NAME}_relationship_template.json`

### Step 2: Apply Template to Other Variations

#### Option A: Using the Notebook (Interactive)

1. **Change configuration** to new variation:
   ```python
   VARIATION = 1  # New variation
   ```

2. **Run all cells** up to the scene graph editor

3. **Load Template**: Click the "Load Template" button
   - Automatically applies relationships at corresponding gripper transitions
   - Review the results in the editor

4. **Adjust if needed**: Manually fix any mismatches

5. **Save Scene Graph**: Click "Save Scene Graph" button

#### Option B: Using Batch Script (Automated)

Process multiple variations at once:

```bash
cd /workspace/src/data_collection

python apply_template_batch.py \
    --task stack_cups \
    --variations 1 2 3 4 5 \
    --episode 0 \
    --camera front
```

**This automatically generates for each variation:**
- ✅ Scene graph JSON with relationships
- ✅ Overlay video (RGB + mask + scene graph)
- ✅ Mask video (vivid colors)
- ✅ Encoded mask video (for training)
- ✅ Object color map JSON

**Speed options:**
```bash
# Skip video generation (faster, only creates scene graphs)
python apply_template_batch.py \
    --task stack_cups \
    --variations 1 2 3 4 5 \
    --skip-videos
```

## Template File Format

The template JSON structure:

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
    },
    {
      "transition_index": 1,
      "from_state": "closed",
      "to_state": "open",
      "relationships": [
        {
          "object1": "robot",
          "object2": "cup_1",
          "type": "holding"
        }
      ]
    }
  ]
}
```

## Use Cases

### 1. Same Task, Different Variations
**Example**: `stack_cups` with different initial cup positions
- Template captures: when gripper closes → robot grasps cup_1
- Applies to all variations regardless of cup position

### 2. Similar Tasks (with adaptation)
**Example**: `stack_cups` → `stack_blocks`
- Copy template file
- Edit object names in template JSON
- Apply to new task

### 3. Batch Processing Datasets
Process entire dataset efficiently with all visualizations:
```bash
# Annotate variation 0 manually for each task
# Then batch process variations 1-10 (includes videos)
for task in stack_cups stack_blocks pick_and_place; do
    python apply_template_batch.py \
        --task $task \
        --variations 1 2 3 4 5 6 7 8 9 10
done

# Result: Scene graphs + videos for all variations automatically generated
```

## Advanced Usage

### Programmatic Template Application

```python
import sys
sys.path.insert(0, '/workspace/src')
import utils.scene_graph_utils as utils

# Load data
demo, image_data = utils.load_episode_data('stack_cups', 1, 0)
gripper_states = utils.extract_gripper_states(demo)

# Build scene graph (simplified - see notebook for full version)
scene_graph = {...}  # Your scene graph structure

# Apply template
template_path = '/workspace/datasets/rlbench/stack_cups/stack_cups_relationship_template.json'
utils.apply_relationship_template(template_path, scene_graph, gripper_states)

# Save result
output_file = '/workspace/datasets/rlbench/stack_cups/variation1/stack_cups_scene_graph.json'
with open(output_file, 'w') as f:
    json.dump(scene_graph, f, indent=2)
```

### Custom Template Editing

You can manually edit template files to:
- Add new relationship types
- Modify transition conditions
- Combine templates from multiple variations

## API Reference

### `save_relationship_template(scene_graph, gripper_states, output_path)`
Extract and save relationship template from annotated scene graph.

**Args:**
- `scene_graph` (dict): Annotated scene graph with relationships
- `gripper_states` (list[bool]): Gripper closed states per frame
- `output_path` (str): Where to save template JSON

**Returns:**
- Template dict

### `apply_relationship_template(template_path, scene_graph, gripper_states)`
Apply template to new scene graph based on gripper transitions.

**Args:**
- `template_path` (str): Path to template JSON file
- `scene_graph` (dict): Scene graph to modify (in-place)
- `gripper_states` (list[bool]): Gripper closed states for this variation

**Returns:**
- Modified scene_graph

## Tips & Best Practices

1. **Choose a representative variation for the template**
   - Pick variation 0 or one with typical behavior
   - Ensure all gripper transitions are clear

2. **Use clear relationship names**
   - `approaching`, `holding`, `on`, `above`, etc.
   - Consistent naming helps reusability

3. **Review auto-applied templates**
   - Always check results after batch processing
   - Edge cases may need manual adjustment

4. **Version control your templates**
   - Track template files in git
   - Document any manual edits

5. **Test on one variation first**
   - Before batch processing, verify template works on 1-2 variations
   - Adjust template if needed

6. **Video generation can be slow**
   - Use `--skip-videos` flag if you only need scene graphs
   - Generate videos later for selected variations only

7. **See batch_examples.sh for more usage examples**
   - Located in `src/data_collection/batch_examples.sh`
   - Contains ready-to-use command patterns

## Troubleshooting

### Template not found
**Problem**: "Template not found" error
**Solution**: 
1. Check template file exists at `{DATASET_PATH}/{TASK_NAME}/{TASK_NAME}_relationship_template.json`
2. Ensure you saved template using "Save as Template" button
3. Use `--template` flag to specify custom path

### Transition mismatch
**Problem**: "Transition type mismatch" warnings
**Solution**:
- Verify gripper transitions match between variations
- Check if variation has different number of transitions
- May need separate templates for different task scenarios

### Relationships not applied correctly
**Problem**: Template applies but relationships seem wrong
**Solution**:
1. Check gripper state change frames match expected transitions
2. Review template JSON to ensure relationships are correct
3. Manually adjust after template application
4. Consider creating task-specific template for edge cases

## File Structure

```
/workspace/
├── datasets/rlbench/
│   └── stack_cups/
│       ├── info.json                                    # Task configuration
│       ├── stack_cups_relationship_template.json       # Template file
│       ├── variation0/
│       │   ├── stack_cups_scene_graph.json            # Manually annotated
│       │   ├── episode_overlay.mp4                     # RGB + mask + scene graph
│       │   ├── episode_mask.mp4                        # Vivid mask video
│       │   ├── episode_mask_encoded.mp4                # Encoded mask (training)
│       │   ├── object_color_map.json                   # Object colors
│       │   └── ...
│       ├── variation1/
│       │   ├── stack_cups_scene_graph.json            # Auto-generated from template
│       │   ├── episode_overlay.mp4                     # Auto-generated
│       │   ├── episode_mask.mp4                        # Auto-generated
│       │   ├── episode_mask_encoded.mp4                # Auto-generated
│       │   ├── object_color_map.json                   # Auto-generated
│       │   └── ...
│       └── ...
└── src/
    ├── utils/scene_graph_utils.py                      # Template functions
    └── data_collection/
        ├── RLBench_scene_graph_collection.ipynb       # Interactive editor
        ├── apply_template_batch.py                     # Batch processing script
        ├── batch_examples.sh                           # Usage examples
        └── RELATIONSHIP_TEMPLATE_README.md            # This file
```

## Example: Complete Workflow

```bash
# 1. Annotate reference variation in notebook
#    - Open RLBench_scene_graph_collection.ipynb
#    - Set VARIATION = 0
#    - Run all cells, annotate relationships
#    - Click "Save as Template"

# 2. Batch process other variations (includes video generation)
cd /workspace/src/data_collection
python apply_template_batch.py \
    --task stack_cups \
    --variations 1 2 3 \
    --camera front

# 3. Verify results
ls -l /workspace/datasets/rlbench/stack_cups/variation*/stack_cups_scene_graph.json
ls -l /workspace/datasets/rlbench/stack_cups/variation*/episode_overlay.mp4

# 4. View videos (optional - download from container)
# Videos are in MP4 format and can be played in any media player
```

## Contributing

To improve the template system:
1. Add new relationship types to `info.json`
2. Extend template matching logic in `scene_graph_utils.py`
3. Update notebook UI for new features
4. Document changes in this README

## License

See project root LICENSE file.
