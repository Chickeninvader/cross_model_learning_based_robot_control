# Environment Creation Guide

Comprehensive reference for creating custom robosuite environments for cross-embodiment learning and manipulation research.

## Key Features

- **Arena-Aware Placement:** Objects are automatically placed correctly for each arena type (bins, table, pegs, etc.)
- **Automatic Size Validation:** Object sizes are validated based on arena type and count - errors prevent invalid configurations
- **18+ Object Types:** Primitives (box, ball, cylinder, capsule), household items (bottle, milk, bread), and tools (hammer, cone, pot)
- **6 Arena Types:** Table, bins, pegs, empty, wipe, multi-table
- **Multiple Layouts:** Random, grid, circle, line, or exact positions
- **Advanced Placement:** UniformRandomSampler, SequentialCompositeSampler for complex scenes
- **Built-in Robosuite Environments:** Access to Lift, PickPlace, Stack, NutAssembly, Door, Wipe, and more

## Basic Command Structure

```bash
python src/create_env.py --save NAME [OPTIONS]
```

**Required:** `--save NAME` - Save configuration with this name

## Most Common Options

### Objects

```bash
--num-objects N              # Number of objects (default: 3)
--type TYPE [TYPE ...]       # Object types (can specify multiple)
--sizes SIZE [SIZE ...]      # Individual sizes (cycles if fewer than objects)
--colors COLOR [COLOR ...]   # Colors or 'random'
```

**Object Types:**
- **Primitives:** `box`, `ball`, `cylinder`, `capsule`
- **Household:** `bottle`, `can`, `milk`, `bread`, `cereal`, `lemon`
- **Nuts/Tools:** `square_nut`, `round_nut`, `cone`, `hammer`, `pot`

### Arena Types

```bash
--arena-type TYPE           # Choose workspace type
```

**Available Arenas:**
- `table` - Standard flat table (default)
- `bins` - Two side-by-side bins for sorting tasks
- `pegs` - Table with fixed pegs
- `empty` - Empty space, no table
- `wipe` - Table with surface markers
- `multi_table` - Multiple tables

### Layout Modes

```bash
--layout MODE               # How to arrange objects
```

**Layout Options:**
- `random` - Random placement (default)
- `grid` - Grid pattern (use with `--grid-size RxC`)
- `circle` - Circular arrangement
- `line` - Line arrangement

### Robot Selection

```bash
--robot ROBOT              # Robot type (default: Panda)
```

**Available:** `Panda`, `Sawyer`

**Note:** While the project currently supports Panda and Sawyer, robosuite includes additional robots (Baxter, iiwa, Jaco, Kinova3, UR5e, Xarm7, Tiago) that can be integrated if needed.

## Complete Object and Arena Reference

### All Available Objects

#### Primitive Objects (Customizable Size and Color)

- **box** - Cubic/rectangular box
  - Size: single value (cube) or [x, y, z] (rectangular)
  - Default: 0.025m
  - Example: `--type box --size 0.03` or `--sizes 0.02 0.03 0.04`

- **ball** (or **sphere**) - Spherical object
  - Size: radius
  - Default: 0.025m
  - Example: `--type ball --size 0.02`

- **cylinder** - Cylindrical object
  - Size: [radius, height] or single value (height = 2×radius)
  - Default: 0.025m
  - Example: `--type cylinder --size 0.03`

- **capsule** - Capsule (cylinder with rounded ends)
  - Size: [radius, half-length] or single value
  - Default: 0.025m
  - Example: `--type capsule --size 0.02`

#### Household Objects (Pre-defined Meshes)

- **bottle** - Plastic water bottle
  - Fixed appearance, realistic mesh
  - Approx size: 0.05m height

- **can** - Soda/cola can
  - Fixed appearance with texture
  - Approx size: 0.04m height

- **milk** - Milk carton
  - Rectangular carton with texture
  - Approx size: 0.06m height

- **bread** - Bread loaf
  - Soft bread appearance
  - Approx size: 0.08m length

- **cereal** - Cereal box
  - Rectangular box with texture
  - Approx size: 0.12m height

- **lemon** - Lemon fruit
  - Ellipsoidal shape with texture
  - Approx size: 0.03m

#### Tools and Nuts

- **square_nut** - Square nut for assembly tasks
  - Square hole in center
  - Used in NutAssembly environment
  - Approx size: 0.04m

- **round_nut** - Round nut for assembly tasks
  - Circular hole in center
  - Used in NutAssembly environment
  - Approx size: 0.04m

- **hammer** - Hammer tool (composite object)
  - Handle + head assembly
  - Approx size: 0.15m length

- **cone** - Traffic cone shape
  - Conical composite object
  - Approx size: 0.08m height

- **pot** - Pot with handles
  - Cylindrical pot with two handles
  - Approx size: 0.1m diameter

### All Available Arenas

#### Table Arena (Default)
- **Type:** `table`
- **Description:** Standard flat table surface
- **Size:** Customizable (default: 0.8×0.8×0.05m)
- **Placement Range:** ±0.15m (X and Y)
- **Best For:** General manipulation, pick-and-place tasks
- **Objects Supported:** All objects
- **Example:**
  ```bash
  python src/create_env.py --arena-type table --num-objects 5 --type box ball --save table_env
  ```

#### Bins Arena
- **Type:** `bins`
- **Description:** Two side-by-side bins for sorting tasks
- **Size:** Each bin ~0.39×0.49m (customizable via table-size)
- **Placement Range:** Objects placed in bin1 (left bin) by default
  - X range: ±0.145m from bin center
  - Y range: ±0.195m from bin center
- **Best For:** Sorting tasks, bin picking
- **Objects Supported:** All objects (size limits apply - see size validation)
- **Special:** Boundary checking ensures objects stay within bin walls
- **Example:**
  ```bash
  python src/create_env.py --arena-type bins --num-objects 6 --type ball --sizes 0.02 --save sorting_task
  ```

#### Pegs Arena
- **Type:** `pegs`
- **Description:** Table with fixed vertical pegs
- **Size:** 0.8×0.8m table with pegs
- **Placement Range:** ±0.12m (tighter to avoid pegs)
- **Best For:** Peg insertion, obstacle avoidance
- **Objects Supported:** All objects, especially round_nut and square_nut
- **Example:**
  ```bash
  python src/create_env.py --arena-type pegs --num-objects 2 --type round_nut square_nut --save peg_task
  ```

#### Empty Arena
- **Type:** `empty`
- **Description:** Empty space with no surfaces
- **Size:** Unlimited
- **Placement Range:** Objects float at z=0.5m height
- **Best For:** Testing physics, aerial manipulation concepts
- **Objects Supported:** All objects (will float in mid-air)
- **Example:**
  ```bash
  python src/create_env.py --arena-type empty --num-objects 3 --type box --save floating_objects
  ```

#### Wipe Arena
- **Type:** `wipe`
- **Description:** Table with surface markers for wiping tasks
- **Size:** 0.8×0.8m table with visual markers
- **Placement Range:** ±0.15m
- **Best For:** Surface cleaning, wiping demonstrations
- **Objects Supported:** All objects
- **Example:**
  ```bash
  python src/create_env.py --arena-type wipe --num-objects 2 --type cylinder --save wipe_task
  ```

#### Multi-Table Arena
- **Type:** `multi_table`
- **Description:** Multiple tables in the workspace
- **Size:** Customizable table positions
- **Placement Range:** Objects placed on first table by default
- **Best For:** Multi-surface manipulation, transfer tasks
- **Objects Supported:** All objects
- **Note:** Advanced users can customize table positions in code
- **Example:**
  ```bash
  python src/create_env.py --arena-type multi_table --num-objects 4 --type box ball --save multi_surface
  ```

## Common Usage Examples

### Basic Environments

```bash
# Simple random objects
python src/create_env.py --num-objects 5 --type box --save my_env

# Mixed object types
python src/create_env.py --num-objects 6 --type box ball cylinder --save mixed

# Household items
python src/create_env.py --num-objects 4 --type bottle can milk bread --save kitchen
```

### Different Sizes

```bash
# IMPORTANT: Use --sizes (plural) for different sizes
python src/create_env.py --num-objects 3 --type box --sizes 0.02 0.05 0.08 --save varied_boxes

# All same size (use --size singular)
python src/create_env.py --num-objects 5 --type ball --size 0.03 --save same_balls
```

### Arena Variations

```bash
# Bins for sorting tasks (objects automatically placed inside bins)
python src/create_env.py --arena-type bins --num-objects 6 --type ball --save sorting_task

# Empty space (floating objects)
python src/create_env.py --arena-type empty --num-objects 3 --type box --save floating

# Table with pegs (tighter placement to avoid pegs)
python src/create_env.py --arena-type pegs --num-objects 2 --type round_nut --save peg_task
```

**Note:** Object placement is automatically adjusted for each arena type:
- **bins**: Objects placed inside bin1 with proper bin dimensions (0.39x0.49m)
  - Placement range: ±0.145m (X), ±0.195m (Y)
  - Reference position: (0.1, -0.25, 0.8)
  - **Boundary checking enabled**: Entire object (including size) stays within bin
  - Z-offset: 0.02m above bin bottom
- **empty**: Objects float in mid-air at z=0.5m
- **pegs**: Tighter placement (±0.12m) to avoid fixed pegs
- **table/wipe**: Standard table surface placement (±0.15m)

**Important for Bins:** The placement system ensures the entire object boundary stays within the bin walls. This means larger objects will be placed closer to the center to prevent them from clipping through bin walls and falling to the ground.

## Understanding Size Limits

The system **automatically calculates safe size limits** based on:
1. **Arena type** - Bins have less space than tables
2. **Object count** - More objects require smaller individual sizes
3. **Arena dimensions** - Custom table sizes affect limits

**Formula:** `max_size = usable_space / (sqrt(object_count) * packing_factor)`

This means:
- **Doubling the objects** reduces max size by ~40% (not 50%)
- **Larger arenas** allow larger objects
- **Bins arena** is most restrictive (smallest workspace)

### Example Size Limits

**Bins Arena (0.39×0.49m):**
- 1 object: max 0.081m (8.1cm)
- 3 objects: max 0.047m (4.7cm) ← Your command with 0.05m will FAIL
- 5 objects: max 0.036m (3.6cm)
- 10 objects: max 0.026m (2.6cm)

**Table Arena (0.8×0.8m):**
- 3 objects: max 0.096m (9.6cm)
- 5 objects: max 0.075m (7.5cm)
- 10 objects: max 0.053m (5.3cm)

### Layout Patterns

```bash
# Grid layout
python src/create_env.py --layout grid --grid-size 3x3 --type ball --save grid_env

# Circular arrangement
python src/create_env.py --layout circle --num-objects 6 --type box --save circle_env

# Line arrangement
python src/create_env.py --layout line --num-objects 5 --type cylinder --save line_env
```

### Color Customization

```bash
# Random colors
python src/create_env.py --num-objects 5 --type box --colors random --save colorful

# Specific colors (cycles if fewer than objects)
python src/create_env.py --num-objects 6 --type ball --colors red blue green --save rgb_balls
```

**Available Colors:** red, green, blue, yellow, cyan, magenta, orange, purple, white, gray, black, random

## Using Your Environment

After creating an environment, use it with:

```bash
# Teleoperation with SpaceMouse
python src/run_simulation.py --mode teleop --env NAME

# Automated simulation
python src/run_simulation.py --mode sim --env NAME --episodes 10
```

## Advanced Options (Less Common)

### Exact Positioning

```bash
--positions "X,Y" "X,Y" ...   # Exact XY positions for each object
```

Example:
```bash
python src/create_env.py --positions "0.1,0.1" "-0.1,0.0" "0.0,-0.1" --save exact_pos
```

### Table Customization

```bash
--table-size X Y Z            # Table dimensions (default: 0.8 0.8 0.05)
--table-height H              # Table height (default: 0.8)
--density D                   # Object density kg/m³ (default: 100)
```

### Custom Placement Ranges

By default, placement ranges are automatically adjusted for each arena type. You can override:

```bash
--placement-range R          # Symmetric range ±R meters
--x-range MIN MAX           # Custom X range
--y-range MIN MAX           # Custom Y range
```

Examples:
```bash
# Tighter placement in bins
python src/create_env.py --arena-type bins --x-range -0.05 0.05 --y-range -0.05 0.05 --save tight_bins

# Wider spread on table
python src/create_env.py --placement-range 0.25 --save wide_spread
```

## Preset Environments

```bash
--preset NAME               # Use a preset configuration
```

**Available Presets:** `stacking`, `sorting`, `obstacles`, `pick_place`

Example:
```bash
python src/create_env.py --preset stacking --save my_stacking_task
```

## Tips for Cross-Embodiment Learning

1. **Variety over complexity:** Use multiple simple objects rather than complex single objects
2. **Different sizes:** Use `--sizes` to create varied object dimensions
3. **Mixed types:** Combine different object types for better generalization
4. **Arena diversity:** Create environments with different arenas (table, bins, empty)
5. **Color variation:** Use random colors or diverse color palettes
6. **Object size for bins:** Keep objects smaller (<0.05m) when using bins arena to ensure proper placement

## Quick Reference Card

```bash
# Most useful flags for data collection:
--save NAME                              # Required: save name
--num-objects N                          # How many objects
--type box ball bottle can               # Object types (space-separated)
--sizes 0.02 0.05 0.08                   # Individual sizes
--colors random                          # Random colors
--arena-type table                       # Arena type
--layout random                          # Layout mode
--robot Panda                            # Robot type
```

## Example: Complete Environment for Play Data

```bash
python src/create_env.py \
  --save play_data_env \
  --num-objects 8 \
  --type box ball cylinder bottle can \
  --sizes 0.02 0.03 0.04 0.05 \
  --colors random \
  --arena-type table \
  --layout random \
  --robot Panda
```

This creates a diverse environment with 8 objects of mixed types and sizes, perfect for collecting varied manipulation data.

## Troubleshooting

### Size Validation Errors

The system will **automatically reject** configurations with objects too large for the arena:

```bash
# This WILL FAIL - 0.05m exceeds limit for 3 objects in bins (0.047m)
python src/create_env.py --arena-type bins --num-objects 3 --sizes 0.01 0.02 0.05 --save bad

# Error message will show:
# ERROR: Object sizes too large for bins arena (0.39m × 0.49m)!
# Object 3 (box_2): size 0.050m exceeds limit 0.047m
# Suggestions:
#   1. Reduce object sizes to ≤0.047m: --sizes 0.047 0.047 0.047
#   2. Use fewer objects (fewer objects = larger max size allowed)
```

**Solutions:**

1. **Reduce sizes** (recommended):
   ```bash
   python src/create_env.py --arena-type bins --num-objects 3 --sizes 0.01 0.02 0.04 --save good_bins
   ```

2. **Use fewer objects** (allows larger sizes):
   ```bash
   python src/create_env.py --arena-type bins --num-objects 2 --sizes 0.01 0.05 --save fewer_bins
   # With 2 objects, max size is 0.057m, so 0.05m is valid
   ```

3. **Use table arena** (more space):
   ```bash
   python src/create_env.py --arena-type table --num-objects 3 --sizes 0.01 0.02 0.05 --save table_env
   # With 3 objects on table, max size is 0.096m, so 0.05m is valid
   ```

4. **Larger bins** (custom dimensions):
   ```bash
   python src/create_env.py --arena-type bins --table-size 0.6 0.6 0.82 --num-objects 3 --sizes 0.01 0.02 0.05 --save big_bins
   ```

### Objects Not Reachable

If the robot can't reach objects:
- For bins: Objects are automatically placed in bin1 (left bin)
- For table: Increase placement range with `--placement-range 0.2`
- Check robot type matches your setup (`--robot Panda` or `--robot Sawyer`)

## Attribute Relationships

Understanding how different parameters affect each other:

### Arena Type ↔ Max Object Size
- **bins** (0.39×0.49m): Smallest workspace → strictest limits
- **pegs** (0.8×0.8m with obstacles): Medium workspace → medium limits
- **table** (0.8×0.8m): Largest workspace → most permissive

### Object Count ↔ Max Object Size
**Inverse square root relationship:**
```
Objects:  1     2     3     5     10
Bins max: 8.1cm 5.7cm 4.7cm 3.6cm 2.6cm  (bins)
Table max: 16.6cm 11.7cm 9.6cm 7.5cm 5.3cm (table)
```

**Scaling rule:** To allow objects **2x larger**, use **4x fewer** objects

### Arena Dimensions ↔ Max Object Size
**Direct proportional relationship:**
```
# Default bins (0.39×0.49m), 3 objects
max_size = 0.047m

# Larger bins (0.6×0.6m), 3 objects
max_size = 0.073m  (55% larger arena → 55% larger objects)
```

### Example: Planning Object Sizes

**Goal:** Place 5 ball objects in bins

1. Check limit: `5 objects in bins → max 0.036m (3.6cm)`
2. Choose sizes: `--sizes 0.01 0.02 0.025 0.03 0.035`
3. All ≤ 0.036m → Valid!

**Goal:** Place 5 objects sized 0.05m

1. Too large for bins (limit 0.036m)
2. Options:
   - Reduce to 2 objects (limit becomes 0.057m)
   - Switch to table arena (limit becomes 0.075m)
   - Increase bin size to 0.6×0.6m (limit becomes 0.056m)

## Advanced Placement Methods

### Placement Samplers

The environment creation system uses robosuite's placement sampler framework:

#### UniformRandomSampler
- **Purpose:** Random placement within specified ranges
- **Features:**
  - Configurable X and Y ranges
  - Optional rotation (fixed, range, or full random)
  - Boundary checking (ensure objects stay within range)
  - Collision avoidance (prevents overlapping objects)
- **Usage:** Automatically used with `--layout random` (default)
- **Customization:**
  ```bash
  # Custom placement ranges
  python src/create_env.py --x-range -0.2 0.2 --y-range -0.1 0.1 --save custom_range

  # Symmetric range
  python src/create_env.py --placement-range 0.25 --save wide_spread
  ```

#### SequentialCompositeSampler
- **Purpose:** Chain multiple samplers for complex scenes
- **Features:**
  - Place objects sequentially
  - Reference-based placement (place relative to other objects)
  - On-top placement (stack objects)
- **Usage:** For advanced users via code modification
- **Example Use Cases:**
  - Stacking tasks (place objects on top of each other)
  - Spatial relationships (place object A near object B)
  - Multi-stage scenes (fixtures + movable objects)

### Arena-Specific Placement Behavior

Each arena type has optimized placement defaults:

| Arena Type | Reference Position | X Range | Y Range | Z Offset | Boundary Check |
|------------|-------------------|---------|---------|----------|----------------|
| table      | (0, 0, 0.8)       | ±0.15m  | ±0.15m  | 0.01m    | No             |
| bins       | (0.1, -0.25, 0.8) | ±0.145m | ±0.195m | 0.02m    | Yes            |
| pegs       | (0, 0, 0.8)       | ±0.12m  | ±0.12m  | 0.01m    | No             |
| empty      | (0, 0, 0.5)       | ±0.15m  | ±0.15m  | 0.0m     | No             |
| wipe       | (0, 0, 0.8)       | ±0.15m  | ±0.15m  | 0.01m    | No             |
| multi_table| first table pos   | ±0.15m  | ±0.15m  | 0.01m    | No             |

## Built-in Robosuite Environments

For comparison and learning, robosuite includes these pre-built manipulation environments:

### Single-Arm Tasks

#### Lift
- **Description:** Lift a cube to a target height
- **Objects:** Single cube
- **Arena:** Table
- **Objective:** Pick up cube and lift above threshold
- **Use Case:** Basic pick-and-place learning

#### PickPlace
- **Description:** Pick objects from one bin and place in another
- **Objects:** Milk, bread, cereal, can (household items)
- **Arena:** Bins (two side-by-side)
- **Objective:** Transfer objects between bins
- **Use Case:** Sorting, bin picking

#### Stack
- **Description:** Stack colored cubes in correct order
- **Objects:** Multiple cubes (red, green)
- **Arena:** Table
- **Objective:** Stack red cube on green cube
- **Use Case:** Stacking, spatial reasoning

#### NutAssembly
- **Description:** Fit nuts onto pegs
- **Objects:** Square nut, round nut
- **Arena:** Table with pegs
- **Objective:** Place nuts on matching pegs
- **Use Case:** Precision insertion, peg-in-hole

#### Door
- **Description:** Open a door with handle
- **Objects:** Door with hinge and handle
- **Arena:** Table/fixed door
- **Objective:** Grasp handle and open door
- **Use Case:** Articulated object manipulation

#### Wipe
- **Description:** Wipe surface with marks
- **Objects:** Wiping tool/cube
- **Arena:** Table with visual markers
- **Objective:** Cover marked areas
- **Use Case:** Surface coverage, wiping motions

#### ToolHang
- **Description:** Hang tool on stand
- **Objects:** Tool/wrench, hanging stand
- **Arena:** Table
- **Objective:** Hook tool onto stand
- **Use Case:** Precision placement, hooking

### Two-Arm Tasks

#### TwoArmLift
- **Description:** Lift object with two arms
- **Objects:** Pot, cube
- **Arena:** Table
- **Objective:** Coordinate two arms to lift
- **Use Case:** Bi-manual coordination

#### TwoArmPegInHole
- **Description:** Insert peg while stabilizing with other arm
- **Objects:** Peg, plate with hole
- **Arena:** Table
- **Objective:** Precise insertion with stabilization
- **Use Case:** Bi-manual precision tasks

#### TwoArmHandover
- **Description:** Pass object from one arm to another
- **Objects:** Hammer, pot
- **Arena:** Table
- **Objective:** Transfer object between arms
- **Use Case:** Hand-off coordination

#### TwoArmTransport
- **Description:** Transport object together
- **Objects:** Box, pot
- **Arena:** Multiple tables
- **Objective:** Move object using both arms
- **Use Case:** Collaborative transport

### Using Built-in Environments

While you create custom environments with `create_env.py`, you can study or use built-in environments:

```bash
# Run built-in environment with teleoperation
python src/run_simulation.py --mode teleop --robot Panda --task Lift

# Built-in environments for reference:
# - Lift, PickPlace, Stack, NutAssembly
# - Door, Wipe, ToolHang
# - TwoArmLift, TwoArmPegInHole, TwoArmHandover, TwoArmTransport
```

## Quick Environment Creation Workflows

### Workflow 1: Quick Prototyping
```bash
# Start with preset
python src/create_env.py --preset pick_place --save test1

# Test immediately
python src/run_simulation.py --mode sim --env test1 --episodes 5

# Adjust and iterate
python src/create_env.py --num-objects 8 --type box ball --save test2
```

### Workflow 2: Cross-Embodiment Data Collection
```bash
# Create diverse environments
python src/create_env.py --num-objects 6 --type box ball cylinder --colors random \
  --arena-type table --save dataset_env1

python src/create_env.py --num-objects 4 --type bottle can milk bread \
  --arena-type bins --save dataset_env2

python src/create_env.py --num-objects 5 --type box cylinder --layout circle \
  --arena-type pegs --save dataset_env3

# Collect data from each
python src/run_simulation.py --mode teleop --env dataset_env1
python src/run_simulation.py --mode teleop --env dataset_env2
python src/run_simulation.py --mode teleop --env dataset_env3
```

### Workflow 3: Task-Specific Environment
```bash
# For stacking tasks
python src/create_env.py --type box --sizes 0.05 0.04 0.03 --colors red green blue \
  --layout exact --positions "0,0" "0.06,0" "0.12,0" --save stacking_aligned

# For sorting tasks
python src/create_env.py --arena-type bins --num-objects 8 \
  --type ball box cylinder --colors random --save sorting_diverse

# For precision tasks
python src/create_env.py --arena-type pegs --type round_nut square_nut \
  --num-objects 2 --save precision_assembly
```

## Environment Configuration Files

All environments are saved as JSON in `data/environments/`. You can:

1. **Edit configurations manually:**
   ```bash
   vim data/environments/my_env.json
   ```

2. **Copy and modify existing configs:**
   ```bash
   cp data/environments/env1.json data/environments/env_modified.json
   # Edit env_modified.json
   ```

3. **Share configurations:**
   ```bash
   # Environment configs are portable
   git add data/environments/my_custom_env.json
   git commit -m "Add custom environment for grasping experiments"
   ```

### Configuration Structure
```json
{
  "robot": "Panda",
  "placement_mode": "random",
  "placement_params": {
    "x_range": [-0.15, 0.15],
    "y_range": [-0.15, 0.15]
  },
  "objects": [
    {
      "type": "box",
      "size": 0.03,
      "color": [1.0, 0.0, 0.0, 1.0],
      "name": "box_0",
      "density": 100.0
    }
  ],
  "arena_type": "table",
  "table_config": {
    "size": [0.8, 0.8, 0.05],
    "height": 0.8,
    "friction": [1.0, 0.005, 0.0001]
  },
  "metadata": {
    "name": "my_env",
    "created": "2025-01-25T10:30:00",
    "version": "1.0"
  }
}
```
