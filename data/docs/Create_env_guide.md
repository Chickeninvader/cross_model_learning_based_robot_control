# Environment Creation Guide

Quick reference for creating custom robosuite environments for cross-embodiment learning.

## Key Features

- **Arena-Aware Placement:** Objects are automatically placed correctly for each arena type (bins, table, pegs, etc.)
- **Automatic Size Validation:** Object sizes are validated based on arena type and count - errors prevent invalid configurations
- **18 Object Types:** Primitives (box, ball), household items (bottle, milk), and tools (hammer, cone)
- **6 Arena Types:** Table, bins, pegs, empty, wipe, multi-table
- **Multiple Layouts:** Random, grid, circle, line, or exact positions

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
