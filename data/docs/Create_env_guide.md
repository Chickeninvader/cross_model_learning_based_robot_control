# Environment Creation Guide

Quick reference for creating custom robosuite environments.

## Basic Usage

```bash
python src/create_env.py --save NAME [OPTIONS]
```

## Complete Command with All Options

```bash
python src/create_env.py \
  --save NAME \
  --robot {Panda|Sawyer} \
  --num-objects N \
  --type {box|ball|cylinder|capsule} \
  --size SIZE \
  --sizes SIZE1 SIZE2 ... \
  --colors {random|red|green|blue|yellow|cyan|magenta|orange|purple|white|gray|black} \
  --density DENSITY \
  --layout {random|grid|circle|line} \
  --placement-range RANGE \
  --positions "X,Y" "X,Y" ... \
  --grid-size RxC \
  --spacing SPACING \
  --radius RADIUS \
  --line-length LENGTH \
  --line-axis {x|y} \
  --table-size X Y Z \
  --table-height HEIGHT \
  --table-friction SLIDING TORSIONAL ROLLING \
  --preset {stacking|sorting|obstacles|pick_place}
```

## Quick Examples

```bash
# Simple random environment
python src/create_env.py --num-objects 5 --type box --colors random --save my_env

# Grid layout
python src/create_env.py --layout grid --grid-size 4x4 --spacing 0.1 --type ball --save grid_env

# Custom table and objects
python src/create_env.py --table-size 1.2 1.2 0.05 --num-objects 10 --density 50 --save large_env

# Preset environment
python src/create_env.py --preset stacking --save stacking_task
```

## Using the Environment

```bash
# Teleoperation with SpaceMouse
python src/run_simulation.py --mode teleop --env my_env

# Record demonstrations
python src/run_simulation.py --mode teleop --env my_env --record --num-episodes 5

# Replay demonstrations
python src/replay_demo.py data/recordings/my_env_TIMESTAMP/
```

## Default Values

| Option | Default | Description |
|--------|---------|-------------|
| `--robot` | Panda | Robot type (Panda or Sawyer) |
| `--num-objects` | 3 | Number of objects |
| `--type` | box | Object type |
| `--size` | 0.025 | Object size in meters |
| `--density` | 100 | Object density (kg/m³) |
| `--layout` | random | Placement layout |
| `--placement-range` | 0.15 | Random placement range (m) |
| `--table-size` | 0.8 0.8 0.05 | Table dimensions (m) |
| `--table-height` | 0.8 | Table height (m) |
| `--table-friction` | 1.0 0.005 0.0001 | Friction values |

## Object Types

- `box` - Cube or rectangular block
- `ball` / `sphere` - Round object
- `cylinder` - Tube/column
- `capsule` - Pill-shaped (rounded cylinder)

## Placement Layouts

- `random` - Random within range
- `grid` - Regular grid (use `--grid-size` and `--spacing`)
- `circle` - Circular arrangement (use `--radius`)
- `line` - Linear arrangement (use `--line-length` and `--line-axis`)
- Exact positions with `--positions "x,y" "x,y" ...`

## Colors

`random`, `red`, `green`, `blue`, `yellow`, `cyan`, `magenta`, `orange`, `purple`, `white`, `gray`, `black`

## Typical Values

- **Size**: 0.02 (small), 0.025 (medium), 0.04 (large)
- **Density**: 50 (light), 100 (standard), 500+ (heavy)
- **Spacing**: 0.08 (tight), 0.1 (standard), 0.15 (wide)

## Help Commands

```bash
# Show all options
python src/create_env.py --help

# List preset environments
python src/create_env.py --list-presets
```

## Output

Environment configurations are saved to `data/environments/NAME.json`
