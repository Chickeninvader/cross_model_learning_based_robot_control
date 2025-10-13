# RoboSim - Robosuite Custom Environment Project

Professional robosuite simulation setup for robotics research with custom environments, SpaceMouse teleoperation, demonstration recording, and replay capabilities.

## Quick Start

### 1. Create Custom Environment

```bash
# Simple environment with random objects
python src/create_env.py --num-objects 5 --type box --colors random --save my_env
```

### 2. Run with SpaceMouse

```bash
# Teleoperation mode
python src/run_simulation.py --mode teleop --env my_env
```

### 3. Record Demonstrations

```bash
# Record demonstrations with SpaceMouse
python src/run_simulation.py --mode teleop --env my_env --record --num-episodes 5
```

### 4. Replay Demonstrations

```bash
# Replay recorded demonstrations
python src/replay_demo.py data/recordings/my_env_YYYYMMDD_HHMMSS/
```

---

## Project Structure

```
robosim/
├── src/                        # Source code
│   ├── run_simulation.py      # Main simulation runner
│   ├── create_env.py          # Environment creator
│   ├── replay_demo.py         # Demonstration replay
│   └── custom_spacemouse.py   # SpaceMouse driver
├── data/
│   ├── environments/          # Environment JSON configurations
│   ├── recordings/            # Recorded demonstrations (HDF5)
│   └── simulations/           # Simulation recordings
├── docs/
│   ├── ENV_CREATION.md        # Environment creation guide
│   ├── install_robosuite.sh   # Installation script
│   └── setup_spacemouse_permissions.sh
├── external/
│   └── robosuite/             # Robosuite installation
└── README.md                  # This file
```

---

## Complete Workflow

### Create → Teleoperate → Record → Replay

```bash
# 1. Create custom environment
python src/create_env.py \
  --num-objects 5 \
  --type ball \
  --colors random \
  --save my_demo_env

# 2. Run teleoperation with SpaceMouse
python src/run_simulation.py \
  --mode teleop \
  --env my_demo_env \
  --record \
  --num-episodes 3

# 3. Replay demonstrations
python src/replay_demo.py data/recordings/my_demo_env_YYYYMMDD_HHMMSS/
```

---

## Creating Custom Environments

### Basic Usage

```bash
# Full command with all options
python src/create_env.py \
  --save NAME \
  --robot {Panda|Sawyer} \
  --num-objects N \
  --type {box|ball|cylinder|capsule} \
  --colors {random|red|green|blue|yellow|cyan|magenta|...} \
  --layout {random|grid|circle|line} \
  --table-size X Y Z \
  --preset {stacking|sorting|obstacles|pick_place}
```

### Quick Examples

```bash
# Random placement
python src/create_env.py --num-objects 5 --type box --colors random --save my_env

# Grid layout (4x4)
python src/create_env.py --layout grid --grid-size 4x4 --spacing 0.1 --type ball --save grid_env

# Custom table size
python src/create_env.py --table-size 1.2 1.2 0.05 --num-objects 10 --save large_env

# Preset environment
python src/create_env.py --preset stacking --save stacking_task
```

### Object Types

- `box` - Cube or rectangular block
- `ball` / `sphere` - Round object
- `cylinder` - Tube/column
- `capsule` - Pill-shaped (rounded cylinder)

### Placement Layouts

- `random` - Random within range
- `grid` - Regular grid (use `--grid-size` and `--spacing`)
- `circle` - Circular arrangement (use `--radius`)
- `line` - Linear arrangement (use `--line-length` and `--line-axis`)
- Exact positions with `--positions "x,y" "x,y" ...`

### Preset Environments

```bash
python src/create_env.py --list-presets  # List available presets

# Available presets:
#   stacking   - Graduated blocks for stacking tasks
#   sorting    - Mixed objects for color/shape sorting
#   obstacles  - Obstacle course with targets
#   pick_place - Simple pick and place with 3 boxes
```

---

## Running Simulations

### Teleoperation Mode (SpaceMouse)

```bash
# Free play with custom environment
python src/run_simulation.py --mode teleop --env my_env

# Record demonstrations
python src/run_simulation.py --mode teleop --env my_env --record --num-episodes 5

# Standard robosuite task
python src/run_simulation.py --mode teleop --robot Panda --task Lift

# Adjust sensitivity
python src/run_simulation.py --mode teleop --env my_env --pos-sensitivity 3.0 --rot-sensitivity 2.0
```

### Automated Simulation Mode

```bash
# Run episodes with random policy
python src/run_simulation.py --mode sim --env my_env --episodes 10

# Run with recording
python src/run_simulation.py --mode sim --env my_env --episodes 20 --record
```

---

## SpaceMouse Controls

### Basic Controls

- **Move SpaceMouse**: Control end-effector position (X, Y, Z)
- **Rotate SpaceMouse**: Control end-effector orientation
- **Button 1**: Toggle gripper open/close
- **Menu Button**: End episode (when recording)
- **Ctrl+C**: Exit teleoperation

### Sensitivity Settings

```bash
--pos-sensitivity 3.0    # Position sensitivity (default: 3.0)
--rot-sensitivity 2.0    # Rotation sensitivity (default: 2.0)
```

---

## Recording & Replay

### Recording Demonstrations

```bash
# Record 5 episodes
python src/run_simulation.py \
  --mode teleop \
  --env my_env \
  --record \
  --num-episodes 5 \
  --demo-dir data/recordings

# Specify horizon (max steps)
python src/run_simulation.py \
  --mode teleop \
  --env my_env \
  --record \
  --horizon 5000
```

**What Gets Saved:**
- Individual episodes in `episode_0/`, `episode_1/`, etc.
- Each episode contains:
  - `demo.npz` - States, actions, rewards, timestamps
  - `model.xml` - MuJoCo model file
  - `metadata.json` - Episode information
- `demos.hdf5` - Combined HDF5 dataset

### Replay Demonstrations

```bash
# Replay all episodes in directory
python src/replay_demo.py data/recordings/my_env_YYYYMMDD_HHMMSS/

# Replay specific episode
python src/replay_demo.py --demo-dir data/recordings/my_env_YYYYMMDD_HHMMSS/ --episode 0

# List available demonstrations
python src/replay_demo.py --list --demo-dir data/recordings/

# Adjust playback speed
python src/replay_demo.py data/recordings/my_env_YYYYMMDD_HHMMSS/ --speed 2.0
```

---

## Supported Robots & Tasks

### Robots

- **Panda** (Franka Emika) - 7-DOF, default
- **Sawyer** (Rethink Robotics) - 7-DOF

### Standard Robosuite Tasks

- **Lift** - Pick up and lift a cube
- **Stack** - Stack blocks on top of each other
- **PickPlace** - Pick object and place in target location
- **NutAssembly** - Assemble nut onto peg
- **Door** - Open a door with handle
- **Wipe** - Wipe a surface clean

---

## Performance Settings

### Control Frequency
- Default: 50 Hz (20ms per step)
- Provides responsive robot control
- Configurable via `--control-freq` flag

### Episode Horizon
- Default: 5000 steps (100 seconds at 50Hz)
- Configurable via `--horizon` flag

### Visualization
- End-effector marker (red sphere) enabled by default
- Detailed action debug output during recording/replay

---

## Tips and Best Practices

### Object Sizing
- **Small** (0.015-0.02m) - Precision tasks
- **Medium** (0.025-0.03m) - Standard manipulation
- **Large** (0.04-0.05m) - Easier to grasp

### Placement
- **Grid** - Systematic coverage
- **Circle** - Radial reaching tasks
- **Random** - Varied training scenarios
- **Exact** - Repeatable experiments

### Table Configuration
- **Larger tables** (1.0+ m) - More workspace
- **Higher tables** (0.9+ m) - Easier reach
- **Lower friction** - Dynamic, sliding objects
- **Higher friction** - Stable, precise manipulation

### Object Density
- **Light** (50 kg/m³) - Easy to push
- **Medium** (100 kg/m³) - Balanced
- **Heavy** (500+ kg/m³) - Difficult to move, good for obstacles

---

## Troubleshooting

### Objects falling through table
```bash
--table-height 0.85
```

### Robot can't reach objects
```bash
--placement-range 0.12
--table-size 0.7 0.7 0.05
```

### Objects too heavy to move
```bash
--density 50
```

### Objects sliding too much
```bash
--table-friction 1.5 0.01 0.001
```

### Simulation running slow
```bash
--num-objects 3
--type ball  # Simpler geometry than box or cylinder
```

### SpaceMouse Not Detected
```bash
# Check if device is connected
lsusb | grep "3Dconnexion"

# Re-run permissions script
sudo docs/setup_spacemouse_permissions.sh

# Reconnect device (unplug and plug back in)
```

---

## Complete Examples

### Example 1: Dense Grid Environment

```bash
# Create grid environment
python src/create_env.py \
  --layout grid \
  --grid-size 5x5 \
  --spacing 0.08 \
  --type ball \
  --size 0.02 \
  --colors random \
  --save dense_grid

# Test with teleoperation
python src/run_simulation.py --mode teleop --env dense_grid

# Record demonstrations
python src/run_simulation.py --mode teleop --env dense_grid --record --num-episodes 10
```

### Example 2: Precision Task

```bash
# Create environment with exact positions
python src/create_env.py \
  --positions "0.15,0.0" "0.0,0.15" "-0.15,0.0" "0.0,-0.15" \
  --type capsule \
  --sizes 0.015 0.015 0.015 0.015 \
  --colors red green blue yellow \
  --density 80 \
  --table-friction 1.5 0.008 0.0002 \
  --save precision_task

# Run teleoperation
python src/run_simulation.py --mode teleop --env precision_task
```

### Example 3: Large Workspace

```bash
# Create large table with many objects
python src/create_env.py \
  --table-size 1.2 1.2 0.05 \
  --table-height 0.85 \
  --num-objects 15 \
  --placement-range 0.3 \
  --type box ball cylinder \
  --colors random \
  --density 80 \
  --save large_workspace

# Run and record
python src/run_simulation.py --mode teleop --env large_workspace --record
```

---

## Command Reference

### Environment Creation
```bash
python src/create_env.py --help            # Show all options
python src/create_env.py --list-presets    # List preset environments
```

### Simulation
```bash
python src/run_simulation.py --help        # Show all options
```

### Replay
```bash
python src/replay_demo.py --help           # Show replay options
python src/replay_demo.py --list           # List recordings
```

---

## Documentation

- **[ENV_CREATION.md](data/docs/Create_env_guide.md)** - Complete environment creation guide
- **[install_robosuite.sh](misc/install_robosuite.sh)** - Installation script
- **[setup_spacemouse_permissions.sh](misc/setup_spacemouse_permissions.sh)** - SpaceMouse setup

---

## File Locations

- **Environment configs**: `data/environments/*.json`
- **Recordings**: `data/recordings/*/`
- **Scripts**: `src/`
- **Documentation**: `docs/`

---

## Requirements

- Python 3.8+
- Robosuite
- SpaceMouse device (for teleoperation)
- MuJoCo physics engine

---

## Default Values

| Option | Default | Description |
|--------|---------|-------------|
| `--robot` | Panda | Robot type |
| `--num-objects` | 3 | Number of objects |
| `--type` | box | Object type |
| `--size` | 0.025 | Object size (m) |
| `--density` | 100 | Object density (kg/m³) |
| `--layout` | random | Placement layout |
| `--placement-range` | 0.15 | Random placement range (m) |
| `--table-size` | 0.8 0.8 0.05 | Table dimensions (m) |
| `--table-height` | 0.8 | Table height (m) |
| `--table-friction` | 1.0 0.005 0.0001 | Friction values |
| `--control-freq` | 50 | Control frequency (Hz) |
| `--horizon` | 5000 | Episode max steps |

---

## License

This project uses the robosuite framework. Please refer to robosuite's license for usage terms.

---

**Version:** 2.0
**Last Updated:** October 10, 2025
