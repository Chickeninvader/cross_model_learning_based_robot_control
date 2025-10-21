#!/usr/bin/env python3
"""
Unified Environment Creator for Robosuite
Create custom environments with flexible options for objects, placement, and layouts

Usage:
    # Basic random environment
    python create_env.py --num-objects 5 --type box ball --colors random --save my_env

    # Different sized objects
    python create_env.py --num-objects 3 --type box --sizes 0.02 0.05 0.08 --save varied

    # Bins arena for sorting
    python create_env.py --arena-type bins --num-objects 6 --type ball --save sorting

    # Household items
    python create_env.py --num-objects 4 --type bottle can milk bread --save kitchen

    # Grid layout
    python create_env.py --layout grid --grid-size 3x3 --type ball --save grid_env

For complete documentation, see data/docs/Create_env_guide.md
"""

import argparse
import numpy as np
import json
from pathlib import Path
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import (
    TableArena,
    BinsArena,
    EmptyArena,
    MultiTableArena,
    PegsArena,
    WipeArena,
)
from robosuite.models.objects import (
    # Primitive objects
    BoxObject,
    BallObject,
    CylinderObject,
    CapsuleObject,
    # XML objects
    BottleObject,
    CanObject,
    LemonObject,
    MilkObject,
    BreadObject,
    CerealObject,
    SquareNutObject,
    RoundNutObject,
    # Composite objects
    ConeObject,
    HammerObject,
    PotWithHandlesObject,
)
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler


class UnifiedCustomEnv(ManipulationEnv):
    """
    Unified custom environment supporting all placement and configuration options
    """

    def __init__(
        self,
        robots,
        object_configs,
        placement_mode='random',  # 'random', 'exact', 'grid', 'circle', 'line'
        placement_params=None,
        arena_type='table',  # 'table' or 'bins'
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        bin1_pos=(0.1, -0.25, 0.8),  # Position of bin1 for BinsArena
        placement_initializer=None,
        **kwargs
    ):
        """
        Args:
            robots: Robot type
            object_configs: List of object configurations
            placement_mode: How to place objects
            placement_params: Parameters for placement (ranges, positions, etc.)
            arena_type: Type of arena ('table' or 'bins')
            table_full_size: Table dimensions (for TableArena)
            table_friction: Table friction parameters
            bin1_pos: Position of first bin (for BinsArena)
            placement_initializer: Custom placement initializer (optional)
        """
        # Store arena configuration
        self.arena_type = arena_type
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.bin1_pos = bin1_pos
        self.table_offset = np.array((0, 0, 0.8))

        # Store object and placement configuration
        self.object_configs = object_configs
        self.placement_mode = placement_mode
        self.placement_params = placement_params or {}
        self.placement_initializer = placement_initializer

        # Object lists
        self.objects = []
        self.object_body_ids = []

        super().__init__(
            robots=robots,
            env_configuration="default",
            controller_configs=None,
            base_types="default",
            gripper_types="default",
            initialization_noise="default",
            use_camera_obs=kwargs.get('use_camera_obs', False),
            has_renderer=kwargs.get('has_renderer', True),
            has_offscreen_renderer=kwargs.get('has_offscreen_renderer', False),
            render_camera=kwargs.get('render_camera', None),
            render_collision_mesh=kwargs.get('render_collision_mesh', False),
            render_visual_mesh=kwargs.get('render_visual_mesh', True),
            render_gpu_device_id=kwargs.get('render_gpu_device_id', -1),
            control_freq=kwargs.get('control_freq', 20),
            lite_physics=kwargs.get('lite_physics', True),
            horizon=kwargs.get('horizon', 1000),
            ignore_done=kwargs.get('ignore_done', False),
            hard_reset=kwargs.get('hard_reset', True),
            camera_names=kwargs.get('camera_names', 'agentview'),
            camera_heights=kwargs.get('camera_heights', 256),
            camera_widths=kwargs.get('camera_widths', 256),
            camera_depths=kwargs.get('camera_depths', False),
            camera_segmentations=kwargs.get('camera_segmentations', None),
            renderer=kwargs.get('renderer', 'mjviewer'),
            renderer_config=kwargs.get('renderer_config', None),
            seed=kwargs.get('seed', None),
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Clear objects list to prevent duplicates on reset
        self.objects = []
        self.object_body_ids = []

        # Adjust robot base pose to be positioned correctly relative to the table
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Create arena based on type
        if self.arena_type == 'empty':
            mujoco_arena = EmptyArena()
        elif self.arena_type == 'bins':
            mujoco_arena = BinsArena(
                bin1_pos=self.bin1_pos,
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
            )
        elif self.arena_type == 'pegs':
            mujoco_arena = PegsArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )
        elif self.arena_type == 'wipe':
            mujoco_arena = WipeArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )
        elif self.arena_type == 'multi_table':
            # MultiTableArena requires table_offsets
            # Default to 2 tables side by side
            table_offsets = getattr(self, 'table_offsets', [[0.5, 0, 0.8], [-0.5, 0, 0.8]])
            mujoco_arena = MultiTableArena(
                table_offsets=table_offsets,
                table_full_sizes=self.table_full_size,
                table_frictions=self.table_friction,
            )
        else:
            # Default to table arena
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_friction=self.table_friction,
                table_offset=self.table_offset,
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Create objects
        for config in self.object_configs:
            obj = self._create_object(config)
            if obj:
                self.objects.append(obj)

        # Setup placement sampler
        if self.placement_initializer is not None:
            # Use provided initializer
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objects)
        elif self.placement_mode == 'random':
            # Get arena-specific placement configuration
            placement_config = self._get_arena_placement_config()

            # Debug output
            print(f"\nDEBUG Placement Config:")
            print(f"  Arena type: {self.arena_type}")
            print(f"  Table full size: {self.table_full_size}")
            print(f"  Reference pos: {placement_config['reference_pos']}")
            print(f"  X range: {placement_config['x_range']}")
            print(f"  Y range: {placement_config['y_range']}")
            print(f"  Z offset: {placement_config['z_offset']}")
            print(f"  Ensure boundary: {placement_config.get('ensure_boundary', False)}")

            # Create random placement sampler with arena-aware settings
            x_range = self.placement_params.get('x_range', placement_config['x_range'])
            y_range = self.placement_params.get('y_range', placement_config['y_range'])

            # For bins, ensure entire object boundary stays within range
            # For other arenas, just check center position
            ensure_boundary = placement_config.get('ensure_boundary', False)

            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objects,
                x_range=x_range,
                y_range=y_range,
                rotation=None,
                rotation_axis='z',
                ensure_object_boundary_in_range=ensure_boundary,
                ensure_valid_placement=True,
                reference_pos=placement_config['reference_pos'],
                z_offset=placement_config['z_offset'],
            )
        else:
            # Exact placement - no sampler needed
            self.placement_initializer = None

        # Setup task
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _get_arena_placement_config(self):
        """
        Get arena-specific placement configuration.
        Returns dict with reference_pos, x_range, y_range, z_offset
        """
        if self.arena_type == 'bins':
            # Bins arena: place objects in the first bin
            # Use the table_full_size to determine bin dimensions
            # Based on robosuite's PickPlace implementation
            bin_x_half = self.table_full_size[0] / 2 - 0.05
            bin_y_half = self.table_full_size[1] / 2 - 0.05

            return {
                'reference_pos': np.array(self.bin1_pos),
                'x_range': [-bin_x_half, bin_x_half],
                'y_range': [-bin_y_half, bin_y_half],
                'z_offset': 0.02,  # Small offset to ensure objects spawn above bin bottom
                'ensure_boundary': True,  # Ensure entire object stays within bin bounds
            }

        elif self.arena_type == 'empty':
            # Empty arena: place objects at a fixed height
            return {
                'reference_pos': np.array([0, 0, 0.5]),  # Mid-air placement
                'x_range': [-0.15, 0.15],
                'y_range': [-0.15, 0.15],
                'z_offset': 0.0,
                'ensure_boundary': False,
            }

        elif self.arena_type == 'pegs':
            # Pegs arena: place objects on table, avoiding pegs
            # PegsArena has pegs, so use smaller range
            return {
                'reference_pos': self.table_offset,
                'x_range': [-0.12, 0.12],
                'y_range': [-0.12, 0.12],
                'z_offset': 0.01,
                'ensure_boundary': False,
            }

        elif self.arena_type == 'wipe':
            # Wipe arena: regular table placement
            return {
                'reference_pos': self.table_offset,
                'x_range': [-0.15, 0.15],
                'y_range': [-0.15, 0.15],
                'z_offset': 0.01,
                'ensure_boundary': False,
            }

        elif self.arena_type == 'multi_table':
            # Multi-table: place on first table
            table_offsets = getattr(self, 'table_offsets', [[0.5, 0, 0.8], [-0.5, 0, 0.8]])
            first_table = np.array(table_offsets[0])
            return {
                'reference_pos': first_table,
                'x_range': [-0.15, 0.15],
                'y_range': [-0.15, 0.15],
                'z_offset': 0.01,
                'ensure_boundary': False,
            }

        else:
            # Default table arena
            return {
                'reference_pos': self.table_offset,
                'x_range': [-0.15, 0.15],
                'y_range': [-0.15, 0.15],
                'z_offset': 0.01,
                'ensure_boundary': False,
            }

    def _create_object(self, config):
        """Create object from configuration"""
        obj_type = config.get('type', 'box')
        name = config.get('name', f"{obj_type}_{len(self.objects)}")
        size = config.get('size', 0.025)
        color = config.get('color', [1.0, 0.0, 0.0, 1.0])
        density = config.get('density', 100.0)

        # Primitive objects (support custom size and color)
        if obj_type == 'box':
            if isinstance(size, list):
                size_array = size
            else:
                size_array = [size] * 3
            return BoxObject(
                name=name,
                size_min=size_array,
                size_max=size_array,
                rgba=color,
                density=density,
            )
        elif obj_type == 'ball':
            return BallObject(
                name=name,
                size=[size if not isinstance(size, list) else size[0]],
                rgba=color,
                density=density,
            )
        elif obj_type == 'cylinder':
            height = size * 2 if not isinstance(size, list) else size[1]
            radius = size if not isinstance(size, list) else size[0]
            return CylinderObject(
                name=name,
                size=[radius, height],
                rgba=color,
                density=density,
            )
        elif obj_type == 'capsule':
            height = size * 2 if not isinstance(size, list) else size[1]
            radius = size * 0.5 if not isinstance(size, list) else size[0]
            return CapsuleObject(
                name=name,
                size=[radius, height],
                rgba=color,
                density=density,
            )

        # XML objects (pre-defined objects, use default appearance)
        elif obj_type == 'bottle':
            return BottleObject(name=name)
        elif obj_type == 'can':
            return CanObject(name=name)
        elif obj_type == 'lemon':
            return LemonObject(name=name)
        elif obj_type == 'milk':
            return MilkObject(name=name)
        elif obj_type == 'bread':
            return BreadObject(name=name)
        elif obj_type == 'cereal':
            return CerealObject(name=name)
        elif obj_type == 'square_nut':
            return SquareNutObject(name=name)
        elif obj_type == 'round_nut':
            return RoundNutObject(name=name)

        # Composite objects
        elif obj_type == 'cone':
            return ConeObject(name=name)
        elif obj_type == 'hammer':
            return HammerObject(name=name)
        elif obj_type == 'pot':
            return PotWithHandlesObject(name=name)

        return None

    def _setup_references(self):
        """
        Sets up references to important components
        """
        super()._setup_references()

        # Store object body IDs for easy access
        self.object_body_ids = [
            self.sim.model.body_name2id(obj.root_body) for obj in self.objects
        ]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment
        """
        return super()._setup_observables()

    def _reset_internal(self):
        """
        Resets simulation internal configurations
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            if self.placement_mode == 'random' and self.placement_initializer:
                # Random placement using sampler
                object_placements = self.placement_initializer.sample()

                # Debug: print object placements
                print(f"\nDEBUG Object Placements:")
                for obj_pos, obj_quat, obj in object_placements.values():
                    print(f"  {obj.name}: pos=({obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f})")

                # Loop through all objects and reset their positions
                for obj_pos, obj_quat, obj in object_placements.values():
                    self.sim.data.set_joint_qpos(
                        obj.joints[0],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)])
                    )
            else:
                # Exact placement - use arena-aware reference position
                placement_config = self._get_arena_placement_config()
                reference_pos = placement_config['reference_pos']
                z_offset = placement_config['z_offset']

                positions = self.placement_params.get('positions', [])
                for i, obj in enumerate(self.objects):
                    if i < len(positions):
                        pos = positions[i]
                        # Position is [x, y] or [x, y, z]
                        if len(pos) == 2:
                            abs_pos = [pos[0], pos[1], reference_pos[2] + z_offset]
                        else:
                            abs_pos = [pos[0], pos[1], pos[2] if pos[2] > 1.0 else reference_pos[2] + pos[2]]

                        quat = [1, 0, 0, 0]
                        self.sim.data.set_joint_qpos(
                            obj.joints[0],
                            np.concatenate([abs_pos, quat])
                        )

    def reward(self, action=None):
        """Simple proximity-based reward"""
        reward = 0.0
        try:
            # Get gripper position
            eef_site_id = self.robots[0].eef_site_id
            if isinstance(eef_site_id, dict):
                gripper_site_id = list(eef_site_id.values())[0]
            elif hasattr(eef_site_id, '__iter__') and not isinstance(eef_site_id, (str, int)):
                gripper_site_id = eef_site_id[0]
            else:
                gripper_site_id = eef_site_id

            gripper_pos = self.sim.data.site_xpos[gripper_site_id]

            # Reward based on proximity to any object
            for body_id in self.object_body_ids:
                obj_pos = self.sim.data.body_xpos[body_id]
                dist = np.linalg.norm(gripper_pos - obj_pos)
                if dist < 0.05:
                    reward += 0.1
        except:
            pass

        return reward

    def _check_success(self):
        """Check if task is successful"""
        return False


# ============================================================================
# Helper Functions
# ============================================================================

def parse_color(color_str):
    """Parse color string to RGBA"""
    color_map = {
        'red': [1.0, 0.0, 0.0, 1.0],
        'green': [0.0, 1.0, 0.0, 1.0],
        'blue': [0.0, 0.0, 1.0, 1.0],
        'yellow': [1.0, 1.0, 0.0, 1.0],
        'cyan': [0.0, 1.0, 1.0, 1.0],
        'magenta': [1.0, 0.0, 1.0, 1.0],
        'orange': [1.0, 0.5, 0.0, 1.0],
        'purple': [0.5, 0.0, 1.0, 1.0],
        'white': [1.0, 1.0, 1.0, 1.0],
        'gray': [0.5, 0.5, 0.5, 1.0],
        'black': [0.1, 0.1, 0.1, 1.0],
    }
    return color_map.get(color_str.lower(), [1.0, 0.0, 0.0, 1.0])


def generate_grid_positions(rows, cols, spacing):
    """Generate grid layout positions"""
    positions = []
    total_width = (cols - 1) * spacing
    total_height = (rows - 1) * spacing
    start_x = -total_width / 2
    start_y = -total_height / 2

    for row in range(rows):
        for col in range(cols):
            x = start_x + col * spacing
            y = start_y + row * spacing
            positions.append([x, y])
    return positions


def generate_circle_positions(num_objects, radius):
    """Generate circular layout positions"""
    positions = []
    for i in range(num_objects):
        angle = 2 * np.pi * i / num_objects
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions.append([x, y])
    return positions


def generate_line_positions(num_objects, length, axis):
    """Generate line layout positions"""
    positions = []
    spacing = length / (num_objects - 1) if num_objects > 1 else 0
    start = -length / 2

    for i in range(num_objects):
        if axis == 'x':
            positions.append([start + i * spacing, 0.0])
        else:
            positions.append([0.0, start + i * spacing])
    return positions


def get_preset_config(preset_name):
    """Get configuration for preset environments"""
    presets = {
        'stacking': {
            'description': 'Stacking task with graduated blocks',
            'objects': [
                {'type': 'box', 'size': [0.05, 0.05, 0.02], 'color': [0.8, 0.2, 0.2, 1.0]},
                {'type': 'box', 'size': [0.04, 0.04, 0.02], 'color': [0.2, 0.8, 0.2, 1.0]},
                {'type': 'box', 'size': [0.03, 0.03, 0.02], 'color': [0.2, 0.2, 0.8, 1.0]},
                {'type': 'box', 'size': [0.025, 0.025, 0.02], 'color': [0.8, 0.8, 0.2, 1.0]},
            ],
            'placement_mode': 'random',
            'placement_params': {'x_range': [-0.15, 0.15], 'y_range': [-0.15, 0.15]}
        },
        'sorting': {
            'description': 'Color sorting with mixed objects',
            'objects': [
                {'type': 'ball', 'size': 0.02, 'color': [1, 0, 0, 1]},
                {'type': 'ball', 'size': 0.02, 'color': [1, 0, 0, 1]},
                {'type': 'box', 'size': 0.025, 'color': [0, 1, 0, 1]},
                {'type': 'box', 'size': 0.025, 'color': [0, 1, 0, 1]},
                {'type': 'cylinder', 'size': 0.02, 'color': [0, 0, 1, 1]},
                {'type': 'cylinder', 'size': 0.02, 'color': [0, 0, 1, 1]},
            ],
            'placement_mode': 'random',
            'placement_params': {'x_range': [-0.25, 0.25], 'y_range': [-0.25, 0.25]}
        },
        'obstacles': {
            'description': 'Obstacle course with targets',
            'objects': [
                {'type': 'cylinder', 'size': [0.03, 0.1], 'color': [0.5, 0.5, 0.5, 1], 'density': 1000},
                {'type': 'cylinder', 'size': [0.03, 0.1], 'color': [0.5, 0.5, 0.5, 1], 'density': 1000},
                {'type': 'cylinder', 'size': [0.03, 0.1], 'color': [0.5, 0.5, 0.5, 1], 'density': 1000},
                {'type': 'cylinder', 'size': [0.03, 0.1], 'color': [0.5, 0.5, 0.5, 1], 'density': 1000},
                {'type': 'ball', 'size': 0.025, 'color': [1, 0, 0, 1]},
                {'type': 'ball', 'size': 0.025, 'color': [0, 1, 0, 1]},
                {'type': 'ball', 'size': 0.025, 'color': [0, 0, 1, 1]},
            ],
            'placement_mode': 'random',
            'placement_params': {'x_range': [-0.2, 0.2], 'y_range': [-0.15, 0.15]}
        },
        'pick_place': {
            'description': 'Simple pick and place with 3 boxes',
            'objects': [
                {'type': 'box', 'size': 0.03, 'color': [1, 0, 0, 1]},
                {'type': 'box', 'size': 0.03, 'color': [0, 1, 0, 1]},
                {'type': 'box', 'size': 0.03, 'color': [0, 0, 1, 1]},
            ],
            'placement_mode': 'random',
            'placement_params': {'x_range': [-0.15, 0.15], 'y_range': [-0.15, 0.15]}
        }
    }
    return presets.get(preset_name)


def get_max_object_size(arena_type, num_objects, table_size=None):
    """
    Calculate maximum safe object size based on arena type, object count, and arena dimensions.

    The calculation considers:
    - Arena type (bins have less space than tables)
    - Object count (more objects = smaller max size due to packing constraints)
    - Actual arena dimensions (if custom table size provided)

    Args:
        arena_type: Type of arena ('bins', 'table', etc.)
        num_objects: Number of objects to place
        table_size: Optional table dimensions [x, y, z] for more accurate calculation

    Returns:
        max_size: Maximum object size in meters
    """
    if arena_type == 'bins':
        # Bins arena: Calculate based on actual bin dimensions
        # Default: 0.39x0.49m, usable area after margins: ~0.29x0.39m
        if table_size:
            # Custom bin size provided
            usable_x = (table_size[0] / 2 - 0.05) * 2  # Account for margins
            usable_y = (table_size[1] / 2 - 0.05) * 2
        else:
            # Default bin size
            usable_x = 0.29
            usable_y = 0.39

        # Estimate objects per row/col based on square packing
        # Use conservative packing factor (1.5x object size per slot for spacing)
        min_usable = min(usable_x, usable_y)

        # Dynamic calculation: max_size = usable_space / (sqrt(N) * packing_factor)
        import math
        packing_factor = 2.0  # Conservative: each object needs 2x its size in space
        max_size_calculated = min_usable / (math.sqrt(num_objects) * packing_factor)

        # Clamp to reasonable bounds
        max_size_calculated = max(0.015, min(0.08, max_size_calculated))

        return max_size_calculated

    elif arena_type == 'pegs':
        # Pegs arena: smaller placement area due to pegs (±0.12m = 0.24m usable)
        if table_size:
            # Use actual table size
            usable = min(table_size[0], table_size[1]) * 0.3  # 30% of table for safety
        else:
            usable = 0.24

        import math
        packing_factor = 2.0
        max_size_calculated = usable / (math.sqrt(num_objects) * packing_factor)
        return max(0.02, min(0.10, max_size_calculated))

    else:
        # Table and other arenas: larger workspace (±0.15m = 0.30m usable)
        if table_size:
            usable = min(table_size[0], table_size[1]) * 0.375  # 37.5% of table
        else:
            usable = 0.30

        import math
        packing_factor = 1.8  # Less conservative for larger spaces
        max_size_calculated = usable / (math.sqrt(num_objects) * packing_factor)
        return max(0.03, min(0.20, max_size_calculated))


def validate_object_sizes(objects, arena_type, table_size=None):
    """
    Validate that object sizes are appropriate for the arena.

    Args:
        objects: List of object configurations
        arena_type: Type of arena
        table_size: Optional table dimensions [x, y, z]

    Raises:
        ValueError: If any object size exceeds the safe limit
    """
    num_objects = len(objects)
    max_size = get_max_object_size(arena_type, num_objects, table_size)

    errors = []
    for i, obj in enumerate(objects):
        size = obj.get('size', 0.025)

        # Get the maximum dimension of the object
        if isinstance(size, list):
            max_dimension = max(size)
        else:
            max_dimension = size

        if max_dimension > max_size:
            errors.append(
                f"  Object {i+1} ({obj['name']}): size {max_dimension:.3f}m exceeds limit {max_size:.3f}m"
            )

    if errors:
        # Format table size info
        table_info = ""
        if table_size:
            table_info = f" ({table_size[0]:.2f}m × {table_size[1]:.2f}m)"

        error_msg = f"\n{'='*70}\n"
        error_msg += f"ERROR: Object sizes too large for {arena_type} arena{table_info}!\n"
        error_msg += f"\nConfiguration:\n"
        error_msg += f"  Arena: {arena_type}\n"
        if table_size:
            error_msg += f"  Arena dimensions: {table_size[0]:.2f}m × {table_size[1]:.2f}m\n"
        error_msg += f"  Object count: {num_objects}\n"
        error_msg += f"  Max safe size: {max_size:.3f}m ({max_size*100:.1f}cm)\n"
        error_msg += "\nProblematic objects:\n"
        error_msg += "\n".join(errors)
        error_msg += f"\n\nSuggestions:\n"
        error_msg += f"1. Reduce object sizes to ≤{max_size:.3f}m:\n"
        error_msg += f"   --sizes " + " ".join([f"{max_size:.3f}"] * min(3, num_objects)) + "\n"
        error_msg += f"2. Use fewer objects (fewer objects = larger max size allowed)\n"

        if arena_type == 'bins':
            error_msg += f"3. Use table arena instead: --arena-type table\n"
            error_msg += f"4. Increase bin size: --table-size 0.6 0.6 0.82\n"

        error_msg += f"\nDynamic size limits for {arena_type} arena{table_info}:\n"
        for n in [1, 2, 3, 5, 7, 10]:
            if n <= num_objects + 3:  # Show relevant range
                limit = get_max_object_size(arena_type, n, table_size)
                error_msg += f"  {n:2d} objects: max {limit:.3f}m ({limit*100:.1f}cm)\n"

        error_msg += f"\nNote: Max size = (usable_space / sqrt(N)) / packing_factor\n"
        error_msg += f"      More objects = smaller max size (packing constraint)\n"
        error_msg += f"{'='*70}\n"
        raise ValueError(error_msg)


def build_config_from_args(args):
    """Build environment configuration from command line arguments"""
    arena_type = args.arena_type if hasattr(args, 'arena_type') else 'table'

    # Set default table size based on arena type
    if arena_type == 'bins':
        default_table_size = [0.39, 0.49, 0.82]  # Bins-specific dimensions
    else:
        default_table_size = [0.8, 0.8, 0.05]   # Standard table dimensions

    config = {
        'robot': args.robot,
        'placement_mode': 'random',
        'placement_params': {},
        'objects': [],
        'arena_type': arena_type,
        'table_config': {
            'size': args.table_size if args.table_size else default_table_size,
            'height': args.table_height,
            'friction': args.table_friction if args.table_friction else [1.0, 5e-3, 1e-4]
        },
        'bin_config': {
            'bin1_pos': args.bin1_pos if hasattr(args, 'bin1_pos') and args.bin1_pos else [0.1, -0.25, 0.8]
        }
    }

    # Determine placement mode and positions
    if args.preset:
        # Load preset configuration
        preset = get_preset_config(args.preset)
        if preset:
            config.update(preset)
            return config
        else:
            print(f"Warning: Unknown preset '{args.preset}', using custom config")

    if args.positions:
        # Exact positions specified
        config['placement_mode'] = 'exact'
        positions = []
        for pos_str in args.positions:
            x, y = map(float, pos_str.split(','))
            positions.append([x, y])
        config['placement_params']['positions'] = positions
        num_objects = len(positions)

    elif args.layout == 'grid':
        # Grid layout
        config['placement_mode'] = 'exact'
        rows, cols = map(int, args.grid_size.split('x'))
        positions = generate_grid_positions(rows, cols, args.spacing)
        config['placement_params']['positions'] = positions
        num_objects = len(positions)

    elif args.layout == 'circle':
        # Circle layout
        config['placement_mode'] = 'exact'
        positions = generate_circle_positions(args.num_objects, args.radius)
        config['placement_params']['positions'] = positions
        num_objects = args.num_objects

    elif args.layout == 'line':
        # Line layout
        config['placement_mode'] = 'exact'
        positions = generate_line_positions(args.num_objects, args.line_length, args.line_axis)
        config['placement_params']['positions'] = positions
        num_objects = args.num_objects

    else:
        # Random placement (default)
        config['placement_mode'] = 'random'
        config['placement_params'] = {}

        # Set custom ranges if provided
        if args.x_range:
            config['placement_params']['x_range'] = args.x_range
        if args.y_range:
            config['placement_params']['y_range'] = args.y_range

        # Use placement_range if provided and custom ranges not set
        if args.placement_range is not None:
            if 'x_range' not in config['placement_params']:
                config['placement_params']['x_range'] = [-args.placement_range, args.placement_range]
            if 'y_range' not in config['placement_params']:
                config['placement_params']['y_range'] = [-args.placement_range, args.placement_range]

        # If neither is set, arena-specific defaults will be used in _get_arena_placement_config()

        num_objects = args.num_objects

    # Build object configurations
    # Use list cycling to repeat types if fewer than num_objects
    if args.type:
        obj_types = [args.type[i % len(args.type)] for i in range(num_objects)]
    else:
        obj_types = ['box'] * num_objects

    # Parse colors
    if args.colors == 'random':
        colors = [[np.random.random(), np.random.random(), np.random.random(), 1.0]
                  for _ in range(num_objects)]
    elif args.colors:
        # Cycle through provided colors
        colors = [parse_color(args.colors[i % len(args.colors)]) for i in range(num_objects)]
    else:
        default_colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        colors = [parse_color(default_colors[i % len(default_colors)]) for i in range(num_objects)]

    # Parse sizes - cycle through provided sizes
    if args.sizes:
        sizes = [args.sizes[i % len(args.sizes)] for i in range(num_objects)]
    else:
        sizes = [args.size] * num_objects

    # Create object configurations
    config['objects'] = []

    # Generate simple sequential names without timestamps to avoid MuJoCo naming conflicts
    for i in range(num_objects):
        obj_type = obj_types[i]
        # Use simple sequential names - no timestamp needed
        simple_name = f"{obj_type}_{i}"

        obj_config = {
            'type': obj_type,
            'size': sizes[i],
            'color': colors[i],
            'name': simple_name,
            'density': args.density
        }
        config['objects'].append(obj_config)

    # Validate object sizes for the arena type
    arena_type = config['arena_type']
    num_objects = len(config['objects'])
    table_size = config['table_config'].get('size')

    # Should already be correct from config building, but double-check
    if table_size is None:
        table_size = [0.39, 0.49, 0.82] if arena_type == 'bins' else [0.8, 0.8, 0.05]

    max_size = get_max_object_size(arena_type, num_objects, table_size)

    validate_object_sizes(config['objects'], arena_type, table_size)

    # Print validation success message
    print(f"\n✓ Size validation passed!")
    print(f"  Arena: {arena_type}")
    print(f"  Objects: {num_objects}")
    print(f"  Arena dimensions: {table_size[0]:.2f}m × {table_size[1]:.2f}m")
    print(f"  Max safe size: {max_size:.3f}m ({max_size*100:.1f}cm)")

    return config


def save_config(config, name):
    """Save configuration to file"""
    config_dir = Path("data/environments")
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / f"{name}.json"

    # Add metadata
    from datetime import datetime
    config['metadata'] = {
        'name': name,
        'created': datetime.now().isoformat(),
        'version': '1.0'
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Configuration saved: {name}")
    print(f"  Location: {config_file}")
    print(f"\nUse it with:")
    print(f"  python src/run_simulation.py --mode sim --env {name} --episodes 10")


def run_environment(config, args):
    """Create and run environment from configuration"""

    print(f"\n{'='*60}")
    print("Custom Environment")
    print(f"{'='*60}")
    print(f"Description: {config.get('description', 'Custom environment')}")
    print(f"Robot: {config['robot']}")
    print(f"Arena: {config.get('arena_type', 'table')}")
    print(f"Placement: {config['placement_mode']}")
    print(f"Objects: {len(config['objects'])}")
    print(f"{'='*60}\n")

    # Print object details
    print("Objects:")
    for i, obj in enumerate(config['objects']):
        size_str = f"{obj['size']:.3f}m" if isinstance(obj['size'], (int, float)) else str(obj['size'])
        print(f"  {i+1}. {obj['name']:15s} {obj['type']:10s} {size_str}")

    if config['placement_mode'] == 'exact' and 'positions' in config['placement_params']:
        print("\nPositions:")
        for i, pos in enumerate(config['placement_params']['positions']):
            print(f"  {i+1}. ({pos[0]:+.3f}, {pos[1]:+.3f})")

    print(f"\n{'='*60}\n")

    # Create environment
    arena_type = config.get('arena_type', 'table')
    bin_config = config.get('bin_config', {})
    table_config = config.get('table_config', {})

    # Get table dimensions from config (should already be set correctly)
    table_full_size = table_config.get('size')
    if table_full_size is None:
        # Fallback: use arena-specific defaults
        table_full_size = [0.39, 0.49, 0.82] if arena_type == 'bins' else [0.8, 0.8, 0.05]

    print(f"DEBUG: Arena type: {arena_type}")
    print(f"DEBUG: Table dimensions: {table_full_size}")

    env = UnifiedCustomEnv(
        robots=config['robot'],
        object_configs=config['objects'],
        placement_mode=config['placement_mode'],
        placement_params=config['placement_params'],
        arena_type=arena_type,
        table_full_size=tuple(table_full_size),
        bin1_pos=tuple(bin_config.get('bin1_pos', [0.1, -0.25, 0.8])),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=50,
        horizon=1000,
    )

    print("Environment created successfully!")
    print("Running simulation... (Press Ctrl+C to stop)\n")

    try:
        obs = env.reset()
        env.render()

        step_count = 0
        import time

        while True:
            action = np.random.randn(env.robots[0].action_dim) * 0.02
            obs, reward, done, info = env.step(action)
            env.render()

            # Add small sleep to prevent CPU spinning
            time.sleep(0.01)

            step_count += 1
            if step_count % 100 == 0:
                print(f"Step: {step_count}")

            if done:
                print(f"\nEpisode complete at step {step_count}")
                obs = env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        env.close()
        print("Environment closed")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Environment Creator - Create custom robosuite environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMMON EXAMPLES:

  # Basic environments
  python create_env.py --num-objects 5 --type box ball --colors random --save my_env
  python create_env.py --num-objects 4 --type bottle can milk --save kitchen

  # Different sizes (IMPORTANT: use --sizes plural)
  python create_env.py --num-objects 3 --type box --sizes 0.02 0.05 0.08 --save varied

  # Different arenas
  python create_env.py --arena-type bins --num-objects 6 --type ball --save sorting
  python create_env.py --arena-type pegs --num-objects 2 --type round_nut --save pegs

  # Layout patterns
  python create_env.py --layout grid --grid-size 3x3 --type ball --save grid_env

  # Use your environment
  python src/run_simulation.py --mode teleop --env my_env

For complete documentation: data/docs/Create_env_guide.md
Environments saved to: data/environments/
        """
    )

    # ========== PRESET OR CUSTOM ==========
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--preset", type=str,
                           choices=['stacking', 'sorting', 'obstacles', 'pick_place'],
                           help="Use a preset environment configuration")

    # ========== PLACEMENT MODE ==========
    parser.add_argument("--layout", type=str, default='random',
                       choices=['random', 'grid', 'circle', 'line'],
                       help="Object placement layout (default: random)")

    # ========== EXACT POSITIONS ==========
    parser.add_argument("--positions", nargs='+', metavar="X,Y",
                       help="Exact XY positions (e.g., '0.1,0.1' '0.2,0.0')")

    # ========== GRID OPTIONS ==========
    parser.add_argument("--grid-size", type=str, default="3x3",
                       help="Grid size (e.g., '3x3', '2x4') for grid layout")
    parser.add_argument("--spacing", type=float, default=0.1,
                       help="Grid spacing in meters (default: 0.1)")

    # ========== CIRCLE OPTIONS ==========
    parser.add_argument("--radius", type=float, default=0.15,
                       help="Circle radius in meters (default: 0.15)")

    # ========== LINE OPTIONS ==========
    parser.add_argument("--line-length", type=float, default=0.3,
                       help="Line length in meters (default: 0.3)")
    parser.add_argument("--line-axis", type=str, default='x', choices=['x', 'y'],
                       help="Line axis (default: x)")

    # ========== OBJECT PROPERTIES ==========
    parser.add_argument("--num-objects", type=int, default=3,
                       help="Number of objects (for random/circle/line layouts)")
    parser.add_argument("--type", nargs='+',
                       choices=['box', 'ball', 'sphere', 'cylinder', 'capsule',
                                'bottle', 'can', 'lemon', 'milk', 'bread', 'cereal',
                                'square_nut', 'round_nut', 'cone', 'hammer', 'pot'],
                       help="Object types: primitives (box, ball, cylinder, capsule), "
                            "household items (bottle, can, lemon, milk, bread, cereal), "
                            "nuts (square_nut, round_nut), tools (cone, hammer, pot)")
    parser.add_argument("--size", type=float, default=0.025,
                       help="Default object size in meters (default: 0.025)")
    parser.add_argument("--sizes", nargs='+', type=float,
                       help="Individual object sizes in meters (repeats if fewer than objects)")
    parser.add_argument("--colors", nargs='+',
                       help="Colors: 'random' or names (red, green, blue, yellow, cyan, magenta, orange, purple, white, gray, black)")
    parser.add_argument("--density", type=float, default=100.0,
                       help="Object density in kg/m³ (default: 100, light=50, heavy=500)")

    # ========== RANDOM PLACEMENT OPTIONS ==========
    parser.add_argument("--placement-range", type=float, default=None,
                       help="Placement range from center for random layout in meters (default: auto-adjusted per arena)")
    parser.add_argument("--x-range", nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help="Custom X placement range (overrides --placement-range)")
    parser.add_argument("--y-range", nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help="Custom Y placement range (overrides --placement-range)")

    # ========== ARENA OPTIONS ==========
    parser.add_argument("--arena-type", type=str, default="table",
                       choices=["table", "bins", "empty", "pegs", "wipe", "multi_table"],
                       help="Arena type: table, bins, empty, pegs, wipe, or multi_table")
    parser.add_argument("--table-size", nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help="Table size [x, y, z] in meters (default: 0.8 0.8 0.05)")
    parser.add_argument("--table-height", type=float, default=0.8,
                       help="Table height from ground in meters (default: 0.8)")
    parser.add_argument("--table-friction", nargs=3, type=float, metavar=('SLIDING', 'TORSIONAL', 'ROLLING'),
                       help="Table friction [sliding, torsional, rolling] (default: 1.0 0.005 0.0001)")
    parser.add_argument("--bin1-pos", nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help="Position of first bin for bins arena (default: 0.1 -0.5 0.8)")

    # ========== ROBOT ==========
    parser.add_argument("--robot", type=str, default="Panda",
                       choices=["Panda", "Sawyer"],
                       help="Robot type: Panda (7-DOF, Franka Emika) or Sawyer (7-DOF, Rethink Robotics)")

    # ========== SAVE ==========
    parser.add_argument("--save", type=str, metavar="NAME",
                       help="Save configuration with this name (REQUIRED)")

    # ========== INFO ==========
    parser.add_argument("--list-presets", action="store_true",
                       help="List available preset environments")

    args = parser.parse_args()

    # Handle special commands
    if args.list_presets:
        print("\nAvailable Preset Environments:")
        print("="*60)
        presets = ['stacking', 'sorting', 'obstacles', 'pick_place']
        for preset in presets:
            config = get_preset_config(preset)
            print(f"\n  {preset:15s} - {config['description']}")
            print(f"                    {len(config['objects'])} objects")
        print("\n" + "="*60)
        print("\nUse with: python src/create_env.py --preset <name> --save <name>")
        return

    # Validate required arguments
    if not args.save:
        parser.error("--save is required (use --save NAME to save configuration)")

    # Build configuration
    config = build_config_from_args(args)

    # Print configuration summary
    print(f"\n{'='*60}")
    print("Environment Configuration")
    print(f"{'='*60}")
    print(f"Name: {args.save}")
    print(f"Robot: {config['robot']}")
    print(f"Placement: {config['placement_mode']}")
    print(f"Objects: {len(config['objects'])}")

    # Print object details
    print(f"\nObjects:")
    for i, obj in enumerate(config['objects']):
        obj_type = obj['type']
        size_str = f"{obj['size']:.3f}m" if isinstance(obj['size'], (int, float)) else str(obj['size'])
        color = obj.get('color', [1, 0, 0, 1])
        color_name = [k for k, v in {
            'red': [1.0, 0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0], 'yellow': [1.0, 1.0, 0.0, 1.0],
            'cyan': [0.0, 1.0, 1.0, 1.0], 'magenta': [1.0, 0.0, 1.0, 1.0]
        }.items() if v == color] or ['custom']
        print(f"  {i+1}. {obj_type:10s} {size_str:8s} {color_name[0]}")

    if config['placement_mode'] == 'exact' and 'positions' in config['placement_params']:
        print(f"\nPositions:")
        for i, pos in enumerate(config['placement_params']['positions'][:5]):
            print(f"  {i+1}. ({pos[0]:+.3f}, {pos[1]:+.3f})")
        if len(config['placement_params']['positions']) > 5:
            print(f"  ... and {len(config['placement_params']['positions']) - 5} more")

    print(f"{'='*60}\n")

    # Save configuration
    save_config(config, args.save)

    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print(f"\n1. Use with SpaceMouse teleoperation:")
    print(f"   python src/run_simulation.py --mode teleop --env {args.save}")
    print(f"\n2. Run automated simulation:")
    print(f"   python src/run_simulation.py --mode sim --env {args.save} --episodes 10")
    print(f"\n3. Test with standard Lift environment:")
    print(f"   python src/run_simulation.py --mode teleop --robot Panda --task Lift")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
