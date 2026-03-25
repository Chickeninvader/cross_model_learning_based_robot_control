from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from pyrep.const import RenderMode
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    JointVelocity,
)
from rlbench.action_modes.gripper_action_modes import Discrete


def build_obs_config(image_size: list[int], renderer: str) -> ObservationConfig:
    """Build ObservationConfig for low-dim + wrist/front RGB only."""
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)
    obs_config.joint_forces = False
    # Some RLBench robot grippers (e.g. UR5 variants) do not expose touch sensors.
    # Disable this observation channel so scene.get_observation() doesn't call
    # gripper.get_touch_sensor_forces() and raise NotImplementedError.
    obs_config.gripper_touch_forces = False

    cameras = [obs_config.wrist_camera, obs_config.front_camera]
    for cam in cameras:
        cam.set_all(False)
        cam.rgb = True
        cam.depth = False
        cam.mask = False
        cam.point_cloud = False
        cam.image_size = image_size
        cam.depth_in_meters = False
        cam.masks_as_one_channel = False

    if renderer == "opengl":
        render_mode = RenderMode.OPENGL
    elif renderer == "opengl3":
        render_mode = RenderMode.OPENGL3
    else:
        raise ValueError(f"Unknown renderer: {renderer}")

    for cam in cameras:
        cam.render_mode = render_mode

    return obs_config


def build_action_mode(action_mode: str) -> MoveArmThenGripper:
    if action_mode == "ee_planning":
        arm_mode = EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False)
    elif action_mode == "ee_ik":
        arm_mode = EndEffectorPoseViaIK(absolute_mode=True, collision_checking=False)
    elif action_mode == "joint_velocity":
        arm_mode = JointVelocity()
    else:
        raise ValueError(f"Unknown action mode: {action_mode}")

    return MoveArmThenGripper(arm_action_mode=arm_mode, gripper_action_mode=Discrete())


def write_json(path: str, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_var_seed_dir(path: Path) -> bool:
    return re.match(r"^var-?\d+_seed-?\d+$", path.name) is not None


def _is_policy_state_dir(path: Path) -> bool:
    if not path.is_dir() or "_" not in path.name:
        return False
    for child in path.iterdir():
        if child.is_dir() and _is_var_seed_dir(child):
            return True
    return False


def discover_task_roots(aggregate_root: Path, fallback_task_name: str) -> list[tuple[str, Path]]:
    """Return (task_name, task_root) pairs for aggregation.

    Supports both layouts:
      1) <aggregate_root>/<policy>_<state>/varX_seedY     (single task)
      2) <aggregate_root>/<task>/<policy>_<state>/...     (multi task)
    """
    if not aggregate_root.exists():
        raise FileNotFoundError(f"Aggregate root does not exist: {aggregate_root}")

    if any(_is_policy_state_dir(p) for p in aggregate_root.iterdir() if p.is_dir()):
        return [(fallback_task_name, aggregate_root)]

    task_roots: list[tuple[str, Path]] = []
    for child in sorted([p for p in aggregate_root.iterdir() if p.is_dir()]):
        if any(_is_policy_state_dir(p) for p in child.iterdir() if p.is_dir()):
            task_roots.append((child.name, child))
    return task_roots
