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
    """Build ObservationConfig for low-dim + RGB (+ front mask for runtime color cues)."""
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
    # Keep wrist mask disabled to reduce payload; enable front mask so eval.py
    # can infer object colors from runtime snapshots.
    obs_config.front_camera.mask = True

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


def _read_first_task_string(dataset_root: str | None) -> str | None:
    if not dataset_root:
        return None
    tasks_path = Path(dataset_root).expanduser() / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return None

    try:
        import pyarrow.parquet as pq  # type: ignore

        table = pq.read_table(tasks_path, columns=[], use_threads=False)
        # The index written by pandas is usually restored as "__index_level_0__".
        # Read full table if needed to access index-like column.
        if "__index_level_0__" not in table.column_names:
            table = pq.read_table(tasks_path, use_threads=False)
        if "__index_level_0__" in table.column_names and table.num_rows > 0:
            value = table["__index_level_0__"][0].as_py()
            return str(value) if value is not None else None
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(tasks_path)
        if len(df.index) > 0:
            return str(df.index[0])
    except Exception:
        return None
    return None


def infer_expected_with_context_prompt(
    *,
    dataset_root: str | None,
    train_dataset_root: str | None,
) -> dict[str, object]:
    """Infer whether eval should use context prompt to match training style."""
    out: dict[str, object] = {
        "expected_with_context_prompt": None,
        "inference_source": None,
        "sample_task_text": None,
    }

    sample = _read_first_task_string(dataset_root)
    if sample:
        out["sample_task_text"] = sample
        out["expected_with_context_prompt"] = sample.startswith(
            "This prompt describes a robotic manipulation task using scene graphs."
        )
        out["inference_source"] = "dataset_tasks_parquet"
        return out

    root_for_heuristic = str(train_dataset_root or dataset_root or "")
    lowered = root_for_heuristic.lower()
    if "lerobot_trial_3" in lowered or "lerobot_trial3" in lowered:
        out["expected_with_context_prompt"] = True
        out["inference_source"] = "dataset_root_heuristic_trial3"
    elif "lerobot_trial_2" in lowered or "lerobot_trial2" in lowered:
        out["expected_with_context_prompt"] = False
        out["inference_source"] = "dataset_root_heuristic_trial2"

    return out


def _resolve_pretrained_model_dir(checkpoint_path: str | Path) -> Path | None:
    """Resolve a user checkpoint path to a LeRobot pretrained_model directory."""
    p = Path(checkpoint_path).expanduser()

    if (p / "config.json").is_file():
        return p

    if (p / "pretrained_model" / "config.json").is_file():
        return p / "pretrained_model"

    last = p / "checkpoints" / "last" / "pretrained_model"
    if (last / "config.json").is_file():
        return last

    ckpt_root = p / "checkpoints"
    if not ckpt_root.is_dir():
        return None

    candidates: list[Path] = []
    for ckpt_dir in ckpt_root.iterdir():
        if not ckpt_dir.is_dir():
            continue
        pm = ckpt_dir / "pretrained_model"
        if (pm / "config.json").is_file():
            candidates.append(pm)

    if not candidates:
        return None

    def _score(pm_dir: Path) -> tuple[int, str]:
        name = pm_dir.parent.name
        return (int(name) if name.isdigit() else -1, name)

    candidates.sort(key=_score)
    return candidates[-1]


def load_checkpoint_train_dataset_metadata(checkpoint_path: str | None) -> dict[str, str | None]:
    """Read dataset provenance from checkpoint train_config.json if available."""
    out: dict[str, str | None] = {
        "checkpoint_path_input": checkpoint_path,
        "pretrained_model_dir": None,
        "train_config_path": None,
        "train_dataset_repo_id": None,
        "train_dataset_root": None,
    }
    if not checkpoint_path:
        return out

    pm_dir = _resolve_pretrained_model_dir(checkpoint_path)
    if pm_dir is None:
        return out

    out["pretrained_model_dir"] = str(pm_dir)
    train_config_path = pm_dir / "train_config.json"
    if not train_config_path.is_file():
        return out

    out["train_config_path"] = str(train_config_path)
    try:
        with open(train_config_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return out

    dataset_cfg = payload.get("dataset", {}) if isinstance(payload, dict) else {}
    if isinstance(dataset_cfg, dict):
        repo_id = dataset_cfg.get("repo_id")
        root = dataset_cfg.get("root")
        out["train_dataset_repo_id"] = str(repo_id) if repo_id is not None else None
        out["train_dataset_root"] = str(root) if root is not None else None
    return out
