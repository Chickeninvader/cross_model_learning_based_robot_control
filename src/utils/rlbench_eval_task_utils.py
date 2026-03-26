from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.rlbench_eval_utils import quat_angle_deg
from src.utils.rlbench_utils import extract_eef_state, extract_joint_state
from src.utils.scene_graph_language import generate_task_descriptions
from src.utils.scene_graph_utils import (
    base_object_type,
    decode_mask_array,
    discover_object_mapping_from_handles,
    rgb_to_color_name,
    scan_mask_handles_from_mask_arrays,
)

RELATIONSHIP_TASK_STEPS_CACHE: dict[tuple[str, bool], list[str]] = {}
SUPPORTED_ROBOTS: set[str] = {"panda", "jaco", "mico", "sawyer", "ur5"}
ROBOT_SETUP_ALIASES: dict[str, str] = {
    "franka": "panda",
    "franka_panda": "panda",
}


def normalize_robot_setup(robot_setup: str) -> str:
    normalized = str(robot_setup).strip().lower()
    return ROBOT_SETUP_ALIASES.get(normalized, normalized)


def validate_action_mode_for_robot(action_mode: str, robot_setup: str) -> None:
    if robot_setup != "panda" and action_mode == "joint_velocity":
        raise ValueError(
            "Joint-state evaluation (action_mode=joint_velocity) is only supported "
            "for robot_setup=panda (Franka). Use ee_planning/ee_ik for other robots."
        )


def is_robot_or_gripper_object(obj_name: str, meta: dict | None) -> bool:
    text_parts = [str(obj_name)]
    if isinstance(meta, dict):
        for k in ("type", "display_name"):
            v = meta.get(k)
            if isinstance(v, str):
                text_parts.append(v)
    text = " ".join(text_parts).lower()
    return ("robot" in text) or ("gripper" in text)


def sanitize_object_meta(obj_name: str, meta: dict) -> dict:
    cleaned: dict[str, object] = {}
    base_type = meta.get("type")
    if isinstance(base_type, str) and base_type.strip():
        cleaned["type"] = base_type.strip()
    else:
        cleaned["type"] = str(obj_name).strip()

    display_name = meta.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        cleaned["display_name"] = display_name.strip()

    attrs_raw = meta.get("attributes")
    attrs: dict[str, str] = {}
    if isinstance(attrs_raw, dict):
        for k, v in attrs_raw.items():
            if isinstance(k, str) and isinstance(v, str) and v.strip():
                attrs[k] = v.strip()

    color_top = meta.get("color")
    if isinstance(color_top, str) and color_top.strip():
        attrs["color"] = color_top.strip()

    if is_robot_or_gripper_object(obj_name, cleaned):
        attrs.pop("color", None)

    if attrs:
        cleaned["attributes"] = attrs
    return cleaned


def scan_mask_handles_from_observations(observations: list, max_frames: int = 5) -> list[int]:
    mask_arrays = [
        np.asarray(getattr(obs, "front_mask"))
        for obs in observations[:max_frames]
        if getattr(obs, "front_mask", None) is not None
    ]
    return scan_mask_handles_from_mask_arrays(mask_arrays, max_frames=max_frames)


def runtime_objects_meta_from_observations(
    task: str,
    rlbench_root: str | Path,
    variation: int,
    planner_obs: list,
    *,
    max_frames: int = 8,
    min_pixels: int = 40,
) -> dict[str, dict]:
    task_dir = Path(rlbench_root) / task
    info_path = task_dir / "info.json"
    if not info_path.is_file() or not planner_obs:
        return {}
    if getattr(planner_obs[0], "front_mask", None) is None:
        return {}

    try:
        with open(info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
    except Exception:
        return {}

    raw_objects = info.get("objects", {})
    raw_mapping = info.get("object_mapping", {})
    if not isinstance(raw_objects, dict) or not isinstance(raw_mapping, dict):
        return {}

    ordered_object_names: list[str] = []
    for key in sorted(raw_objects.keys(), key=lambda x: int(str(x))):
        obj = raw_objects.get(key, {})
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).strip()
        if name:
            ordered_object_names.append(name)
    if not ordered_object_names:
        return {}

    reference_mapping: dict[int, str] = {}
    for handle_id, name in raw_mapping.items():
        try:
            reference_mapping[int(handle_id)] = str(name)
        except Exception:
            continue
    if not reference_mapping:
        return {}

    actual_handles = scan_mask_handles_from_observations(planner_obs, max_frames=max_frames)
    if not actual_handles:
        return {}

    object_mapping = discover_object_mapping_from_handles(reference_mapping, actual_handles)
    handles_by_object: dict[str, list[int]] = {n: [] for n in ordered_object_names}
    for h, name in object_mapping.items():
        if name in handles_by_object:
            handles_by_object[name].append(int(h))
    for name in handles_by_object:
        handles_by_object[name].sort()

    per_object_rgb_samples: dict[str, list[np.ndarray]] = {n: [] for n in ordered_object_names}
    n_frames = min(len(planner_obs), max_frames)
    for obs in planner_obs[:n_frames]:
        rgb = getattr(obs, "front_rgb", None)
        mask = getattr(obs, "front_mask", None)
        if rgb is None or mask is None:
            continue

        rgb_frame = np.asarray(rgb)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
            continue
        rgb_frame = rgb_frame[:, :, :3]
        if rgb_frame.dtype != np.uint8:
            if np.issubdtype(rgb_frame.dtype, np.floating) and float(np.max(rgb_frame)) <= 1.0:
                rgb_frame = np.clip(rgb_frame * 255.0, 0, 255).astype(np.uint8)
            else:
                rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)

        decoded = decode_mask_array(np.asarray(mask))
        for name in ordered_object_names:
            handle_ids = handles_by_object.get(name, [])
            if not handle_ids:
                continue
            obj_mask = np.isin(decoded, handle_ids)
            if int(obj_mask.sum()) < min_pixels:
                continue
            pixels = rgb_frame[obj_mask]
            if pixels.size == 0:
                continue
            per_object_rgb_samples[name].append(np.median(pixels, axis=0))

    out: dict[str, dict] = {}
    for name in ordered_object_names:
        meta: dict[str, object] = {"type": base_object_type(name)}
        samples = per_object_rgb_samples.get(name, [])
        if samples:
            rgb_med = np.median(np.vstack(samples), axis=0)
            rgb_triplet = np.clip(np.round(rgb_med), 0, 255).astype(np.uint8)
            color_name = rgb_to_color_name(rgb_triplet)
            if not is_robot_or_gripper_object(name, meta):
                meta["attributes"] = {"color": color_name}
        out[name] = sanitize_object_meta(name, meta)
    return out


def task_steps_from_transitions(
    transitions: list[dict],
    objects_meta: dict[str, dict],
    task: str,
    *,
    use_context_prompt: bool = False,
) -> list[str]:
    episodes: list[dict[str, list[dict]]] = []
    begin_sg: list[dict] = []
    for transition in transitions:
        if not isinstance(transition, dict):
            continue
        rels = transition.get("relationships", [])
        if not isinstance(rels, list):
            rels = []
        end_sg = [r for r in rels if isinstance(r, dict)]
        episodes.append({"begin_sg": list(begin_sg), "end_sg": end_sg})
        begin_sg = end_sg
    if not episodes:
        return []
    task_steps = generate_task_descriptions(
        episodes,
        objects_meta,
        task,
        use_context=use_context_prompt,
    )
    return [str(s).strip() for s in task_steps if str(s).strip()]


def load_objects_meta(
    task: str,
    rlbench_root: str | Path,
    template_path: Path,
    variation: int,
) -> dict[str, dict]:
    root = Path(rlbench_root)
    task_dir = root / task
    sg_candidates = [
        task_dir / f"variation{variation}" / f"{task}_scene_graph.json",
        template_path.parent / f"variation{variation}" / f"{task}_scene_graph.json",
    ]
    for sg_path in sg_candidates:
        if not sg_path.is_file():
            continue
        try:
            with open(sg_path, "r", encoding="utf-8") as handle:
                sg_data = json.load(handle)
            objects = sg_data.get("objects", {})
            if not isinstance(objects, dict):
                continue
            objects_meta: dict[str, dict] = {}
            for obj_name, obj_meta in objects.items():
                if not isinstance(obj_meta, dict):
                    continue
                key = str(obj_name).strip()
                if key:
                    objects_meta[key] = sanitize_object_meta(key, obj_meta)
            if objects_meta:
                return objects_meta
        except Exception:
            continue

    info_path = template_path.parent / "info.json"
    if info_path.is_file():
        try:
            with open(info_path, "r", encoding="utf-8") as handle:
                info_data = json.load(handle)
            raw_objects = info_data.get("objects", {})
            if isinstance(raw_objects, dict):
                objects_meta = {}
                for obj in raw_objects.values():
                    if not isinstance(obj, dict):
                        continue
                    name = str(obj.get("name", "")).strip()
                    if name:
                        objects_meta[name] = sanitize_object_meta(name, obj)
                return objects_meta
        except Exception:
            return {}
    return {}


def resolve_relationship_template_path(
    task: str,
    rlbench_root: str | Path,
    explicit: str | None,
) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    p = Path(rlbench_root) / task / f"{task}_relationship_template.json"
    return p if p.is_file() else None


def load_relationship_transitions(template_path: Path) -> list[dict]:
    with open(template_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    transitions = data.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return []
    return transitions


def infer_task_steps_from_relationship_template(
    template_path: Path,
    task: str,
    *,
    rlbench_root: str | Path,
    variation: int,
    use_context_prompt: bool = False,
) -> list[str]:
    try:
        with open(template_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    transitions = data.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return []
    objects_meta = load_objects_meta(
        task=task,
        rlbench_root=rlbench_root,
        template_path=template_path,
        variation=variation,
    )
    return task_steps_from_transitions(
        transitions,
        objects_meta,
        task,
        use_context_prompt=use_context_prompt,
    )


def resolve_task_steps_from_relationship_template(
    task: str,
    rlbench_root: str | Path,
    explicit_template_path: str | None,
    *,
    use_context_prompt: bool = False,
    variation: int = 0,
) -> list[str]:
    template_path = resolve_relationship_template_path(task, rlbench_root, explicit_template_path)
    if template_path is None:
        return [""]

    cache_key = (f"{template_path.resolve()}::var{variation}", bool(use_context_prompt))
    cached = RELATIONSHIP_TASK_STEPS_CACHE.get(cache_key)
    if cached is None:
        cached = infer_task_steps_from_relationship_template(
            template_path,
            task,
            rlbench_root=rlbench_root,
            variation=variation,
            use_context_prompt=use_context_prompt,
        )
        RELATIONSHIP_TASK_STEPS_CACHE[cache_key] = cached
    return cached if cached else [""]


def gripper_change_indices(observations: list, threshold: float = 0.5) -> list[int]:
    if not observations:
        return []
    gripper = np.asarray([float(obs.gripper_open) for obs in observations], dtype=np.float64)
    indices: list[int] = []
    for idx in range(1, gripper.shape[0]):
        prev_open = gripper[idx - 1] > threshold
        curr_open = gripper[idx] > threshold
        if prev_open != curr_open:
            indices.append(idx)
    return indices


def segment_boundaries(change_indices: list[int], n_segments: int, last_index: int) -> list[int]:
    needed = max(0, n_segments - 1)
    boundaries = list(change_indices[:needed])
    while len(boundaries) < needed:
        boundaries.append(last_index)
    return boundaries


def segment_ranges(boundaries: list[int], n_segments: int, last_index: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(n_segments):
        end = boundaries[idx] if idx < len(boundaries) else last_index
        end = int(max(start, min(end, last_index)))
        ranges.append((start, end))
        start = min(end + 1, last_index)
    return ranges


def segmentation_from_relationship_template(
    observations: list,
    transitions: list[dict],
    last_index: int,
    threshold: float = 0.5,
) -> tuple[int, list[int]]:
    n = len(transitions)
    if n <= 1:
        return max(1, n), []
    gripper_open = [float(obs.gripper_open) > threshold for obs in observations]
    boundaries: list[int] = []
    search_from = 1
    for i in range(n - 1):
        t = transitions[i]
        from_open = str(t.get("from_state", "")).lower() == "open"
        to_open = str(t.get("to_state", "")).lower() == "open"
        found: int | None = None
        for idx in range(max(search_from, 1), len(observations)):
            if gripper_open[idx - 1] == gripper_open[idx]:
                continue
            if gripper_open[idx - 1] == from_open and gripper_open[idx] == to_open:
                found = idx
                break
        if found is None:
            for idx in range(max(search_from, 1), len(observations)):
                if gripper_open[idx - 1] != gripper_open[idx]:
                    found = idx
                    break
        if found is None:
            found = last_index
        found = int(max(0, min(found, last_index)))
        boundaries.append(found)
        search_from = found + 1
    return n, boundaries


def instructions_per_segment(
    n_segments: int,
    task_steps: list[str],
    task_description: str,
) -> list[str]:
    td = (task_description or "").strip()
    if n_segments <= 0:
        return []
    if n_segments == 1:
        return [td if td else (task_steps[0] if task_steps else "")]
    steps = [s.strip() for s in task_steps if s.strip()]
    if len(steps) >= n_segments:
        return steps[:n_segments]
    if not steps:
        return [td] * n_segments if td else [""] * n_segments
    out = list(steps)
    while len(out) < n_segments:
        out.append(out[-1])
    return out[:n_segments]


def segment_state_metric(expert_final_obs, policy_final_obs) -> dict[str, float]:
    expert_eef = extract_eef_state(expert_final_obs).astype(np.float64)
    policy_eef = extract_eef_state(policy_final_obs).astype(np.float64)
    expert_joint = extract_joint_state(expert_final_obs).astype(np.float64)
    policy_joint = extract_joint_state(policy_final_obs).astype(np.float64)
    return {
        "eef_l2": float(np.linalg.norm(policy_eef - expert_eef)),
        "joint_l2": float(np.linalg.norm(policy_joint - expert_joint)),
        "pos_l2": float(np.linalg.norm(policy_eef[:3] - expert_eef[:3])),
        "rot_deg": quat_angle_deg(policy_eef[3:7], expert_eef[3:7]),
    }


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))
