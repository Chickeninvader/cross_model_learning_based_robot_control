#!/usr/bin/env python3
"""
Convert RLBench variations into **LeRobot v3 datasets** in two action spaces:

- **EEF dataset**: `observation.state` + `action` (delta EEF)
- **Joint dataset**: `observation.state` + `action` (joint velocity)

The CLI produces merged outputs (no per-variation folders) named:

- `datasets/lerobot/<task_name>_eef/`
- `datasets/lerobot/<task_name>_joint/`

Key differences from the original converter
--------------------------------------------
* **No mask / depth channels** -- GR00T is RGB-only; we keep ``front_rgb`` and
  ``wrist_rgb`` only.
* **Natural-language task descriptions** derived from scene-graph transitions
  (ConceptGraphs-style), stored as the ``annotation.human.action.task_description``
  column that GR00T reads via ``task_index -> tasks.parquet -> "task"`` mapping.
* **GR00T annotation columns** added to the data parquet:
    - ``annotation.human.action.task_description`` (int64 -> task index)
    - ``annotation.human.action.task_name`` (int64 -> task index for short name)
    - ``annotation.human.validity`` (int64 -> always "Valid")
    - ``next.reward`` (float64 -> 0.0, with 1.0 at the last frame)
* **Per-episode stats** include ``q01`` / ``q99`` percentiles to match the GR00T
  LeRobot v3 schema.

Usage
-----
    python src/data_collection/convert_rlbench_to_lerobot.py \\
        --task_name stack_cups \\
        --rlbench_root datasets/rlbench \\
        --output_root datasets/lerobot \\
        --fps 10

The script reads:
    datasets/rlbench/<task_name>/variation<num>/
        episodes/episode0/front_rgb/*.png
        episodes/episode0/wrist_rgb/*.png
        episodes/episode0/low_dim_obs.pkl
        <task_name>_scene_graph.json
        object_color_map.json

And writes merged LeRobot-v3 datasets to:
    datasets/lerobot/<task_name>_eef/
    datasets/lerobot/<task_name>_joint/
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to sys.path to allow imports from src package
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import re

from src.utils.rlbench_utils import (
    build_episode_stats_groot,
    compute_delta_eef_actions,
    compute_joint_velocity_actions,
    compute_episode_boundaries,
    compute_feature_stats_groot,
    compute_image_stats_placeholder,
    compute_bool_stats_groot,
    extract_eef_state,
    extract_joint_state,
    find_transitions,
    images_to_video,
    load_rlbench_demo,
    load_scene_graph,
)
from src.utils.scene_graph_language import (
    generate_task_descriptions,
)

# -- Codec/pix-fmt used for output videos -----------------------------------
VIDEO_CODEC = "libopenh264"
PIX_FMT = "yuv420p"

# -- Camera names (GR00T convention) ----------------------------------------
# Map RLBench camera dirs to GR00T-style observation keys
CAMERAS: Dict[str, str] = {
    "front_rgb": "observation.images.front_rgb",
    "wrist_rgb": "observation.images.wrist_rgb",
}


def _is_robot_or_gripper_object(obj_name: str, meta: dict[str, Any] | None) -> bool:
    """Return True if object looks like robot/gripper related."""
    text_parts = [str(obj_name)]
    if isinstance(meta, dict):
        for k in ("type", "display_name"):
            v = meta.get(k)
            if isinstance(v, str):
                text_parts.append(v)
    text = " ".join(text_parts).lower()
    return ("robot" in text) or ("gripper" in text)


def _strip_robot_gripper_color_attributes(objects_meta: dict[str, Any]) -> dict[str, Any]:
    """Remove color attribute for robot/gripper-related objects."""
    sanitized: dict[str, Any] = {}
    for obj_name, meta in (objects_meta or {}).items():
        if not isinstance(meta, dict):
            sanitized[obj_name] = meta
            continue

        meta_copy = dict(meta)
        if _is_robot_or_gripper_object(obj_name, meta_copy):
            attrs = meta_copy.get("attributes")
            if isinstance(attrs, dict) and "color" in attrs:
                attrs_copy = dict(attrs)
                attrs_copy.pop("color", None)
                if attrs_copy:
                    meta_copy["attributes"] = attrs_copy
                else:
                    meta_copy.pop("attributes", None)
        sanitized[obj_name] = meta_copy
    return sanitized


# ===========================================================================
# Main conversion logic
# ===========================================================================

def convert(
    task_name: str,
    variation: int,
    rlbench_root: str,
    output_root: str,
    fps: int = 10,
    episode_index_in_rlbench: int = 0,
    use_context_prompt: bool = False,
    action_space: str = "eef",
):
    """Convert one RLBench variation into a LeRobot v3 dataset.

    Parameters
    ----------
    task_name : str
        RLBench task, e.g. ``"stack_cups"``.
    variation : int
        Variation index.
    rlbench_root : str
        Root containing ``<task>/variation<N>/...``.
    output_root : str
        Where to write the LeRobot dataset.
    fps : int
        Frames per second.
    episode_index_in_rlbench : int
        Which RLBench episode directory to read (default 0).
    use_context_prompt : bool
        If True, prepend the full ConceptGraphs-style scene context to each
        task description.  Default False (concise imperative only).
    action_space : str
        Which control/action space to export:
          - "eef"   : observation.state + action (delta EEF)  [x y z qx qy qz qw gripper]
            - "joint" : observation.state + action (joint velocity) [q0..q6 gripper]
    """
    # -- Paths ---------------------------------------------------------------
    var_dir = os.path.join(rlbench_root, task_name, f"variation{variation}")
    ep_dir = os.path.join(var_dir, "episodes", f"episode{episode_index_in_rlbench}")
    front_rgb_dir = os.path.join(ep_dir, "front_rgb")
    wrist_rgb_dir = os.path.join(ep_dir, "wrist_rgb")
    pkl_path = os.path.join(ep_dir, "low_dim_obs.pkl")
    sg_path = os.path.join(var_dir, f"{task_name}_scene_graph.json")
    color_map_path = os.path.join(var_dir, "object_color_map.json")

    dataset_name = f"{task_name}_variation{variation}"
    out_dir = os.path.join(output_root, dataset_name)

    print(f"[INFO] Converting {var_dir} -> {out_dir}")

    # -- Load sources --------------------------------------------------------
    observations = load_rlbench_demo(pkl_path)
    scene_graph = load_scene_graph(sg_path)
    with open(color_map_path) as f:
        object_color_map = json.load(f)

    transitions = find_transitions(scene_graph)

    # Frame count = observation count (1:1 mapping, no offset).
    # Extra images in the directory (if any) are ignored.
    n_obs = len(observations)
    n_frames_total = n_obs
    print(f"[INFO] Observations: {n_obs} (= number of frames to convert)")

    # Episode boundaries are based on the scene graph, clamped to [0, n_obs-1].
    episodes = compute_episode_boundaries(scene_graph, transitions, n_total_images=n_obs)
    n_episodes = len(episodes)
    print(f"[INFO] Found {len(transitions)} transitions -> {n_episodes} episodes")
    for _ti, _t in enumerate(transitions):
        print(f"       Transition {_ti} at frame {_t['frame_id']}")
    for _ei, _ep in enumerate(episodes):
        print(f"       Episode {_ei}: frames {_ep['start_frame']}-{_ep['end_frame']} "
              f"({_ep['end_frame'] - _ep['start_frame'] + 1} frames)")

    action_space = (action_space or "eef").strip().lower()
    if action_space not in {"eef", "joint"}:
        raise ValueError(f"Unknown action_space: {action_space}. Expected 'eef' or 'joint'.")

    # Pre-compute ALL states for the full observation list
    if action_space == "eef":
        state_feature_name = "observation.state"
        all_states = np.stack([extract_eef_state(o) for o in observations], axis=0)  # (n_obs, 8)

        def _compute_actions(obs_slice: list) -> np.ndarray:
            return compute_delta_eef_actions(obs_slice)
    else:
        # NOTE: SmolVLA training in LeRobot expects the low-dim state key to be
        # exactly `observation.state`. For joint-space datasets we therefore
        # store joint positions under `observation.state`.
        state_feature_name = "observation.state"
        all_states = np.stack([extract_joint_state(o) for o in observations], axis=0)  # (n_obs, 8)

        def _compute_actions(obs_slice: list) -> np.ndarray:
            return compute_joint_velocity_actions(obs_slice, fps=fps)

    # NOTE: actions are computed per-episode below (not globally),
    # because relative actions would be wrong at episode boundaries.

    objects_meta = _strip_robot_gripper_color_attributes(scene_graph.get("objects", {}))

    # -- Generate task descriptions from scene-graph transitions -------------
    task_descriptions: List[str] = generate_task_descriptions(
        episodes, objects_meta, task_name, use_context=use_context_prompt,
    )
    # Short task-name (like "StackCups" in GR00T)
    task_short_name = task_name.replace("_", " ").title().replace(" ", "")
    # Validity label
    validity_label = "Valid"

    # Build full task list: [desc_0, desc_1, ..., task_short_name, validity_label]
    # GR00T structure: different annotation types share the same task table.
    unique_descs = list(dict.fromkeys(task_descriptions))  # dedup, preserve order
    all_task_strings: List[str] = unique_descs + [task_short_name, validity_label]
    task_str_to_idx = {s: i for i, s in enumerate(all_task_strings)}

    # Per-description index for task_description column
    task_desc_idx_map = {desc: task_str_to_idx[desc] for desc in unique_descs}
    task_name_idx = task_str_to_idx[task_short_name]
    validity_idx = task_str_to_idx[validity_label]

    print(f"[INFO] Task descriptions ({len(unique_descs)} unique):")
    for d_ in unique_descs:
        print(f"       [{task_str_to_idx[d_]}] {d_[:120]}{'...' if len(d_) > 120 else ''}")

    # -- Per-episode data collection -----------------------------------------
    parquet_rows: List[dict] = []
    episode_meta_rows: List[dict] = []
    global_index = 0

    # Episode-level scene-graph info stored in info.json
    episode_sg_info: List[dict] = []

    total_frames = 0
    camera_names = list(CAMERAS.values())

    for ep_idx, ep in enumerate(episodes):
        start_frame = ep["start_frame"]
        end_frame = ep["end_frame"]
        n_frames = end_frame - start_frame + 1
        desc = task_descriptions[ep_idx]
        desc_task_idx = task_desc_idx_map[desc]

        print(f"  Episode {ep_idx}: frames {start_frame}-{end_frame} ({n_frames} frames)")

        # -- Observation slice (1:1 with frames, no offset) ----------------
        obs_start = max(0, start_frame)
        obs_end = min(n_obs - 1, end_frame)
        n_obs_frames = obs_end - obs_start + 1

        ep_states = all_states[obs_start : obs_end + 1]

        # Compute actions WITHIN this episode only (not globally)
        ep_obs_slice = observations[obs_start : obs_end + 1]
        ep_actions = _compute_actions(ep_obs_slice)  # (n_obs_frames, 8)

        # Safety: if episode boundary extends past obs range, pad/trim
        if n_obs_frames < n_frames:
            pad = n_frames - n_obs_frames
            ep_states = np.concatenate([ep_states, np.tile(ep_states[-1:], (pad, 1))], axis=0)
            ep_actions = np.concatenate([ep_actions, np.tile(ep_actions[-1:], (pad, 1))], axis=0)
        elif n_obs_frames > n_frames:
            ep_states = ep_states[:n_frames]
            ep_actions = ep_actions[:n_frames]

        # Debug: report obs alignment for this episode
        print(f"    obs[{obs_start}..{obs_end}] ({n_obs_frames} obs), "
              f"n_frames={n_frames}, padded={max(0, n_frames - n_obs_frames)}")
        if n_obs_frames > 0:
            print(f"    gripper[first]={ep_states[0, 7]:.1f}, gripper[last]={ep_states[min(n_obs_frames, n_frames)-1, 7]:.1f}")
            a_label = "delta_pos" if action_space == "eef" else "joint_vel"
            print(f"    action {a_label} range: {ep_actions[:, :3].min(0)} to {ep_actions[:, :3].max(0)}")

        # -- Create videos (RGB only, no mask) -------------------------------
        chunk_idx = 0
        file_idx = ep_idx

        for cam_dir, cam_key in CAMERAS.items():
            img_dir = os.path.join(ep_dir, cam_dir)
            vid_out = os.path.join(
                out_dir, "videos", cam_key,
                f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4",
            )
            print(f"    Creating video {vid_out}")
            images_to_video(img_dir, vid_out, start_frame, end_frame, fps,
                            codec=VIDEO_CODEC, pix_fmt=PIX_FMT)

        # -- Build parquet rows ----------------------------------------------
        timestamps = np.arange(n_frames, dtype=np.float64) / fps
        frame_indices = np.arange(n_frames, dtype=np.int64)
        done_flags = np.zeros(n_frames, dtype=bool)
        done_flags[-1] = True
        rewards = np.zeros(n_frames, dtype=np.float64)
        rewards[-1] = 1.0
        global_indices = np.arange(global_index, global_index + n_frames, dtype=np.int64)
        episode_indices = np.full(n_frames, ep_idx, dtype=np.int64)
        task_indices = np.full(n_frames, desc_task_idx, dtype=np.int64)
        task_desc_indices = np.full(n_frames, desc_task_idx, dtype=np.int64)
        task_name_indices_arr = np.full(n_frames, task_name_idx, dtype=np.int64)
        validity_indices = np.full(n_frames, validity_idx, dtype=np.int64)

        for i in range(n_frames):
            row = {
                state_feature_name: ep_states[i].tolist(),
                "action": ep_actions[i].tolist(),
                "timestamp": float(timestamps[i]),
                "annotation.human.action.task_description": int(task_desc_indices[i]),
                "task_index": int(task_indices[i]),
                "annotation.human.action.task_name": int(task_name_indices_arr[i]),
                "annotation.human.validity": int(validity_indices[i]),
                "episode_index": int(episode_indices[i]),
                "index": int(global_indices[i]),
                "next.reward": float(rewards[i]),
                "next.done": bool(done_flags[i]),
            }
            parquet_rows.append(row)

        # -- Episode metadata row --------------------------------------------
        ep_duration = n_frames / fps
        ep_meta: Dict[str, Any] = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": 0,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + n_frames,
            "tasks": np.array([desc], dtype=object),
            "length": n_frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        # Per-video columns
        for cam_key in camera_names:
            ep_meta[f"videos/{cam_key}/chunk_index"] = chunk_idx
            ep_meta[f"videos/{cam_key}/file_index"] = file_idx
            ep_meta[f"videos/{cam_key}/from_timestamp"] = 0.0
            ep_meta[f"videos/{cam_key}/to_timestamp"] = float(ep_duration)

        # Per-episode stats (GR00T format with q01/q99)
        ep_stats = build_episode_stats_groot(
            states=ep_states,
            actions=ep_actions,
            timestamps=timestamps,
            done_flags=done_flags,
            rewards=rewards,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            global_indices=global_indices,
            task_indices=task_indices,
            task_desc_indices=task_desc_indices,
            task_name_indices=task_name_indices_arr,
            validity_indices=validity_indices,
            camera_names=camera_names,
            n_camera_frames=[n_frames] * len(camera_names),
            state_feature_name=state_feature_name,
            action_feature_name="action",
        )
        ep_meta.update(ep_stats)
        episode_meta_rows.append(ep_meta)

        # Track scene-graph info for this episode
        episode_sg_info.append({
            "episode_index": ep_idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "begin_scene_graph": ep["begin_sg"],
            "end_scene_graph": ep["end_sg"],
            "task_description": desc,
        })

        total_frames += n_frames
        global_index += n_frames

    # =======================================================================
    # Write data parquet
    # =======================================================================
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(parquet_rows)
    data_parquet_path = os.path.join(data_dir, "file-000.parquet")
    df.to_parquet(data_parquet_path, index=False)
    print(f"[INFO] Wrote {data_parquet_path}  ({len(df)} rows)")

    # =======================================================================
    # Write meta/tasks.parquet
    # =======================================================================
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(all_task_strings)))},
        index=all_task_strings,
    )
    tasks_df.to_parquet(os.path.join(meta_dir, "tasks.parquet"))
    print(f"[INFO] Wrote tasks.parquet ({len(all_task_strings)} entries)")

    # =======================================================================
    # Write meta/episodes/chunk-000/file-000.parquet
    # =======================================================================
    ep_meta_dir = os.path.join(meta_dir, "episodes", "chunk-000")
    os.makedirs(ep_meta_dir, exist_ok=True)
    ep_df = pd.DataFrame(episode_meta_rows)
    ep_df.to_parquet(os.path.join(ep_meta_dir, "file-000.parquet"), index=False)
    print(f"[INFO] Wrote episodes parquet")

    # =======================================================================
    # Write meta/info.json
    # =======================================================================
    from PIL import Image
    sample_img = Image.open(os.path.join(front_rgb_dir, "0.png"))
    img_w, img_h = sample_img.size

    splits = {
        "train": f"0:{n_episodes}",
    }

    def _video_feature(shape: list):
        return {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(fps),
                "video.codec": VIDEO_CODEC.replace("lib", ""),
                "video.pix_fmt": PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    # State / action motor names
    if action_space == "eef":
        state_motor_names = [
            "eef_x", "eef_y", "eef_z",
            "eef_qx", "eef_qy", "eef_qz", "eef_qw",
            "gripper_open",
        ]
        action_motor_names = [
            "delta_eef_x", "delta_eef_y", "delta_eef_z",
            "delta_eef_qx", "delta_eef_qy", "delta_eef_qz", "delta_eef_qw",
            "gripper_open",
        ]
    else:
        state_motor_names = [f"joint_{i}" for i in range(7)] + ["gripper_open"]
        action_motor_names = [f"joint_vel_{i}" for i in range(7)] + ["gripper_open"]

    info = {
        "codebase_version": "v3.0",
        "robot_type": "rlbench_franka",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_task_strings),
        "chunks_size": 1000,
        "fps": fps,
        "splits": splits,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            state_feature_name: {
                "dtype": "float32",
                "shape": [8],
                "names": state_motor_names,
                "fps": fps,
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": action_motor_names,
                "fps": fps,
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1],
                "fps": fps,
            },
            "annotation.human.action.task_description": {
                "dtype": "int64",
                "shape": [1],
                "fps": fps,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "fps": fps,
            },
            "annotation.human.action.task_name": {
                "dtype": "int64",
                "shape": [1],
                "fps": fps,
            },
            "annotation.human.validity": {
                "dtype": "int64",
                "shape": [1],
                "fps": fps,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "fps": fps,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "fps": fps,
            },
            "next.reward": {
                "dtype": "float64",
                "shape": [1],
                "fps": fps,
            },
            "next.done": {
                "dtype": "bool",
                "shape": [1],
                "fps": fps,
            },
            # Cameras (RGB only -- no mask/depth)
            "observation.images.front_rgb": _video_feature([img_h, img_w, 3]),
            "observation.images.wrist_rgb": _video_feature([img_h, img_w, 3]),
        },
        # Extra RLBench-specific metadata
        "object_color_map": object_color_map,
        "scene_graph_objects": objects_meta,
        "episodes_scene_graph": episode_sg_info,
    }

    info_path = os.path.join(meta_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"[INFO] Wrote {info_path}")

    # =======================================================================
    # Write meta/stats.json  (global stats -- GR00T format)
    # =======================================================================
    # Rebuild global arrays from parquet data for exact consistency
    all_state_arr = np.stack([r[state_feature_name] for r in parquet_rows], axis=0).astype(np.float64)
    all_action_arr = np.stack([r["action"] for r in parquet_rows], axis=0).astype(np.float64)
    all_timestamps = np.array([r["timestamp"] for r in parquet_rows], dtype=np.float64)
    all_task_idx = np.array([r["task_index"] for r in parquet_rows], dtype=np.float64)
    all_ep_idx = np.array([r["episode_index"] for r in parquet_rows], dtype=np.float64)
    all_g_idx = np.array([r["index"] for r in parquet_rows], dtype=np.float64)
    all_done = np.array([r["next.done"] for r in parquet_rows], dtype=np.float64)
    all_reward = np.array([r["next.reward"] for r in parquet_rows], dtype=np.float64)
    all_desc_idx = np.array([r["annotation.human.action.task_description"] for r in parquet_rows], dtype=np.float64)
    all_name_idx = np.array([r["annotation.human.action.task_name"] for r in parquet_rows], dtype=np.float64)
    all_valid_idx = np.array([r["annotation.human.validity"] for r in parquet_rows], dtype=np.float64)

    stats: Dict[str, Any] = {}
    stats_map = {
        state_feature_name: all_state_arr,
        "action": all_action_arr,
        "timestamp": all_timestamps.reshape(-1, 1),
        "annotation.human.action.task_description": all_desc_idx.reshape(-1, 1),
        "task_index": all_task_idx.reshape(-1, 1),
        "annotation.human.action.task_name": all_name_idx.reshape(-1, 1),
        "annotation.human.validity": all_valid_idx.reshape(-1, 1),
        "episode_index": all_ep_idx.reshape(-1, 1),
        "index": all_g_idx.reshape(-1, 1),
        "next.reward": all_reward.reshape(-1, 1),
    }
    for feat_name, arr in stats_map.items():
        stats[feat_name] = compute_feature_stats_groot(arr)

    # Bool feature
    stats["next.done"] = compute_bool_stats_groot(np.array([r["next.done"] for r in parquet_rows]))

    # Image placeholder stats
    for cam_key in camera_names:
        stats[cam_key] = compute_image_stats_placeholder(total_frames)

    stats_path = os.path.join(meta_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[INFO] Wrote {stats_path}")

    print(f"\n[DONE] Dataset '{dataset_name}' (action_space={action_space}) written to {out_dir}")
    print(f"       {n_episodes} episodes, {total_frames} total frames, {fps} fps")
    print(f"       Tasks: {all_task_strings}")
    return out_dir


def _concat_videos_ffmpeg(
    input_videos: List[str],
    output_path: str,
    *,
    fps: int,
    codec: str = VIDEO_CODEC,
    pix_fmt: str = PIX_FMT,
) -> str:
    """Concatenate multiple MP4 files into one."""
    import subprocess
    import tempfile

    if not input_videos:
        raise ValueError("No input videos provided for concatenation")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = f.name
        for p in input_videos:
            f.write(f"file '{os.path.abspath(p)}'\n")

    try:
        cmd_copy = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-an",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            subprocess.run(cmd_copy, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError:
            cmd_reencode = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-an",
                "-c:v",
                codec,
                "-pix_fmt",
                pix_fmt,
                "-r",
                str(fps),
                "-movflags",
                "+faststart",
                output_path,
            ]
            subprocess.run(cmd_reencode, check=True, capture_output=True)
            return output_path
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


# ===========================================================================
# Merge all per-variation datasets into one
# ===========================================================================

def merge_all_variations(
    task_name: str,
    input_root: str,
    output_root: str | None = None,
    output_name: str | None = None,
):
    """Merge all ``<task_name>_variation*`` datasets under *input_root*
    into a single combined dataset.

    Output videos are stored as **a single MP4 per camera key**:
        ``videos/<video_key>/chunk-000/file-000.mp4``

    Each episode row in ``meta/episodes/...`` points into that video file via
    ``from_timestamp`` / ``to_timestamp``.

    Parameters
    ----------
    task_name : str
        RLBench task name, e.g. ``"put_rubbish_in_bin"``.
    input_root : str
        Directory containing the per-variation datasets
        (``<task_name>_variation0/``, ``<task_name>_variation1/``, …).
    output_root : str or None
        Where to write the merged dataset.  Defaults to *input_root*.
    output_name : str or None
        Name for the output dataset directory.
        Defaults to ``<task_name>_all``.

    Returns
    -------
    str
        Path to the merged dataset directory.
    """
    if output_root is None:
        output_root = input_root

    def _concat_videos_ffmpeg(
        input_videos: List[str],
        output_path: str,
        *,
        fps: int,
        codec: str = VIDEO_CODEC,
        pix_fmt: str = PIX_FMT,
    ) -> str:
        """Concatenate multiple MP4 files into one.

        Tries a fast remux (`-c copy`) first. If that fails, falls back to
        re-encoding with the requested codec/pix-fmt.
        """
        import subprocess
        import tempfile

        if not input_videos:
            raise ValueError("No input videos provided for concatenation")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build concat-demuxer list file.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            list_path = f.name
            for p in input_videos:
                f.write(f"file '{os.path.abspath(p)}'\n")

        try:
            # 1) Preferred: remux (no re-encode)
            cmd_copy = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-an",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                output_path,
            ]
            try:
                subprocess.run(cmd_copy, check=True, capture_output=True)
                return output_path
            except subprocess.CalledProcessError:
                # 2) Fallback: re-encode to enforce consistency
                cmd_reencode = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_path,
                    "-an",
                    "-c:v",
                    codec,
                    "-pix_fmt",
                    pix_fmt,
                    "-r",
                    str(fps),
                    "-movflags",
                    "+faststart",
                    output_path,
                ]
                subprocess.run(cmd_reencode, check=True, capture_output=True)
                return output_path
        finally:
            try:
                os.unlink(list_path)
            except OSError:
                pass

    # Discover variation directories
    pattern = re.compile(rf"^{re.escape(task_name)}_variation(\d+)$")
    variations = [
        d for d in os.listdir(input_root)
        if pattern.match(d) and os.path.isdir(os.path.join(input_root, d))
    ]
    variations.sort(key=lambda x: int(re.search(r"(\d+)$", x).group(1)))

    if not variations:
        print(f"[WARN] merge_all_variations: no variations found for '{task_name}' in {input_root}")
        return None

    dataset_name = output_name or f"{task_name}_all"
    out_dir = os.path.join(output_root, dataset_name)
    print(f"\n{'='*60}")
    print(f"[MERGE] Merging {len(variations)} variation(s) -> {out_dir}")
    print(f"{'='*60}")
    for v in variations:
        print(f"       - {v}")

    # -- Collect data from each per-variation dataset ------------------------
    all_data_rows: List[pd.DataFrame] = []
    all_episode_meta: List[dict] = []
    all_task_strings: List[str] = []
    all_task_string_set: set = set()

    ref_info = _load_json(os.path.join(input_root, variations[0], "meta", "info.json"))
    fps = ref_info["fps"]
    features = ref_info["features"]
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    # For merged videos: one file per camera; collect per-episode inputs in order.
    cam_to_video_inputs: Dict[str, List[str]] = {k: [] for k in camera_keys}

    global_episode_idx = 0
    global_frame_idx = 0
    total_frames = 0
    variation_summaries: List[dict] = []

    for var_name in variations:
        ds_dir = os.path.join(input_root, var_name)
        info = _load_json(os.path.join(ds_dir, "meta", "info.json"))
        if int(info.get("fps", fps)) != int(fps):
            raise ValueError(
                f"All variations must use the same fps. Expected {fps}, got {info.get('fps')} in {ds_dir}"
            )
        tasks_df = pd.read_parquet(os.path.join(ds_dir, "meta", "tasks.parquet"))
        episodes_df = pd.read_parquet(
            os.path.join(ds_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
        )
        data_df = pd.read_parquet(
            os.path.join(ds_dir, "data", "chunk-000", "file-000.parquet")
        )

        n_episodes = info["total_episodes"]
        n_frames = info["total_frames"]

        # Collect unique task strings
        for task_str in tasks_df.index.tolist():
            if task_str not in all_task_string_set:
                all_task_strings.append(task_str)
                all_task_string_set.add(task_str)

        ep_offset = global_episode_idx
        frame_offset = global_frame_idx

        # Re-index data
        df = data_df.copy()
        df["episode_index"] = df["episode_index"] + ep_offset
        df["index"] = df["index"] + frame_offset
        all_data_rows.append(df)

        # Re-index episodes and collect per-episode video files for concat
        episodes_df = episodes_df.sort_values("episode_index")
        for _, ep_row in episodes_df.iterrows():
            old_ep_idx = int(ep_row["episode_index"])
            new_ep_idx = old_ep_idx + ep_offset
            chunk_idx = 0

            length = int(ep_row["length"])
            new_from = (
                global_frame_idx
                + int(ep_row["dataset_from_index"])
                - int(episodes_df.iloc[0]["dataset_from_index"])
            )
            new_to = new_from + length

            # For the single merged video file, every episode points to file_index=0
            # and uses timestamps to select its segment.
            video_from_ts = new_from / fps
            video_to_ts = new_to / fps

            ep_meta: Dict[str, Any] = {
                "episode_index": new_ep_idx,
                "data/chunk_index": chunk_idx,
                "data/file_index": 0,
                "dataset_from_index": new_from,
                "dataset_to_index": new_to,
                "length": length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            if "tasks" in ep_row.index:
                # Normalize to LeRobot-style: a single-element list/array of task strings.
                # Older exports stored extra strings like task-name/validity; keep only the first.
                tasks_val = ep_row["tasks"]
                first_task = None
                try:
                    if tasks_val is not None and len(tasks_val) > 0:
                        first_task = tasks_val[0]
                except Exception:
                    first_task = None
                if first_task is not None:
                    ep_meta["tasks"] = np.array([str(first_task)], dtype=object)

            for cam_key in camera_keys:
                old_chunk = int(ep_row.get(f"videos/{cam_key}/chunk_index", 0))
                old_file = int(ep_row.get(f"videos/{cam_key}/file_index", old_ep_idx))
                ep_meta[f"videos/{cam_key}/chunk_index"] = chunk_idx
                ep_meta[f"videos/{cam_key}/file_index"] = 0
                ep_meta[f"videos/{cam_key}/from_timestamp"] = float(video_from_ts)
                ep_meta[f"videos/{cam_key}/to_timestamp"] = float(video_to_ts)

                src_video = os.path.join(
                    ds_dir,
                    "videos",
                    cam_key,
                    f"chunk-{old_chunk:03d}",
                    f"file-{old_file:03d}.mp4",
                )
                if not os.path.exists(src_video):
                    raise FileNotFoundError(f"Missing expected source video: {src_video}")
                cam_to_video_inputs[cam_key].append(src_video)

            # Copy per-episode stats
            for col in ep_row.index:
                if col.startswith("stats/"):
                    ep_meta[col] = ep_row[col]

            all_episode_meta.append(ep_meta)

        variation_summaries.append({
            "name": var_name, "episodes": n_episodes, "frames": n_frames,
            "ep_offset": ep_offset, "frame_offset": frame_offset,
        })
        global_episode_idx += n_episodes
        global_frame_idx += n_frames
        total_frames += n_frames

    total_episodes = global_episode_idx
    print(f"[MERGE] Total: {total_episodes} episodes, {total_frames} frames")

    # -- Unified task table --------------------------------------------------
    task_str_to_idx = {s: i for i, s in enumerate(all_task_strings)}

    # -- Remap task indices in data ------------------------------------------
    merged_data = pd.concat(all_data_rows, ignore_index=True)
    for vi, var_name in enumerate(variations):
        ds_dir = os.path.join(input_root, var_name)
        old_tasks_df = pd.read_parquet(os.path.join(ds_dir, "meta", "tasks.parquet"))
        old_idx_to_str = {
            int(row["task_index"]): ts for ts, row in old_tasks_df.iterrows()
        }
        vs = variation_summaries[vi]
        frame_start, frame_end = vs["frame_offset"], vs["frame_offset"] + vs["frames"]
        mask = (merged_data["index"] >= frame_start) & (merged_data["index"] < frame_end)
        for col in [
            "annotation.human.action.task_description", "task_index",
            "annotation.human.action.task_name", "annotation.human.validity",
        ]:
            if col in merged_data.columns:
                old_vals = merged_data.loc[mask, col].values
                merged_data.loc[mask, col] = np.array([
                    task_str_to_idx.get(old_idx_to_str.get(int(v), ""), int(v))
                    for v in old_vals
                ])

    merged_data["index"] = np.arange(len(merged_data), dtype=np.int64)

    # -- Build single video per camera --------------------------------------
    print(f"[MERGE] Building single video per camera (file-000.mp4) ...")
    for cam_key in camera_keys:
        out_video = os.path.join(
            out_dir,
            "videos",
            cam_key,
            "chunk-000",
            "file-000.mp4",
        )
        _concat_videos_ffmpeg(cam_to_video_inputs[cam_key], out_video, fps=fps)

    # -- Write data parquet --------------------------------------------------
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    merged_data.to_parquet(os.path.join(data_dir, "file-000.parquet"), index=False)
    print(f"[MERGE] Wrote data parquet ({len(merged_data)} rows)")

    # -- Write meta/tasks.parquet --------------------------------------------
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    pd.DataFrame(
        {"task_index": list(range(len(all_task_strings)))}, index=all_task_strings
    ).to_parquet(os.path.join(meta_dir, "tasks.parquet"))

    # -- Write meta/episodes parquet -----------------------------------------
    ep_meta_dir = os.path.join(meta_dir, "episodes", "chunk-000")
    os.makedirs(ep_meta_dir, exist_ok=True)
    pd.DataFrame(all_episode_meta).to_parquet(
        os.path.join(ep_meta_dir, "file-000.parquet"), index=False
    )

    # -- Write meta/info.json ------------------------------------------------
    info_out = dict(ref_info)
    info_out["total_episodes"] = total_episodes
    info_out["total_frames"] = total_frames
    info_out["total_tasks"] = len(all_task_strings)
    info_out["splits"] = {"train": f"0:{total_episodes}"}
    for key in ["object_color_map", "scene_graph_objects", "episodes_scene_graph"]:
        info_out.pop(key, None)
    info_out["merged_from"] = [v["name"] for v in variation_summaries]
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info_out, f, indent=4)

    # -- Write meta/stats.json -----------------------------------------------
    stats: Dict[str, Any] = {}
    for feat_name, feat_spec in (ref_info.get("features") or {}).items():
        dtype = (feat_spec or {}).get("dtype")
        if dtype == "video":
            stats[feat_name] = compute_image_stats_placeholder(total_frames)
            continue

        if feat_name not in merged_data.columns:
            continue

        if dtype == "bool":
            stats[feat_name] = compute_bool_stats_groot(merged_data[feat_name].values)
            continue

        shape = (feat_spec or {}).get("shape", [1])
        is_vector = False
        try:
            if isinstance(shape, list) and len(shape) > 0 and int(shape[0]) > 1:
                is_vector = True
        except Exception:
            is_vector = False

        if is_vector:
            arr = np.stack(merged_data[feat_name].values).astype(np.float64)
        else:
            arr = merged_data[feat_name].values.astype(np.float64).reshape(-1, 1)
        stats[feat_name] = compute_feature_stats_groot(arr)

    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[DONE] Merged dataset '{dataset_name}' written to {out_dir}")
    print(f"       {total_episodes} episodes, {total_frames} frames from {len(variations)} variations")
    return out_dir


def merge_all_tasks_datasets(
    input_root: str,
    action_space: str,
    output_root: str | None = None,
    output_name: str | None = None,
):
    """Merge all task datasets for one action space into one dataset.

    Expects task datasets already exported under *input_root* with names like
    ``<task>_eef`` or ``<task>_joint``.
    """
    if output_root is None:
        output_root = input_root

    action_space = (action_space or "").strip().lower()
    if action_space not in {"eef", "joint"}:
        raise ValueError(f"Unknown action_space: {action_space}")

    dataset_name = output_name or f"all_task_{action_space}"

    candidates = [
        d for d in os.listdir(input_root)
        if d.endswith(f"_{action_space}")
        and d != dataset_name
        and os.path.isdir(os.path.join(input_root, d))
        and os.path.exists(os.path.join(input_root, d, "meta", "info.json"))
        and os.path.exists(os.path.join(input_root, d, "meta", "tasks.parquet"))
        and os.path.exists(os.path.join(input_root, d, "meta", "episodes", "chunk-000", "file-000.parquet"))
        and os.path.exists(os.path.join(input_root, d, "data", "chunk-000", "file-000.parquet"))
    ]
    candidates.sort()

    if not candidates:
        print(f"[WARN] No task datasets found to merge for action_space='{action_space}' under: {input_root}")
        return None

    out_dir = os.path.join(output_root, dataset_name)
    print(f"\n{'='*60}")
    print(f"[MERGE] Merging {len(candidates)} task dataset(s) -> {out_dir}")
    print(f"{'='*60}")
    for d in candidates:
        print(f"       - {d}")

    ref_info = _load_json(os.path.join(input_root, candidates[0], "meta", "info.json"))
    fps = ref_info["fps"]
    features = ref_info["features"]
    camera_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    all_data_rows: List[pd.DataFrame] = []
    all_episode_meta: List[dict] = []
    all_task_strings: List[str] = []
    all_task_string_set: set = set()
    cam_to_video_inputs: Dict[str, List[str]] = {k: [] for k in camera_keys}

    global_episode_idx = 0
    global_frame_idx = 0
    total_frames = 0
    dataset_summaries: List[dict] = []

    for ds_name in candidates:
        ds_dir = os.path.join(input_root, ds_name)
        info = _load_json(os.path.join(ds_dir, "meta", "info.json"))
        if int(info.get("fps", fps)) != int(fps):
            raise ValueError(
                f"All datasets must use the same fps. Expected {fps}, got {info.get('fps')} in {ds_dir}"
            )

        tasks_df = pd.read_parquet(os.path.join(ds_dir, "meta", "tasks.parquet"))
        episodes_df = pd.read_parquet(
            os.path.join(ds_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
        )
        data_df = pd.read_parquet(
            os.path.join(ds_dir, "data", "chunk-000", "file-000.parquet")
        )

        n_episodes = int(info.get("total_episodes", len(episodes_df)))
        n_frames = int(info.get("total_frames", len(data_df)))

        for task_str in tasks_df.index.tolist():
            if task_str not in all_task_string_set:
                all_task_strings.append(task_str)
                all_task_string_set.add(task_str)

        ep_offset = global_episode_idx
        frame_offset = global_frame_idx

        df = data_df.copy()
        df["episode_index"] = df["episode_index"] + ep_offset
        df["index"] = df["index"] + frame_offset
        all_data_rows.append(df)

        episodes_df = episodes_df.sort_values("episode_index")
        base_from = int(episodes_df.iloc[0]["dataset_from_index"]) if len(episodes_df) > 0 else 0

        for _, ep_row in episodes_df.iterrows():
            old_ep_idx = int(ep_row["episode_index"])
            new_ep_idx = old_ep_idx + ep_offset
            chunk_idx = 0

            length = int(ep_row["length"])
            new_from = global_frame_idx + int(ep_row["dataset_from_index"]) - base_from
            new_to = new_from + length
            video_from_ts = new_from / fps
            video_to_ts = new_to / fps

            ep_meta: Dict[str, Any] = {
                "episode_index": new_ep_idx,
                "data/chunk_index": chunk_idx,
                "data/file_index": 0,
                "dataset_from_index": new_from,
                "dataset_to_index": new_to,
                "length": length,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
            if "tasks" in ep_row.index:
                tasks_val = ep_row["tasks"]
                first_task = None
                try:
                    if tasks_val is not None and len(tasks_val) > 0:
                        first_task = tasks_val[0]
                except Exception:
                    first_task = None
                if first_task is not None:
                    ep_meta["tasks"] = np.array([str(first_task)], dtype=object)

            for cam_key in camera_keys:
                old_chunk = int(ep_row.get(f"videos/{cam_key}/chunk_index", 0))
                old_file = int(ep_row.get(f"videos/{cam_key}/file_index", 0))
                ep_meta[f"videos/{cam_key}/chunk_index"] = chunk_idx
                ep_meta[f"videos/{cam_key}/file_index"] = 0
                ep_meta[f"videos/{cam_key}/from_timestamp"] = float(video_from_ts)
                ep_meta[f"videos/{cam_key}/to_timestamp"] = float(video_to_ts)

                src_video = os.path.join(
                    ds_dir,
                    "videos",
                    cam_key,
                    f"chunk-{old_chunk:03d}",
                    f"file-{old_file:03d}.mp4",
                )
                if not os.path.exists(src_video):
                    raise FileNotFoundError(f"Missing expected source video: {src_video}")
                cam_to_video_inputs[cam_key].append(src_video)

            for col in ep_row.index:
                if col.startswith("stats/"):
                    ep_meta[col] = ep_row[col]

            all_episode_meta.append(ep_meta)

        old_idx_to_str = {
            int(row["task_index"]): task_str for task_str, row in tasks_df.iterrows()
        }
        dataset_summaries.append({
            "name": ds_name,
            "episodes": n_episodes,
            "frames": n_frames,
            "ep_offset": ep_offset,
            "frame_offset": frame_offset,
            "old_idx_to_str": old_idx_to_str,
        })
        global_episode_idx += n_episodes
        global_frame_idx += n_frames
        total_frames += n_frames

    total_episodes = global_episode_idx
    print(f"[MERGE] Total: {total_episodes} episodes, {total_frames} frames")

    task_str_to_idx = {s: i for i, s in enumerate(all_task_strings)}

    merged_data = pd.concat(all_data_rows, ignore_index=True)
    for ds in dataset_summaries:
        frame_start = ds["frame_offset"]
        frame_end = ds["frame_offset"] + ds["frames"]
        mask = (merged_data["index"] >= frame_start) & (merged_data["index"] < frame_end)
        old_idx_to_str = ds["old_idx_to_str"]
        for col in [
            "annotation.human.action.task_description", "task_index",
            "annotation.human.action.task_name", "annotation.human.validity",
        ]:
            if col in merged_data.columns:
                old_vals = merged_data.loc[mask, col].values
                merged_data.loc[mask, col] = np.array([
                    task_str_to_idx.get(old_idx_to_str.get(int(v), ""), int(v))
                    for v in old_vals
                ])

    merged_data["index"] = np.arange(len(merged_data), dtype=np.int64)

    print("[MERGE] Building single video per camera (file-000.mp4) ...")
    for cam_key in camera_keys:
        out_video = os.path.join(
            out_dir,
            "videos",
            cam_key,
            "chunk-000",
            "file-000.mp4",
        )
        _concat_videos_ffmpeg(cam_to_video_inputs[cam_key], out_video, fps=fps)

    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    merged_data.to_parquet(os.path.join(data_dir, "file-000.parquet"), index=False)
    print(f"[MERGE] Wrote data parquet ({len(merged_data)} rows)")

    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    pd.DataFrame(
        {"task_index": list(range(len(all_task_strings)))}, index=all_task_strings
    ).to_parquet(os.path.join(meta_dir, "tasks.parquet"))

    ep_meta_dir = os.path.join(meta_dir, "episodes", "chunk-000")
    os.makedirs(ep_meta_dir, exist_ok=True)
    pd.DataFrame(all_episode_meta).to_parquet(
        os.path.join(ep_meta_dir, "file-000.parquet"), index=False
    )

    info_out = dict(ref_info)
    info_out["total_episodes"] = total_episodes
    info_out["total_frames"] = total_frames
    info_out["total_tasks"] = len(all_task_strings)
    info_out["splits"] = {"train": f"0:{total_episodes}"}
    for key in ["object_color_map", "scene_graph_objects", "episodes_scene_graph"]:
        info_out.pop(key, None)
    info_out["merged_from"] = [d["name"] for d in dataset_summaries]
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info_out, f, indent=4)

    stats: Dict[str, Any] = {}
    for feat_name, feat_spec in (ref_info.get("features") or {}).items():
        dtype = (feat_spec or {}).get("dtype")
        if dtype == "video":
            stats[feat_name] = compute_image_stats_placeholder(total_frames)
            continue

        if feat_name not in merged_data.columns:
            continue

        if dtype == "bool":
            stats[feat_name] = compute_bool_stats_groot(merged_data[feat_name].values)
            continue

        shape = (feat_spec or {}).get("shape", [1])
        is_vector = False
        try:
            if isinstance(shape, list) and len(shape) > 0 and int(shape[0]) > 1:
                is_vector = True
        except Exception:
            is_vector = False

        if is_vector:
            arr = np.stack(merged_data[feat_name].values).astype(np.float64)
        else:
            arr = merged_data[feat_name].values.astype(np.float64).reshape(-1, 1)
        stats[feat_name] = compute_feature_stats_groot(arr)

    with open(os.path.join(meta_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print(f"\n[DONE] Merged dataset '{dataset_name}' written to {out_dir}")
    print(f"       {total_episodes} episodes, {total_frames} frames from {len(candidates)} task datasets")
    return out_dir


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Convert RLBench variations to LeRobot v3 and output merged datasets without per-variation folders. "
            "Produces <task>_eef and/or <task>_joint under --output_root."
        )
    )
    p.add_argument("--task_name", type=str, default=None,
                   help="RLBench task name, e.g. stack_cups (required for conversion mode)")
    p.add_argument(
        "--variation",
        type=int,
        help="Variation number. If omitted, processes ALL variations.",
    )
    p.add_argument("--rlbench_root", type=str, default="datasets/rlbench",
                   help="Root directory containing RLBench datasets")
    p.add_argument("--output_root", type=str, default="datasets/lerobot",
                   help="Root directory for output LeRobot datasets")
    p.add_argument("--fps", type=int, default=20,
                   help="Frames per second for the output dataset")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode index inside the RLBench variation directory")
    p.add_argument("--use_context_prompt", action="store_true",
                   help="Use ConceptGraphs-style context in task descriptions")
    p.add_argument(
        "--action_space",
        type=str,
        default="both",
        choices=["eef", "joint", "both"],
        help="Which dataset(s) to create (default: both).",
    )
    p.add_argument(
        "--strict_variations",
        action="store_true",
        help="Fail immediately if any variation conversion fails. Default behavior is to skip failed variations.",
    )
    p.add_argument(
        "--merge_all_tasks_root",
        type=str,
        default=None,
        help=(
            "Merge mode: root directory containing per-task datasets (e.g. <task>_eef/<task>_joint). "
            "When set, converter skips per-task conversion and creates all_task_<action_space> dataset(s)."
        ),
    )
    p.add_argument(
        "--merged_output_name",
        type=str,
        default=None,
        help="Optional output dataset name in merge mode. If omitted, uses all_task_<action_space>.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Merge-only mode across tasks: builds all_task_eef / all_task_joint.
    if args.merge_all_tasks_root:
        merge_root = args.merge_all_tasks_root
        if not os.path.isdir(merge_root):
            raise SystemExit(f"Merge root directory not found: {merge_root}")

        action_spaces = ["eef", "joint"] if args.action_space == "both" else [args.action_space]
        merge_failed = False
        print(f"[INFO] Merge-all-tasks root: {merge_root}")
        print(f"[INFO] Action spaces to merge: {action_spaces}")

        for action_space in action_spaces:
            out_name = args.merged_output_name or f"all_task_{action_space}"
            try:
                merged_path = merge_all_tasks_datasets(
                    input_root=merge_root,
                    action_space=action_space,
                    output_root=args.output_root,
                    output_name=out_name,
                )
                if merged_path is None:
                    merge_failed = True
            except Exception as e:
                merge_failed = True
                print(f"[WARN] Failed to merge action_space='{action_space}': {e}")

        raise SystemExit(1 if merge_failed else 0)

    if not args.task_name:
        raise SystemExit("--task_name is required unless --merge_all_tasks_root is provided.")

    task_dir = os.path.join(args.rlbench_root, args.task_name)
    if not os.path.isdir(task_dir):
        raise SystemExit(f"Task directory not found: {task_dir}")

    if args.variation is not None:
        variation_nums = [int(args.variation)]
    else:
        variations = [d for d in os.listdir(task_dir) if d.startswith("variation")]
        variations = sorted(variations, key=lambda x: int(x.replace("variation", "")))
        variation_nums = [int(v.replace("variation", "")) for v in variations]

    if not variation_nums:
        raise SystemExit(f"No variations found under: {task_dir}")

    action_spaces = ["eef", "joint"] if args.action_space == "both" else [args.action_space]

    os.makedirs(args.output_root, exist_ok=True)

    print(f"[INFO] Variations to process: {variation_nums}")
    print(f"[INFO] Action spaces to export: {action_spaces}")
    print(f"[INFO] Strict variation mode: {args.strict_variations}")

    overall_failed = False

    for action_space in action_spaces:
        final_name = f"{args.task_name}_{action_space}"
        print(f"\n{'='*60}")
        print(f"[EXPORT] Building merged dataset: {final_name}")
        print(f"{'='*60}")

        with tempfile.TemporaryDirectory(
            prefix=f".{args.task_name}_{action_space}_tmp_",
            dir=args.output_root,
        ) as tmp_root:
            successful_vars = []
            failed_vars = []
            for var_num in variation_nums:
                try:
                    convert(
                        task_name=args.task_name,
                        variation=var_num,
                        rlbench_root=args.rlbench_root,
                        output_root=tmp_root,
                        fps=args.fps,
                        episode_index_in_rlbench=args.episode,
                        use_context_prompt=args.use_context_prompt,
                        action_space=action_space,
                    )
                    successful_vars.append(var_num)
                except Exception as e:
                    failed_vars.append(var_num)
                    print(f"[WARN] Skipping variation {var_num} ({action_space}) due to error: {e}")
                    if args.strict_variations:
                        raise

            if not successful_vars:
                print(
                    f"[WARN] No successful variations for task '{args.task_name}' "
                    f"in action_space='{action_space}'. Skipping merge."
                )
                overall_failed = True
                continue

            merge_all_variations(
                task_name=args.task_name,
                input_root=tmp_root,
                output_root=args.output_root,
                output_name=final_name,
            )
            print(
                f"[INFO] action_space='{action_space}' summary: "
                f"success={len(successful_vars)}, failed={len(failed_vars)}"
            )
            if failed_vars:
                print(f"[INFO] Failed variations ({action_space}): {failed_vars}")

    if overall_failed:
        raise SystemExit(1)