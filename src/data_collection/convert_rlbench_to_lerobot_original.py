#!/usr/bin/env python3
"""
Convert an RLBench dataset (one variation of one task) into LeRobot v3 format,
splitting the trajectory into multiple episodes based on scene-graph transitions.

Usage
-----
    python src/convert_rlbench_to_lerobot.py \
        --task_name stack_cups \
        --variation 1 \
        --rlbench_root datasets/rlbench \
        --output_root datasets/lerobot \
        --fps 10

The script reads:
    datasets/rlbench/<task_name>/variation<num>/
        episodes/episode0/front_rgb/*.png
        episodes/episode0/wrist_rgb/*.png
        episodes/episode0/low_dim_obs.pkl
        episode_mask.mp4
        <task_name>_scene_graph.json
        object_color_map.json

And writes a LeRobot-v3 dataset to:
    datasets/lerobot/<task_name>_variation<num>/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to sys.path to allow imports from src package
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.rlbench_utils import (
    build_episode_stats,
    compute_eef_actions,
    compute_episode_boundaries,
    compute_feature_stats,
    compute_image_stats_placeholder,
    compute_bool_stats,
    compute_obs_offset,
    extract_eef_state,
    find_transitions,
    images_to_video,
    load_rlbench_demo,
    load_scene_graph,
    obs_index_for_frame,
    slice_video,
)

# ── Codec/pix-fmt used for output videos ──────────────────────────────────
VIDEO_CODEC = "libx264"
PIX_FMT = "yuv420p"


# ═══════════════════════════════════════════════════════════════════════════
# Main conversion logic
# ═══════════════════════════════════════════════════════════════════════════

def convert(
    task_name: str,
    variation: int,
    rlbench_root: str,
    output_root: str,
    fps: int = 10,
    episode_index_in_rlbench: int = 0,
):
    # ── Paths ─────────────────────────────────────────────────────────────
    var_dir = os.path.join(rlbench_root, task_name, f"variation{variation}")
    ep_dir = os.path.join(var_dir, "episodes", f"episode{episode_index_in_rlbench}")
    front_rgb_dir = os.path.join(ep_dir, "front_rgb")
    wrist_rgb_dir = os.path.join(ep_dir, "wrist_rgb")
    pkl_path = os.path.join(ep_dir, "low_dim_obs.pkl")
    mask_video_path = os.path.join(var_dir, "episode_mask.mp4")
    sg_path = os.path.join(var_dir, f"{task_name}_scene_graph.json")
    color_map_path = os.path.join(var_dir, "object_color_map.json")

    dataset_name = f"{task_name}_variation{variation}"
    out_dir = os.path.join(output_root, dataset_name)

    print(f"[INFO] Converting {var_dir} → {out_dir}")

    # ── Load sources ──────────────────────────────────────────────────────
    observations = load_rlbench_demo(pkl_path)
    scene_graph = load_scene_graph(sg_path)
    with open(color_map_path) as f:
        object_color_map = json.load(f)

    transitions = find_transitions(scene_graph)
    episodes = compute_episode_boundaries(scene_graph, transitions)
    n_episodes = len(episodes)
    print(f"[INFO] Found {len(transitions)} transitions → {n_episodes} episodes")

    # Number of images / obs for alignment
    n_images = len(os.listdir(front_rgb_dir))
    n_obs = len(observations)
    obs_offset = compute_obs_offset(n_images, n_obs)
    print(f"[INFO] Images: {n_images}, Observations: {n_obs}, obs_offset: {obs_offset}")

    # Pre-compute ALL EEF states and actions for the full observation list
    all_states = np.stack([extract_eef_state(o) for o in observations], axis=0)  # (n_obs, 8)
    all_actions = compute_eef_actions(observations)  # (n_obs, 8)

    # Scene-graph first frame (mask video frame 0 = this image index)
    sg_first_frame = scene_graph["frames"][0]["frame_id"]

    # ── Determine task descriptions ───────────────────────────────────────
    task_descriptions: List[str] = []
    for ep in episodes:
        begin_str = json.dumps(ep["begin_sg"])
        end_str = json.dumps(ep["end_sg"])
        desc = f"{task_name}: {begin_str} → {end_str}"
        task_descriptions.append(desc)

    # ── Per-episode data collection ───────────────────────────────────────
    # We accumulate rows and write a single parquet later.
    parquet_rows: List[dict] = []
    episode_meta_rows: List[dict] = []
    global_index = 0  # monotonically increasing across all episodes

    # Episode-level scene-graph info stored in info.json later
    episode_sg_info: List[dict] = []

    # Stats accumulators (across all episodes)
    all_ep_states: List[np.ndarray] = []
    all_ep_actions: List[np.ndarray] = []
    all_ep_done: List[np.ndarray] = []
    all_ep_timestamps: List[np.ndarray] = []
    all_ep_frame_idx: List[np.ndarray] = []
    all_ep_episode_idx: List[np.ndarray] = []
    all_ep_index: List[np.ndarray] = []
    all_ep_task_idx: List[np.ndarray] = []

    total_frames = 0
    total_video_frames_front = 0
    total_video_frames_wrist = 0
    total_video_frames_mask = 0

    for ep_idx, ep in enumerate(episodes):
        start_frame = ep["start_frame"]
        end_frame = ep["end_frame"]
        n_frames = end_frame - start_frame + 1
        task_idx = ep_idx  # one task per transition-episode

        print(f"  Episode {ep_idx}: frames {start_frame}–{end_frame} ({n_frames} frames)")

        # ── Map frame range to observation indices ────────────────────────
        obs_start = obs_index_for_frame(start_frame, obs_offset)
        obs_end = obs_index_for_frame(end_frame, obs_offset)
        # Clamp to valid range
        obs_start = max(0, obs_start)
        obs_end = min(n_obs - 1, obs_end)
        n_obs_frames = obs_end - obs_start + 1

        ep_states = all_states[obs_start : obs_end + 1]       # (n_obs_frames, 8)
        ep_actions = all_actions[obs_start : obs_end + 1]      # (n_obs_frames, 8)

        # If obs count < image count for this segment, we pad/repeat last
        if n_obs_frames < n_frames:
            pad = n_frames - n_obs_frames
            ep_states = np.concatenate([ep_states, np.tile(ep_states[-1:], (pad, 1))], axis=0)
            ep_actions = np.concatenate([ep_actions, np.tile(ep_actions[-1:], (pad, 1))], axis=0)
        elif n_obs_frames > n_frames:
            ep_states = ep_states[:n_frames]
            ep_actions = ep_actions[:n_frames]

        # ── Create videos ─────────────────────────────────────────────────
        chunk_idx = 0
        file_idx = ep_idx  # one file per episode

        for cam_name, img_dir in [
            ("observation.images.front_rgb", front_rgb_dir),
            ("observation.images.wrist_rgb", wrist_rgb_dir),
        ]:
            vid_out = os.path.join(
                out_dir, "videos", cam_name,
                f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4",
            )
            print(f"    Creating video {vid_out}")
            images_to_video(img_dir, vid_out, start_frame, end_frame, fps,
                            codec=VIDEO_CODEC, pix_fmt=PIX_FMT)

        # ── Mask video – slice from episode_mask.mp4 ─────────────────────
        mask_vid_out = os.path.join(
            out_dir, "videos", "observation.images.mask",
            f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4",
        )
        print(f"    Slicing mask video → {mask_vid_out}")
        slice_video(
            mask_video_path, mask_vid_out,
            start_frame, end_frame,
            video_first_frame=sg_first_frame,
            fps=fps, codec=VIDEO_CODEC, pix_fmt=PIX_FMT,
        )

        # ── Build parquet rows ────────────────────────────────────────────
        timestamps = np.arange(n_frames, dtype=np.float32) / fps
        frame_indices = np.arange(n_frames, dtype=np.int64)
        done_flags = np.zeros(n_frames, dtype=bool)
        done_flags[-1] = True
        global_indices = np.arange(global_index, global_index + n_frames, dtype=np.int64)
        episode_indices = np.full(n_frames, ep_idx, dtype=np.int64)
        task_indices = np.full(n_frames, task_idx, dtype=np.int64)

        for i in range(n_frames):
            parquet_rows.append({
                "observation.state": ep_states[i].tolist(),
                "action": ep_actions[i].tolist(),
                "episode_index": int(episode_indices[i]),
                "frame_index": int(frame_indices[i]),
                "timestamp": float(timestamps[i]),
                "next.done": bool(done_flags[i]),
                "index": int(global_indices[i]),
                "task_index": int(task_indices[i]),
            })

        # Accumulate for global stats
        all_ep_states.append(ep_states)
        all_ep_actions.append(ep_actions)
        all_ep_done.append(done_flags)
        all_ep_timestamps.append(timestamps)
        all_ep_frame_idx.append(frame_indices)
        all_ep_episode_idx.append(episode_indices)
        all_ep_index.append(global_indices)
        all_ep_task_idx.append(task_indices)

        # ── Episode metadata row ──────────────────────────────────────────
        ep_duration = n_frames / fps
        ep_meta: Dict[str, Any] = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": 0,  # all data in one parquet file
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + n_frames,
            "tasks": np.array(task_descriptions[ep_idx:ep_idx + 1], dtype=object),
            "length": n_frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        # Per-video columns
        for cam in ["observation.images.front_rgb",
                     "observation.images.wrist_rgb",
                     "observation.images.mask"]:
            ep_meta[f"videos/{cam}/chunk_index"] = chunk_idx
            ep_meta[f"videos/{cam}/file_index"] = file_idx
            ep_meta[f"videos/{cam}/from_timestamp"] = 0.0
            ep_meta[f"videos/{cam}/to_timestamp"] = float(ep_duration)

        # Per-episode stats
        ep_stats = build_episode_stats(
            states=ep_states,
            actions=ep_actions,
            episode_indices=episode_indices,
            frame_indices=frame_indices,
            timestamps=timestamps,
            done_flags=done_flags,
            global_indices=global_indices,
            task_indices=task_indices,
            n_front_frames=n_frames,
            n_wrist_frames=n_frames,
            n_mask_frames=n_frames,
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
        })

        total_frames += n_frames
        total_video_frames_front += n_frames
        total_video_frames_wrist += n_frames
        total_video_frames_mask += n_frames
        global_index += n_frames

    # ═════════════════════════════════════════════════════════════════════
    # Write data parquet
    # ═════════════════════════════════════════════════════════════════════
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    os.makedirs(data_dir, exist_ok=True)
    df = pd.DataFrame(parquet_rows)
    data_parquet_path = os.path.join(data_dir, "file-000.parquet")
    df.to_parquet(data_parquet_path, index=False)
    print(f"[INFO] Wrote {data_parquet_path}  ({len(df)} rows)")

    # ═════════════════════════════════════════════════════════════════════
    # Write meta/tasks.parquet
    # ═════════════════════════════════════════════════════════════════════
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    tasks_df = pd.DataFrame({
        "task_index": list(range(len(task_descriptions))),
    }, index=task_descriptions)
    tasks_df.to_parquet(os.path.join(meta_dir, "tasks.parquet"))
    print(f"[INFO] Wrote tasks.parquet")

    # ═════════════════════════════════════════════════════════════════════
    # Write meta/episodes/chunk-000/file-000.parquet
    # ═════════════════════════════════════════════════════════════════════
    ep_meta_dir = os.path.join(meta_dir, "episodes", "chunk-000")
    os.makedirs(ep_meta_dir, exist_ok=True)
    ep_df = pd.DataFrame(episode_meta_rows)
    ep_df.to_parquet(os.path.join(ep_meta_dir, "file-000.parquet"), index=False)
    print(f"[INFO] Wrote episodes parquet")

    # ═════════════════════════════════════════════════════════════════════
    # Write meta/info.json
    # ═════════════════════════════════════════════════════════════════════
    # Determine image shape from first image
    from PIL import Image
    sample_img = Image.open(os.path.join(front_rgb_dir, "0.png"))
    img_w, img_h = sample_img.size

    splits = {}
    for ep_idx in range(n_episodes):
        splits[f"episode_{ep_idx}"] = f"{ep_idx}:{ep_idx + 1}"
    splits["train"] = f"0:{n_episodes}"

    def _video_feature(name: str, shape: list):
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

    info = {
        "codebase_version": "v3.0",
        "robot_type": "rlbench_franka",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_descriptions),
        "chunks_size": 1000,
        "fps": fps,
        "splits": splits,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.front_rgb": _video_feature(
                "front_rgb", [img_h, img_w, 3]
            ),
            "observation.images.wrist_rgb": _video_feature(
                "wrist_rgb", [img_h, img_w, 3]
            ),
            "observation.images.mask": _video_feature(
                "mask", [img_h, img_w, 3]
            ),
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": [
                        "eef_x", "eef_y", "eef_z",
                        "eef_qx", "eef_qy", "eef_qz", "eef_qw",
                        "gripper_open",
                    ]
                },
                "fps": fps,
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": [
                        "eef_x", "eef_y", "eef_z",
                        "eef_qx", "eef_qy", "eef_qz", "eef_qw",
                        "gripper_open",
                    ]
                },
                "fps": fps,
            },
            "episode_index": {
                "dtype": "int64", "shape": [1], "names": None, "fps": fps,
            },
            "frame_index": {
                "dtype": "int64", "shape": [1], "names": None, "fps": fps,
            },
            "timestamp": {
                "dtype": "float32", "shape": [1], "names": None, "fps": fps,
            },
            "next.done": {
                "dtype": "bool", "shape": [1], "names": None, "fps": fps,
            },
            "index": {
                "dtype": "int64", "shape": [1], "names": None, "fps": fps,
            },
            "task_index": {
                "dtype": "int64", "shape": [1], "names": None, "fps": fps,
            },
        },
        # Extra RLBench-specific metadata
        "object_color_map": object_color_map,
        "scene_graph_objects": scene_graph.get("objects", {}),
        "episodes_scene_graph": episode_sg_info,
    }

    info_path = os.path.join(meta_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"[INFO] Wrote {info_path}")

    # ═════════════════════════════════════════════════════════════════════
    # Write meta/stats.json  (global stats over the whole dataset)
    # ═════════════════════════════════════════════════════════════════════
    cat_states = np.concatenate(all_ep_states, axis=0)
    cat_actions = np.concatenate(all_ep_actions, axis=0)
    cat_done = np.concatenate(all_ep_done, axis=0)
    cat_timestamps = np.concatenate(all_ep_timestamps, axis=0)
    cat_frame_idx = np.concatenate(all_ep_frame_idx, axis=0)
    cat_episode_idx = np.concatenate(all_ep_episode_idx, axis=0)
    cat_index = np.concatenate(all_ep_index, axis=0)
    cat_task_idx = np.concatenate(all_ep_task_idx, axis=0)

    stats = {
        "observation.state": compute_feature_stats(cat_states),
        "action": compute_feature_stats(cat_actions),
        "next.done": compute_bool_stats(cat_done),
        "episode_index": compute_feature_stats(cat_episode_idx.reshape(-1, 1).astype(np.float64)),
        "frame_index": compute_feature_stats(cat_frame_idx.reshape(-1, 1).astype(np.float64)),
        "timestamp": compute_feature_stats(cat_timestamps.reshape(-1, 1).astype(np.float64)),
        "index": compute_feature_stats(cat_index.reshape(-1, 1).astype(np.float64)),
        "task_index": compute_feature_stats(cat_task_idx.reshape(-1, 1).astype(np.float64)),
        "observation.images.front_rgb": compute_image_stats_placeholder(total_video_frames_front),
        "observation.images.wrist_rgb": compute_image_stats_placeholder(total_video_frames_wrist),
        "observation.images.mask": compute_image_stats_placeholder(total_video_frames_mask),
    }

    stats_path = os.path.join(meta_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"[INFO] Wrote {stats_path}")

    print(f"\n[DONE] Dataset '{dataset_name}' written to {out_dir}")
    print(f"       {n_episodes} episodes, {total_frames} total frames, {fps} fps")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert an RLBench variation dataset to LeRobot v3 format."
    )
    p.add_argument("--task_name", type=str, required=True,
                   help="RLBench task name, e.g. stack_cups")
    p.add_argument("--variation", type=int, required=True,
                   help="Variation number, e.g. 1")
    p.add_argument("--rlbench_root", type=str, default="datasets/rlbench",
                   help="Root directory containing RLBench datasets")
    p.add_argument("--output_root", type=str, default="datasets/lerobot",
                   help="Root directory for output LeRobot datasets")
    p.add_argument("--fps", type=int, default=10,
                   help="Frames per second for the output dataset")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode index inside the RLBench variation directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        task_name=args.task_name,
        variation=args.variation,
        rlbench_root=args.rlbench_root,
        output_root=args.output_root,
        fps=args.fps,
        episode_index_in_rlbench=args.episode,
    )
