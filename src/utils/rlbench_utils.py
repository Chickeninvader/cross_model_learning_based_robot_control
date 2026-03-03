"""
Utility functions for converting RLBench datasets to LeRobot v3 format.
"""

import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Pickle loader – works without rlbench / pyrep installed
# ---------------------------------------------------------------------------

class _DummyClass:
    """Stand-in for any rlbench / pyrep class during unpickling."""
    pass


class RLBenchUnpickler(pickle.Unpickler):
    """Unpickler that replaces missing rlbench/pyrep classes with dummies."""

    def find_class(self, module: str, name: str):
        if "rlbench" in module or "pyrep" in module:
            # Return a fresh dummy so different classes don't collide
            cls = type(name, (_DummyClass,), {"__module__": module})
            return cls
        return super().find_class(module, name)


def load_rlbench_demo(pkl_path: str) -> list:
    """Load a RLBench Demo pickle and return a list of Observation objects.

    Each observation has (at minimum):
        .gripper_pose        np.ndarray (7,)  – xyz + quaternion
        .gripper_open        float             – 0.0 or 1.0
        .joint_positions     np.ndarray (7,)
        .joint_velocities    np.ndarray (7,)
        .gripper_joint_positions  np.ndarray (2,)
    """
    with open(pkl_path, "rb") as f:
        demo = RLBenchUnpickler(f).load()
    return demo._observations


# ---------------------------------------------------------------------------
# 2. Observation ↔ state / action helpers
# ---------------------------------------------------------------------------

def extract_eef_state(obs) -> np.ndarray:
    """Return EEF state as float32 array of shape (8,).

    [x, y, z, qx, qy, qz, qw, gripper_open]
    """
    pose = np.asarray(obs.gripper_pose, dtype=np.float32)      # (7,)
    grip = np.float32(obs.gripper_open)                         # scalar
    return np.append(pose, grip)


def compute_eef_actions(observations: list) -> np.ndarray:
    """Compute actions for every timestep.

    action[t] = EEF state at t+1  (next-step target).
    action[-1] = EEF state at -1  (repeat last).
    Returns shape (T, 8) float32.
    """
    states = np.stack([extract_eef_state(o) for o in observations], axis=0)
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


# ---------------------------------------------------------------------------
# 3. Scene-graph helpers
# ---------------------------------------------------------------------------

def _rel_key(relationships: list) -> str:
    """Canonical hashable key for a list of relationship dicts."""
    return json.dumps(sorted(json.dumps(r, sort_keys=True) for r in relationships))


def load_scene_graph(sg_path: str) -> dict:
    with open(sg_path) as f:
        return json.load(f)


def find_transitions(scene_graph: dict) -> List[dict]:
    """Return a list of transition dicts found in the scene graph.

    Each dict:
        frame_id   – the first frame_id with the *new* relationship set
        from_rels  – relationships just before the transition
        to_rels    – relationships at frame_id
    """
    frames = scene_graph["frames"]
    transitions: List[dict] = []
    prev_key: Optional[str] = None
    prev_rels: list = []

    for frame in frames:
        rels = frame["relationships"]
        key = _rel_key(rels)
        if prev_key is not None and key != prev_key:
            transitions.append({
                "frame_id": frame["frame_id"],
                "from_rels": prev_rels,
                "to_rels": rels,
            })
        prev_key = key
        prev_rels = rels

    return transitions


def compute_episode_boundaries(
    scene_graph: dict,
    transitions: List[dict],
) -> List[dict]:
    """Split a scene-graph timeline into episodes at each transition.

    Returns a list of episode dicts:
        start_frame  – inclusive first image index
        end_frame    – inclusive last image index
        begin_sg     – relationships at the start of the segment
        end_sg       – goal relationships (the state after the transition)
    """
    frames = scene_graph["frames"]
    first_frame = frames[0]["frame_id"]
    last_frame = frames[-1]["frame_id"]

    # Build boundary frame list  [first, t0, t1, ..., last+1]
    boundaries = [first_frame] + [t["frame_id"] for t in transitions] + [last_frame + 1]

    episodes = []
    for i in range(len(transitions)):
        start = boundaries[i]
        end = boundaries[i + 1] - 1       # inclusive last frame of this segment
        # begin_sg = relationship state of the segment (before the transition)
        begin_sg = transitions[i]["from_rels"]
        # end_sg = relationship state after the transition (the goal)
        end_sg = transitions[i]["to_rels"]
        episodes.append({
            "start_frame": start,
            "end_frame": end,
            "begin_sg": begin_sg,
            "end_sg": end_sg,
        })

    return episodes


# ---------------------------------------------------------------------------
# 4. Video creation helpers
# ---------------------------------------------------------------------------

def images_to_video(
    image_dir: str,
    output_path: str,
    frame_start: int,
    frame_end: int,
    fps: int = 10,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
) -> str:
    """Encode a range of numbered PNGs [frame_start..frame_end] into an MP4.

    Images are expected as ``<image_dir>/<index>.png``.
    Returns the output_path.
    """
    import tempfile, shutil

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_frames = frame_end - frame_start + 1

    # Symlink images into a temp dir with sequential 0-based naming
    # so ffmpeg's -start_number / %d pattern works cleanly.
    tmp_dir = tempfile.mkdtemp(prefix="lerobot_vid_")
    abs_image_dir = os.path.abspath(image_dir)
    try:
        for i, idx in enumerate(range(frame_start, frame_end + 1)):
            src = os.path.join(abs_image_dir, f"{idx}.png")
            dst = os.path.join(tmp_dir, f"{i:06d}.png")
            os.symlink(src, dst)

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "%06d.png"),
            "-frames:v", str(n_frames),
            "-c:v", codec,
            "-pix_fmt", pix_fmt,
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        shutil.rmtree(tmp_dir)

    return output_path


def slice_video(
    input_video: str,
    output_path: str,
    frame_start: int,
    frame_end: int,
    video_first_frame: int,
    fps: int = 10,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
) -> str:
    """Extract frames [frame_start..frame_end] from *input_video* (whose first
    frame corresponds to ``video_first_frame``) and write a new MP4.

    Uses time-based seeking so no re-decode of the full file is necessary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    offset_start = frame_start - video_first_frame
    offset_end = frame_end - video_first_frame
    ss = offset_start / fps
    n_frames = offset_end - offset_start + 1

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{ss:.6f}",
        "-i", input_video,
        "-frames:v", str(n_frames),
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# ---------------------------------------------------------------------------
# 5. Stats computation
# ---------------------------------------------------------------------------

def compute_feature_stats(values: np.ndarray) -> dict:
    """Compute min / max / mean / std / count for a numeric feature.

    ``values`` shape is (N,) or (N, D).
    Returns a dict suitable for stats.json.
    """
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0, ddof=0).tolist(),
        "count": [int(values.shape[0])],
    }


def compute_feature_stats_groot(values: np.ndarray) -> dict:
    """Compute min / max / mean / std / q01 / q99 / count for a numeric feature.

    Same as :func:`compute_feature_stats` but adds ``q01``/``q99`` percentiles
    to match the GR00T / LeRobot v3 episodes-parquet schema.
    """
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0, ddof=0).tolist(),
        "q01": np.percentile(values, 1, axis=0).tolist(),
        "q99": np.percentile(values, 99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def compute_image_stats_placeholder(n_frames: int) -> dict:
    """Return a fixed placeholder image stats dict (values in [0,1])."""
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [n_frames],
    }


def compute_bool_stats(values: np.ndarray) -> dict:
    arr = values.astype(np.float64)
    return {
        "min": [bool(arr.min())],
        "max": [bool(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std(ddof=0))],
        "count": [int(len(arr))],
    }


def compute_bool_stats_groot(values: np.ndarray) -> dict:
    """Bool stats with q01/q99 to match GR00T episodes-parquet schema."""
    arr = values.astype(np.float64)
    return {
        "min": [bool(arr.min())],
        "max": [bool(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std(ddof=0))],
        "q01": [float(np.percentile(arr, 1))],
        "q99": [float(np.percentile(arr, 99))],
        "count": [int(len(arr))],
    }


# ---------------------------------------------------------------------------
# 6. Observation-index alignment
# ---------------------------------------------------------------------------

def compute_obs_offset(n_images: int, n_obs: int) -> int:
    """Return the image-index offset for observation 0.

    Assumes observations align with the *last* n_obs images,
    i.e.  obs[i] ↔ image[i + offset].
    """
    return n_images - n_obs


def obs_index_for_frame(frame_idx: int, obs_offset: int) -> int:
    """Map an image frame index to the corresponding observation index."""
    return frame_idx - obs_offset


# ---------------------------------------------------------------------------
# 7. Per-episode stat helpers for the episodes-parquet
# ---------------------------------------------------------------------------

def _to_nested_list(val):
    """Convert numpy arrays to nested python lists for parquet storage."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return [_to_nested_list(v) for v in val]
    return val


def build_episode_stats(
    states: np.ndarray,
    actions: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    done_flags: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
    n_front_frames: int,
    n_wrist_frames: int,
    n_mask_frames: int,
) -> dict:
    """Return a flat dict of per-episode statistics keyed like the aloha dataset."""
    d: Dict[str, Any] = {}

    feature_map = {
        "observation.state": states,
        "action": actions,
        "episode_index": episode_indices.reshape(-1, 1).astype(np.float64),
        "frame_index": frame_indices.reshape(-1, 1).astype(np.float64),
        "timestamp": timestamps.reshape(-1, 1).astype(np.float64),
        "index": global_indices.reshape(-1, 1).astype(np.float64),
        "task_index": task_indices.reshape(-1, 1).astype(np.float64),
    }

    for feat_name, arr in feature_map.items():
        stats = compute_feature_stats(arr)
        for stat_key in ("min", "max", "mean", "std", "count"):
            d[f"stats/{feat_name}/{stat_key}"] = _to_nested_list(stats[stat_key])

    # done
    done_stats = compute_bool_stats(done_flags)
    for stat_key in ("min", "max", "mean", "std", "count"):
        d[f"stats/next.done/{stat_key}"] = _to_nested_list(done_stats[stat_key])

    # image stats (placeholder)
    for cam_name, n_fr in [
        ("observation.images.front_rgb", n_front_frames),
        ("observation.images.wrist_rgb", n_wrist_frames),
        ("observation.images.mask", n_mask_frames),
    ]:
        img_stats = compute_image_stats_placeholder(n_fr)
        for stat_key in ("min", "max", "mean", "std", "count"):
            d[f"stats/{cam_name}/{stat_key}"] = _to_nested_list(img_stats[stat_key])

    return d


def build_episode_stats_groot(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    done_flags: np.ndarray,
    rewards: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
    task_desc_indices: np.ndarray,
    task_name_indices: np.ndarray,
    validity_indices: np.ndarray,
    camera_names: List[str],
    n_camera_frames: List[int],
) -> dict:
    """Return a flat dict of per-episode statistics matching GR00T format.

    Includes q01/q99 percentiles and all annotation columns expected by the
    GR00T LeRobot v3 dataset schema.
    """
    d: Dict[str, Any] = {}

    feature_map = {
        "observation.state": states,
        "action": actions,
        "timestamp": timestamps.reshape(-1, 1).astype(np.float64),
        "episode_index": episode_indices.reshape(-1, 1).astype(np.float64),
        "index": global_indices.reshape(-1, 1).astype(np.float64),
        "task_index": task_indices.reshape(-1, 1).astype(np.float64),
        "annotation.human.action.task_description": task_desc_indices.reshape(-1, 1).astype(np.float64),
        "annotation.human.action.task_name": task_name_indices.reshape(-1, 1).astype(np.float64),
        "annotation.human.validity": validity_indices.reshape(-1, 1).astype(np.float64),
        "next.reward": rewards.reshape(-1, 1).astype(np.float64),
    }

    for feat_name, arr in feature_map.items():
        stats = compute_feature_stats_groot(arr)
        for stat_key in ("mean", "std", "min", "max", "q01", "q99", "count"):
            d[f"stats/{feat_name}/{stat_key}"] = _to_nested_list(stats[stat_key])

    # done (bool)
    done_stats = compute_bool_stats_groot(done_flags)
    for stat_key in ("mean", "std", "min", "max", "q01", "q99", "count"):
        d[f"stats/next.done/{stat_key}"] = _to_nested_list(done_stats[stat_key])

    # image stats (placeholder) – one per camera
    for cam_name, n_fr in zip(camera_names, n_camera_frames):
        img_stats = compute_image_stats_placeholder(n_fr)
        for stat_key in ("min", "max", "mean", "std", "count"):
            d[f"stats/{cam_name}/{stat_key}"] = _to_nested_list(img_stats[stat_key])

    return d
