"""RLBench rollout save/load helpers used by inference and evaluation scripts."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from typing import Any

import numpy as np
from PIL import Image

from rlbench.backend import utils as rlbench_backend_utils
from rlbench.backend.const import (
    DEPTH_SCALE,
    FRONT_DEPTH_FOLDER,
    FRONT_MASK_FOLDER,
    FRONT_RGB_FOLDER,
    IMAGE_FORMAT,
    LOW_DIM_PICKLE,
    WRIST_DEPTH_FOLDER,
    WRIST_MASK_FOLDER,
    WRIST_RGB_FOLDER,
)

from src.utils.rlbench_utils import extract_eef_state, extract_joint_state, images_to_video


def _mkdirs(*dirs: str) -> None:
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def _ensure_action_array(actions: list[np.ndarray] | np.ndarray | None) -> np.ndarray:
    if actions is None:
        return np.zeros((0, 8), dtype=np.float64)

    arr = np.asarray(actions, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 8), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 8:
        pad = np.full((arr.shape[0], 8 - arr.shape[1]), np.nan, dtype=np.float64)
        arr = np.concatenate([arr, pad], axis=1)
    if arr.shape[1] > 8:
        arr = arr[:, :8]
    return arr


def _pad_or_trim_1d(values: list[Any] | np.ndarray | None, n: int, *, dtype) -> np.ndarray:
    if values is None:
        return np.zeros((n,), dtype=dtype)
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.shape[0] >= n:
        return arr[:n]
    if arr.shape[0] == 0:
        return np.zeros((n,), dtype=dtype)
    out = np.zeros((n,), dtype=dtype)
    out[: arr.shape[0]] = arr
    out[arr.shape[0] :] = arr[-1]
    return out


def save_observations(
    observations: list,
    save_dir: str,
    *,
    save_videos: bool = True,
    video_fps: int = 10,
    video_codec: str = "libopenh264",
    video_pix_fmt: str = "yuv420p",
    save_depth: bool = True,
    save_mask: bool = True,
    rgb_cameras: tuple[str, ...] = ("wrist", "front"),
    remove_camera_rgb_folders: tuple[str, ...] = (),
) -> None:
    """Save RLBench observations in dataset-style layout.

    Notes
    -----
    This mirrors the previous inference behavior and mutates observations by
    setting per-frame image/depth/mask arrays to ``None`` before pickling.
    """

    cam_specs = [
        ("wrist", WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER),
        ("front", FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER),
    ]
    cam_specs = [spec for spec in cam_specs if spec[0] in set(rgb_cameras)]

    all_dirs: list[str] = []
    for _, rgb_dir, depth_dir, mask_dir in cam_specs:
        all_dirs.append(os.path.join(save_dir, rgb_dir))
        if save_depth:
            all_dirs.append(os.path.join(save_dir, depth_dir))
        if save_mask:
            all_dirs.append(os.path.join(save_dir, mask_dir))
    _mkdirs(*all_dirs)

    for i, obs in enumerate(observations):
        for cam_name, rgb_folder, depth_folder, mask_folder in cam_specs:
            rgb_data = getattr(obs, f"{cam_name}_rgb", None)
            depth_data = getattr(obs, f"{cam_name}_depth", None)
            mask_data = getattr(obs, f"{cam_name}_mask", None)

            if rgb_data is not None:
                Image.fromarray(rgb_data).save(os.path.join(save_dir, rgb_folder, IMAGE_FORMAT % i))

            if save_depth and depth_data is not None:
                depth_img = rlbench_backend_utils.float_array_to_rgb_image(
                    depth_data,
                    scale_factor=DEPTH_SCALE,
                )
                depth_img.save(os.path.join(save_dir, depth_folder, IMAGE_FORMAT % i))

            if save_mask and mask_data is not None:
                mask_img = Image.fromarray((mask_data * 255).astype(np.uint8))
                mask_img.save(os.path.join(save_dir, mask_folder, IMAGE_FORMAT % i))

        for cam_name, _, _, _ in cam_specs:
            for attr_suffix in ("_rgb", "_depth", "_point_cloud", "_mask"):
                setattr(obs, f"{cam_name}{attr_suffix}", None)

    with open(os.path.join(save_dir, LOW_DIM_PICKLE), "wb") as handle:
        pickle.dump(observations, handle)

    if not save_videos:
        return

    n_frames = len(observations)
    if n_frames <= 0:
        return

    frame_start, frame_end = 0, n_frames - 1
    for cam_name, rgb_folder, _, _ in cam_specs:
        rgb_dir = os.path.join(save_dir, rgb_folder)
        video_out = os.path.join(save_dir, f"{rgb_folder}.mp4")
        try:
            images_to_video(
                rgb_dir,
                video_out,
                frame_start=frame_start,
                frame_end=frame_end,
                fps=video_fps,
                codec=video_codec,
                pix_fmt=video_pix_fmt,
            )
        except Exception as exc:  # nosec: B110
            print(f"  Warning: failed to create video for {rgb_folder}: {exc}")

        if cam_name in set(remove_camera_rgb_folders):
            shutil.rmtree(rgb_dir, ignore_errors=True)


def build_rollout_arrays(
    observations: list,
    *,
    actions: list[np.ndarray] | np.ndarray | None = None,
    rewards: list[float] | np.ndarray | None = None,
    dones: list[bool] | np.ndarray | None = None,
    grasped_target: list[bool] | np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build aligned numeric arrays for a rollout.

    Returns a dict with:
    - ``eef_state`` (T, 8)
    - ``joint_state`` (T, 8)
    - ``gripper_open`` (T,)
    - ``action`` (N, 8) for step-level actions
    - ``action_aligned`` (T, 8) with action[t] aligned to resulting observation t
    - ``reward`` (N,)
    - ``done`` (N,)
    - ``grasped_target`` (T,)
    """
    n_obs = len(observations)
    if n_obs > 0:
        eef_state = np.stack([extract_eef_state(obs) for obs in observations], axis=0).astype(np.float32)
        joint_state = np.stack([extract_joint_state(obs) for obs in observations], axis=0).astype(np.float32)
    else:
        eef_state = np.zeros((0, 8), dtype=np.float32)
        joint_state = np.zeros((0, 8), dtype=np.float32)

    action = _ensure_action_array(actions)
    n_steps = action.shape[0]

    reward = _pad_or_trim_1d(rewards, n_steps, dtype=np.float32)
    done = _pad_or_trim_1d(dones, n_steps, dtype=np.bool_)
    grasped = _pad_or_trim_1d(grasped_target, n_obs, dtype=np.bool_)

    action_aligned = np.full((n_obs, 8), np.nan, dtype=np.float64)
    if n_obs > 1 and n_steps > 0:
        n_align = min(n_steps, n_obs - 1)
        action_aligned[1 : 1 + n_align] = action[:n_align]

    return {
        "eef_state": eef_state,
        "joint_state": joint_state,
        "gripper_open": eef_state[:, 7] if n_obs > 0 else np.zeros((0,), dtype=np.float32),
        "action": action,
        "action_aligned": action_aligned,
        "reward": reward,
        "done": done,
        "grasped_target": grasped,
    }


def save_rollout_bundle(
    observations: list,
    save_dir: str,
    *,
    actions: list[np.ndarray] | np.ndarray | None = None,
    rewards: list[float] | np.ndarray | None = None,
    dones: list[bool] | np.ndarray | None = None,
    grasped_target: list[bool] | np.ndarray | None = None,
    info: dict[str, Any] | None = None,
    save_videos: bool = True,
) -> dict[str, np.ndarray]:
    """Save rollout images+pickle and structured arrays/json sidecars."""
    os.makedirs(save_dir, exist_ok=True)

    arrays = build_rollout_arrays(
        observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        grasped_target=grasped_target,
    )
    np.savez_compressed(os.path.join(save_dir, "rollout_arrays.npz"), **arrays)

    if info is not None:
        with open(os.path.join(save_dir, "rollout_info.json"), "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, ensure_ascii=False)

    save_observations(observations, save_dir, save_videos=save_videos)
    return arrays


def save_rollout_sidecars(
    observations: list,
    save_dir: str,
    *,
    actions: list[np.ndarray] | np.ndarray | None = None,
    rewards: list[float] | np.ndarray | None = None,
    dones: list[bool] | np.ndarray | None = None,
    grasped_target: list[bool] | np.ndarray | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Save only rollout arrays/json sidecars (no image or pickle export)."""
    os.makedirs(save_dir, exist_ok=True)

    arrays = build_rollout_arrays(
        observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        grasped_target=grasped_target,
    )
    np.savez_compressed(os.path.join(save_dir, "rollout_arrays.npz"), **arrays)

    if info is not None:
        with open(os.path.join(save_dir, "rollout_info.json"), "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2, ensure_ascii=False)

    return arrays
