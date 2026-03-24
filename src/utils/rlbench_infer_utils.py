"""Shared helper utilities for RLBench policy inference scripts."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def split_task_description_into_steps(task_description: str) -> list[str]:
    """Split a single task-description string into ordered instruction steps."""
    text = (task_description or "").strip()
    if not text:
        return [""]

    if "|" in text:
        parts = [part.strip() for part in text.split("|")]
    elif "\n" in text:
        parts = [part.strip() for part in text.splitlines()]
    elif ";" in text:
        parts = [part.strip() for part in text.split(";")]
    else:
        parts = [part.strip() for part in re.split(r"(?<=\.)\s+", text)]

    parts = [part for part in parts if part]
    return parts if parts else [text]


def set_policy_task_description(policy: Any, task_description: str) -> None:
    """Best-effort update of policy language instruction if supported."""
    if hasattr(policy, "task_description"):
        setattr(policy, "task_description", task_description)


def eef_pose_change(prev_obs: Any, curr_obs: Any) -> tuple[float, float]:
    """Return (position_delta_m, rotation_delta_rad) between two observations."""
    prev_pose = np.asarray(prev_obs.gripper_pose, dtype=np.float64).reshape(-1)
    curr_pose = np.asarray(curr_obs.gripper_pose, dtype=np.float64).reshape(-1)
    if prev_pose.shape[0] < 7 or curr_pose.shape[0] < 7:
        raise ValueError("Expected gripper_pose to have 7 elements (xyz + quat).")

    pos_delta = float(np.linalg.norm(curr_pose[:3] - prev_pose[:3]))

    q_prev = prev_pose[3:7]
    q_curr = curr_pose[3:7]
    q_prev_norm = np.linalg.norm(q_prev)
    q_curr_norm = np.linalg.norm(q_curr)
    if q_prev_norm < 1e-12 or q_curr_norm < 1e-12:
        rot_delta = 0.0
    else:
        q_prev = q_prev / q_prev_norm
        q_curr = q_curr / q_curr_norm
        dot = float(np.dot(q_prev, q_curr))
        dot = abs(dot)
        dot = max(-1.0, min(1.0, dot))
        rot_delta = float(2.0 * np.arccos(dot))

    return pos_delta, rot_delta
