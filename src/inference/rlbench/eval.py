"""Evaluate a trained policy against the RLBench planner (generic task).

Segment count follows `datasets/rlbench/<task>/<task>_relationship_template.json`
when available: each entry in ``transitions`` is one sub-episode (e.g. lamp_on has
one transition → one episode; put_rubbish_in_bin has two → two episodes).
Boundaries align to gripper changes that match each template transition
(open/closed), not raw gripper-event count vs. instruction steps.

Fallback (no template or --no_relationship_template): segment count from
instruction steps and first N-1 gripper changes (legacy).

Outputs: planner/policy rollouts, per-run metrics, CSV, summary JSON.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path for local imports.
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
    JointVelocity,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

from src.utils.rlbench_policy import LeRobotPolicy
from src.utils.rlbench_infer_utils import split_task_description_into_steps
from src.utils.rlbench_eval_utils import (
    capture_task_env_state,
    collect_planner_rollout,
    planner_actions_from_observations,
    restore_task_env_state,
    rollout_policy,
)
from src.utils.rlbench_rollout_io import save_observations, save_rollout_sidecars
from src.utils.rlbench_utils import extract_eef_state, extract_joint_state
from src.utils.rlbench_eval_helpers import (
    parse_bool,
    parse_float,
    parse_int,
    mean_or_none,
    std_or_none,
)


def _build_obs_config(image_size: list[int], renderer: str) -> ObservationConfig:
    """Build ObservationConfig for low-dim + wrist/front RGB only."""
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)
    obs_config.joint_forces = False

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


def _build_action_mode(action_mode: str) -> MoveArmThenGripper:
    if action_mode == "ee_planning":
        arm_mode = EndEffectorPoseViaPlanning(absolute_mode=True, collision_checking=False)
    elif action_mode == "ee_ik":
        arm_mode = EndEffectorPoseViaIK(absolute_mode=True, collision_checking=False)
    elif action_mode == "joint_velocity":
        arm_mode = JointVelocity()
    else:
        raise ValueError(f"Unknown action mode: {action_mode}")

    return MoveArmThenGripper(arm_action_mode=arm_mode, gripper_action_mode=Discrete())


def _write_json(path: str, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: str, rows: list[dict]) -> None:
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


def aggregate_batch_results(aggregate_root: str, task_name: str = "rlbench_task") -> dict:
    """Aggregate already-generated batch outputs without rerunning inference."""
    root = Path(aggregate_root)
    if not root.exists():
        raise FileNotFoundError(f"Aggregate root does not exist: {aggregate_root}")

    run_summaries: list[dict] = []
    run_rows: list[dict] = []
    episode_rows: list[dict] = []

    policy_state_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    var_seed_re = re.compile(r"^var(-?\d+)_seed(-?\d+)$")

    for policy_state_dir in policy_state_dirs:
        name = policy_state_dir.name
        if "_" not in name:
            continue
        policy, state = name.rsplit("_", 1)

        for eval_dir in sorted([p for p in policy_state_dir.iterdir() if p.is_dir()]):
            match = var_seed_re.match(eval_dir.name)
            if match is None:
                continue
            variation = int(match.group(1))
            seed = int(match.group(2))

            csv_path = eval_dir / "per_run_metrics.csv"
            if not csv_path.exists():
                continue

            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

            per_run_episode_rows: dict[int, list[dict]] = {}
            for row in rows:
                success = parse_bool(row.get("policy_success", False))
                row_joint = parse_float(row.get("joint_l2"))
                row_pos = parse_float(row.get("pos_l2"))
                row_rot = parse_float(row.get("rot_deg"))
                row_eef = parse_float(row.get("eef_l2"))
                run_index = parse_int(row.get("run_index"))
                if run_index is None:
                    # Fallback for legacy CSVs without run_index.
                    run_index = 0

                enriched_episode = {
                    "policy": policy,
                    "state": state,
                    "variation": variation,
                    "seed": seed,
                    "run_index": run_index,
                    "episode_index": parse_int(row.get("episode_index")),
                    "instruction": row.get("instruction"),
                    "policy_success": success,
                    "joint_l2": row_joint,
                    "pos_l2": row_pos,
                    "rot_deg": row_rot,
                    "eef_l2": row_eef,
                    "eval_dir": str(eval_dir),
                }
                episode_rows.append(enriched_episode)
                per_run_episode_rows.setdefault(run_index, []).append(enriched_episode)

            eval_run_metrics: list[dict] = []
            for run_index, run_eps in sorted(per_run_episode_rows.items(), key=lambda kv: kv[0]):
                run_episode_successes = [float(bool(ep.get("policy_success", False))) for ep in run_eps]
                run_joint_l2 = [ep.get("joint_l2") for ep in run_eps]
                run_pos_l2 = [ep.get("pos_l2") for ep in run_eps]
                run_rot_deg = [ep.get("rot_deg") for ep in run_eps]
                run_eef_l2 = [ep.get("eef_l2") for ep in run_eps]
                all_success = (
                    float(all(v > 0.5 for v in run_episode_successes))
                    if run_episode_successes
                    else None
                )
                metric = {
                    "policy": policy,
                    "state": state,
                    "variation": variation,
                    "seed": seed,
                    "run_index": run_index,
                    "num_episodes": len(run_eps),
                    "episode_success_rate": mean_or_none(run_episode_successes),
                    "all_episode_success_rate": all_success,
                    "joint_l2_mean": mean_or_none(run_joint_l2),
                    "pos_l2_mean": mean_or_none(run_pos_l2),
                    "rot_deg_mean": mean_or_none(run_rot_deg),
                    "eef_l2_mean": mean_or_none(run_eef_l2),
                    "eval_dir": str(eval_dir),
                }
                eval_run_metrics.append(metric)
                run_rows.append(metric)

            run_summaries.append(
                {
                    "policy": policy,
                    "state": state,
                    "variation": variation,
                    "seed": seed,
                    "num_runs": len(eval_run_metrics),
                    "num_episodes": len(rows),
                    "episode_success_rate_mean": _mean_or_none(
                        [r.get("episode_success_rate") for r in eval_run_metrics]
                    ),
                    "episode_success_rate_overall": _mean_or_none(
                        [float(bool(ep.get("policy_success", False))) for ep in episode_rows if ep["eval_dir"] == str(eval_dir)]
                    ),
                    "all_episode_success_rate_mean": _mean_or_none(
                        [r.get("all_episode_success_rate") for r in eval_run_metrics]
                    ),
                    "joint_l2_mean": _mean_or_none([r.get("joint_l2_mean") for r in eval_run_metrics]),
                    "joint_l2_overall": _mean_or_none(
                        [ep.get("joint_l2") for ep in episode_rows if ep["eval_dir"] == str(eval_dir)]
                    ),
                    "pos_l2_mean": _mean_or_none([r.get("pos_l2_mean") for r in eval_run_metrics]),
                    "pos_l2_overall": _mean_or_none(
                        [ep.get("pos_l2") for ep in episode_rows if ep["eval_dir"] == str(eval_dir)]
                    ),
                    "rot_deg_mean": _mean_or_none([r.get("rot_deg_mean") for r in eval_run_metrics]),
                    "rot_deg_overall": _mean_or_none(
                        [ep.get("rot_deg") for ep in episode_rows if ep["eval_dir"] == str(eval_dir)]
                    ),
                    "eval_dir": str(eval_dir),
                }
            )

    def _group_runs(key: str) -> list[dict]:
        groups: dict[str, list[dict]] = {}
        for row in run_rows:
            groups.setdefault(str(row[key]), []).append(row)

        summaries: list[dict] = []
        for group_key, rows in groups.items():
            group_eval_dirs = {str(r["eval_dir"]) for r in rows}
            group_episodes = [ep for ep in episode_rows if str(ep[key]) == group_key]
            summaries.append(
                {
                    key: group_key,
                    "num_eval_dirs": len(group_eval_dirs),
                    "num_runs": len(rows),
                    "num_episodes": len(group_episodes),
                    "episode_success_rate_mean": _mean_or_none([r.get("episode_success_rate") for r in rows]),
                    "episode_success_rate_std": std_or_none([r.get("episode_success_rate") for r in rows]),
                    "episode_success_rate_overall": _mean_or_none(
                        [float(bool(ep.get("policy_success", False))) for ep in group_episodes]
                    ),
                    "all_episode_success_rate_mean": _mean_or_none(
                        [r.get("all_episode_success_rate") for r in rows]
                    ),
                    "joint_l2_mean": _mean_or_none([r.get("joint_l2_mean") for r in rows]),
                    "joint_l2_overall": _mean_or_none([ep.get("joint_l2") for ep in group_episodes]),
                    "pos_l2_mean": _mean_or_none([r.get("pos_l2_mean") for r in rows]),
                    "pos_l2_overall": _mean_or_none([ep.get("pos_l2") for ep in group_episodes]),
                    "rot_deg_mean": _mean_or_none([r.get("rot_deg_mean") for r in rows]),
                    "rot_deg_overall": _mean_or_none([ep.get("rot_deg") for ep in group_episodes]),
                }
            )
        summaries.sort(
            key=lambda r: (
                -(r.get("all_episode_success_rate_mean") or -1.0),
                r.get("joint_l2_mean") if r.get("joint_l2_mean") is not None else float("inf"),
            )
        )
        return summaries

    pair_groups: dict[tuple[str, str], list[dict]] = {}
    for row in run_rows:
        pair_groups.setdefault((str(row["policy"]), str(row["state"])), []).append(row)

    by_policy_state: list[dict] = []
    for (policy, state), rows in pair_groups.items():
        group_eval_dirs = {str(r["eval_dir"]) for r in rows}
        group_episodes = [
            ep
            for ep in episode_rows
            if str(ep["policy"]) == policy and str(ep["state"]) == state
        ]
        by_policy_state.append(
            {
                "policy": policy,
                "state": state,
                "num_eval_dirs": len(group_eval_dirs),
                "num_runs": len(rows),
                "num_episodes": len(group_episodes),
                "episode_success_rate_mean": _mean_or_none([r.get("episode_success_rate") for r in rows]),
                "episode_success_rate_std": std_or_none([r.get("episode_success_rate") for r in rows]),
                "episode_success_rate_overall": _mean_or_none(
                    [float(bool(ep.get("policy_success", False))) for ep in group_episodes]
                ),
                "all_episode_success_rate_mean": _mean_or_none(
                    [r.get("all_episode_success_rate") for r in rows]
                ),
                "joint_l2_mean": _mean_or_none([r.get("joint_l2_mean") for r in rows]),
                "joint_l2_overall": _mean_or_none([ep.get("joint_l2") for ep in group_episodes]),
                "pos_l2_mean": _mean_or_none([r.get("pos_l2_mean") for r in rows]),
                "pos_l2_overall": _mean_or_none([ep.get("pos_l2") for ep in group_episodes]),
                "rot_deg_mean": _mean_or_none([r.get("rot_deg_mean") for r in rows]),
                "rot_deg_overall": _mean_or_none([ep.get("rot_deg") for ep in group_episodes]),
            }
        )
    by_policy_state.sort(
        key=lambda r: (
            -(r.get("all_episode_success_rate_mean") or -1.0),
            r.get("joint_l2_mean") if r.get("joint_l2_mean") is not None else float("inf"),
        )
    )

    # Convenience table for "policy x action x episode" reporting.
    # In this benchmark pipeline:
    #   eef   -> ee_planning
    #   joint -> joint_velocity
    action_mode_map = {
        "eef": "ee_planning",
        "joint": "joint_velocity",
    }
    combo_episode_groups: dict[tuple[str, str, int], list[dict]] = {}
    for ep in episode_rows:
        ep_idx = ep.get("episode_index")
        if ep_idx is None:
            continue
        key = (str(ep.get("policy")), str(ep.get("state")), int(ep_idx))
        combo_episode_groups.setdefault(key, []).append(ep)

    overall_metrics_rows: list[dict] = []
    for (policy, state, episode_index), rows in sorted(
        combo_episode_groups.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        run_keys = {
            (
                int(parse_int(r.get("variation")) or 0),
                int(parse_int(r.get("seed")) or 0),
                int(parse_int(r.get("run_index")) or 0),
            )
            for r in rows
        }
        eval_dirs = {str(r.get("eval_dir")) for r in rows}
        overall_metrics_rows.append(
            {
                "policy": policy,
                "state": state,
                "action_mode": action_mode_map.get(state, "unknown"),
                "episode_index": episode_index,
                "num_eval_dirs": len(eval_dirs),
                "num_runs": len(run_keys),
                "num_episode_rows": len(rows),
                "episode_success_rate": _mean_or_none(
                    [float(bool(ep.get("policy_success", False))) for ep in rows]
                ),
                "joint_l2_mean": _mean_or_none([ep.get("joint_l2") for ep in rows]),
                "pos_l2_mean": _mean_or_none([ep.get("pos_l2") for ep in rows]),
                "rot_deg_mean": _mean_or_none([ep.get("rot_deg") for ep in rows]),
                "eef_l2_mean": _mean_or_none([ep.get("eef_l2") for ep in rows]),
            }
        )

    overall = {
        "num_eval_dirs": len(run_summaries),
        "num_runs": len(run_rows),
        "num_episodes": len(episode_rows),
        "episode_success_rate_mean": mean_or_none([r.get("episode_success_rate") for r in run_rows]),
        "episode_success_rate_std": std_or_none([r.get("episode_success_rate") for r in run_rows]),
        "episode_success_rate_overall": mean_or_none(
            [float(bool(ep.get("policy_success", False))) for ep in episode_rows]
        ),
        "all_episode_success_rate_mean": mean_or_none(
            [r.get("all_episode_success_rate") for r in run_rows]
        ),
        "joint_l2_mean": mean_or_none([r.get("joint_l2_mean") for r in run_rows]),
        "joint_l2_overall": mean_or_none([ep.get("joint_l2") for ep in episode_rows]),
        "pos_l2_mean": mean_or_none([r.get("pos_l2_mean") for r in run_rows]),
        "pos_l2_overall": mean_or_none([ep.get("pos_l2") for ep in episode_rows]),
        "rot_deg_mean": mean_or_none([r.get("rot_deg_mean") for r in run_rows]),
        "rot_deg_overall": mean_or_none([ep.get("rot_deg") for ep in episode_rows]),
        "eef_l2_mean": mean_or_none([r.get("eef_l2_mean") for r in run_rows]),
        "eef_l2_overall": mean_or_none([ep.get("eef_l2") for ep in episode_rows]),
    }

    return {
        "task": task_name,
        "aggregate_root": str(root),
        "num_eval_dirs": len(run_summaries),
        "num_runs": len(run_rows),
        "num_episode_rows": len(episode_rows),
        "run_details": sorted(
            run_summaries,
            key=lambda r: (str(r["policy"]), str(r["state"]), int(r["variation"]), int(r["seed"])),
        ),
        "per_run_details": sorted(
            run_rows,
            key=lambda r: (
                str(r["policy"]),
                str(r["state"]),
                int(r["variation"]),
                int(r["seed"]),
                int(r["run_index"]),
            ),
        ),
        "comparison": {
            "overall": overall,
            "by_policy_state": by_policy_state,
            "by_policy": _group_runs("policy"),
            "by_state": _group_runs("state"),
        },
        "overall_metrics_rows": overall_metrics_rows,
    }


def _instruction_steps(task_description: str) -> list[str]:
    steps = split_task_description_into_steps(task_description or "")
    cleaned = [s.strip() for s in steps if s.strip()]
    return cleaned if cleaned else [""]


def _gripper_change_indices(observations: list, threshold: float = 0.5) -> list[int]:
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


def _segment_boundaries(change_indices: list[int], n_segments: int, last_index: int) -> list[int]:
    needed = max(0, n_segments - 1)
    boundaries = list(change_indices[:needed])
    while len(boundaries) < needed:
        boundaries.append(last_index)
    return boundaries


def _segment_ranges(boundaries: list[int], n_segments: int, last_index: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(n_segments):
        end = boundaries[idx] if idx < len(boundaries) else last_index
        end = int(max(start, min(end, last_index)))
        ranges.append((start, end))
        start = end
    return ranges


def _resolve_relationship_template_path(
    task: str,
    rlbench_root: str | Path,
    explicit: str | None,
) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    p = Path(rlbench_root) / task / f"{task}_relationship_template.json"
    return p if p.is_file() else None


def _load_relationship_transitions(template_path: Path) -> list[dict]:
    with open(template_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    transitions = data.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return []
    return transitions


def _segmentation_from_relationship_template(
    observations: list,
    transitions: list[dict],
    last_index: int,
    threshold: float = 0.5,
) -> tuple[int, list[int]]:
    """Number of segments = len(transitions). Boundaries end segment 0..n-2."""
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


def _instructions_per_segment(
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


def _quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    qa = np.asarray(q_a, dtype=np.float64).reshape(-1)
    qb = np.asarray(q_b, dtype=np.float64).reshape(-1)
    qa_norm = np.linalg.norm(qa)
    qb_norm = np.linalg.norm(qb)
    if qa_norm < 1e-12 or qb_norm < 1e-12:
        return 0.0
    qa = qa / qa_norm
    qb = qb / qb_norm
    dot = float(np.dot(qa, qb))
    dot = abs(max(-1.0, min(1.0, dot)))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _segment_state_metric(expert_final_obs, policy_final_obs) -> dict[str, float]:
    expert_eef = extract_eef_state(expert_final_obs).astype(np.float64)
    policy_eef = extract_eef_state(policy_final_obs).astype(np.float64)
    expert_joint = extract_joint_state(expert_final_obs).astype(np.float64)
    policy_joint = extract_joint_state(policy_final_obs).astype(np.float64)

    return {
        "eef_l2": float(np.linalg.norm(policy_eef - expert_eef)),
        "joint_l2": float(np.linalg.norm(policy_joint - expert_joint)),
        "pos_l2": float(np.linalg.norm(policy_eef[:3] - expert_eef[:3])),
        "rot_deg": _quat_angle_deg(policy_eef[3:7], expert_eef[3:7]),
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def _save_episode_outputs(
    observations: list,
    episode_dir: str,
    *,
    grasped_target: list[bool] | np.ndarray,
    actions_step: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    info: dict,
) -> None:
    save_rollout_sidecars(
        observations,
        episode_dir,
        actions=actions_step,
        rewards=rewards,
        dones=dones,
        grasped_target=grasped_target,
        info=info,
    )

    # save_observations mutates observations in-place, so save from deep copies.
    save_observations(
        [deepcopy(obs) for obs in observations],
        episode_dir,
        save_videos=True,
        save_depth=False,
        save_mask=False,
        rgb_cameras=("wrist", "front"),
        remove_camera_rgb_folders=("front",),
    )


def run_evaluation(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --aggregate_only is used.")
    if not args.dataset_root:
        raise ValueError("--dataset_root is required unless --aggregate_only is used.")

    os.makedirs(args.save_path, exist_ok=True)

    obs_config = _build_obs_config(list(args.image_size), args.renderer)
    action_mode = _build_action_mode(args.action_mode)

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        headless=True,
    )
    env.launch()
    print(f"RLBench launched for evaluation (task={args.task}, variation={args.variation})")

    run_metrics: list[dict] = []
    run_rows: list[dict] = []

    rlbench_root = Path(args.rlbench_root).resolve() if args.rlbench_root else (_project_root / "datasets" / "rlbench")
    template_path = _resolve_relationship_template_path(
        args.task, rlbench_root, args.relationship_template
    )
    template_transitions: list[dict] | None = None
    if not getattr(args, "no_relationship_template", False) and template_path is not None:
        template_transitions = _load_relationship_transitions(template_path)
        if template_transitions:
            print(
                f"Segmentation from relationship template: {template_path} "
                f"({len(template_transitions)} transition(s) → {len(template_transitions)} episode(s))"
            )
        else:
            template_transitions = None
    if template_transitions is None and not getattr(args, "no_relationship_template", False):
        expected = rlbench_root / args.task / f"{args.task}_relationship_template.json"
        print(
            f"Warning: relationship template missing or empty ({expected}); "
            "using instruction-step count + gripper boundaries."
        )

    try:
        task_class = task_file_to_task_class(args.task)
        task_env = env.get_task(task_class)

        policy = LeRobotPolicy(
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset_root,
            # Important: do NOT bind the full multi-step prompt at init.
            # We set per-instruction text right before each segment rollout.
            task_description="",
            action_mode=args.action_mode,
            device=args.device,
        )

        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            task_env.set_variation(args.variation)
            print(f"\n[{run_idx + 1}/{args.runs}] seed={run_seed}")

            run_dir = os.path.join(args.save_path, f"run_{run_idx:03d}")
            os.makedirs(run_dir, exist_ok=True)

            planner_obs, planner_grasped, planner_demo, planner_state_snaps = collect_planner_rollout(
                task_env,
                max_attempts=args.planner_max_attempts,
            )

            # Reset to exactly the same initial state sampled for planner rollout.
            descriptions, _initial_obs = task_env.reset_to_demo(planner_demo)
            task_description = args.task_description
            if task_description is None:
                task_description = descriptions[0] if descriptions else ""

            task_steps = _instruction_steps(task_description)
            last_expert_idx = max(0, len(planner_obs) - 1)
            gripper_changes = _gripper_change_indices(planner_obs)

            if template_transitions:
                n_segments, boundaries = _segmentation_from_relationship_template(
                    planner_obs, template_transitions, last_expert_idx
                )
                segmentation_source = "relationship_template"
            else:
                n_segments = max(1, len(task_steps))
                boundaries = _segment_boundaries(gripper_changes, n_segments, last_expert_idx)
                segmentation_source = "instruction_steps_gripper"
                if len(gripper_changes) < max(0, n_segments - 1):
                    print(
                        "  Warning: expert gripper transitions fewer than task steps; "
                        "reusing final expert state for remaining segments."
                    )

            segment_instructions = _instructions_per_segment(
                n_segments, task_steps, task_description
            )
            ranges = _segment_ranges(boundaries, n_segments, last_expert_idx)

            initial_state = capture_task_env_state(task_env)
            # RLBench demo callback does not provide a snapshot for the very first
            # observation, so fill it from reset_to_demo() state.
            if planner_state_snaps:
                if planner_state_snaps[0] is None:
                    planner_state_snaps[0] = initial_state

            episode_metrics: list[dict] = []

            for ep_idx in range(n_segments):
                step_text = segment_instructions[ep_idx]
                seg_start, seg_end = ranges[ep_idx]
                print(f"  ep={ep_idx} instruction={step_text}")

                planner_episode_dir = os.path.join(run_dir, "planner", "episodes", f"episode{ep_idx}")
                policy_episode_dir = os.path.join(run_dir, "policy", "episodes", f"episode{ep_idx}")

                planner_ep_obs = planner_obs[seg_start : seg_end + 1]
                planner_ep_grasp = planner_grasped[seg_start : seg_end + 1]

                planner_actions_full = planner_actions_from_observations(planner_ep_obs)
                planner_actions_step = (
                    planner_actions_full[1:]
                    if planner_actions_full.shape[0] > 1
                    else np.zeros((0, 8), dtype=np.float64)
                )
                planner_rewards = np.zeros((planner_actions_step.shape[0],), dtype=np.float32)
                planner_dones = np.zeros((planner_actions_step.shape[0],), dtype=np.bool_)
                if planner_rewards.shape[0] > 0:
                    planner_rewards[-1] = 1.0
                    planner_dones[-1] = True

                _save_episode_outputs(
                    planner_ep_obs,
                    planner_episode_dir,
                    grasped_target=planner_ep_grasp,
                    actions_step=planner_actions_step,
                    rewards=planner_rewards,
                    dones=planner_dones,
                    info={
                        "source": "planner",
                        "run_index": run_idx,
                        "seed": run_seed,
                        "variation": args.variation,
                        "episode_index": ep_idx,
                        "instruction": step_text,
                        "segment_start_index": seg_start,
                        "segment_end_index": seg_end,
                    },
                )

                # Always restore from planner-aligned state for every episode
                # (including episode 0) to keep planner/policy start states matched.
                policy_start_state = (
                    planner_state_snaps[seg_start] if seg_start < len(planner_state_snaps) else None
                )
                if policy_start_state is None:
                    policy_start_state = initial_state
                policy_start_obs = restore_task_env_state(task_env, policy_start_state)

                policy_rollout = rollout_policy(
                    task_env,
                    policy,
                    initial_obs=policy_start_obs,
                    task_description=step_text,
                    max_steps_per_instruction=args.max_steps,
                    stop_on_success=False,
                )

                policy_actions_step = np.asarray(policy_rollout["actions"], dtype=np.float64)
                if policy_actions_step.ndim == 1 and policy_actions_step.size > 0:
                    policy_actions_step = policy_actions_step.reshape(1, -1)
                if policy_actions_step.size == 0:
                    policy_actions_step = np.zeros((0, 8), dtype=np.float64)

                policy_rewards = np.asarray(policy_rollout["rewards"], dtype=np.float32)
                policy_dones = np.asarray(policy_rollout["dones"], dtype=np.bool_)

                _save_episode_outputs(
                    policy_rollout["observations"],
                    policy_episode_dir,
                    grasped_target=policy_rollout["grasped_target"],
                    actions_step=policy_actions_step,
                    rewards=policy_rewards,
                    dones=policy_dones,
                    info={
                        "source": "policy",
                        "run_index": run_idx,
                        "seed": run_seed,
                        "variation": args.variation,
                        "episode_index": ep_idx,
                        "instruction": step_text,
                        "policy_success": bool(policy_rollout["success"]),
                        "policy_errors": policy_rollout["errors"],
                    },
                )

                expert_final_obs = planner_obs[seg_end]
                policy_final_obs = policy_rollout["observations"][-1]
                dist = _segment_state_metric(expert_final_obs, policy_final_obs)

                episode_metric = {
                    "run_index": run_idx,
                    "seed": run_seed,
                    "variation": args.variation,
                    "episode_index": ep_idx,
                    "instruction": step_text,
                    "segment_start_index": seg_start,
                    "segment_end_index": seg_end,
                    "policy_success": bool(policy_rollout["success"]),
                    "policy_terminated": bool(policy_rollout["terminated"]),
                    "policy_error_count": int(len(policy_rollout["errors"])),
                    "eef_l2": dist["eef_l2"],
                    "joint_l2": dist["joint_l2"],
                    "pos_l2": dist["pos_l2"],
                    "rot_deg": dist["rot_deg"],
                }
                episode_metrics.append(episode_metric)
                run_rows.append(dict(episode_metric))

                print(
                    "  "
                    f"ep={ep_idx} success={episode_metric['policy_success']} "
                    f"joint_l2={episode_metric['joint_l2']:.6f} "
                    f"pos_l2={episode_metric['pos_l2']:.6f}"
                )

            run_metric = {
                "run_index": run_idx,
                "seed": run_seed,
                "variation": args.variation,
                "task_description": task_description,
                "task_steps": task_steps,
                "segment_instructions": segment_instructions,
                "segmentation_source": segmentation_source,
                "relationship_template_path": str(template_path) if template_path else None,
                "num_template_transitions": len(template_transitions) if template_transitions else None,
                "expert_gripper_change_indices": gripper_changes,
                "segment_boundaries": boundaries,
                "episode_metrics": episode_metrics,
            }
            run_metrics.append(run_metric)

            _write_json(os.path.join(run_dir, "metrics.json"), run_metric)

    finally:
        env.shutdown()

    max_episodes = max((len(r.get("episode_metrics", [])) for r in run_metrics), default=0)
    per_episode_summary: list[dict] = []
    for ep_idx in range(max_episodes):
        ep_rows = [
            ep
            for run in run_metrics
            for ep in run.get("episode_metrics", [])
            if int(ep.get("episode_index", -1)) == ep_idx
        ]
        per_episode_summary.append(
            {
                "episode_index": ep_idx,
                "num_samples": len(ep_rows),
                "policy_success_rate": _mean_or_none([float(bool(ep["policy_success"])) for ep in ep_rows]),
                "eef_l2_mean": _mean_or_none([ep.get("eef_l2") for ep in ep_rows]),
                "joint_l2_mean": _mean_or_none([ep.get("joint_l2") for ep in ep_rows]),
                "pos_l2_mean": _mean_or_none([ep.get("pos_l2") for ep in ep_rows]),
                "rot_deg_mean": _mean_or_none([ep.get("rot_deg") for ep in ep_rows]),
            }
        )

    all_episode_success = []
    for run in run_metrics:
        eps = run.get("episode_metrics", [])
        if not eps:
            continue
        all_episode_success.append(float(all(bool(ep.get("policy_success", False)) for ep in eps)))

    summary = {
        "task": args.task,
        "variation": args.variation,
        "runs": args.runs,
        "seed": args.seed,
        "action_mode": args.action_mode,
        "num_episodes": max_episodes,
        "num_instruction_episodes": max_episodes,
        "all_episode_success_rate": _mean_or_none(all_episode_success),
        "per_episode": per_episode_summary,
    }

    _write_json(os.path.join(args.save_path, "summary.json"), summary)
    _write_json(os.path.join(args.save_path, "per_run_metrics.json"), run_metrics)
    _write_csv(os.path.join(args.save_path, "per_run_metrics.csv"), run_rows)

    print("\nEvaluation complete.")
    print(f"Saved outputs to: {args.save_path}")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy vs RLBench planner for an RLBench task."
    )
    parser.add_argument("--task", type=str, default="put_rubbish_in_bin")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=150)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument(
        "--rlbench_root",
        type=str,
        default=None,
        help="Directory containing per-task folders (default: <repo>/datasets/rlbench).",
    )
    parser.add_argument(
        "--relationship_template",
        type=str,
        default=None,
        help="Explicit path to <task>_relationship_template.json (overrides rlbench_root lookup).",
    )
    parser.add_argument(
        "--no_relationship_template",
        action="store_true",
        help="Ignore relationship template; use instruction steps + first gripper changes.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="Pick up paper. Release paper, then place paper in trash bin.",
    )
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--action_mode",
        type=str,
        default="joint_velocity",
        choices=["ee_planning", "ee_ik", "joint_velocity"],
    )
    parser.add_argument("--renderer", type=str, default="opengl", choices=["opengl", "opengl3"])
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--arm_max_velocity", type=float, default=1.0)
    parser.add_argument("--arm_max_acceleration", type=float, default=4.0)
    parser.add_argument("--planner_max_attempts", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="output/rlbench_eval/default_task")
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip RLBench evaluation and only aggregate existing outputs under --aggregate_root.",
    )
    parser.add_argument(
        "--aggregate_root",
        type=str,
        default=None,
        help="Root containing policy/state evaluation folders to aggregate.",
    )
    parser.add_argument(
        "--aggregate_output_json",
        type=str,
        default=None,
        help="Where to write aggregate JSON report (default: <aggregate_root>/detailed_summary.json).",
    )
    parser.add_argument(
        "--aggregate_output_csv",
        type=str,
        default=None,
        help="Where to write aggregate per-run CSV report (default: <aggregate_root>/detailed_runs.csv).",
    )
    parser.add_argument(
        "--aggregate_output_overall_csv",
        type=str,
        default=None,
        help=(
            "Where to write aggregate policy/action CSV report "
            "(default: <aggregate_root>/overall_metrics.csv)."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.aggregate_only:
        aggregate_root = parsed_args.aggregate_root or parsed_args.save_path
        payload = aggregate_batch_results(aggregate_root, task_name=parsed_args.task)

        output_json = parsed_args.aggregate_output_json
        if output_json is None:
            output_json = str(Path(aggregate_root) / "detailed_summary.json")
        _write_json(output_json, payload)

        output_csv = parsed_args.aggregate_output_csv
        if output_csv is None:
            output_csv = str(Path(aggregate_root) / "detailed_runs.csv")
        _write_csv(output_csv, payload.get("run_details", []))

        output_overall_csv = parsed_args.aggregate_output_overall_csv
        if output_overall_csv is None:
            output_overall_csv = str(Path(aggregate_root) / "overall_metrics.csv")
        _write_csv(output_overall_csv, payload.get("overall_metrics_rows", []))

        print("\nAggregate-only summary complete.")
        print(f"Aggregate root: {aggregate_root}")
        print(f"Detailed JSON : {output_json}")
        print(f"Detailed CSV  : {output_csv}")
        print(f"Overall CSV   : {output_overall_csv}")
        print(json.dumps(payload.get("comparison", {}), indent=2))
    else:
        run_evaluation(parsed_args)
