"""Evaluate a trained policy against the RLBench planner on put_rubbish_in_bin.

This script performs segmented evaluation for each run:
1) Collect an expert planner rollout.
2) Split task description into ordered instruction steps.
3) Detect expert gripper state-change boundaries.
4) Save planner rollouts as episode0..episodeN-1 based on those boundaries.
5) For each instruction i, restore the simulator to expert boundary i-1 and
    run policy for a fixed rollout horizon.
6) Save policy rollouts as episode0..episodeN-1 and compare policy final state
    against expert segment-end state.

Outputs are written under `--save_path`, including planner/policy rollouts,
per-run metrics, a merged CSV, and a summary JSON.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
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

from src.inference.rlbench.infer_rlbench import LeRobotPolicy
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
            descriptions, initial_obs = task_env.reset_to_demo(planner_demo)
            task_description = args.task_description
            if task_description is None:
                task_description = descriptions[0] if descriptions else ""

            task_steps = _instruction_steps(task_description)
            n_segments = len(task_steps)
            last_expert_idx = max(0, len(planner_obs) - 1)

            gripper_changes = _gripper_change_indices(planner_obs)
            boundaries = _segment_boundaries(gripper_changes, n_segments, last_expert_idx)
            if len(gripper_changes) < max(0, n_segments - 1):
                print(
                    "  Warning: expert gripper transitions fewer than task steps; "
                    "reusing final expert state for remaining segments."
                )

            ranges = _segment_ranges(boundaries, n_segments, last_expert_idx)

            initial_state = capture_task_env_state(task_env)
            policy_start_states: list[dict] = [initial_state]
            for ep_idx in range(1, n_segments):
                boundary_idx = boundaries[ep_idx - 1]
                snap = planner_state_snaps[boundary_idx] if boundary_idx < len(planner_state_snaps) else None
                if snap is None:
                    snap = policy_start_states[-1]
                policy_start_states.append(snap)

            episode_metrics: list[dict] = []

            for ep_idx, step_text in enumerate(task_steps):
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

                if ep_idx == 0:
                    policy_start_obs = initial_obs
                else:
                    policy_start_obs = restore_task_env_state(task_env, policy_start_states[ep_idx])

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
        description="Evaluate trained policy vs RLBench planner for put_rubbish_in_bin."
    )
    parser.add_argument("--task", type=str, default="put_rubbish_in_bin")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=150)

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
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
    parser.add_argument("--save_path", type=str, default="output/rlbench_eval/put_rubbish_in_bin")

    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
