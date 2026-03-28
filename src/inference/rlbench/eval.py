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
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path for local imports.
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

from src.utils.rlbench_policy import LeRobotPolicy
from src.utils.rlbench_eval_utils import (
    capture_task_env_state,
    collect_planner_rollout,
    planner_actions_from_observations,
    restore_task_env_state,
    rollout_policy,
)
from src.utils.rlbench_rollout_io import save_observations, save_rollout_sidecars
from src.utils.rlbench_utils import extract_eef_state, extract_joint_state
from src.utils.rlbench_eval_aggregate import aggregate_batch_results
from src.utils.scene_graph_language import generate_task_descriptions
from src.utils.rlbench_eval_setup import (
    build_action_mode,
    build_obs_config,
    write_csv,
    write_json,
)

_RELATIONSHIP_TASK_STEPS_CACHE: dict[tuple[str, bool], list[str]] = {}
_SUPPORTED_ROBOTS: set[str] = {"panda", "jaco", "mico", "sawyer", "ur5"}
_ROBOT_SETUP_ALIASES: dict[str, str] = {
    "franka": "panda",
    "franka_panda": "panda",
}


def _normalize_robot_setup(robot_setup: str) -> str:
    normalized = str(robot_setup).strip().lower()
    return _ROBOT_SETUP_ALIASES.get(normalized, normalized)


def _validate_action_mode_for_robot(action_mode: str, robot_setup: str) -> None:
    # Joint-state policy was trained specifically for Franka/Panda.
    if robot_setup != "panda" and action_mode == "joint_velocity":
        raise ValueError(
            "Joint-state evaluation (action_mode=joint_velocity) is only supported "
            "for robot_setup=panda (Franka). Use ee_planning/ee_ik for other robots."
        )


def _resolve_task_steps_from_relationship_template(
    task: str,
    rlbench_root: str | Path,
    explicit_template_path: str | None,
    *,
    use_context_prompt: bool = False,
) -> list[str]:
    template_path = _resolve_relationship_template_path(task, rlbench_root, explicit_template_path)
    if template_path is None:
        return [""]

    cache_key = (str(template_path.resolve()), bool(use_context_prompt))
    cached = _RELATIONSHIP_TASK_STEPS_CACHE.get(cache_key)
    if cached is None:
        cached = _infer_task_steps_from_relationship_template(
            template_path,
            task,
            use_context_prompt=use_context_prompt,
        )
        _RELATIONSHIP_TASK_STEPS_CACHE[cache_key] = cached
    return cached if cached else [""]


def _infer_task_steps_from_relationship_template(
    template_path: Path,
    task: str,
    *,
    use_context_prompt: bool = False,
) -> list[str]:
    """Generate ordered per-segment instructions from relationship template."""
    try:
        with open(template_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []

    transitions = data.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        return []

    info_path = template_path.parent / "info.json"
    objects_meta: dict[str, dict[str, str]] = {}
    if info_path.is_file():
        try:
            with open(info_path, "r", encoding="utf-8") as handle:
                info_data = json.load(handle)
            raw_objects = info_data.get("objects", {})
            if isinstance(raw_objects, dict):
                for obj in raw_objects.values():
                    if not isinstance(obj, dict):
                        continue
                    name = str(obj.get("name", "")).strip()
                    if name:
                        objects_meta[name] = {"type": name}
        except Exception:
            objects_meta = {}

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
    cleaned_steps = [str(s).strip() for s in task_steps if str(s).strip()]
    return cleaned_steps


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
        # Boundaries are segment END indices, so next segment starts at end+1.
        # This avoids reusing the transition frame as both previous-end and next-start.
        start = min(end + 1, last_index)
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


def _evaluate_single_task(
    task_name: str,
    env: Environment,
    policy: "LeRobotPolicy",
    args: argparse.Namespace,
    rlbench_root: Path,
    save_path: str,
) -> dict:
    """Run evaluation for one task.  Env and policy stay alive across calls."""

    os.makedirs(save_path, exist_ok=True)

    template_path = _resolve_relationship_template_path(
        task_name, rlbench_root, args.relationship_template
    )
    template_transitions = _load_relationship_transitions(template_path) if template_path else []
    print(
        f"\nSegmentation from relationship template: {template_path} "
        f"({len(template_transitions)} transition(s) → {len(template_transitions)} episode(s))"
    )
    if not template_transitions:
        raise ValueError(f"Relationship template missing or empty: {template_path}")

    task_steps = _resolve_task_steps_from_relationship_template(
        task_name,
        rlbench_root,
        args.relationship_template,
        use_context_prompt=args.with_context_prompt,
    )
    task_description = " ".join(task_steps).strip()

    task_class = task_file_to_task_class(task_name)
    task_env = env.get_task(task_class)

    run_metrics: list[dict] = []
    run_rows: list[dict] = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        task_env.set_variation(args.variation)
        print(f"\n[task={task_name}] [{run_idx + 1}/{args.runs}] seed={run_seed}")

        run_dir = os.path.join(save_path, f"run_{run_idx:03d}")
        os.makedirs(run_dir, exist_ok=True)

        planner_obs, planner_grasped, planner_demo, planner_state_snaps = collect_planner_rollout(
            task_env,
            max_attempts=args.planner_max_attempts,
        )

        _descriptions, _initial_obs = task_env.reset_to_demo(planner_demo)
        last_expert_idx = max(0, len(planner_obs) - 1)
        gripper_changes = _gripper_change_indices(planner_obs)

        n_segments, boundaries = _segmentation_from_relationship_template(
            planner_obs, template_transitions, last_expert_idx
        )

        segment_instructions = _instructions_per_segment(
            n_segments, task_steps, task_description
        )
        ranges = _segment_ranges(boundaries, n_segments, last_expert_idx)

        initial_state = capture_task_env_state(task_env)
        if planner_state_snaps:
            if planner_state_snaps[0] is None:
                planner_state_snaps[0] = initial_state

        offset = max(0, int(args.policy_start_offset))

        episode_metrics: list[dict] = []

        for ep_idx in range(n_segments):
            step_text = segment_instructions[ep_idx]
            seg_start, seg_end = ranges[ep_idx]

            # Advance the effective start by offset frames so the policy
            # begins from a snapshot captured during the live planner run
            # rather than the imperfect reset_to_demo reconstruction.
            effective_start = min(seg_start + offset, seg_end)
            print(f"  ep={ep_idx} instruction={step_text} range=[{seg_start},{seg_end}] effective_start={effective_start}")

            planner_episode_dir = os.path.join(run_dir, "planner", "episodes", f"episode{ep_idx}")
            policy_episode_dir = os.path.join(run_dir, "policy", "episodes", f"episode{ep_idx}")

            planner_ep_obs = planner_obs[effective_start : seg_end + 1]
            planner_ep_grasp = planner_grasped[effective_start : seg_end + 1]

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
                    "effective_start_index": effective_start,
                    "segment_end_index": seg_end,
                    "policy_start_offset": offset,
                },
            )

            policy_start_state = planner_state_snaps[effective_start] if effective_start < len(planner_state_snaps) else None
            policy_start_source = f"planner_state_snaps[{effective_start}]"
            if policy_start_state is None:
                policy_start_state = initial_state
                policy_start_source = "initial_state_fallback"

            if args.debug_snapshot_restore:
                print(
                    f"[restore-select] run={run_idx} ep={ep_idx} seg_start={seg_start} "
                    f"effective_start={effective_start} seg_end={seg_end} "
                    f"source={policy_start_source} "
                    f"snapshots_len={len(planner_state_snaps)}"
                )

            policy_start_obs = restore_task_env_state(
                task_env,
                policy_start_state,
                debug_restore=args.debug_snapshot_restore,
                debug_label=f"run{run_idx}_ep{ep_idx}",
            )

            if args.debug_snapshot_restore and effective_start < len(planner_obs):
                planner_start_obs = planner_obs[effective_start]
                planner_start_eef = extract_eef_state(planner_start_obs).astype(np.float64)
                policy_start_eef = extract_eef_state(policy_start_obs).astype(np.float64)
                planner_start_joint = extract_joint_state(planner_start_obs).astype(np.float64)
                policy_start_joint = extract_joint_state(policy_start_obs).astype(np.float64)
                print(
                    f"[restore-compare] run={run_idx} ep={ep_idx} "
                    f"eef_l2={float(np.linalg.norm(policy_start_eef - planner_start_eef)):.6f} "
                    f"joint_l2={float(np.linalg.norm(policy_start_joint - planner_start_joint)):.6f} "
                    f"gripper_planner={float(getattr(planner_start_obs, 'gripper_open', np.nan)):.4f} "
                    f"gripper_policy={float(getattr(policy_start_obs, 'gripper_open', np.nan)):.4f}"
                )

            policy_rollout = rollout_policy(
                task_env,
                policy,
                initial_obs=policy_start_obs,
                task_description=step_text,
                max_steps_per_instruction=args.max_steps,
                stop_on_success=False,
                binarize_gripper_action=args.binarize_gripper_action,
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
                "robot_setup": args.robot_setup,
                "episode_index": ep_idx,
                "task_description": task_description,
                "instruction": step_text,
                "segment_start_index": seg_start,
                "effective_start_index": effective_start,
                "segment_end_index": seg_end,
                "policy_start_offset": offset,
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
            "robot_setup": args.robot_setup,
            "task_description": task_description,
            "task_steps": task_steps,
            "segment_instructions": segment_instructions,
            "relationship_template_path": str(template_path) if template_path else None,
            "num_template_transitions": len(template_transitions) if template_transitions else None,
            "expert_gripper_change_indices": gripper_changes,
            "segment_boundaries": boundaries,
            "episode_metrics": episode_metrics,
        }
        run_metrics.append(run_metric)
        write_json(os.path.join(run_dir, "metrics.json"), run_metric)

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
        "task": task_name,
        "variation": args.variation,
        "runs": args.runs,
        "seed": args.seed,
        "robot_setup": args.robot_setup,
        "action_mode": args.action_mode,
        "num_episodes": max_episodes,
        "num_instruction_episodes": max_episodes,
        "all_episode_success_rate": _mean_or_none(all_episode_success),
        "per_episode": per_episode_summary,
    }

    write_json(os.path.join(save_path, "summary.json"), summary)
    write_json(os.path.join(save_path, "per_run_metrics.json"), run_metrics)
    write_csv(os.path.join(save_path, "per_run_metrics.csv"), run_rows)

    print(f"\nTask {task_name} evaluation complete.")
    print(f"Saved outputs to: {save_path}")
    print(json.dumps(summary, indent=2))
    return summary


def run_evaluation(args: argparse.Namespace) -> None:
    """Main entry point.  Loads model and env once, then iterates tasks."""

    args.robot_setup = _normalize_robot_setup(args.robot_setup)
    if args.robot_setup not in _SUPPORTED_ROBOTS:
        supported = ", ".join(sorted(_SUPPORTED_ROBOTS))
        raise ValueError(
            f"Unsupported robot_setup='{args.robot_setup}'. Supported: {supported}."
        )
    _validate_action_mode_for_robot(args.action_mode, args.robot_setup)

    task_list: list[str] = []
    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not task_list:
        task_list = [args.task]

    obs_config = build_obs_config(list(args.image_size), args.renderer)
    action_mode = build_action_mode(args.action_mode)

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        arm_max_velocity=args.arm_max_velocity,
        arm_max_acceleration=args.arm_max_acceleration,
        robot_setup=args.robot_setup,
        headless=not args.debug,
    )
    env.launch()
    print(f"RLBench launched (tasks={task_list}, robot_setup={args.robot_setup})")

    policy = LeRobotPolicy(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        task_description="",
        action_mode=args.action_mode,
        device=args.device,
    )

    rlbench_root = Path(args.rlbench_root).resolve() if args.rlbench_root else (_project_root / "datasets" / "rlbench")

    all_summaries: list[dict] = []
    for task_name in task_list:
        task_save_path = os.path.join(args.save_path, task_name) if len(task_list) > 1 else args.save_path
        try:
            summary = _evaluate_single_task(
                task_name, env, policy, args, rlbench_root, task_save_path,
            )
            all_summaries.append(summary)
        except Exception as exc:
            print(f"\n[ERROR] Task {task_name} failed: {exc}")
            all_summaries.append({"task": task_name, "error": str(exc)})

    env.shutdown()

    if len(task_list) > 1:
        write_json(os.path.join(args.save_path, "all_tasks_summary.json"), all_summaries)
        print(f"\nAll {len(task_list)} task(s) complete.  Summary: {args.save_path}/all_tasks_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy vs RLBench planner for an RLBench task."
    )
    parser.add_argument("--task", type=str, default="put_rubbish_in_bin",
                        help="Single task name (use --tasks for multi-task batch).")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task list.  Model + env load once and iterate tasks sequentially.",
    )
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)

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
        "--with_context_prompt",
        action="store_true",
        help="Use ConceptGraphs-style context prompt for per-segment instructions.",
    )
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--action_mode",
        type=str,
        default="joint_velocity",
        choices=["ee_planning", "ee_ik", "joint_velocity"],
    )
    parser.add_argument(
        "--robot_setup",
        type=str,
        default="panda",
        help=(
            "RLBench robot setup. Supports panda, jaco, mico, sawyer, ur5 "
            "(aliases: franka, franka_panda -> panda). "
            "Non-panda robots are restricted to EEF action modes."
        ),
    )
    parser.add_argument("--renderer", type=str, default="opengl", choices=["opengl", "opengl3"])
    parser.add_argument("--image_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--arm_max_velocity", type=float, default=1.0)
    parser.add_argument("--arm_max_acceleration", type=float, default=4.0)
    parser.add_argument("--planner_max_attempts", type=int, default=10)
    parser.add_argument(
        "--binarize_gripper_action",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Snap policy gripper action (last dim) to nearest 0/1 before env.step().",
    )
    parser.add_argument(
        "--policy_start_offset",
        type=int,
        default=1,
        help=(
            "Number of planner frames to skip before the policy starts. "
            "Offset >= 1 ensures the policy restores from a snapshot captured "
            "during the live planner run instead of the imperfect reset_to_demo "
            "reconstruction (default: 1)."
        ),
    )
    parser.add_argument("--save_path", type=str, default="output/rlbench_eval/default_task")
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with RLBench renderer visible and verbose logging.",
    )
    parser.add_argument(
        "--debug_snapshot_restore",
        action="store_true",
        help="Print detailed snapshot/restore diagnostics around policy episode starts.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    run_evaluation(parsed_args)
