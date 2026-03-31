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
from src.utils.rlbench_eval_task_utils import (
    SUPPORTED_ROBOTS,
    gripper_change_indices,
    instructions_per_segment,
    load_relationship_transitions,
    mean_or_none,
    normalize_robot_setup,
    resolve_relationship_template_path,
    resolve_task_steps_from_relationship_template,
    runtime_objects_meta_from_observations,
    segment_ranges,
    segment_state_metric,
    segmentation_from_relationship_template,
    task_steps_from_transitions,
    validate_action_mode_for_robot,
)
from src.utils.rlbench_eval_setup import (
    build_action_mode,
    build_obs_config,
    infer_expected_with_context_prompt,
    load_checkpoint_train_dataset_metadata,
    write_csv,
    write_json,
)
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
    eval_metadata: dict[str, object],
) -> dict:
    """Run evaluation for one task.  Env and policy stay alive across calls."""

    os.makedirs(save_path, exist_ok=True)

    template_path = resolve_relationship_template_path(
        task_name, rlbench_root, args.relationship_template
    )
    template_transitions = load_relationship_transitions(template_path) if template_path else []
    print(
        f"\nSegmentation from relationship template: {template_path} "
        f"({len(template_transitions)} transition(s) → {len(template_transitions)} episode(s))"
    )
    if not template_transitions:
        raise ValueError(f"Relationship template missing or empty: {template_path}")

    fallback_task_steps = resolve_task_steps_from_relationship_template(
        task_name,
        rlbench_root,
        args.relationship_template,
        use_context_prompt=args.with_context_prompt,
        variation=args.variation,
    )

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

        runtime_objects_meta = runtime_objects_meta_from_observations(
            task=task_name,
            rlbench_root=rlbench_root,
            variation=args.variation,
            planner_obs=planner_obs,
        )
        runtime_task_steps = task_steps_from_transitions(
            template_transitions,
            runtime_objects_meta,
            task_name,
            use_context_prompt=args.with_context_prompt,
        )
        task_steps = runtime_task_steps if runtime_task_steps else fallback_task_steps
        task_description = " ".join(task_steps).strip()
        _descriptions, _initial_obs = task_env.reset_to_demo(planner_demo)
        last_expert_idx = max(0, len(planner_obs) - 1)
        gripper_changes = gripper_change_indices(planner_obs)

        n_segments, boundaries = segmentation_from_relationship_template(
            planner_obs, template_transitions, last_expert_idx
        )
        segment_instructions = instructions_per_segment(
            n_segments, task_steps, task_description
        )
        ranges = segment_ranges(boundaries, n_segments, last_expert_idx)

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
            dist = segment_state_metric(expert_final_obs, policy_final_obs)

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
            "with_context_prompt": bool(args.with_context_prompt),
            "requested_with_context_prompt": eval_metadata.get("requested_with_context_prompt"),
            "expected_with_context_prompt": eval_metadata.get("expected_with_context_prompt"),
            "prompt_inference_source": eval_metadata.get("inference_source"),
            "checkpoint": args.checkpoint,
            "dataset_root": args.dataset_root,
            "pretrained_model_dir": eval_metadata.get("pretrained_model_dir"),
            "train_config_path": eval_metadata.get("train_config_path"),
            "train_dataset_repo_id": eval_metadata.get("train_dataset_repo_id"),
            "train_dataset_root": eval_metadata.get("train_dataset_root"),
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
                "policy_success_rate": mean_or_none([float(bool(ep["policy_success"])) for ep in ep_rows]),
                "eef_l2_mean": mean_or_none([ep.get("eef_l2") for ep in ep_rows]),
                "joint_l2_mean": mean_or_none([ep.get("joint_l2") for ep in ep_rows]),
                "pos_l2_mean": mean_or_none([ep.get("pos_l2") for ep in ep_rows]),
                "rot_deg_mean": mean_or_none([ep.get("rot_deg") for ep in ep_rows]),
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
        "with_context_prompt": bool(args.with_context_prompt),
        "requested_with_context_prompt": eval_metadata.get("requested_with_context_prompt"),
        "expected_with_context_prompt": eval_metadata.get("expected_with_context_prompt"),
        "prompt_inference_source": eval_metadata.get("inference_source"),
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "pretrained_model_dir": eval_metadata.get("pretrained_model_dir"),
        "train_config_path": eval_metadata.get("train_config_path"),
        "train_dataset_repo_id": eval_metadata.get("train_dataset_repo_id"),
        "train_dataset_root": eval_metadata.get("train_dataset_root"),
        "num_episodes": max_episodes,
        "num_instruction_episodes": max_episodes,
        "all_episode_success_rate": mean_or_none(all_episode_success),
        "per_episode": per_episode_summary,
    }

    write_json(os.path.join(save_path, "summary.json"), summary)
    write_json(os.path.join(save_path, "per_run_metrics.json"), run_metrics)
    write_csv(os.path.join(save_path, "per_run_metrics.csv"), run_rows)

    print(f"\nTask {task_name} evaluation complete.")
    print(f"Saved outputs to: {save_path}")
    print(json.dumps(summary, indent=2))
    return summary


def _norm_compare_path(path: str | None) -> str:
    if not path:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def _load_task_summary_json(task_save_path: str) -> dict | None:
    summary_path = os.path.join(task_save_path, "summary.json")
    if not os.path.isfile(summary_path):
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _existing_summary_matches_args(
    summary: dict,
    task_name: str,
    args: argparse.Namespace,
) -> bool:
    """True if on-disk summary.json matches current eval knobs (safe to skip re-run)."""
    if summary.get("task") != task_name:
        return False
    try:
        if int(summary.get("runs", -1)) != int(args.runs):
            return False
        if int(summary.get("variation", -(2**31))) != int(args.variation):
            return False
        if int(summary.get("seed", -(2**31))) != int(args.seed):
            return False
    except (TypeError, ValueError):
        return False
    if summary.get("robot_setup") != args.robot_setup:
        return False
    if summary.get("action_mode") != args.action_mode:
        return False
    if bool(summary.get("with_context_prompt")) != bool(args.with_context_prompt):
        return False
    if args.checkpoint is not None:
        ck_s = summary.get("checkpoint")
        if ck_s is None:
            return False
        if _norm_compare_path(str(ck_s)) != _norm_compare_path(args.checkpoint):
            return False
    if args.dataset_root is not None:
        ds_s = summary.get("dataset_root")
        if ds_s is None:
            return False
        if _norm_compare_path(str(ds_s)) != _norm_compare_path(args.dataset_root):
            return False
    return True


def run_evaluation(args: argparse.Namespace) -> None:
    """Main entry point.  Loads model and env once, then iterates tasks."""

    if args.aggregate_only or args.aggregate_root:
        if not args.aggregate_root:
            raise ValueError("--aggregate_only requires --aggregate_root.")
        aggregate_root = Path(args.aggregate_root).expanduser()
        report = aggregate_batch_results(str(aggregate_root), task_name=args.task)

        output_json = (
            Path(args.aggregate_output_json).expanduser()
            if args.aggregate_output_json
            else aggregate_root / "detailed_summary.json"
        )
        output_csv = (
            Path(args.aggregate_output_csv).expanduser()
            if args.aggregate_output_csv
            else aggregate_root / "detailed_runs.csv"
        )
        output_overall_csv = (
            Path(args.aggregate_output_overall_csv).expanduser()
            if args.aggregate_output_overall_csv
            else aggregate_root / "overall_metrics.csv"
        )

        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_overall_csv.parent.mkdir(parents=True, exist_ok=True)
        write_json(str(output_json), report)
        write_csv(str(output_csv), list(report.get("per_run_details", [])))
        write_csv(str(output_overall_csv), list(report.get("overall_metrics_rows", [])))
        print(f"Aggregate summary JSON: {output_json}")
        print(f"Aggregate per-run CSV: {output_csv}")
        print(f"Aggregate overall CSV: {output_overall_csv}")
        return

    args.robot_setup = normalize_robot_setup(args.robot_setup)
    if args.robot_setup not in SUPPORTED_ROBOTS:
        supported = ", ".join(sorted(SUPPORTED_ROBOTS))
        raise ValueError(
            f"Unsupported robot_setup='{args.robot_setup}'. Supported: {supported}."
        )
    validate_action_mode_for_robot(args.action_mode, args.robot_setup)

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
    eval_metadata: dict[str, object] = {
        "requested_with_context_prompt": bool(args.with_context_prompt),
        "checkpoint_path_input": args.checkpoint,
        "dataset_root_input": args.dataset_root,
    }
    eval_metadata.update(load_checkpoint_train_dataset_metadata(args.checkpoint))
    prompt_mode_meta = infer_expected_with_context_prompt(
        dataset_root=args.dataset_root,
        train_dataset_root=eval_metadata.get("train_dataset_root"),
    )
    eval_metadata.update(prompt_mode_meta)
    expected_with_context = prompt_mode_meta.get("expected_with_context_prompt")
    if isinstance(expected_with_context, bool) and bool(args.with_context_prompt) != expected_with_context:
        print(
            "[eval] with_context_prompt mismatch detected; "
            f"requested={bool(args.with_context_prompt)} expected={expected_with_context}. "
            "Auto-aligning to expected mode from dataset metadata."
        )
        args.with_context_prompt = expected_with_context
    eval_metadata["with_context_prompt"] = bool(args.with_context_prompt)
    all_summaries: list[dict] = []
    for task_name in task_list:
        task_save_path = os.path.join(args.save_path, task_name) if len(task_list) > 1 else args.save_path
        if args.skip_completed:
            cached = _load_task_summary_json(task_save_path)
            if cached is not None and _existing_summary_matches_args(cached, task_name, args):
                print(
                    f"\n[skip_completed] Task {task_name}: reusing {task_save_path}/summary.json "
                    f"(runs={args.runs} variation={args.variation} seed={args.seed})."
                )
                all_summaries.append(cached)
                continue
            if cached is not None:
                print(
                    f"\n[skip_completed] Task {task_name}: existing summary.json does not match "
                    "current settings; re-evaluating."
                )
        try:
            summary = _evaluate_single_task(
                task_name, env, policy, args, rlbench_root, task_save_path, eval_metadata,
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
            "RLBench robot: panda, jaco, mico, sawyer, ur5 "
            "(aliases: franka, franka_panda -> panda). "
            "Non-panda arms cannot use joint_velocity; use ee_planning or ee_ik. "
            "scripts/core/eval.sh passes this via --robot_setup."
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
        "--skip_completed",
        action="store_true",
        help=(
            "If task output dir already has summary.json matching this run's "
            "task, runs, variation, seed, robot_setup, action_mode, with_context_prompt, "
            "checkpoint, and dataset_root, skip re-evaluating that task."
        ),
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip policy/env evaluation and only aggregate existing outputs from --aggregate_root.",
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
