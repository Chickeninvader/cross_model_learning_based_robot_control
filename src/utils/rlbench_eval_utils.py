"""Utilities for RLBench planner-vs-policy evaluation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.utils.rlbench_infer_utils import set_policy_task_description, split_task_description_into_steps
from src.utils.rlbench_utils import extract_eef_state, extract_joint_state


def _states_from_observations(observations: list) -> tuple[np.ndarray, np.ndarray]:
    if len(observations) == 0:
        return (
            np.zeros((0, 8), dtype=np.float32),
            np.zeros((0, 8), dtype=np.float32),
        )
    eef = np.stack([extract_eef_state(obs) for obs in observations], axis=0).astype(np.float32)
    joint = np.stack([extract_joint_state(obs) for obs in observations], axis=0).astype(np.float32)
    return eef, joint


def planner_actions_from_observations(observations: list) -> np.ndarray:
    """Extract planner ``joint_position_action`` from observation.misc.

    Returns
    -------
    np.ndarray
        Array of shape (T, 8), with NaNs where action is unavailable.
    """
    actions: list[np.ndarray] = []
    for obs in observations:
        action = None
        misc = getattr(obs, "misc", None)
        if isinstance(misc, dict):
            action = misc.get("joint_position_action")

        if action is None:
            actions.append(np.full((8,), np.nan, dtype=np.float64))
            continue

        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.shape[0] < 8:
            pad = np.full((8 - arr.shape[0],), np.nan, dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > 8:
            arr = arr[:8]
        actions.append(arr)

    if not actions:
        return np.zeros((0, 8), dtype=np.float64)
    return np.stack(actions, axis=0)


def _pad_or_trim_bool(values: list[bool] | np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.bool_).reshape(-1)
    if arr.shape[0] >= n:
        return arr[:n]
    if arr.shape[0] == 0:
        return np.zeros((n,), dtype=np.bool_)
    out = np.zeros((n,), dtype=np.bool_)
    out[: arr.shape[0]] = arr
    out[arr.shape[0] :] = arr[-1]
    return out


def is_target_grasped(task_env: Any, target_attr: str = "rubbish") -> bool:
    """Return True if the active gripper currently holds the task target object."""
    grasped_objects = list(task_env._robot.gripper.get_grasped_objects())
    if not grasped_objects:
        return False

    target_obj = getattr(task_env._task, target_attr, None)
    if target_obj is None:
        for obj in task_env._task.get_graspable_objects():
            try:
                if target_attr.lower() in obj.get_name().lower():
                    target_obj = obj
                    break
            except Exception:
                continue

    if target_obj is not None:
        try:
            target_handle = target_obj.get_handle()
            for obj in grasped_objects:
                if obj.get_handle() == target_handle:
                    return True
        except Exception:
            pass

    # Name-based fallback when handle comparisons are unavailable.
    for obj in grasped_objects:
        try:
            if target_attr.lower() in obj.get_name().lower():
                return True
        except Exception:
            continue
    return False


def capture_task_env_state(task_env: Any) -> dict[str, Any]:
    """Capture a restorable task+robot simulator state snapshot.

    In addition to CoppeliaSim configuration trees, we also capture
    PyRep's Python-side gripper grasp bookkeeping (``_grasped_objects``
    and ``_old_parents``).  Without this, restoring the sim tree alone
    leaves PyRep unaware of the grasp, so the first physics step after
    restore drops the object.
    """
    gripper = task_env._robot.gripper
    grasped_objects = list(gripper._grasped_objects)
    old_parents = list(gripper._old_parents)

    return {
        "arm": task_env._robot.arm.get_configuration_tree(),
        "gripper": gripper.get_configuration_tree(),
        "task": task_env._task.get_state(),
        "grasped_objects": grasped_objects,
        "old_parents": old_parents,
    }


def restore_task_env_state(
    task_env: Any,
    state: dict[str, Any],
    settle_steps: int = 2,
    *,
    debug_restore: bool = False,
    debug_label: str | None = None,
) -> Any:
    """Restore a previously captured task+robot simulator snapshot and return observation."""
    scene = task_env._scene
    scene.pyrep.set_configuration_tree(state["arm"])
    scene.pyrep.set_configuration_tree(state["gripper"])

    task_state = state["task"]
    tag = f"[restore:{debug_label}] " if debug_label else "[restore] "
    if debug_restore:
        try:
            current_obj_count = len(task_env._task.get_base().get_objects_in_tree(exclude_base=False))
        except Exception:
            current_obj_count = -1
        expected_obj_count = int(task_state[1]) if isinstance(task_state, tuple) and len(task_state) > 1 else -1
        print(
            f"{tag}before task.restore_state expected_objects={expected_obj_count} "
            f"current_objects={current_obj_count} settle_steps={int(settle_steps)}"
        )
    # Restore task tree (direct config-tree to avoid object-count guard).
    if isinstance(task_state, tuple) and len(task_state) > 0:
        scene.pyrep.set_configuration_tree(task_state[0])
    else:
        task_env._task.restore_state(task_state)

    # Restore grasp state.  The task-tree restore above can pull grasped
    # objects back to their original parent, undoing the gripper-tree
    # restore.  We fix this by explicitly re-parenting each grasped
    # object under the gripper attach point AFTER all trees are restored.
    gripper = task_env._robot.gripper
    saved_grasped = state.get("grasped_objects", [])
    saved_parents = state.get("old_parents", [])

    # Clear any stale Python-side grasp state first.
    gripper._grasped_objects = []
    gripper._old_parents = []

    if saved_grasped:
        for obj, old_parent in zip(saved_grasped, saved_parents):
            if not obj.still_exists():
                continue
            gripper._grasped_objects.append(obj)
            gripper._old_parents.append(old_parent)
            # Force object back under gripper attach point (task-tree
            # restore may have moved it to the task hierarchy).
            obj.set_parent(gripper._attach_point, keep_in_place=True)
        if debug_restore:
            names = [o.get_name() for o in gripper._grasped_objects]
            print(f"{tag}restored grasped_objects={names}")

    task_env._robot.arm.set_joint_target_velocities([0] * len(task_env._robot.arm.joints))
    gripper.set_joint_target_velocities([0] * len(gripper.joints))

    # Skip settle steps when holding an object — physics steps can
    # break the freshly-restored contact before the policy even starts.
    effective_settle = 0 if gripper._grasped_objects else settle_steps
    for _ in range(max(0, int(effective_settle))):
        scene.pyrep.step()
        task_env._task.step()

    obs = task_env.get_observation()
    if debug_restore:
        print(f"{tag}after restore gripper_open={float(getattr(obs, 'gripper_open', np.nan)):.4f}")
    return obs


def collect_planner_rollout(
    task_env: Any,
    max_attempts: int = 10,
    *,
    save_dir: str | None = None,
    save_info: dict[str, Any] | None = None,
    save_observations_kwargs: dict[str, Any] | None = None,
) -> tuple[list, list[bool], Any, list[dict[str, Any] | None]]:
    """Collect one live planner demo and aligned per-observation grasp flags.

    If ``save_dir`` is provided, the function also saves:
            - RLBench-style rollout files via ``save_observations``
      - sidecar arrays/json (`rollout_arrays.npz`, `rollout_info.json`)
    """
    grasp_flags_each_step: list[bool] = []
    state_snapshots_each_step: list[dict[str, Any]] = []

    def _on_step(_obs):
        grasp_flags_each_step.append(is_target_grasped(task_env))
        state_snapshots_each_step.append(capture_task_env_state(task_env))

    demo, = task_env.get_demos(
        amount=1,
        live_demos=True,
        max_attempts=max_attempts,
        callable_each_step=_on_step,
    )

    observations = [obs for obs in demo]
    n_obs = len(observations)

    # Callback is not called for the very first observation in scene.get_demo.
    # Start with an explicit False (task starts ungrasped), then align lengths.
    grasp_flags = [False] + grasp_flags_each_step

    state_snapshots: list[dict[str, Any] | None] = [None] + state_snapshots_each_step

    if save_dir is not None:
        from src.utils.rlbench_rollout_io import save_observations, save_rollout_sidecars

        os.makedirs(save_dir, exist_ok=True)
        planner_actions_full = planner_actions_from_observations(observations)
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

        info = {
            "source": "planner",
            "action_source": "observation.misc.joint_position_action",
        }
        if save_info:
            info.update(save_info)

        save_rollout_sidecars(
            observations,
            save_dir,
            actions=planner_actions_step,
            rewards=planner_rewards,
            dones=planner_dones,
            grasped_target=grasp_flags,
            info=info,
        )
        save_observations(
            observations,
            save_dir,
            **(save_observations_kwargs or {}),
        )

    return observations, grasp_flags, demo, state_snapshots


def rollout_policy(
    task_env: Any,
    policy: Any,
    initial_obs: Any,
    task_description: str,
    max_steps_per_instruction: int,
    *,
    stop_on_success: bool = True,
    binarize_gripper_action: bool = True,
) -> dict[str, Any]:
    """Run policy rollout with optional multi-instruction prompt splitting."""
    observations = [initial_obs]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    grasped_target: list[bool] = [is_target_grasped(task_env)]
    errors: list[str] = []

    success = False
    terminated = False

    task_steps = split_task_description_into_steps(task_description)
    for step_text in task_steps:
        set_policy_task_description(policy, step_text)
        policy.reset()

        for _ in range(max_steps_per_instruction):
            action = np.asarray(policy.predict(observations[-1]), dtype=np.float64).reshape(-1)
            if action.shape[0] < 8:
                pad = np.full((8 - action.shape[0],), np.nan, dtype=np.float64)
                action = np.concatenate([action, pad], axis=0)
            elif action.shape[0] > 8:
                action = action[:8]

            # RLBench gripper command is expected to be binary-ish (open/close).
            # Snap to nearest {0,1} to avoid weak half-close commands.
            if binarize_gripper_action and action.shape[0] >= 8 and np.isfinite(action[7]):
                action[7] = float(np.round(np.clip(action[7], 0.0, 1.0)))

            try:
                obs, reward, done = task_env.step(action)
            except Exception as exc:  # nosec: B110
                errors.append(str(exc))
                terminated = True
                break

            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))

            if obs is not None:
                observations.append(obs)
                grasped_target.append(is_target_grasped(task_env))
            else:
                grasped_target.append(grasped_target[-1])

            if reward > 0:
                success = True
            if done or (success and stop_on_success):
                terminated = bool(done)
                break

        if terminated or success:
            break

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "grasped_target": grasped_target,
        "success": bool(success),
        "terminated": bool(terminated),
        "task_steps": task_steps,
        "errors": errors,
    }


def detect_gripper_transitions(gripper_open: np.ndarray, threshold: float = 0.5) -> dict[str, list[int]]:
    """Find close/open transitions over observation index timeline."""
    gripper_open = np.asarray(gripper_open, dtype=np.float64).reshape(-1)
    close_indices: list[int] = []
    open_indices: list[int] = []

    for idx in range(1, gripper_open.shape[0]):
        prev_is_open = gripper_open[idx - 1] > threshold
        curr_is_open = gripper_open[idx] > threshold
        if prev_is_open and (not curr_is_open):
            close_indices.append(idx)
        elif (not prev_is_open) and curr_is_open:
            open_indices.append(idx)

    return {"close": close_indices, "open": open_indices}


def first_pick_index(
    gripper_open: np.ndarray,
    grasped_target: np.ndarray,
    threshold: float = 0.5,
) -> int | None:
    """Pick boundary: first close transition where target is actually grasped."""
    transitions = detect_gripper_transitions(gripper_open, threshold=threshold)
    close_indices = transitions["close"]
    if not close_indices:
        return None

    for idx in close_indices:
        if idx < grasped_target.shape[0] and bool(grasped_target[idx]):
            return int(idx)

    return int(close_indices[0])


def first_release_index(
    gripper_open: np.ndarray,
    *,
    start_idx: int | None,
    threshold: float = 0.5,
) -> int | None:
    transitions = detect_gripper_transitions(gripper_open, threshold=threshold)
    for idx in transitions["open"]:
        if start_idx is None or idx > start_idx:
            return int(idx)
    return None


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
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


def _quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Backward-compatible alias for existing private call sites."""
    return quat_angle_deg(q_a, q_b)


def _aligned_mean_l2(seq_a: np.ndarray, seq_b: np.ndarray) -> float | None:
    n = min(seq_a.shape[0], seq_b.shape[0])
    if n <= 0:
        return None
    deltas = np.linalg.norm(seq_a[:n] - seq_b[:n], axis=1)
    return float(np.mean(deltas))


def compute_put_rubbish_metrics(
    planner_observations: list,
    planner_grasped_target: list[bool] | np.ndarray,
    policy_observations: list,
    policy_grasped_target: list[bool] | np.ndarray,
    policy_success: bool,
    *,
    gripper_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute two-phase metrics for put_rubbish_in_bin evaluation."""
    planner_eef, planner_joint = _states_from_observations(planner_observations)
    policy_eef, policy_joint = _states_from_observations(policy_observations)

    planner_grasped = _pad_or_trim_bool(planner_grasped_target, planner_eef.shape[0])
    policy_grasped = _pad_or_trim_bool(policy_grasped_target, policy_eef.shape[0])

    planner_gripper = planner_eef[:, 7] if planner_eef.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
    policy_gripper = policy_eef[:, 7] if policy_eef.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

    planner_pick_idx = first_pick_index(planner_gripper, planner_grasped, threshold=gripper_threshold)
    policy_pick_idx = first_pick_index(policy_gripper, policy_grasped, threshold=gripper_threshold)

    planner_release_idx = first_release_index(
        planner_gripper,
        start_idx=planner_pick_idx,
        threshold=gripper_threshold,
    )
    policy_release_idx = first_release_index(
        policy_gripper,
        start_idx=policy_pick_idx,
        threshold=gripper_threshold,
    )

    phase1_eef_l2: float | None = None
    phase1_joint_l2: float | None = None
    phase1_pos_l2: float | None = None
    phase1_rot_deg: float | None = None

    if planner_pick_idx is not None and policy_pick_idx is not None:
        planner_pick_eef = planner_eef[planner_pick_idx]
        policy_pick_eef = policy_eef[policy_pick_idx]
        planner_pick_joint = planner_joint[planner_pick_idx]
        policy_pick_joint = policy_joint[policy_pick_idx]

        phase1_eef_l2 = _l2(planner_pick_eef, policy_pick_eef)
        phase1_joint_l2 = _l2(planner_pick_joint, policy_pick_joint)
        phase1_pos_l2 = _l2(planner_pick_eef[:3], policy_pick_eef[:3])
        phase1_rot_deg = _quat_angle_deg(planner_pick_eef[3:7], policy_pick_eef[3:7])

    planner_phase2_start = planner_pick_idx if planner_pick_idx is not None else 0
    policy_phase2_start = policy_pick_idx if policy_pick_idx is not None else 0

    planner_phase2_eef = planner_eef[planner_phase2_start:]
    policy_phase2_eef = policy_eef[policy_phase2_start:]
    planner_phase2_joint = planner_joint[planner_phase2_start:]
    policy_phase2_joint = policy_joint[policy_phase2_start:]

    phase2_traj_eef_l2_mean = _aligned_mean_l2(policy_phase2_eef, planner_phase2_eef)
    phase2_traj_joint_l2_mean = _aligned_mean_l2(policy_phase2_joint, planner_phase2_joint)

    phase2_final_eef_l2: float | None = None
    phase2_final_joint_l2: float | None = None
    if planner_phase2_eef.shape[0] > 0 and policy_phase2_eef.shape[0] > 0:
        phase2_final_eef_l2 = _l2(policy_phase2_eef[-1], planner_phase2_eef[-1])
        phase2_final_joint_l2 = _l2(policy_phase2_joint[-1], planner_phase2_joint[-1])

    policy_pick_grasped = (
        bool(policy_grasped[policy_pick_idx])
        if (policy_pick_idx is not None and policy_pick_idx < policy_grasped.shape[0])
        else False
    )

    policy_release_detected = policy_release_idx is not None
    policy_release_after_pick = (
        policy_release_detected
        and (policy_pick_idx is None or (policy_release_idx is not None and policy_release_idx > policy_pick_idx))
    )
    policy_release_while_holding = False
    if policy_release_idx is not None and policy_release_idx > 0:
        hold_idx = min(policy_release_idx - 1, policy_grasped.shape[0] - 1)
        policy_release_while_holding = bool(policy_grasped[hold_idx])

    return {
        "phase_indices": {
            "planner_pick_idx": planner_pick_idx,
            "policy_pick_idx": policy_pick_idx,
            "planner_release_idx": planner_release_idx,
            "policy_release_idx": policy_release_idx,
        },
        "phase1": {
            "policy_pick_event_detected": policy_pick_idx is not None,
            "policy_pick_grasped_target": policy_pick_grasped,
            "eef_l2": phase1_eef_l2,
            "joint_l2": phase1_joint_l2,
            "pos_l2": phase1_pos_l2,
            "rot_deg": phase1_rot_deg,
        },
        "phase2": {
            "policy_release_event_detected": policy_release_detected,
            "policy_release_after_pick": bool(policy_release_after_pick),
            "policy_release_while_holding": bool(policy_release_while_holding),
            "traj_eef_l2_mean": phase2_traj_eef_l2_mean,
            "traj_joint_l2_mean": phase2_traj_joint_l2_mean,
            "final_eef_l2": phase2_final_eef_l2,
            "final_joint_l2": phase2_final_joint_l2,
        },
        "final": {
            "policy_success": bool(policy_success),
        },
    }


def _get_nested(dct: dict[str, Any], *keys: str) -> Any:
    current: Any = dct
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "n": int(arr.shape[0]),
    }


def aggregate_put_rubbish_metrics(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-run metrics into report-friendly summary statistics."""

    def collect_bool_rate(*path: str) -> float | None:
        vals: list[bool] = []
        for metric in run_metrics:
            value = _get_nested(metric, *path)
            if value is None:
                continue
            vals.append(bool(value))
        if not vals:
            return None
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    def collect_float_summary(*path: str) -> dict[str, Any]:
        vals: list[float] = []
        for metric in run_metrics:
            value = _get_nested(metric, *path)
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            vals.append(value)
        return _float_summary(vals)

    return {
        "num_runs": len(run_metrics),
        "rates": {
            "policy_success_rate": collect_bool_rate("final", "policy_success"),
            "phase1_pick_event_rate": collect_bool_rate("phase1", "policy_pick_event_detected"),
            "phase1_pick_grasp_rate": collect_bool_rate("phase1", "policy_pick_grasped_target"),
            "phase2_release_event_rate": collect_bool_rate("phase2", "policy_release_event_detected"),
            "phase2_release_after_pick_rate": collect_bool_rate("phase2", "policy_release_after_pick"),
            "phase2_release_while_holding_rate": collect_bool_rate("phase2", "policy_release_while_holding"),
        },
        "distance": {
            "phase1_eef_l2": collect_float_summary("phase1", "eef_l2"),
            "phase1_joint_l2": collect_float_summary("phase1", "joint_l2"),
            "phase1_pos_l2": collect_float_summary("phase1", "pos_l2"),
            "phase1_rot_deg": collect_float_summary("phase1", "rot_deg"),
            "phase2_traj_eef_l2_mean": collect_float_summary("phase2", "traj_eef_l2_mean"),
            "phase2_traj_joint_l2_mean": collect_float_summary("phase2", "traj_joint_l2_mean"),
            "phase2_final_eef_l2": collect_float_summary("phase2", "final_eef_l2"),
            "phase2_final_joint_l2": collect_float_summary("phase2", "final_joint_l2"),
        },
    }


def flatten_put_rubbish_run_metric(run_metric: dict[str, Any]) -> dict[str, Any]:
    """Flatten one run's nested metric dict for CSV export."""
    return {
        "run_index": run_metric.get("run_index"),
        "seed": run_metric.get("seed"),
        "variation": run_metric.get("variation"),
        "policy_success": _get_nested(run_metric, "final", "policy_success"),
        "phase1_pick_event": _get_nested(run_metric, "phase1", "policy_pick_event_detected"),
        "phase1_pick_grasped": _get_nested(run_metric, "phase1", "policy_pick_grasped_target"),
        "phase1_eef_l2": _get_nested(run_metric, "phase1", "eef_l2"),
        "phase1_joint_l2": _get_nested(run_metric, "phase1", "joint_l2"),
        "phase1_pos_l2": _get_nested(run_metric, "phase1", "pos_l2"),
        "phase1_rot_deg": _get_nested(run_metric, "phase1", "rot_deg"),
        "phase2_release_event": _get_nested(run_metric, "phase2", "policy_release_event_detected"),
        "phase2_release_after_pick": _get_nested(run_metric, "phase2", "policy_release_after_pick"),
        "phase2_release_while_holding": _get_nested(run_metric, "phase2", "policy_release_while_holding"),
        "phase2_traj_eef_l2_mean": _get_nested(run_metric, "phase2", "traj_eef_l2_mean"),
        "phase2_traj_joint_l2_mean": _get_nested(run_metric, "phase2", "traj_joint_l2_mean"),
        "phase2_final_eef_l2": _get_nested(run_metric, "phase2", "final_eef_l2"),
        "phase2_final_joint_l2": _get_nested(run_metric, "phase2", "final_joint_l2"),
    }
