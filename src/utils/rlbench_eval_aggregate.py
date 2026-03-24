from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from src.utils.rlbench_eval_helpers import (
    mean_or_none,
    parse_bool,
    parse_float,
    parse_int,
    std_or_none,
)
from src.utils.rlbench_eval_setup import discover_task_roots


def _rank_key(row: dict) -> tuple[float, float]:
    return (
        -(row.get("all_episode_success_rate_mean") or -1.0),
        row.get("joint_l2_mean") if row.get("joint_l2_mean") is not None else float("inf"),
    )


def _group_runs(run_rows: list[dict], episode_rows: list[dict], key: str) -> list[dict]:
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
                "episode_success_rate_mean": mean_or_none(
                    [r.get("episode_success_rate") for r in rows]
                ),
                "episode_success_rate_std": std_or_none(
                    [r.get("episode_success_rate") for r in rows]
                ),
                "episode_success_rate_overall": mean_or_none(
                    [float(bool(ep.get("policy_success", False))) for ep in group_episodes]
                ),
                "all_episode_success_rate_mean": mean_or_none(
                    [r.get("all_episode_success_rate") for r in rows]
                ),
                "joint_l2_mean": mean_or_none([r.get("joint_l2_mean") for r in rows]),
                "joint_l2_overall": mean_or_none([ep.get("joint_l2") for ep in group_episodes]),
                "pos_l2_mean": mean_or_none([r.get("pos_l2_mean") for r in rows]),
                "pos_l2_overall": mean_or_none([ep.get("pos_l2") for ep in group_episodes]),
                "rot_deg_mean": mean_or_none([r.get("rot_deg_mean") for r in rows]),
                "rot_deg_overall": mean_or_none([ep.get("rot_deg") for ep in group_episodes]),
            }
        )
    summaries.sort(key=_rank_key)
    return summaries


def aggregate_batch_results(aggregate_root: str, task_name: str = "rlbench_task") -> dict:
    """Aggregate already-generated batch outputs without rerunning inference."""
    root = Path(aggregate_root)
    task_roots = discover_task_roots(root, fallback_task_name=task_name)
    if not task_roots:
        return {
            "task": task_name,
            "aggregate_root": str(root),
            "num_eval_dirs": 0,
            "num_runs": 0,
            "num_episode_rows": 0,
            "run_details": [],
            "per_run_details": [],
            "comparison": {
                "overall": {},
                "by_task": [],
                "by_policy_state": [],
                "by_policy": [],
                "by_state": [],
            },
            "overall_metrics_rows": [],
        }

    run_summaries: list[dict] = []
    run_rows: list[dict] = []
    episode_rows: list[dict] = []
    var_seed_re = re.compile(r"^var(-?\d+)_seed(-?\d+)$")

    for task, task_root in task_roots:
        policy_state_dirs = sorted([p for p in task_root.iterdir() if p.is_dir()])
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

                task_description_by_run: dict[int, str] = {}
                metrics_json_path = eval_dir / "metrics.json"
                if metrics_json_path.exists():
                    try:
                        with open(metrics_json_path, "r", encoding="utf-8") as handle:
                            metrics_payload = json.load(handle)
                        desc = str(metrics_payload.get("task_description", "")).strip()
                        if desc:
                            for ep in metrics_payload.get("episode_metrics", []):
                                run_idx = parse_int(ep.get("run_index"))
                                if run_idx is not None:
                                    task_description_by_run[run_idx] = desc
                    except Exception as exc:
                        print(f"Warning: failed reading {metrics_json_path}: {exc}")

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
                        run_index = 0

                    enriched_episode = {
                        "task": task,
                        "policy": policy,
                        "state": state,
                        "variation": variation,
                        "seed": seed,
                        "run_index": run_index,
                        "episode_index": parse_int(row.get("episode_index")),
                        "task_description": (
                            str(row.get("task_description")).strip()
                            if row.get("task_description") is not None
                            else task_description_by_run.get(run_index, "")
                        ),
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
                        "task": task,
                        "policy": policy,
                        "state": state,
                        "variation": variation,
                        "seed": seed,
                        "run_index": run_index,
                        "task_description": (
                            task_description_by_run.get(run_index, "")
                            or next(
                                (
                                    str(ep.get("task_description", "")).strip()
                                    for ep in run_eps
                                    if str(ep.get("task_description", "")).strip()
                                ),
                                "",
                            )
                        ),
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

                eval_episode_rows = [ep for ep in episode_rows if ep["eval_dir"] == str(eval_dir)]
                run_summaries.append(
                    {
                        "task": task,
                        "policy": policy,
                        "state": state,
                        "variation": variation,
                        "seed": seed,
                        "task_description": (
                            next(
                                (
                                    str(ep.get("task_description", "")).strip()
                                    for ep in eval_episode_rows
                                    if str(ep.get("task_description", "")).strip()
                                ),
                                "",
                            )
                        ),
                        "num_runs": len(eval_run_metrics),
                        "num_episodes": len(rows),
                        "episode_success_rate_mean": mean_or_none(
                            [r.get("episode_success_rate") for r in eval_run_metrics]
                        ),
                        "episode_success_rate_overall": mean_or_none(
                            [float(bool(ep.get("policy_success", False))) for ep in eval_episode_rows]
                        ),
                        "all_episode_success_rate_mean": mean_or_none(
                            [r.get("all_episode_success_rate") for r in eval_run_metrics]
                        ),
                        "joint_l2_mean": mean_or_none([r.get("joint_l2_mean") for r in eval_run_metrics]),
                        "joint_l2_overall": mean_or_none([ep.get("joint_l2") for ep in eval_episode_rows]),
                        "pos_l2_mean": mean_or_none([r.get("pos_l2_mean") for r in eval_run_metrics]),
                        "pos_l2_overall": mean_or_none([ep.get("pos_l2") for ep in eval_episode_rows]),
                        "rot_deg_mean": mean_or_none([r.get("rot_deg_mean") for r in eval_run_metrics]),
                        "rot_deg_overall": mean_or_none([ep.get("rot_deg") for ep in eval_episode_rows]),
                        "eval_dir": str(eval_dir),
                    }
                )

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
                "episode_success_rate_mean": mean_or_none([r.get("episode_success_rate") for r in rows]),
                "episode_success_rate_std": std_or_none([r.get("episode_success_rate") for r in rows]),
                "episode_success_rate_overall": mean_or_none(
                    [float(bool(ep.get("policy_success", False))) for ep in group_episodes]
                ),
                "all_episode_success_rate_mean": mean_or_none(
                    [r.get("all_episode_success_rate") for r in rows]
                ),
                "joint_l2_mean": mean_or_none([r.get("joint_l2_mean") for r in rows]),
                "joint_l2_overall": mean_or_none([ep.get("joint_l2") for ep in group_episodes]),
                "pos_l2_mean": mean_or_none([r.get("pos_l2_mean") for r in rows]),
                "pos_l2_overall": mean_or_none([ep.get("pos_l2") for ep in group_episodes]),
                "rot_deg_mean": mean_or_none([r.get("rot_deg_mean") for r in rows]),
                "rot_deg_overall": mean_or_none([ep.get("rot_deg") for ep in group_episodes]),
            }
        )
    by_policy_state.sort(key=_rank_key)

    by_task_groups: dict[str, list[dict]] = {}
    for row in run_rows:
        by_task_groups.setdefault(str(row.get("task", task_name)), []).append(row)
    by_task: list[dict] = []
    for task_key, rows in by_task_groups.items():
        group_eval_dirs = {str(r["eval_dir"]) for r in rows}
        group_episodes = [ep for ep in episode_rows if str(ep.get("task", task_name)) == task_key]
        by_task.append(
            {
                "task": task_key,
                "num_eval_dirs": len(group_eval_dirs),
                "num_runs": len(rows),
                "num_episodes": len(group_episodes),
                "episode_success_rate_mean": mean_or_none([r.get("episode_success_rate") for r in rows]),
                "episode_success_rate_std": std_or_none([r.get("episode_success_rate") for r in rows]),
                "episode_success_rate_overall": mean_or_none(
                    [float(bool(ep.get("policy_success", False))) for ep in group_episodes]
                ),
                "all_episode_success_rate_mean": mean_or_none(
                    [r.get("all_episode_success_rate") for r in rows]
                ),
                "joint_l2_mean": mean_or_none([r.get("joint_l2_mean") for r in rows]),
                "joint_l2_overall": mean_or_none([ep.get("joint_l2") for ep in group_episodes]),
                "pos_l2_mean": mean_or_none([r.get("pos_l2_mean") for r in rows]),
                "pos_l2_overall": mean_or_none([ep.get("pos_l2") for ep in group_episodes]),
                "rot_deg_mean": mean_or_none([r.get("rot_deg_mean") for r in rows]),
                "rot_deg_overall": mean_or_none([ep.get("rot_deg") for ep in group_episodes]),
            }
        )
    by_task.sort(
        key=lambda r: (
            *_rank_key(r),
            str(r.get("task", "")),
        )
    )

    action_mode_map = {
        "eef": "ee_planning",
        "joint": "joint_velocity",
    }
    combo_episode_groups: dict[tuple[str, str, str, int], list[dict]] = {}
    for ep in episode_rows:
        ep_idx = ep.get("episode_index")
        if ep_idx is None:
            continue
        key = (
            str(ep.get("task", task_name)),
            str(ep.get("policy")),
            str(ep.get("state")),
            int(ep_idx),
        )
        combo_episode_groups.setdefault(key, []).append(ep)

    overall_metrics_rows: list[dict] = []
    for (task, policy, state, episode_index), rows in sorted(
        combo_episode_groups.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3]),
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
                "task": task,
                "policy": policy,
                "state": state,
                "action_mode": action_mode_map.get(state, "unknown"),
                "episode_index": episode_index,
                "num_eval_dirs": len(eval_dirs),
                "num_runs": len(run_keys),
                "num_episode_rows": len(rows),
                "episode_success_rate": mean_or_none(
                    [float(bool(ep.get("policy_success", False))) for ep in rows]
                ),
                "joint_l2_mean": mean_or_none([ep.get("joint_l2") for ep in rows]),
                "pos_l2_mean": mean_or_none([ep.get("pos_l2") for ep in rows]),
                "rot_deg_mean": mean_or_none([ep.get("rot_deg") for ep in rows]),
                "eef_l2_mean": mean_or_none([ep.get("eef_l2") for ep in rows]),
            }
        )

    overall = {
        "num_tasks": len({str(r.get("task", task_name)) for r in run_rows}),
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
            key=lambda r: (
                str(r.get("task", task_name)),
                str(r["policy"]),
                str(r["state"]),
                int(r["variation"]),
                int(r["seed"]),
            ),
        ),
        "per_run_details": sorted(
            run_rows,
            key=lambda r: (
                str(r.get("task", task_name)),
                str(r["policy"]),
                str(r["state"]),
                int(r["variation"]),
                int(r["seed"]),
                int(r["run_index"]),
            ),
        ),
        "comparison": {
            "overall": overall,
            "by_task": by_task,
            "by_policy_state": by_policy_state,
            "by_policy": _group_runs(run_rows, episode_rows, "policy"),
            "by_state": _group_runs(run_rows, episode_rows, "state"),
        },
        "overall_metrics_rows": overall_metrics_rows,
    }
