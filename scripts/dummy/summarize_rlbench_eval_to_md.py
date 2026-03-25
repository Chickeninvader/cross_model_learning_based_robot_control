#!/usr/bin/env python3
"""
Scan RLBench eval output trees (e.g. output/rlbench_eval_new_*) and write a Markdown
report: per action folder (groot_eef, …), per task, per run (run_000 …), per subtask
(episode segment) with success and pose-error metrics.

Repo cwd: run from repo root or pass absolute paths.

Example:
  python scripts/dummy/summarize_rlbench_eval_to_md.py output/rlbench_eval_new_2
  python scripts/dummy/summarize_rlbench_eval_to_md.py "output/rlbench_eval_new_*" -o /tmp/report.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


METRICS_BLURB = """### Metrics (quick)

**eef_l2** — L2 distance vs expert on the full **7D** EE vector (pos + quat). **pos_l2** — **3D** position only. **rot_deg** — orientation **angle** in degrees (not an L2). Means in the overview are **averages over every run×subtask row** in the logs.
"""


def _r3(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.3f}"


def _success_cell(ok: bool | None) -> str:
    if ok is None:
        return "—"
    return "1" if ok else "0"


def _expand_inputs(raw_paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_paths:
        hits = sorted(glob.glob(raw))
        if hits:
            out.extend(Path(p).resolve() for p in hits)
        elif os.path.exists(raw):
            out.append(Path(raw).resolve())
    # de-dupe preserving order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _parse_metric_path(prm: Path) -> tuple[str, str, str] | None:
    """Return (eval_label, action_dir_name, task_name) or None."""
    parts = prm.parts
    try:
        i = parts.index("all_tasks")
    except ValueError:
        return None
    if i + 2 >= len(parts):
        return None
    n = len(parts)
    eval_root = prm.parents[n - i - 1]
    action = parts[i + 1]
    task = parts[i + 2]
    return eval_root.name, action, task


def _discover_per_run_files(scan_roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in scan_roots:
        root = root.resolve()
        if not root.exists():
            continue
        found.extend(sorted(root.rglob("per_run_metrics.json")))
    return found


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _subtask_title(task: str, episode_index: int) -> str:
    return f"{task.replace('_', ' ')} {episode_index + 1}"


def _mean(nums: list[float]) -> float | None:
    clean = [float(x) for x in nums if x is not None and not (isinstance(x, float) and x != x)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _task_high_level_stats(prm: Path) -> dict | None:
    """Aggregate one task's per_run_metrics (+ optional summary.json) for the overview row."""
    try:
        data = _load_json(prm)
    except OSError:
        return None
    if not isinstance(data, list):
        return None

    eefs: list[float] = []
    poss: list[float] = []
    rots: list[float] = []
    sub_ok: dict[int, list[bool]] = defaultdict(list)
    n_runs = 0

    for run_block in data:
        if not isinstance(run_block, dict):
            continue
        n_runs += 1
        eps = run_block.get("episode_metrics") or []
        if not isinstance(eps, list):
            continue
        for ep in eps:
            if not isinstance(ep, dict):
                continue
            ep_i = int(ep.get("episode_index", 0))
            ps = ep.get("policy_success")
            if isinstance(ps, bool):
                sub_ok[ep_i].append(ps)
            for key, bucket in (
                ("eef_l2", eefs),
                ("pos_l2", poss),
                ("rot_deg", rots),
            ):
                v = ep.get(key)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                    bucket.append(float(v))

    task_succ: float | None = None
    summary_path = prm.parent / "summary.json"
    if summary_path.is_file():
        try:
            summ = _load_json(summary_path)
            if isinstance(summ, dict):
                ts = summ.get("all_episode_success_rate")
                if isinstance(ts, (int, float)) and not (isinstance(ts, float) and ts != ts):
                    task_succ = float(ts)
        except OSError:
            pass

    sub_rates: list[tuple[int, float | None]] = []
    for ep_i in sorted(sub_ok.keys()):
        vals = sub_ok[ep_i]
        r = sum(1 for v in vals if v) / len(vals) if vals else None
        sub_rates.append((ep_i, r))

    sub_compact = "; ".join(f"{i + 1}:{_r3(r)}" for i, r in sub_rates)
    return {
        "n_runs": n_runs,
        "task_succ": task_succ,
        "sub_compact": sub_compact if sub_compact else "—",
        "eef_m": _mean(eefs),
        "pos_m": _mean(poss),
        "rot_m": _mean(rots),
    }


def _build_overview_table(entries: list[tuple[str, str, str, Path]]) -> str:
    rows: list[tuple[str, str, str, dict]] = []
    for eval_label, action, task, prm in sorted(entries, key=lambda t: (t[0], t[1], t[2])):
        stats = _task_high_level_stats(prm)
        if stats is None:
            continue
        rows.append((eval_label, action, task, stats))

    if not rows:
        return "_No per_run_metrics rows found._\n"

    lines = [
        "\n## High-level summary\n\n",
        "| Eval | Policy | Task | runs | task_succ | sub_succ | eef_m | pos_m | rot_m |\n",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |\n",
    ]
    for eval_label, action, task, st in rows:
        lines.append(
            f"| `{eval_label}` | `{action}` | `{task}` | {st['n_runs']} | "
            f"{_r3(st['task_succ'])} | {st['sub_compact']} | "
            f"{_r3(st['eef_m'])} | {_r3(st['pos_m'])} | {_r3(st['rot_m'])} |\n"
        )
    lines.append(
        "\n_task_succ_: all subtasks succeeded in a run, **averaged over runs**. "
        "_sub_succ_: per-segment success rate (`1:0.5` = segment 1 rate 0.5). "
        "_*_m_: mean of logged errors over all segments and runs.\n\n"
        "---\n"
    )
    return "".join(lines)


def build_markdown(entries: list[tuple[str, str, str, Path]]) -> str:
    """
    entries: list of (eval_label, action, task, path_to_per_run_metrics.json)
    """
    lines: list[str] = []
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    lines.append("# RLBench eval report\n")
    lines.append(f"_Generated {now}._\n")
    lines.append(_build_overview_table(entries))
    lines.append("\n## Details\n")
    lines.append(METRICS_BLURB)

    by_eval: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for ev, action, task, prm in entries:
        by_eval[ev].append((action, task, prm))

    for eval_label in sorted(by_eval.keys()):
        lines.append(f"\n### Eval: `{eval_label}`\n")
        by_action: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for action, task, prm in by_eval[eval_label]:
            by_action[action].append((task, prm))

        for action in sorted(by_action.keys()):
            lines.append(f"\n#### Policy: `{action}`\n")
            tasks = sorted(by_action[action], key=lambda x: x[0])
            for task, prm in tasks:
                lines.append(f"\n##### `{task}`\n")
                data = _load_json(prm)
                if not isinstance(data, list):
                    lines.append("_Invalid per_run_metrics.json (expected a list)._\n")
                    continue

                for run_block in data:
                    if not isinstance(run_block, dict):
                        continue
                    run_idx = run_block.get("run_index", "?")
                    seed = run_block.get("seed", "?")
                    lines.append(f"\n###### Run {run_idx} (seed {seed})\n")
                    lines.append(
                        "| Subtask | Instr. | succ | eef | pos | rot° |\n"
                        "| --- | --- | ---: | ---: | ---: | ---: |\n"
                    )
                    eps = run_block.get("episode_metrics") or []
                    if not isinstance(eps, list):
                        eps = []
                    for ep in eps:
                        if not isinstance(ep, dict):
                            continue
                        ep_i = int(ep.get("episode_index", 0))
                        title = _subtask_title(task, ep_i)
                        instr = (ep.get("instruction") or "").replace("|", "\\|")
                        if len(instr) > 50:
                            instr = instr[:47] + "…"
                        lines.append(
                            f"| {title} | {instr} | "
                            f"{_success_cell(ep.get('policy_success'))} | "
                            f"{_r3(ep.get('eef_l2'))} | {_r3(ep.get('pos_l2'))} | "
                            f"{_r3(ep.get('rot_deg'))} |\n"
                        )

    lines.append("\n---\n\n_End of report._\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Markdown report from RLBench eval output (per_run_metrics.json)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Eval root dir(s) or glob(s), e.g. output/rlbench_eval_new_2 or output/rlbench_eval_new_*",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .md path. Default: <first_input>/rlbench_eval_subtasks.md",
    )
    args = parser.parse_args()

    scan_roots = _expand_inputs(args.paths)
    if not scan_roots:
        raise SystemExit("No existing paths resolved from arguments.")

    prm_files = _discover_per_run_files(scan_roots)
    if not args.output:
        out_path = scan_roots[0] / "rlbench_eval_subtasks.md"
    else:
        out_path = Path(args.output).expanduser().resolve()

    entries: list[tuple[str, str, str, Path]] = []
    for prm in prm_files:
        parsed = _parse_metric_path(prm)
        if parsed is None:
            continue
        eval_label, action, task = parsed
        entries.append((eval_label, action, task, prm))

    entries.sort(key=lambda t: (t[0], t[1], t[2], str(t[3])))
    md = build_markdown(entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path} ({len(entries)} task file(s)).")


if __name__ == "__main__":
    main()
