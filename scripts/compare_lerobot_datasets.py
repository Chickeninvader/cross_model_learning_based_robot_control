#!/usr/bin/env python3
"""Compare (and sanity-check) two LeRobot v3 dataset folders.

This is meant for quickly answering:
- "What is missing or extra compared to a standard dataset?"
- "Is my dataset internally consistent (info.json ↔ parquet schemas ↔ videos)?"

It compares:
- Directory layout + required files
- meta/info.json (top-level keys + per-feature specs)
- meta/stats.json (present feature stats + available quantiles)
- Parquet schemas for:
  - data/chunk-*/file-*.parquet
  - meta/episodes/chunk-*/file-*.parquet
  - meta/tasks.parquet

It also runs per-dataset *internal* consistency checks:
- Every non-video feature in info.json exists as a parquet column
- Parquet physical dtypes match the declared info.json dtypes
- Vector features (e.g. shape [8]) really have list length 8 (sampled)
- Episode metadata references existing parquet/video files (sampled)

Usage
-----
python scripts/compare_lerobot_datasets.py \
  --a datasets/lerobot_without_prompt/put_rubbish_in_bin_all_variation0 \
  --b datasets/lerobot/multi_task_1_variation0 \
  --check-videos

Exit codes
----------
- 0: ran successfully (even if differences found)
- 2: missing/invalid inputs (unless --no-strict)
- 3: strict mode failures (errors found)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# -----------------------------
# Small formatting helpers
# -----------------------------

def _p(text: str) -> None:
    print(text)


def _indent(lines: Iterable[str], prefix: str = "  ") -> str:
    return "\n".join(prefix + ln for ln in lines)


def _sorted_str(items: Iterable[str]) -> list[str]:
    return sorted(set(items))


# -----------------------------
# Arrow / parquet helpers
# -----------------------------


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        return pa, pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for schema comparison. Install with `pip install pyarrow` "
            "or use an environment that already has LeRobot dependencies."
        ) from exc


def _arrow_type_str(pa_type: Any) -> str:
    # Compact, stable string representation.
    try:
        return pa_type.to_string()
    except Exception:
        return str(pa_type)


def _arrow_expected_type(info_dtype: str, *, is_list: bool) -> str:
    # Map info.json "dtype" strings to pyarrow physical types.
    # NOTE: LeRobot v3 stores vector features as list<element: T>.
    scalar_map = {
        "float32": "float",
        "float64": "double",
        "int64": "int64",
        "bool": "bool",
        "string": "string",
    }
    if info_dtype not in scalar_map:
        return "(unknown)"
    base = scalar_map[info_dtype]
    return f"list<{base}>" if is_list else base


def _is_video_dtype(info_dtype: str) -> bool:
    return info_dtype in {"video", "image"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _glob_parquet(root: Path, rel_dir: str) -> list[Path]:
    return sorted((root / rel_dir).glob("chunk-*/file-*.parquet"))


def _schema_map_from_parquet(parquet_path: Path) -> dict[str, str]:
    _, pq = _require_pyarrow()
    schema = pq.read_schema(parquet_path)
    out: dict[str, str] = {}
    for name in schema.names:
        out[name] = _arrow_type_str(schema.field(name).type)
    return out


def _sample_list_lengths(parquet_path: Path, column: str, n: int) -> list[int | None]:
    """Return list lengths for first n rows of a list column."""
    _, pq = _require_pyarrow()

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=n, columns=[column]):
        vals = batch.column(0).to_pylist()
        lengths: list[int | None] = []
        for v in vals:
            if v is None:
                lengths.append(None)
            else:
                try:
                    lengths.append(len(v))
                except Exception:
                    lengths.append(None)
        return lengths
    return []


# -----------------------------
# Dataset inspection
# -----------------------------


@dataclass
class DatasetSnapshot:
    root: Path
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    info: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None

    data_files: list[Path] = field(default_factory=list)
    episodes_files: list[Path] = field(default_factory=list)

    data_schema: dict[str, str] = field(default_factory=dict)
    episodes_schema: dict[str, str] = field(default_factory=dict)
    tasks_schema: dict[str, str] = field(default_factory=dict)


def snapshot_dataset(
    root: Path,
    *,
    sample_rows: int = 5,
    sample_episodes: int = 3,
    check_videos: bool = False,
) -> DatasetSnapshot:
    snap = DatasetSnapshot(root=root, ok=True)

    # ---- Required paths ----
    required = [
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
        root / "meta" / "episodes",
        root / "data",
    ]
    for p in required:
        if not p.exists():
            snap.errors.append(f"Missing required path: {p}")

    if snap.errors:
        snap.ok = False
        return snap

    # ---- Load JSON ----
    try:
        snap.info = _read_json(root / "meta" / "info.json")
    except Exception as exc:
        snap.errors.append(f"Failed to read meta/info.json: {exc}")
        snap.ok = False
        return snap

    try:
        snap.stats = _read_json(root / "meta" / "stats.json")
    except Exception as exc:
        snap.errors.append(f"Failed to read meta/stats.json: {exc}")
        snap.ok = False
        return snap

    # ---- Collect parquet files + schemas ----
    snap.data_files = _glob_parquet(root, "data")
    snap.episodes_files = _glob_parquet(root, "meta/episodes")

    if not snap.data_files:
        snap.errors.append("No data parquet files found under data/chunk-*/file-*.parquet")
    if not snap.episodes_files:
        snap.errors.append(
            "No episodes parquet files found under meta/episodes/chunk-*/file-*.parquet"
        )

    # tasks.parquet schema
    try:
        snap.tasks_schema = _schema_map_from_parquet(root / "meta" / "tasks.parquet")
    except Exception as exc:
        snap.errors.append(f"Failed to read tasks.parquet schema: {exc}")

    if snap.data_files:
        try:
            snap.data_schema = _schema_map_from_parquet(snap.data_files[0])
        except Exception as exc:
            snap.errors.append(f"Failed to read data parquet schema: {exc}")

    if snap.episodes_files:
        try:
            snap.episodes_schema = _schema_map_from_parquet(snap.episodes_files[0])
        except Exception as exc:
            snap.errors.append(f"Failed to read episodes parquet schema: {exc}")

    if snap.errors:
        snap.ok = False
        return snap

    # ---- Internal consistency checks: info.json ↔ parquet ----
    info = snap.info or {}
    features: dict[str, Any] = info.get("features", {}) or {}

    # (1) non-video features must exist as columns
    missing_cols: list[str] = []
    for feat_key, feat_spec in features.items():
        dtype = str(feat_spec.get("dtype", ""))
        if _is_video_dtype(dtype):
            continue
        if feat_key not in snap.data_schema:
            missing_cols.append(feat_key)

    if missing_cols:
        snap.errors.append(
            "info.json declares features missing from data parquet columns: "
            + ", ".join(sorted(missing_cols))
        )

    # (2) parquet dtype should match declared dtype (best-effort)
    dtype_mismatches: list[str] = []
    for feat_key, feat_spec in features.items():
        dtype = str(feat_spec.get("dtype", ""))
        shape = feat_spec.get("shape")
        if _is_video_dtype(dtype):
            continue
        if feat_key not in snap.data_schema:
            continue

        # Vector features are stored as list<element: T>
        is_list = isinstance(shape, list) and len(shape) == 1 and int(shape[0]) > 1
        expected = _arrow_expected_type(dtype, is_list=is_list)
        actual = snap.data_schema[feat_key]

        # Normalize a few common spellings from pyarrow.
        actual_norm = actual
        if actual.startswith("list<element: "):
            # list<element: float> -> list<float>
            actual_norm = actual.replace("list<element: ", "list<").replace(">", ">")
            actual_norm = actual_norm.replace("list<", "list<")

        # Expected is a heuristic string; compare loosely.
        if expected != "(unknown)":
            if is_list:
                if "list" not in actual_norm:
                    dtype_mismatches.append(
                        f"{feat_key}: expected list column for shape {shape}, got {actual}"
                    )
                else:
                    # element type check
                    if dtype == "float32" and "float" not in actual_norm:
                        dtype_mismatches.append(
                            f"{feat_key}: expected list<float> (float32), got {actual}"
                        )
                    if dtype == "float64" and "double" not in actual_norm:
                        dtype_mismatches.append(
                            f"{feat_key}: expected list<double> (float64), got {actual}"
                        )
                    if dtype == "int64" and "int64" not in actual_norm:
                        dtype_mismatches.append(
                            f"{feat_key}: expected list<int64>, got {actual}"
                        )
            else:
                # scalar
                if dtype == "float32" and actual != "float":
                    dtype_mismatches.append(f"{feat_key}: expected float (float32), got {actual}")
                if dtype == "float64" and actual != "double":
                    dtype_mismatches.append(f"{feat_key}: expected double (float64), got {actual}")
                if dtype == "int64" and actual != "int64":
                    dtype_mismatches.append(f"{feat_key}: expected int64, got {actual}")
                if dtype == "bool" and actual != "bool":
                    dtype_mismatches.append(f"{feat_key}: expected bool, got {actual}")

    if dtype_mismatches:
        snap.errors.append("Parquet dtype mismatches vs info.json:\n" + _indent(dtype_mismatches))

    # (3) vector length sampling
    if snap.data_files and sample_rows > 0:
        vec_len_issues: list[str] = []
        for feat_key, feat_spec in features.items():
            dtype = str(feat_spec.get("dtype", ""))
            shape = feat_spec.get("shape")
            if _is_video_dtype(dtype):
                continue
            if not (isinstance(shape, list) and len(shape) == 1):
                continue
            want_len = int(shape[0])
            if want_len <= 1:
                continue
            if feat_key not in snap.data_schema:
                continue
            # Only check list-typed columns
            if "list" not in snap.data_schema[feat_key]:
                vec_len_issues.append(
                    f"{feat_key}: expected list values of length {want_len}, but parquet type is {snap.data_schema[feat_key]}"
                )
                continue
            lengths = _sample_list_lengths(snap.data_files[0], feat_key, n=sample_rows)
            bad = [l for l in lengths if l is not None and l != want_len]
            if bad:
                vec_len_issues.append(
                    f"{feat_key}: expected length {want_len}, saw lengths {lengths} (sampled)"
                )
        if vec_len_issues:
            snap.errors.append("Vector length mismatches (sampled):\n" + _indent(vec_len_issues))

    # (4) stats.json keys should cover all features
    stats = snap.stats or {}
    missing_stats = [k for k in features.keys() if k not in stats]
    if missing_stats:
        snap.warnings.append(
            "Features missing from meta/stats.json: " + ", ".join(sorted(missing_stats))
        )

    # (5) video feature metadata key sanity
    for feat_key, feat_spec in features.items():
        if str(feat_spec.get("dtype", "")) != "video":
            continue
        if "info" not in feat_spec and "video_info" in feat_spec:
            snap.warnings.append(
                f"{feat_key}: video feature uses 'video_info' instead of LeRobot's usual 'info' key"
            )
        if "info" not in feat_spec and "video_info" not in feat_spec:
            snap.warnings.append(f"{feat_key}: video feature missing 'info' metadata")

    # (6) sampled episode references: parquet/video existence
    if snap.episodes_files:
        # Use pyarrow for cheap column reads
        _, pq = _require_pyarrow()
        ep_table = pq.read_table(snap.episodes_files[0])
        n_eps = ep_table.num_rows
        if n_eps == 0:
            snap.errors.append("Episodes parquet has 0 rows")
        else:
            to_check = min(sample_episodes, n_eps)
            # Determine video keys from info.json
            video_keys = [k for k, ft in features.items() if str(ft.get("dtype")) == "video"]
            # Column names in episodes parquet use 'videos/<video_key>/chunk_index'
            for i in range(to_check):
                row = ep_table.slice(i, 1).to_pydict()
                # Data parquet reference
                try:
                    d_chunk = int(row["data/chunk_index"][0])
                    d_file = int(row["data/file_index"][0])
                    data_path_fmt = str(info.get("data_path"))
                    data_path = root / data_path_fmt.format(chunk_index=d_chunk, file_index=d_file)
                    if not data_path.exists():
                        snap.errors.append(
                            f"Episode {i}: referenced data parquet missing: {data_path}"
                        )
                except Exception:
                    pass

                if check_videos:
                    vid_path_fmt = str(info.get("video_path", ""))
                    for vk in video_keys:
                        key = vk
                        c_col = f"videos/{key}/chunk_index"
                        f_col = f"videos/{key}/file_index"
                        if c_col not in row or f_col not in row:
                            snap.errors.append(
                                f"Episodes parquet missing video reference columns for {key} ({c_col}, {f_col})"
                            )
                            continue
                        try:
                            v_chunk = int(row[c_col][0])
                            v_file = int(row[f_col][0])
                            vpath = root / vid_path_fmt.format(
                                video_key=key, chunk_index=v_chunk, file_index=v_file
                            )
                            if not vpath.exists():
                                snap.errors.append(
                                    f"Episode {i}: referenced video missing: {vpath}"
                                )
                        except Exception as exc:
                            snap.errors.append(
                                f"Episode {i}: failed to resolve video path for {key}: {exc}"
                            )

    if snap.errors:
        snap.ok = False
    return snap


# -----------------------------
# Diffing helpers
# -----------------------------


def _diff_sets(label: str, a: set[str], b: set[str]) -> list[str]:
    only_a = sorted(a - b)
    only_b = sorted(b - a)
    out: list[str] = []
    if only_a:
        out.append(f"Only in {label}.a: {only_a}")
    if only_b:
        out.append(f"Only in {label}.b: {only_b}")
    return out


def _diff_kv(prefix: str, a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        if k not in a:
            diffs.append(f"{prefix}{k}: missing in A")
        elif k not in b:
            diffs.append(f"{prefix}{k}: missing in B")
        else:
            if a[k] != b[k]:
                diffs.append(f"{prefix}{k}: A={a[k]!r} != B={b[k]!r}")
    return diffs


def compare_snapshots(a: DatasetSnapshot, b: DatasetSnapshot) -> str:
    lines: list[str] = []

    if a.info is None or b.info is None:
        return "Cannot compare: missing info.json in one snapshot"

    info_a = a.info
    info_b = b.info

    # Top-level info.json keys
    lines.append("=== info.json: top-level keys ===")
    top_a = set(info_a.keys())
    top_b = set(info_b.keys())
    only_a = sorted(top_a - top_b)
    only_b = sorted(top_b - top_a)
    if only_a:
        lines.append(f"Only in A: {only_a}")
    if only_b:
        lines.append(f"Only in B: {only_b}")
    common = sorted(top_a & top_b)
    # Avoid noisy diffs for counts/splits by default, but still show if different.
    noisy = {
        "total_episodes",
        "total_frames",
        "total_tasks",
        "splits",
        "features",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
        "merged_from",
    }
    for k in common:
        if k in noisy:
            continue
        if info_a.get(k) != info_b.get(k):
            lines.append(f"Value differs: {k}: A={info_a.get(k)!r} vs B={info_b.get(k)!r}")

    # Features
    lines.append("\n=== info.json: features ===")
    feat_a = info_a.get("features", {}) or {}
    feat_b = info_b.get("features", {}) or {}
    keys_a = set(feat_a.keys())
    keys_b = set(feat_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        if only_a:
            lines.append(f"Features only in A: {only_a}")
        if only_b:
            lines.append(f"Features only in B: {only_b}")

    common_feats = sorted(keys_a & keys_b)
    for k in common_feats:
        fa = dict(feat_a[k])
        fb = dict(feat_b[k])
        # Compare a small stable subset first
        subset_keys = ["dtype", "shape", "names"]
        sa = {sk: fa.get(sk) for sk in subset_keys}
        sb = {sk: fb.get(sk) for sk in subset_keys}
        if sa != sb:
            lines.append(f"Feature differs: {k}: A={sa} vs B={sb}")

        # Video metadata key differences
        if fa.get("dtype") == "video" and fb.get("dtype") == "video":
            meta_key_a = "info" if "info" in fa else ("video_info" if "video_info" in fa else None)
            meta_key_b = "info" if "info" in fb else ("video_info" if "video_info" in fb else None)
            if meta_key_a != meta_key_b:
                lines.append(
                    f"Video metadata key differs for {k}: A uses {meta_key_a}, B uses {meta_key_b}"
                )

    # Parquet schemas
    lines.append("\n=== data parquet schema (first file) ===")
    cols_a = set(a.data_schema.keys())
    cols_b = set(b.data_schema.keys())
    if cols_a != cols_b:
        only_a = sorted(cols_a - cols_b)
        only_b = sorted(cols_b - cols_a)
        if only_a:
            lines.append(f"Columns only in A: {only_a}")
        if only_b:
            lines.append(f"Columns only in B: {only_b}")

    for col in sorted(cols_a & cols_b):
        ta = a.data_schema[col]
        tb = b.data_schema[col]
        if ta != tb:
            lines.append(f"Type differs: {col}: A={ta} vs B={tb}")

    lines.append("\n=== episodes parquet schema (first file) ===")
    ecols_a = set(a.episodes_schema.keys())
    ecols_b = set(b.episodes_schema.keys())
    if ecols_a != ecols_b:
        only_a = sorted(ecols_a - ecols_b)
        only_b = sorted(ecols_b - ecols_a)
        if only_a:
            lines.append(f"Episode columns only in A: {only_a}")
        if only_b:
            lines.append(f"Episode columns only in B: {only_b}")

    # Stats keys
    if a.stats is not None and b.stats is not None:
        lines.append("\n=== stats.json keys ===")
        sk_a = set(a.stats.keys())
        sk_b = set(b.stats.keys())
        if sk_a != sk_b:
            only_a = sorted(sk_a - sk_b)
            only_b = sorted(sk_b - sk_a)
            if only_a:
                lines.append(f"Stats only in A: {only_a}")
            if only_b:
                lines.append(f"Stats only in B: {only_b}")

        # For common stats, compare which quantiles exist
        common_stats = sorted(sk_a & sk_b)
        for key in common_stats:
            a_fields = set((a.stats.get(key) or {}).keys())
            b_fields = set((b.stats.get(key) or {}).keys())
            if a_fields != b_fields:
                # Keep it short; just show missing fields
                miss_a = sorted(b_fields - a_fields)
                miss_b = sorted(a_fields - b_fields)
                if miss_a or miss_b:
                    lines.append(
                        f"Stat fields differ for {key}: missing_in_A={miss_a} missing_in_B={miss_b}"
                    )

    return "\n".join(lines)


def _print_snapshot_summary(snap: DatasetSnapshot, *, title: str) -> None:
    _p(f"\n{'='*80}\n{title}: {snap.root}\n{'='*80}")
    if snap.ok:
        _p("Status: OK")
    else:
        _p("Status: FAIL")

    if snap.info:
        _p(
            "info.json summary: "
            + f"robot_type={snap.info.get('robot_type')!r}, fps={snap.info.get('fps')!r}, "
            + f"total_episodes={snap.info.get('total_episodes')!r}, total_frames={snap.info.get('total_frames')!r}"
        )

        feats = snap.info.get("features", {}) or {}
        _p(f"features: {len(feats)} keys")
        video_keys = [k for k, ft in feats.items() if str(ft.get('dtype')) == 'video']
        _p(f"video features: {video_keys}")

    _p(f"data parquet files: {len(snap.data_files)}")
    _p(f"episodes parquet files: {len(snap.episodes_files)}")

    def _render_bullets(messages: list[str]) -> str:
        out_lines: list[str] = []
        for msg in messages:
            msg_lines = str(msg).splitlines() or [""]
            out_lines.append(f"- {msg_lines[0]}")
            out_lines.extend([f"  {ln}" for ln in msg_lines[1:]])
        return "\n".join(["  " + ln for ln in out_lines])

    if snap.errors:
        _p("\nErrors:")
        _p(_render_bullets(snap.errors))

    if snap.warnings:
        _p("\nWarnings:")
        _p(_render_bullets(snap.warnings))


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=str, required=True, help="Path to dataset A root")
    p.add_argument("--b", type=str, required=True, help="Path to dataset B root")
    p.add_argument(
        "--sample-rows",
        type=int,
        default=5,
        help="How many rows to sample for vector length checks (default: 5)",
    )
    p.add_argument(
        "--sample-episodes",
        type=int,
        default=3,
        help="How many episodes to sample for reference checks (default: 3)",
    )
    p.add_argument(
        "--check-videos",
        action="store_true",
        help="Also check that referenced video files exist (slower).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any errors are found.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    a_root = Path(args.a)
    b_root = Path(args.b)

    if not a_root.exists() or not b_root.exists():
        _p("[ERROR] One of the provided dataset roots does not exist.")
        _p(f"  A: {a_root}")
        _p(f"  B: {b_root}")
        return 2

    snap_a = snapshot_dataset(
        a_root,
        sample_rows=args.sample_rows,
        sample_episodes=args.sample_episodes,
        check_videos=args.check_videos,
    )
    snap_b = snapshot_dataset(
        b_root,
        sample_rows=args.sample_rows,
        sample_episodes=args.sample_episodes,
        check_videos=args.check_videos,
    )

    _print_snapshot_summary(snap_a, title="Dataset A")
    _print_snapshot_summary(snap_b, title="Dataset B")

    _p("\n" + compare_snapshots(snap_a, snap_b))

    if args.strict and (snap_a.errors or snap_b.errors):
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
