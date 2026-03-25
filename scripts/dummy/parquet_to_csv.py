#!/usr/bin/env python3
"""
Convert all parquet files in a LeRobot dataset to CSV format.

This script recursively finds all .parquet files in a task's dataset directory
and converts them to CSV while preserving the directory structure.

Usage
-----
    python scripts/dummy/parquet_to_csv.py --task all_task --lerobot-root datasets/lerobot_trial_3 --output-dir output/csv

The script expects the merged dataset layout produced by
`src/data_collection/convert_rlbench_to_lerobot.py`:

    datasets/lerobot/<task>_eef/
    datasets/lerobot/<task>_joint/

By default it processes both (EEF + joint). Use `--action_space` to export only one.

Backward-compatibility: if the new folders are not found, it will fall back to
older layouts like `<task>_variation*` or `<task>_all` when present.

It converts:
    - data/chunk-*/file-*.parquet
    - meta/episodes/chunk-*/file-*.parquet
    - meta/tasks.parquet

Output structure:
    output/csv/
    └── stack_cups/
        ├── eef/
        │   ├── data/
        │   │   └── chunk-000/
        │   │       └── file-000.csv
        │   └── meta/
        │       ├── episodes/
        │       │   └── chunk-000/
        │       │       └── file-000.csv
        │       └── tasks.csv
        └── joint/
            └── ...
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def find_parquet_files(root_dir: Path) -> list[Path]:
    """Recursively find all .parquet files in a directory.
    
    Parameters
    ----------
    root_dir : Path
        Root directory to search.
        
    Returns
    -------
    list[Path]
        List of paths to .parquet files.
    """
    return sorted(root_dir.rglob("*.parquet"))


def convert_parquet_to_csv(parquet_path: Path, output_path: Path, *, display_root: Path | None = None) -> None:
    """Convert a single parquet file to CSV.
    
    Parameters
    ----------
    parquet_path : Path
        Path to input .parquet file.
    output_path : Path
        Path to output .csv file.
    """
    # Load parquet file
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    df.to_csv(output_path, index=False)
    if display_root is not None:
        try:
            src_disp = parquet_path.relative_to(display_root)
        except Exception:
            src_disp = parquet_path
    else:
        src_disp = parquet_path
    print(f"  ✓ {src_disp} -> {output_path}")


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    task_name: str,
    dataset_tag: str,
) -> int:
    """Process all parquet files in a dataset directory.
    
    Parameters
    ----------
    dataset_dir : Path
        Path to a LeRobot dataset directory (e.g., datasets/lerobot/stack_cups_eef).
    output_dir : Path
        Root output directory for CSV files.
    task_name : str
        Task name (e.g., "stack_cups").
    dataset_tag : str
        Output tag under `output_dir/<task_name>/...` (e.g., "eef" or "joint").
        
    Returns
    -------
    int
        Number of files converted.
    """
    # Find all parquet files (excluding videos)
    parquet_files = [
        f for f in find_parquet_files(dataset_dir)
        if "videos" not in f.parts  # Skip video metadata if any
    ]
    
    if not parquet_files:
        print(f"  [WARN] No parquet files found in {dataset_dir}")
        return 0
    
    print(f"  [INFO] Found {len(parquet_files)} parquet files in {dataset_tag}")
    
    # Convert each parquet file
    for parquet_path in parquet_files:
        # Compute relative path from dataset directory
        rel_path = parquet_path.relative_to(dataset_dir)
        
        # Create corresponding CSV path
        csv_rel_path = rel_path.with_suffix(".csv")
        csv_output_path = output_dir / task_name / dataset_tag / csv_rel_path
        
        # Convert
        convert_parquet_to_csv(parquet_path, csv_output_path, display_root=dataset_dir)
    
    return len(parquet_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert all parquet files in a LeRobot dataset to CSV format."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Base task name (e.g., 'stack_cups'). Will process <task>_eef and/or <task>_joint under --lerobot-root."
        ),
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("datasets/lerobot"),
        help="Root directory containing LeRobot datasets (default: datasets/lerobot).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/parquet_to_csv"),
        help="Output directory for CSV files (default: scripts/parquet_to_csv).",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="both",
        choices=["eef", "joint", "both"],
        help="Which dataset(s) to convert (default: both).",
    )
    parser.add_argument(
        "--variation",
        type=int,
        default=None,
        help="DEPRECATED: variation datasets are no longer produced; kept for backward compatibility.",
    )
    
    args = parser.parse_args()

    if args.variation is not None:
        print("[WARN] --variation is deprecated and ignored for <task>_eef/<task>_joint datasets.")
    
    dataset_dirs: list[Path] = []

    # New layout: <task>_eef and <task>_joint
    eef_dir = args.lerobot_root / f"{args.task}_eef"
    joint_dir = args.lerobot_root / f"{args.task}_joint"
    if args.action_space in {"eef", "both"} and eef_dir.is_dir():
        dataset_dirs.append(eef_dir)
    if args.action_space in {"joint", "both"} and joint_dir.is_dir():
        dataset_dirs.append(joint_dir)

    # Backward compatibility: older layouts
    if not dataset_dirs:
        variation_dirs = sorted(args.lerobot_root.glob(f"{args.task}_variation*"))
        variation_dirs = [d for d in variation_dirs if d.is_dir()]
        if variation_dirs:
            dataset_dirs = variation_dirs

    if not dataset_dirs:
        merged_dir = args.lerobot_root / f"{args.task}_all"
        if merged_dir.is_dir():
            dataset_dirs = [merged_dir]

    if not dataset_dirs:
        exact_dir = args.lerobot_root / args.task
        if exact_dir.is_dir():
            dataset_dirs = [exact_dir]

    if not dataset_dirs:
        print(f"[ERROR] No datasets found for task '{args.task}' under {args.lerobot_root}")
        print(f"        Expected: {eef_dir} and/or {joint_dir}")
        return 1

    print(f"[INFO] Found {len(dataset_dirs)} dataset(s) for task '{args.task}':")
    for d in dataset_dirs:
        print(f"       - {d.name}")
    print()
    
    # Process each dataset
    total_files = 0
    for dataset_dir in dataset_dirs:
        # New scheme: eef/joint tag from suffix
        if dataset_dir.name.endswith("_eef"):
            tag = "eef"
        elif dataset_dir.name.endswith("_joint"):
            tag = "joint"
        elif dataset_dir.name == f"{args.task}_all":
            tag = "all"
        else:
            tag = dataset_dir.name.replace(f"{args.task}_", "") or "merged"

        print(f"[INFO] Processing {tag}...")
        
        n_files = process_dataset(
            dataset_dir,
            args.output_dir,
            args.task,
            tag,
        )
        total_files += n_files
        print()
    
    print(f"[DONE] Converted {total_files} parquet file(s) to CSV")
    print(f"       Output directory: {args.output_dir / args.task}")
    
    return 0


if __name__ == "__main__":
    exit(main())
