#!/usr/bin/env python3
"""
Convert all parquet files in a LeRobot dataset to CSV format.

This script recursively finds all .parquet files in a task's dataset directory
and converts them to CSV while preserving the directory structure.

Usage
-----
    python scripts/parquet_to_csv.py --task stack_cups --lerobot-root datasets/lerobot --output-dir output/csv

The script will process all variations found for the task (e.g., stack_cups_variation0, 
stack_cups_variation1, etc.) and convert:
    - data/chunk-*/file-*.parquet
    - meta/episodes/chunk-*/file-*.parquet
    - meta/tasks.parquet

Output structure:
    output/csv/
    └── stack_cups/
        ├── variation0/
        │   ├── data/
        │   │   └── chunk-000/
        │   │       └── file-000.csv
        │   └── meta/
        │       ├── episodes/
        │       │   └── chunk-000/
        │       │       └── file-000.csv
        │       └── tasks.csv
        └── variation1/
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


def convert_parquet_to_csv(parquet_path: Path, output_path: Path) -> None:
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
    print(f"  ✓ {parquet_path.relative_to(parquet_path.parents[3])} -> {output_path.name}")


def process_variation(
    variation_dir: Path,
    output_dir: Path,
    task_name: str,
    variation: str,
) -> int:
    """Process all parquet files in a variation directory.
    
    Parameters
    ----------
    variation_dir : Path
        Path to variation directory (e.g., datasets/lerobot/stack_cups_variation0).
    output_dir : Path
        Root output directory for CSV files.
    task_name : str
        Task name (e.g., "stack_cups").
    variation : str
        Variation name (e.g., "variation0").
        
    Returns
    -------
    int
        Number of files converted.
    """
    # Find all parquet files (excluding videos)
    parquet_files = [
        f for f in find_parquet_files(variation_dir)
        if "videos" not in f.parts  # Skip video metadata if any
    ]
    
    if not parquet_files:
        print(f"  [WARN] No parquet files found in {variation_dir}")
        return 0
    
    print(f"  [INFO] Found {len(parquet_files)} parquet files in {variation}")
    
    # Convert each parquet file
    for parquet_path in parquet_files:
        # Compute relative path from variation directory
        rel_path = parquet_path.relative_to(variation_dir)
        
        # Create corresponding CSV path
        csv_rel_path = rel_path.with_suffix(".csv")
        csv_output_path = output_dir / task_name / variation / csv_rel_path
        
        # Convert
        convert_parquet_to_csv(parquet_path, csv_output_path)
    
    return len(parquet_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert all parquet files in a LeRobot dataset to CSV format."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., 'stack_cups'). Will process all variations found.",
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
        "--variation",
        type=int,
        default=None,
        help="Process only a specific variation number (optional).",
    )
    
    args = parser.parse_args()
    
    # Find all variation directories for this task
    task_pattern = f"{args.task}_variation*"
    variation_dirs = sorted(args.lerobot_root.glob(task_pattern))
    
    if not variation_dirs:
        print(f"[ERROR] No variations found for task '{args.task}' in {args.lerobot_root}")
        print(f"        Looking for pattern: {task_pattern}")
        return 1
    
    # Filter by specific variation if requested
    if args.variation is not None:
        variation_dirs = [
            d for d in variation_dirs
            if d.name == f"{args.task}_variation{args.variation}"
        ]
        if not variation_dirs:
            print(f"[ERROR] Variation {args.variation} not found for task '{args.task}'")
            return 1
    
    print(f"[INFO] Found {len(variation_dirs)} variation(s) for task '{args.task}':")
    for vdir in variation_dirs:
        print(f"       - {vdir.name}")
    print()
    
    # Process each variation
    total_files = 0
    for variation_dir in variation_dirs:
        variation_name = variation_dir.name.replace(f"{args.task}_", "")
        print(f"[INFO] Processing {variation_name}...")
        
        n_files = process_variation(
            variation_dir,
            args.output_dir,
            args.task,
            variation_name,
        )
        total_files += n_files
        print()
    
    print(f"[DONE] Converted {total_files} parquet file(s) to CSV")
    print(f"       Output directory: {args.output_dir / args.task}")
    
    return 0


if __name__ == "__main__":
    exit(main())
