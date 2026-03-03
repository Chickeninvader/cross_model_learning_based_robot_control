#!/usr/bin/env python3
"""
Download GR00T datasets from Hugging Face to datasets/lerobot folder.
Only downloads single_panda_gripper datasets (2-3 episodes each).
"""

import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download, login

# Get HF token from environment variable (optional but recommended)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✓ Logged in with HF token")
else:
    print("⚠ No HF_TOKEN found. Set it to avoid rate limiting:")
    print("  export HF_TOKEN='your_token_here'")

# Define datasets to download (single panda gripper only)
DATASETS = [
    "GR00T-Dateset/single_panda_gripper-TurnSinkSpout",
    "GR00T-Dateset/single_panda_gripper-OpenSingleDoor",
    "GR00T-Dateset/single_panda_gripper-CoffeeSetupMug",
]

# Output directory
OUTPUT_DIR = Path("datasets") / "lerobot" / "gr00t"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nDownloading {len(DATASETS)} datasets to {OUTPUT_DIR}")
print("-" * 50)

for i, dataset_id in enumerate(DATASETS):
    dataset_name = dataset_id.split("/")[-1]
    output_path = OUTPUT_DIR / dataset_name
    
    print(f"\nDownloading: {dataset_name}")
    print(f"  Output: {output_path}")
    
    try:
        # Only download metadata and a few episodes (0, 1, 2)
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=output_path,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "meta.json",
                "meta/**",
                "videos/chunk-000/*/episode_00000[012].mp4",
                "data/chunk-000/episode_00000[012].parquet",
            ]
        )
        print(f"  ✓ Successfully downloaded")
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        print(f"  Retrying in 10 seconds...")
        time.sleep(10)
        try:
            snapshot_download(
                repo_id=dataset_id,
                repo_type="dataset",
                local_dir=output_path,
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "meta.json",
                    "meta/**",
                    "videos/chunk-000/*/episode_00000[012].mp4",
                    "data/chunk-000/episode_00000[012].parquet",
                ]
            )
            print(f"  ✓ Successfully downloaded on retry")
        except Exception as e2:
            print(f"  ✗ Failed again: {e2}")
    
    # Add delay between downloads to avoid rate limiting
    if i < len(DATASETS) - 1:
        time.sleep(5)

print("\n" + "-" * 50)
print("Download complete!")
