#!/usr/bin/env python3
"""
Script to generate videos from RLBench dataset front_rgb images.
Iterates through all tasks, variations, and episodes, creating videos for each.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def get_image_files(image_dir):
    """Get sorted list of image files in a directory."""
    if not os.path.exists(image_dir):
        return []
    
    images = []
    for f in os.listdir(image_dir):
        if f.endswith('.png') or f.endswith('.jpg'):
            images.append(f)
    
    # Sort by numeric order (0.png, 1.png, etc.)
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return images


def create_video_from_images(image_dir, output_path, fps=30):
    """
    Create a video from a sequence of images.
    
    Args:
        image_dir: Directory containing the image sequence
        output_path: Path to save the output video
        fps: Frames per second for the video
    """
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"  No images found in {image_dir}")
        return False
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        print(f"  Failed to read first image: {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"  Failed to create video writer for {output_path}")
        return False
    
    # Write frames to video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"  Warning: Failed to read image {image_file}")
            continue
        
        # Ensure frame has correct dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
    
    out.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos from RLBench dataset front_rgb images"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/workspace/datasets/rlbench",
        help="Root directory of RLBench dataset"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/workspace/datasets/rlbench_visualize",
        help="Root directory to save output videos"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output videos"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front_rgb",
        help="Camera view to visualize (e.g., front_rgb, wrist_rgb)"
    )
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    
    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get all tasks
    tasks = sorted([d for d in dataset_root.iterdir() if d.is_dir()])
    
    print(f"Found {len(tasks)} tasks")
    print(f"Saving videos to: {output_root}")
    print(f"Using camera view: {args.camera}")
    print()
    
    total_videos = 0
    successful_videos = 0
    
    # Iterate through tasks
    with tqdm(tasks, desc="Tasks") as pbar_tasks:
        for task_dir in pbar_tasks:
            task_name = task_dir.name
            pbar_tasks.set_description(f"Task: {task_name}")
            
            # Get all variations
            variations = sorted([d for d in task_dir.iterdir() if d.is_dir()])
            
            for var_dir in variations:
                var_name = var_dir.name
                
                # Get all episodes
                episodes_dir = var_dir / "episodes"
                if not episodes_dir.exists():
                    continue
                
                episodes = sorted([d for d in episodes_dir.iterdir() if d.is_dir()])
                
                for episode_dir in episodes:
                    episode_name = episode_dir.name
                    
                    # Path to images
                    image_dir = episode_dir / args.camera
                    
                    if not image_dir.exists():
                        continue
                    
                    total_videos += 1
                    
                    # Output video path with flat naming: task_name_variation_episode.mp4
                    video_filename = f"{task_name}_{var_name}_{episode_name}.mp4"
                    video_output_path = output_root / video_filename
                    
                    # Create video
                    if create_video_from_images(str(image_dir), str(video_output_path), args.fps):
                        successful_videos += 1
    
    print()
    print(f"Completed!")
    print(f"Total videos processed: {total_videos}")
    print(f"Successful: {successful_videos}")
    print(f"Failed: {total_videos - successful_videos}")


if __name__ == "__main__":
    main()
