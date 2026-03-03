import os
import cv2
from tqdm import tqdm
import argparse

def extract_frames(video_path, output_dir, every_n=1):
    """
    Extract frames from a video file and save as frame_00000.jpg, frame_00001.jpg, ...
    
    Args:
        video_path (str): Path to input video (.mp4, .avi, etc.)
        output_dir (str): Directory to save frames.
        every_n (int): Save every Nth frame (default 1 = all frames).
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Video: {video_path}")
    print(f"   ➤ Total frames: {total_frames}")
    print(f"   ➤ FPS: {fps:.2f}")
    print(f"   ➤ Saving to: {output_dir}")

    frame_idx = 0
    saved_idx = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n == 0:
                filename = os.path.join(output_dir, f"frame_{saved_idx:05d}.jpg")
                cv2.imwrite(filename, frame)
                saved_idx += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"\n✅ Done! Extracted {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--outdir", required=True, help="Output directory for frames")
    parser.add_argument("--every", type=int, default=1, help="Save every Nth frame (default=1)")
    args = parser.parse_args()

    extract_frames(args.video, args.outdir, args.every)
