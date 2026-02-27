import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import time, json, gc

from lang_sam import LangSAM
from lang_sam.utils import draw_image

# ------------------------------
# Configuration
# ------------------------------
video_path = "Stack_teleop.mp4"
output_dir = "Stack_teleop_output_local_1"

dirs = {
    "overlay": os.path.join(output_dir, "overlay_frames"),
    "mask_npy": os.path.join(output_dir, "mask_arrays"),
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

output_video_path = os.path.join(output_dir, "output_overlay.mp4")
annotation_path = os.path.join(output_dir, "annotations.json")

objects = ["can", "box", "franka_robot", "table"]
relations = ["next to", "on"]

text_prompt = ", ".join(objects)
sam_type = "sam2.1_hiera_large"
box_threshold = 0.3
text_threshold = 0.25

save_every_n_frames = 20  # flush video writer every N frames

# ------------------------------
# Initialize model
# ------------------------------
print("🔧 Loading LangSAM model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LangSAM(sam_type=sam_type)
print(f"✅ Model loaded on {device}.")

# ------------------------------
# Process video
# ------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"🎬 Starting inference on {total_frames} frames...")
start_time = time.time()

with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        with torch.no_grad():
            results = model.predict(
                images_pil=[image_pil],
                texts_prompt=[text_prompt],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )[0]

        if len(results["masks"]) == 0:
            overlay_bgr = frame  # no segmentation
        else:
            overlay = draw_image(
                frame_rgb,
                results["masks"],
                results["boxes"],
                results["scores"],
                results["labels"],
            )
            overlay_bgr = cv2.cvtColor(np.uint8(overlay), cv2.COLOR_RGB2BGR)

        out_video.write(overlay_bgr)

        # Save mask to disk (optional)
        np.save(os.path.join(dirs["mask_npy"], f"mask_{frame_idx:05d}.npy"), results["masks"])

        # Force flush the video writer occasionally
        if frame_idx % save_every_n_frames == 0:
            out_video.release()
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

        # Explicit cleanup
        del results, overlay_bgr, frame_rgb, image_pil, frame
        torch.cuda.empty_cache()
        gc.collect()

        frame_idx += 1
        pbar.update(1)

cap.release()
out_video.release()

# ------------------------------
# Save global annotation
# ------------------------------
annotation_data = {"objects": objects, "relations": relations}
with open(annotation_path, "w") as f:
    json.dump(annotation_data, f, indent=2)

elapsed = time.time() - start_time
print(f"\n✅ Done! {frame_idx} frames processed in {elapsed/60:.2f} min")
print(f"🎞 Output video saved to {output_video_path}")
print(f"🗂️ Annotation file saved to {annotation_path}")
