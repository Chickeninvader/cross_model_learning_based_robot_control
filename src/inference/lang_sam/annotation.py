import json
from pathlib import Path
import PIL.Image as Image

frames_dir = Path("../lang-segment-anything/robot_man_1_output_local_2")
seg_dir = Path("../lang-segment-anything/robot_man_1_output_local_2/segmentation")
anno_path = Path("../lang-segment-anything/robot_man_1_output_local_2/annotations.json")
classes = ["drawer", "box"]
images = []
annotations = []

for idx, frame_file in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
    img_id = idx + 1
    mask_file = seg_dir / frame_file.with_suffix(".png").name

    if not mask_file.exists():
        continue

    img = Image.open(frame_file)
    w, h = img.size
    images.append({
        "id": img_id,
        "file_name": frame_file.name,
        "width": w,
        "height": h
    })

    segments_info = [{"id": i+1, "category_id": i+1} for i in range(len(classes))]
    annotations.append({
        "image_id": img_id,
        "pan_seg_file_name": mask_file.name,
        "segments_info": segments_info
    })

categories = [{"id": i+1, "name": name} for i, name in enumerate(classes)]

dataset = {"images": images, "annotations": annotations, "categories": categories}

with open(anno_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ Saved annotation file at {anno_path}")
