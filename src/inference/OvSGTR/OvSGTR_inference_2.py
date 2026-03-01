"""
OvSGTR inference on RLBench episodes.

Reads RGB frames from an RLBench episode directory, runs OvSGTR for
object detection + scene-graph generation, and produces:
  1. Scene-graph JSON  (same format as the RLBench notebook)
  2. Overlay video     (RGB + bboxes + centres + relationship arrows)
  3. Mask video        (RLBench ground-truth masks, per-object ID encoded)

Usage example:
  PYTHONPATH=~/Desktop:~/Desktop/external/OvSGTR/GroundingDINO/groundingdino
  python OvSGTR_inference_2.py \
      --config  .../OvSGTR/config/GroundingDINO_SwinB_ovdr.py \
      --weights .../OvSGTR/vg-ovdr-swinb-mega-best.pth \
      --dataset_dir .../datasets/rlbench/stack_cups \
      --variation 0 --episode 0 --camera front \
      --output_dir output/OvSGTR --device cuda
"""
import os
import sys
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms as T
from torchvision.ops import nms

# GroundingDINO / OvSGTR imports
from external.OvSGTR.util.slconfig import SLConfig
from external.OvSGTR.GroundingDINO.groundingdino.models.GroundingDINO import build_groundingdino
from external.OvSGTR.util.misc import nested_tensor_from_tensor_list

# Allow importing utils.py when it sits next to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import (
    load_rlbench_images,
    decode_mask_image,
    create_detection_overlay_video,
    create_rlbench_mask_video,
)

# =====================
# Constants & Defaults
# =====================
OBJ_SCORE_THRESH = 0.25
REL_SCORE_THRESH = 0.25
IOU_THRESHOLD = 0.5
TOP_OBJ_KEEP = 5


# =====================
# OvSGTR-specific code
# =====================

def load_model(config_path, ckpt_path, device, objects, relations):
    """Load OvSGTR (GroundingDINO + SGG head) from config + checkpoint."""
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device
    cfg.eval = True
    cfg.do_sgg = True
    cfg.use_text_labels = True

    model, _, postprocessors = build_groundingdino(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.to(device).eval()

    postproc = postprocessors["bbox"].to(device)
    postproc.rln_proj = getattr(model, "rln_proj", None)
    postproc.rln_classifier = getattr(model, "rln_classifier", None)
    postproc.rln_freq_bias = getattr(model, "rln_freq_bias", None)

    postproc.name2classes = {name: i for i, name in enumerate(objects)}
    postproc.name2predicates = {name: i for i, name in enumerate(relations)}

    return model, postproc


def run_inference(model, postproc, frame_bgr, caption, rel_caption, device):
    """Run OvSGTR forward pass on a single BGR frame.

    Returns the raw postprocessor result dict.
    """
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = T.ToTensor()(Image.fromarray(img_rgb))
    samples = nested_tensor_from_tensor_list([img_tensor]).to(device)

    with torch.no_grad():
        outputs = model(samples, captions=[caption], rel_captions=[rel_caption])

    res = postproc(outputs, torch.tensor([[h, w]]).to(device))[0]
    return res


def extract_frame_detections(res, obj_list, rel_list, frame_idx, fps):
    """Process raw OvSGTR result into a structured per-frame dict.

    Returns a dict suitable for ``create_detection_overlay_video`` and for
    building the scene-graph JSON.
    """
    # --- NMS → score filter → top-K ---
    keep = nms(res["boxes"], res["scores"], IOU_THRESHOLD)
    valid = res["scores"][keep] >= OBJ_SCORE_THRESH
    top_indices = keep[valid]
    if len(top_indices) > 0:
        order = torch.argsort(res["scores"][top_indices], descending=True)
        top_indices = top_indices[order[:TOP_OBJ_KEEP]]

    # --- Best detection per object class ---
    best = {}                       # class_name → detection info
    for idx in top_indices.tolist():
        cls_idx = int(res["labels"][idx].item())
        if cls_idx >= len(obj_list):
            continue
        cls_name = obj_list[cls_idx]
        score = float(res["scores"][idx].item())
        if cls_name not in best or score > best[cls_name]["score"]:
            box = res["boxes"][idx].tolist()
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            best[cls_name] = {
                "det_idx": idx,
                "score": round(score, 4),
                "bbox": [int(b) for b in box],
                "position": [cx, cy],
            }

    # --- Relationships ---
    relationships = []
    idx_to_cls = {det["det_idx"]: name for name, det in best.items()}
    graph_data = res.get("graph")
    if graph_data and "all_node_pairs" in graph_data:
        pairs = graph_data["all_node_pairs"]
        probs = graph_data["all_relation"]
        for i, (a, b) in enumerate(pairs):
            ia, ib = int(a.item()), int(b.item())
            if ia in idx_to_cls and ib in idx_to_cls:
                rel_probs = probs[i]
                ridx = int(torch.argmax(rel_probs).item())
                rconf = float(rel_probs[ridx].item())
                if rconf >= REL_SCORE_THRESH and ridx < len(rel_list):
                    relationships.append({
                        "object1": idx_to_cls[ia],
                        "object2": idx_to_cls[ib],
                        "type": rel_list[ridx],
                        "score": round(rconf, 4),
                    })

    # --- Assemble frame dict ---
    frame_data = {
        "frame_id": frame_idx,
        "timestamp": round(frame_idx / fps, 4),
        "objects": {},
        "relationships": relationships,
    }
    for name in obj_list:
        if name in best:
            d = best[name]
            frame_data["objects"][name] = {
                "position": d["position"],
                "visible": True,
                "bbox": d["bbox"],
                "score": d["score"],
            }
        else:
            frame_data["objects"][name] = {
                "position": [0, 0],
                "visible": False,
            }
    return frame_data


def build_scene_graph(all_frame_data, obj_list):
    """Build the scene-graph JSON structure from per-frame detections.

    Follows the same schema as the RLBench notebook's scene-graph JSON:
      { "objects": { ... }, "frames": [ ... ] }
    """
    scene_graph = {"objects": {}, "frames": []}
    for name in obj_list:
        scene_graph["objects"][name] = {
            "type": "robot" if "robot" in name.lower() else "object",
            "name": name,
        }
    for fd in all_frame_data:
        entry = {
            "frame_id": fd["frame_id"],
            "timestamp": fd["timestamp"],
            "objects": {
                name: {
                    "position": od["position"],
                    "visible": od["visible"],
                }
                for name, od in fd["objects"].items()
            },
            "relationships": [
                {"object1": r["object1"], "object2": r["object2"],
                 "type": r["type"]}
                for r in fd["relationships"]
            ],
        }
        scene_graph["frames"].append(entry)
    return scene_graph


# =====================
# Main
# =====================

def main():
    parser = ArgumentParser(description="OvSGTR inference on RLBench episodes")
    parser.add_argument("--config", required=True,
                        help="OvSGTR model config (.py)")
    parser.add_argument("--weights", required=True,
                        help="OvSGTR checkpoint (.pth)")
    parser.add_argument("--dataset_dir", required=True,
                        help="RLBench task root that contains info.json "
                             "(e.g. datasets/rlbench/stack_cups)")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default="front")
    parser.add_argument("--output_dir", default="output/OvSGTR",
                        help="Base output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="FPS used for output videos and timestamps")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Read dataset metadata
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset_dir)
    with open(dataset_path / "info.json") as f:
        info = json.load(f)

    obj_list = [v["name"] for k, v in
                sorted(info["objects"].items(), key=lambda x: int(x[0]))]
    rel_list = [v["name"] for k, v in
                sorted(info["relations"].items(), key=lambda x: int(x[0]))]
    object_mapping = {int(k): v for k, v in info["object_mapping"].items()}

    print(f"Objects : {obj_list}")
    print(f"Relations: {rel_list}")

    # ------------------------------------------------------------------
    # 2. Load image paths
    # ------------------------------------------------------------------
    variation_dir = str(dataset_path / f"variation{args.variation}")
    rgb_paths, mask_paths = load_rlbench_images(
        variation_dir, args.episode, args.camera)
    print(f"RGB frames : {len(rgb_paths)}")
    print(f"Mask frames: {len(mask_paths)}")

    if not rgb_paths:
        print("ERROR: no RGB images found — check paths.")
        return

    # ------------------------------------------------------------------
    # 3. Load OvSGTR model
    # ------------------------------------------------------------------
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    model, postproc = load_model(
        args.config, args.weights, device, obj_list, rel_list)

    caption = ". ".join(obj_list) + "."
    rel_caption = ". ".join(rel_list) + "."

    # ------------------------------------------------------------------
    # 4. Inference loop
    # ------------------------------------------------------------------
    all_frame_data = []
    for i, rgb_path in enumerate(rgb_paths):
        frame = cv2.imread(rgb_path)
        if frame is None:
            print(f"  [warn] cannot read {rgb_path}")
            continue

        res = run_inference(model, postproc, frame, caption, rel_caption,
                            device)
        fd = extract_frame_detections(res, obj_list, rel_list, i, args.fps)
        all_frame_data.append(fd)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(rgb_paths)} frames")

    print(f"Inference complete — {len(all_frame_data)} frames processed.")

    # ------------------------------------------------------------------
    # 5. Outputs
    # ------------------------------------------------------------------
    task_name = dataset_path.name
    out_dir = (Path(args.output_dir) / task_name
               / f"variation{args.variation}"
               / f"episode{args.episode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5a. Scene-graph JSON
    scene_graph = build_scene_graph(all_frame_data, obj_list)
    sg_path = out_dir / f"{task_name}_scene_graph.json"
    with open(sg_path, "w") as f:
        json.dump(scene_graph, f, indent=2)
    print(f"Scene graph saved : {sg_path}")

    # 5b. Overlay video  (RGB + bboxes + centres + arrows + labels)
    overlay_path = str(out_dir / "overlay_video.mp4")
    create_detection_overlay_video(
        rgb_paths, all_frame_data, overlay_path, fps=args.fps)

    # 5c. Mask video  (RLBench GT masks, per-object ID encoded)
    if mask_paths:
        mask_vid_path = str(out_dir / "mask_video.mp4")
        create_rlbench_mask_video(
            mask_paths, object_mapping, obj_list, mask_vid_path, fps=args.fps)
    else:
        print("No mask images found — skipping mask video.")

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()