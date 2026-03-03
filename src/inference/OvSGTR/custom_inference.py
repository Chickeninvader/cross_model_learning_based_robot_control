#!/usr/bin/env python3
"""
Custom inference using GroundingDINO + PostProcess (relations included).
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

# --- repo imports ---
from util.slconfig import SLConfig
from groundingdino.models.GroundingDINO import build_groundingdino
from util.misc import nested_tensor_from_tensor_list
# --------------------


# -------------------------
# Dataset for custom images
# -------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file=None):
        self.img_dir = Path(img_dir)
        self.items = []

        if ann_file and Path(ann_file).exists():
            with open(ann_file, "r") as f:
                data = json.load(f)

            # Handle minimal annotation structure {objects: [...], relations: [...]}
            if isinstance(data, dict) and "objects" in data and "relations" in data:
                self.objects = data["objects"]
                self.relations = data["relations"]
            else:
                raise ValueError("Annotation file must contain {'objects': [...], 'relations': [...]}")

            # Build default dataset items for all .jpg images
            for p in sorted(self.img_dir.glob("*.jpg")):
                self.items.append({
                    "file_name": str(p),
                    "caption": ". ".join(self.objects) + ".",   # e.g. "drawer. box. franka_robot."
                    "rel_caption": ". ".join(self.relations) + "."  # e.g. "next to."
                })
        else:
            # Fallback if no annotation file
            print("⚠️ No annotation JSON found. Using default vocab.")
            self.objects = ["drawer", "box", "franka_robot"]
            self.relations = ["next to"]

            for p in sorted(self.img_dir.glob("*.jpg")):
                self.items.append({
                    "file_name": str(p),
                    "caption": ". ".join(self.objects) + ".",
                    "rel_caption": ". ".join(self.relations) + "."
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["file_name"]).convert("RGB")
        import torchvision.transforms as T
        img_tensor = T.ToTensor()(img)
        h, w = img_tensor.shape[1:]
        target = {
            "image_id": idx,
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
            "caption": item["caption"],
            "rel_caption": item["rel_caption"]
        }
        return img_tensor, target


# -------------------------
# Build model + postprocess
# -------------------------
def load_model(config_path, ckpt_path, device, objects, relations):
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device
    cfg.eval = True
    cfg.do_sgg = True
    cfg.use_text_labels = True

    print("Building model...")
    model, criterion, postprocessors = build_groundingdino(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # ---- replicate main.py linking ----
    rln_proj = getattr(model, "rln_proj", None)
    rln_classifier = getattr(model, "rln_classifier", None)
    rln_freq_bias = getattr(model, "rln_freq_bias", None)

    criterion.rln_proj = rln_proj
    criterion.rln_proj_teacher = None
    criterion.rln_classifier = rln_classifier
    criterion.rln_freq_bias = rln_freq_bias

    postproc = postprocessors["bbox"].to(device)
    postproc.rln_proj = rln_proj
    postproc.rln_classifier = rln_classifier
    postproc.rln_freq_bias = rln_freq_bias

    # -------------------------
    # Load vocab from annotation
    # -------------------------
    postproc.name2classes = {name: i for i, name in enumerate(objects)}
    postproc.name2predicates = {name: i for i, name in enumerate(relations)}

    print(f"✅ Model + PostProcessor initialized with {len(objects)} objects and {len(relations)} relations.")
    return model, postproc


# -------------------------
# Run inference
# -------------------------
def run_inference(model, postproc, dataset, device, output_json="results.json"):
    os.makedirs(Path(output_json).parent, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_results = []
    for img_tensor, target in loader:
        samples = nested_tensor_from_tensor_list([img_tensor.squeeze(0)]).to(device)
        captions = [target["caption"][0]]
        rel_captions = [target["rel_caption"][0]]

        with torch.no_grad():
            outputs = model(samples, captions=captions, rel_captions=rel_captions)
        sizes = torch.stack([target["orig_size"][0]]).to(device)

        results = postproc(outputs, sizes)
        for res in results:
            boxes = res["boxes"].cpu().tolist()
            labels = res["labels"].cpu().tolist()
            scores = res["scores"].cpu().tolist()
            out_item = {
                "file_name": target["image_id"].item(),
                "objects": [{"label": int(l), "bbox": b, "score": s} for l, b, s in zip(labels, boxes, scores)]
            }
            if "graph" in res:
                # triplets: (subject, predicate, object)
                out_item["relations"] = res["graph"]
            all_results.append(out_item)

    # Save results
    def tensor_to_jsonable(obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: tensor_to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [tensor_to_jsonable(v) for v in obj]
        return obj

    with open(output_json, "w") as f:
        json.dump(tensor_to_jsonable(all_results), f, indent=2)

    print(f"💾 Saved {len(all_results)} results to {output_json}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = ArgumentParser()
    ap.add_argument("--config", required=True, help="Config .py file")
    ap.add_argument("--weights", required=True, help="Checkpoint .pth")
    ap.add_argument("--image_dir", required=True, help="Directory with images")
    ap.add_argument("--ann_file", required=True, help="annotations.json file with objects/relations")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="output/custom_results.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load annotation JSON once for vocab
    with open(args.ann_file, "r") as f:
        ann_data = json.load(f)
    objects = ann_data.get("objects", [])
    relations = ann_data.get("relations", [])

    model, postproc = load_model(args.config, args.weights, device, objects, relations)
    dataset = SimpleDataset(args.image_dir, args.ann_file)
    run_inference(model, postproc, dataset, device, args.output)


if __name__ == "__main__":
    main()

"""
module load mamba/latest
module load cuda-10.2.89-gcc-12.1.0
python custom_inference.py \
    --config config/GroundingDINO_SwinB_ovdr.py \
    --weights vg-ovdr-swinb-mega-best.pth \
    --image_dir ../lang-segment-anything/Stack_teleop_output_local_1/original_frames \
    --ann_file ../lang-segment-anything/Stack_teleop_output_local_1/annotations.json \
    --device cuda \
    --output output_Stack_teleop/robot_sgg_results.json

"""