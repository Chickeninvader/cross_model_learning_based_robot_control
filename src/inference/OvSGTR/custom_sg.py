import json
import cv2
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# =====================
# Config
# =====================
RESULT_PATH = "output_Stack_teleop/robot_sgg_results.json"  # output from GroundingDINO inference
ANN_FILE = "../lang-segment-anything/Stack_teleop_output_local_1/annotations.json"  # vocab json
IMAGE_DIR = "../lang-segment-anything/Stack_teleop_output_local_1/original_frames"
OUTPUT_DIR = "output_Stack_teleop/vis_video"
VIDEO_PATH = "output_Stack_teleop/scene_graph_video.mp4"

OBJ_SCORE_THRESH = 0.2
REL_SCORE_THRESH = 0.15
TOP_OBJ_KEEP = 3   # top objects by score
GRAPH_SIZE = (400, 400)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================
# Load vocab from JSON
# =====================
if not os.path.exists(ANN_FILE):
    raise FileNotFoundError(f"Annotation file not found: {ANN_FILE}")

with open(ANN_FILE, "r") as f:
    vocab = json.load(f)

CLASSES = vocab.get("objects", [])
RELATIONS = vocab.get("relations", [])

print(f"📘 Loaded vocab: {len(CLASSES)} objects, {len(RELATIONS)} relations.")
print(f"  Objects: {CLASSES}")
print(f"  Relations: {RELATIONS}")


# =====================
# Utility
# =====================
def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area

    if union == 0:
        return 0.0
    return inter_area / union

def top_objects(objects, iou_thresh=0.6):
    """Filter and sort objects by score; remove high-overlap duplicates."""
    objs = [o for o in objects if o["score"] >= OBJ_SCORE_THRESH]
    objs.sort(key=lambda o: o["score"], reverse=True)

    filtered = []
    for o in objs:
        keep = True
        for f in filtered:
            iou = compute_iou(o["bbox"], f["bbox"])
            if iou > iou_thresh:
                keep = False
                break
        if keep:
            filtered.append(o)
    return filtered[:TOP_OBJ_KEEP]


def top_relations(relations, num_objs):
    """Extract valid high-score relations."""
    if not relations or not isinstance(relations, dict):
        return []
    pairs, probs = relations.get("all_node_pairs", []), relations.get("all_relation", [])
    if not pairs or not probs:
        return []
    valid = []
    for i, (a, b) in enumerate(pairs):
        if a >= num_objs or b >= num_objs:
            continue
        rel_scores = probs[i]
        rel_idx = int(np.argmax(rel_scores))
        rel_score = rel_scores[rel_idx]
        if rel_score >= REL_SCORE_THRESH:
            valid.append((a, b, rel_idx, rel_score))
    return valid


def draw_graph(objects, relations, layout_cache=None, max_rel_per_obj=2):
    """Draw scene graph and return it as an image (numpy array)."""
    objects = top_objects(objects)
    rels = top_relations(relations, len(objects))

    # handle bidirection / duplicate edges
    best_edges = {}
    for a, b, rel_idx, rel_score in rels:
        if (b, a) in best_edges:
            # bidirectional, keep one merged
            prev = best_edges.pop((b, a))
            rel_name = RELATIONS[rel_idx] if rel_idx < len(RELATIONS) else f"rel_{rel_idx}"
            merged = (a, b, f"↔ {rel_name}", max(rel_score, prev[3]))
            best_edges[(a, b)] = merged
        else:
            best_edges[(a, b)] = (a, b, rel_idx, rel_score)

    # Build graph
    G = nx.DiGraph()
    node_names = [f"{CLASSES[obj['label']]}_{i}" for i, obj in enumerate(objects)]
    for name in node_names:
        G.add_node(name)

    for (a, b), (_, _, rel_idx, rel_score) in best_edges.items():
        src = node_names[a]
        dst = node_names[b]
        if isinstance(rel_idx, str):
            rel_name = rel_idx  # already a name like "next to"
        elif isinstance(rel_idx, (int, np.integer)):
            rel_name = RELATIONS[rel_idx] if rel_idx < len(RELATIONS) else f"rel_{rel_idx}"
        else:
            rel_name = str(rel_idx)

        G.add_edge(src, dst, label=f"{rel_name} ({rel_score:.2f})")

    if len(G.nodes) == 0:
        return np.zeros((*GRAPH_SIZE, 3), dtype=np.uint8), layout_cache

    # Fixed layout (stable positions)
    if layout_cache is None or set(G.nodes) != set(layout_cache.keys()):
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = layout_cache

    plt.figure(figsize=(4, 4))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1200, arrows=True, font_size=8)
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=6)
    plt.axis("off")

    graph_path = Path(OUTPUT_DIR) / "temp_graph.png"
    plt.savefig(graph_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    graph_img = cv2.imread(str(graph_path))
    graph_img = cv2.resize(graph_img, GRAPH_SIZE)
    return graph_img, pos


def overlay_graph(frame, graph_img):
    """Overlay graph image in top-right corner of frame."""
    h, w, _ = frame.shape
    gh, gw, _ = graph_img.shape
    x_offset, y_offset = w - gw - 10, 10
    frame[y_offset:y_offset + gh, x_offset:x_offset + gw] = graph_img
    return frame


# =====================
# Main
# =====================
with open(RESULT_PATH, "r") as f:
    results = json.load(f)

if len(results) == 0:
    raise RuntimeError("No results found in the JSON file.")

# prepare video writer
sample_img = cv2.imread(str(Path(IMAGE_DIR) / f"frame_{int(results[0]['file_name']):05d}.jpg")) \
    if isinstance(results[0]["file_name"], int) else cv2.imread(str(Path(IMAGE_DIR) / str(results[0]['file_name'])))
height, width = sample_img.shape[:2]
fps = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (width, height))

layout_cache = None

for idx, item in enumerate(results):
    # image
    file_name = item["file_name"]
    img_name = f"frame_{int(file_name):05d}.jpg" if not str(file_name).endswith(".jpg") else str(file_name)
    img_path = str(Path(IMAGE_DIR) / img_name)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"⚠️ Skipping {img_name}")
        continue

    # draw graph instead of arrows
    graph_img, layout_cache = draw_graph(item["objects"], item.get("relations"), layout_cache)

    # draw object boxes only
    objs = top_objects(item["objects"])
    for i, obj in enumerate(objs):
        x1, y1, x2, y2 = map(int, obj["bbox"])
        cls = CLASSES[obj["label"]] if obj["label"] < len(CLASSES) else f"cls_{obj['label']}"
        score = obj["score"]
        color = (0, int(255 * score), 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    frame_out = overlay_graph(frame, graph_img)
    out_video.write(frame_out)

    cv2.imwrite(str(Path(OUTPUT_DIR) / f"{Path(img_name).stem}_vis.jpg"), frame_out)
    print(f"✅ Frame {idx+1}/{len(results)} processed: {img_name}")

out_video.release()
print(f"\n🎞 Video saved to: {VIDEO_PATH}")
