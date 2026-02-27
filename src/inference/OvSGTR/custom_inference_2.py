import os
import json
import torch
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import torchvision.transforms as T
from torchvision.ops import nms  # Added for IOU filtering

# GroundingDINO imports
from util.slconfig import SLConfig
from groundingdino.models.GroundingDINO import build_groundingdino
from util.misc import nested_tensor_from_tensor_list

# =====================
# Constants & Defaults
# =====================
OBJ_SCORE_THRESH = 0.25
REL_SCORE_THRESH = 0.25
IOU_THRESHOLD = 0.5  # Objects with IOU > 0.5 will be filtered
TOP_OBJ_KEEP = 5

def load_model(config_path, ckpt_path, device, objects, relations):
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

import matplotlib.patheffects as path_effects
import matplotlib
matplotlib.use('Agg')

def draw_graph(res, obj_labels, rel_labels, top_indices, layout_map, graph_size, mode="balanced", top_rel_keep=5):
    G = nx.MultiDiGraph()
    labels = res["labels"]
    scores = res["scores"]

    # --- 1. Compute best score per class (if detected) ---
    best_node_per_class = {name: {"score": 0.0, "idx": None} for name in obj_labels}
    for orig_idx in top_indices:
        cls_idx = int(labels[orig_idx].item())
        full_label = obj_labels[cls_idx]
        conf_score = float(scores[orig_idx].item())
        if conf_score > best_node_per_class[full_label]["score"]:
            best_node_per_class[full_label]["score"] = conf_score
            best_node_per_class[full_label]["idx"] = int(orig_idx)

    idx_to_node = {}
    current_pos = {}

    # --- 2. Fixed circular layout for ALL classes (in normalized [0,1] coords) ---
    num_classes = max(1, len(obj_labels))
    center = np.array([0.5, 0.5])
    radius = 0.35  # smaller radius to reduce node distances
    for i, class_name in enumerate(obj_labels):
        angle = 2 * np.pi * i / num_classes
        pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
        score = best_node_per_class[class_name]["score"]
        short_label = class_name[:6].upper()
        if score > 0:
            node_display_name = f"{short_label}\n{score:.2f}"
        else:
            node_display_name = f"{short_label}\n-"

        G.add_node(node_display_name)
        current_pos[node_display_name] = pos

        # map any detection index for this class to the node_display_name
        det_idx = best_node_per_class[class_name]["idx"]
        if det_idx is not None:
            idx_to_node[int(det_idx)] = node_display_name

    # --- 3. Setup Edges ---
    simple_edge_labels = {}  # (u,v) -> (rel_idx, rel_conf, label_text)
    graph_data = res.get("graph")
    candidate_edges = []  # tuples (idx_a, idx_b, rel_idx, rel_conf)
    if graph_data and "all_node_pairs" in graph_data:
        pairs = graph_data["all_node_pairs"]
        probs = graph_data["all_relation"]
        for i, (a, b) in enumerate(pairs):
            idx_a, idx_b = int(a.item()), int(b.item())
            # Only consider if detection indices map to our fixed class nodes
            if idx_a in idx_to_node and idx_b in idx_to_node:
                rel_probs = probs[i]
                rel_idx = int(torch.argmax(rel_probs).item())
                rel_conf = float(rel_probs[rel_idx].item())
                if rel_conf >= REL_SCORE_THRESH:
                    candidate_edges.append((idx_a, idx_b, rel_idx, rel_conf))

    # Determine edges to keep based on mode
    keep_edges = []
    if mode == "object":
        # find highest scoring object among top_indices
        highest_idx = None
        if len(top_indices) > 0:
            scores = res["scores"]
            # top_indices may be a list of ints
            ti = torch.tensor(top_indices, device=scores.device) if not isinstance(top_indices, torch.Tensor) else top_indices
            highest_idx = int(ti[torch.argmax(scores[ti])].item())
        for (a, b, ridx, rconf) in candidate_edges:
            if highest_idx is not None and (a == highest_idx or b == highest_idx):
                keep_edges.append((a, b, ridx, rconf))
    elif mode == "relation":
        # pick top `top_rel_keep` edges by rel_conf
        candidate_edges.sort(key=lambda x: x[3], reverse=True)
        keep_edges = candidate_edges[:top_rel_keep]
    else:
        # balanced: union of object-focused and top relations
        # object part
        highest_idx = None
        if len(top_indices) > 0:
            scores = res["scores"]
            ti = torch.tensor(top_indices, device=scores.device) if not isinstance(top_indices, torch.Tensor) else top_indices
            highest_idx = int(ti[torch.argmax(scores[ti])].item())
        for (a, b, ridx, rconf) in candidate_edges:
            if highest_idx is not None and (a == highest_idx or b == highest_idx):
                keep_edges.append((a, b, ridx, rconf))
        # add top relation edges
        candidate_edges.sort(key=lambda x: x[3], reverse=True)
        for e in candidate_edges[:top_rel_keep]:
            if e not in keep_edges:
                keep_edges.append(e)

    # Build graph edges and labels, keeping only best relation per (u,v)
    for (idx_a, idx_b, rel_idx, rel_conf) in keep_edges:
        u, v = idx_to_node[idx_a], idx_to_node[idx_b]
        rel_name = rel_labels[rel_idx]
        G.add_edge(u, v)
        prev = simple_edge_labels.get((u, v))
        if prev is None or rel_conf > prev[1]:
            label_text = f"{rel_name}:{rel_conf:.2f}"
            simple_edge_labels[(u, v)] = (rel_idx, rel_conf, label_text)

    # Ensure positions are inside [pad, 1-pad]
    if len(current_pos) > 0:
        pad = 0.08
        for k in list(current_pos.keys()):
            v = current_pos[k]
            v = np.clip(v, -1.0, 1.0)
            # positions already centered around 0.5; just clamp into padded area
            v = np.clip(v, pad, 1 - pad)
            current_pos[k] = v
    if len(G.nodes) == 0:
        return np.zeros((graph_size[0], graph_size[1], 3), dtype=np.uint8)

    # --- 3. Rendering ---
    target_h, target_w = graph_size
    aspect_ratio = target_w / target_h

    # Render at a minimum pixel size to avoid tiny text on low-res overlays
    SS = 2  # supersample multiplier
    MIN_PIXEL = 400  # minimum rendering dimension (px)

    render_h = max(int(target_h * SS), MIN_PIXEL)
    render_w = max(int(render_h * aspect_ratio), MIN_PIXEL)

    # Create figure sized to render_x pixels using a fixed dpi
    FIG_DPI = 100
    fig = plt.figure(figsize=(render_w / FIG_DPI, render_h / FIG_DPI), dpi=FIG_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)

    # Scale visual params with rendering height (reduced sizes for crisper fit)
    node_size = max(150, int(render_h * 3))
    node_edge_width = max(1, int(render_h / 300))
    node_font = max(6, int(render_h / 55))
    edge_font = max(6, int(render_h / 50))
    arrow_mutation = max(12, int(render_h / 18))
    arrow_linewidth = max(1, int(render_h / 200))
    arrow_path_effect = max(1, int(render_h / 120))

    # NOTE: Node markers and node text are drawn later with OpenCV for pixel-crisp
    # rendering. We intentionally skip drawing node circles and labels here to
    # avoid duplicate rendering (matplotlib + OpenCV overlap).

    # Note: arrows and labels will be drawn with OpenCV below for precise
    # pixel placement. We skip drawing arrows in matplotlib to avoid
    # misaligned text and to keep a single rendering backend for edges.

    fig.canvas.draw()
    actual_w, actual_h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8").reshape((actual_h, actual_w, 3))
    plt.close(fig)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Downsample to target overlay size for crisper result
    img_resized = cv2.resize(img_bgr, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)

    # --- Post-process with OpenCV for pixel-crisp node markers and labels ---
    H, W = img_resized.shape[:2]
    # Node visual params (smaller nodes and fonts)
    node_radius = max(30, H // 30)
    node_border = max(1, H // 200)
    font_scale = max(0.35, H / 500.0)
    font_thickness = max(1, H // 250)

    # Draw nodes with filled circle + border for crispness
    for node, pos in current_pos.items():
        x = int(pos[0] * W)
        y = int(pos[1] * H)
        cv2.circle(img_resized, (x, y), node_radius, (180, 127, 31), -1, lineType=cv2.LINE_AA)
        cv2.circle(img_resized, (x, y), node_radius + node_border, (0, 0, 0), node_border, lineType=cv2.LINE_AA)

        # Draw node label centered on node (smaller two-line text)
        lines = str(node).split('\n')
        # reduce spacing so label is on the node
        total_h = 0
        sizes = []
        for ln in lines:
            (w_txt, h_txt), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            sizes.append((w_txt, h_txt))
            total_h += h_txt

        y0 = y - total_h // 2
        for i, ln in enumerate(lines):
            w_txt, h_txt = sizes[i]
            x0 = x - w_txt // 2
            cv2.rectangle(img_resized, (x0 - 3, y0 - 1), (x0 + w_txt + 3, y0 + h_txt + 1), (0, 0, 0), -1)
            cv2.putText(img_resized, ln, (x0, y0 + h_txt), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness, lineType=cv2.LINE_AA)
            y0 += h_txt

    # Draw edges and edge labels using OpenCV so labels align exactly on the
    # drawn edge. Draw lines first (under nodes), then labels on the edge.
    for (u, v), val in simple_edge_labels.items():
        # val = (rel_idx, rel_conf, label_text)
        if u in current_pos and v in current_pos:
            pu = current_pos[u]
            pv = current_pos[v]
            ux_i, uy_i = int(pu[0] * W), int(pu[1] * H)
            vx_i, vy_i = int(pv[0] * W), int(pv[1] * H)

            # draw a black thicker line as outline then colored arrow on top
            th = max(1, int(arrow_linewidth))
            cv2.line(img_resized, (ux_i, uy_i), (vx_i, vy_i), (0, 0, 0), thickness=th + 2, lineType=cv2.LINE_AA)
            cv2.arrowedLine(img_resized, (ux_i, uy_i), (vx_i, vy_i), (0, 255, 255), thickness=th, tipLength=0.12, line_type=cv2.LINE_AA)

            # compute midpoint and small perpendicular offset for label placement
            ux_f, uy_f = pu[0] * W, pu[1] * H
            vx_f, vy_f = pv[0] * W, pv[1] * H
            dx = vx_f - ux_f
            dy = vy_f - uy_f
            dist = np.hypot(dx, dy)
            mid_x = ux_f + dx * 0.5
            mid_y = uy_f + dy * 0.5

            if dist > 1e-6:
                perp_x = -dy / dist
                perp_y = dx / dist
                arc_offset = int(dist * 0.04)  # small offset so label sits on edge
                base_offset = 0
                px = int(perp_x * (base_offset + arc_offset))
                py = int(perp_y * (base_offset + arc_offset))
            else:
                px, py = 0, - (node_radius + 3)

            label_x = int(np.clip(mid_x + px, 0, W - 1))
            label_y = int(np.clip(mid_y + py, 0, H - 1))

            txt = val[2]
            (w_txt, h_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95, max(1, font_thickness))
            x0 = int(label_x - w_txt // 2)
            y0 = int(label_y - h_txt // 2)
            cv2.rectangle(img_resized, (x0 - 3, y0 - 1), (x0 + w_txt + 3, y0 + h_txt + 1), (0, 0, 0), -1)
            cv2.putText(img_resized, txt, (x0, y0 + h_txt), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.95, (0, 255, 255), max(1, font_thickness), lineType=cv2.LINE_AA)

    return img_resized

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    with open(dataset_path / "info.json", "r") as f:
        info = json.load(f)
    
    obj_list = [v["name"] for k, v in sorted(info["objects"].items(), key=lambda x: int(x[0]))]
    rel_list = [v["name"] for k, v in sorted(info["relations"].items(), key=lambda x: int(x[0]))]
    
    # Predefine Fixed Layout Map (Globally for all frames)
    # Positions objects in a circle based on class names
    LAYOUT_MAP = {}
    for i, name in enumerate(obj_list):
        angle = 2 * np.pi * i / len(obj_list)
        LAYOUT_MAP[name] = np.array([np.cos(angle), np.sin(angle)])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, postproc = load_model(args.config, args.weights, device, obj_list, rel_list)
    
    cap = cv2.VideoCapture(str(dataset_path / info["original_video"]))
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use full-screen for scenegraph outputs (one video per full frame)
    gw, gh = width, height

    out_dir = dataset_path / "inference_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create separate video writers: bbox-only, and three scenegraph modes
    bbox_writer = cv2.VideoWriter(str(out_dir / "bbox_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    sg_obj_writer = cv2.VideoWriter(str(out_dir / "scenegraph_object_focused.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    sg_rel_writer = cv2.VideoWriter(str(out_dir / "scenegraph_relation_focused.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    sg_bal_writer = cv2.VideoWriter(str(out_dir / "scenegraph_balanced.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    caption, rel_caption = ". ".join(obj_list) + ".", ". ".join(rel_list) + "."
    
    j = 0

    while cap.isOpened():
        
        ret, frame = cap.read()
        j = j + 1
        if j % 5 == 0:
            print(f"Processed {j} frames.")
            
        if not ret: break

        img_tensor = T.ToTensor()(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        samples = nested_tensor_from_tensor_list([img_tensor]).to(device)

        with torch.no_grad():
            outputs = model(samples, captions=[caption], rel_captions=[rel_caption])
        
        res = postproc(outputs, torch.tensor([[height, width]]).to(device))[0]

        # --- 1. IOU Filtering (NMS) ---
        keep_idxs = nms(res["boxes"], res["scores"], IOU_THRESHOLD)
        
        # --- 2. Score Filtering & Top K ---
        scores = res["scores"][keep_idxs]
        valid_mask = scores >= OBJ_SCORE_THRESH
        top_indices = keep_idxs[valid_mask]
        
        # Sort by score and take Top K
        if len(top_indices) > 0:
            top_indices = top_indices[torch.argsort(res["scores"][top_indices], descending=True)[:TOP_OBJ_KEEP]]

        # --- 3. Scene Graph Overlay ---
        # Render the scene graph at a higher fixed height for crisper text, then downscale
        graph_render_h = 960
        aspect = (gw / gh) if gh > 0 else 1.0
        graph_render_w = max(320, int(graph_render_h * aspect))

        # Prepare bbox-only frame
        bbox_frame = frame.copy()
        for i in (top_indices.tolist() if hasattr(top_indices, 'tolist') else top_indices):
            x1, y1, x2, y2 = map(int, res["boxes"][i].tolist())
            label = obj_list[int(res["labels"][i].item())]
            cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bbox_frame, f"{label}_{i}", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)
        bbox_writer.write(bbox_frame)

        # --- Scenegraph renders for three modes ---
        # Render high-res and downscale for crisper text
        graph_img_obj_high = draw_graph(res, obj_list, rel_list, top_indices.tolist() if hasattr(top_indices, 'tolist') else list(top_indices), LAYOUT_MAP, (graph_render_h, graph_render_w), mode="object", top_rel_keep=TOP_OBJ_KEEP)
        graph_img_rel_high = draw_graph(res, obj_list, rel_list, top_indices.tolist() if hasattr(top_indices, 'tolist') else list(top_indices), LAYOUT_MAP, (graph_render_h, graph_render_w), mode="relation", top_rel_keep=TOP_OBJ_KEEP)
        graph_img_bal_high = draw_graph(res, obj_list, rel_list, top_indices.tolist() if hasattr(top_indices, 'tolist') else list(top_indices), LAYOUT_MAP, (graph_render_h, graph_render_w), mode="balanced", top_rel_keep=TOP_OBJ_KEEP)

        graph_img_obj = cv2.resize(graph_img_obj_high, (gw, gh), interpolation=cv2.INTER_AREA)
        graph_img_rel = cv2.resize(graph_img_rel_high, (gw, gh), interpolation=cv2.INTER_AREA)
        graph_img_bal = cv2.resize(graph_img_bal_high, (gw, gh), interpolation=cv2.INTER_AREA)

        # Place graph overlay centered in a blank frame for separate scenegraph videos
        def place_graph_on_blank(graph_img):
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            y_offset = (height - gh) // 2
            x_offset = (width - gw) // 2
            blank[y_offset:y_offset+gh, x_offset:x_offset+gw] = graph_img
            return blank

        sg_obj_frame = place_graph_on_blank(graph_img_obj)
        sg_rel_frame = place_graph_on_blank(graph_img_rel)
        sg_bal_frame = place_graph_on_blank(graph_img_bal)

        sg_obj_writer.write(sg_obj_frame)
        sg_rel_writer.write(sg_rel_frame)
        sg_bal_writer.write(sg_bal_frame)

    cap.release()
    bbox_writer.release()
    sg_obj_writer.release()
    sg_rel_writer.release()
    sg_bal_writer.release()

if __name__ == "__main__":
    main()