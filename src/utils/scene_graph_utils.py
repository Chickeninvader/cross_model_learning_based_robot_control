import numpy as np
import pickle
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import matplotlib.pyplot as plt
import cv2
import os
import json
import re
import warnings

# Notebook-only imports — silently skip when running as a CLI script
try:
    from ipywidgets import *
    import IPython.display as display
    from IPython.display import Video, clear_output
except ImportError:
    pass

def load_task_info(task_path):
    """Load task configuration from info.json.
    
    Args:
        task_path: Path to task directory (e.g., /workspace/datasets/rlbench/stack_cups)
    
    Returns:
        dict with keys: 'objects', 'relations', 'object_mapping'
    """
    info_path = os.path.join(task_path, 'info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"info.json not found at {info_path}")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Convert object_mapping keys from strings to integers
    if 'object_mapping' in info:
        info['object_mapping'] = {int(k): v for k, v in info['object_mapping'].items()}
    
    # Extract object names list
    if 'objects' in info:
        info['object_names'] = [obj['name'] for obj in sorted(info['objects'].values(), key=lambda x: int(list(info['objects'].keys())[list(info['objects'].values()).index(x)]))]
    
    # Extract relationship types list
    if 'relations' in info:
        info['relationship_types'] = [rel['name'] for rel in sorted(info['relations'].values(), key=lambda x: int(list(info['relations'].keys())[list(info['relations'].values()).index(x)]))]
    
    return info


def extract_gripper_states(demo):
    """Extract gripper open/closed states from demo data.
    
    Args:
        demo: List of observation dictionaries from low_dim_obs.pkl
    
    Returns:
        List of bool (True = gripper closed, False = open), one per frame
    """
    gripper_states = []
    for obs in demo:
        # RLBench gripper_open: 1.0 = open, 0.0 = closed
        gripper_open = obs.gripper_open
        gripper_closed = gripper_open < 0.5  # Threshold at 0.5
        gripper_states.append(gripper_closed)
    return gripper_states


def load_episode_data(task_name, variation, episode, DATASET_PATH='/workspace/rlbench_data', CAMERA='front'):
    """Load episode data including observations and images."""
    episode_path = f"{DATASET_PATH}/{task_name}/variation{variation}/episodes/episode{episode}"
    
    # Load low-dimensional observations
    with open(f"{episode_path}/low_dim_obs.pkl", 'rb') as f:
        demo = pickle.load(f)
    
    # Load image paths
    image_data = {
        'rgb': [],
        'depth': [],
        'mask': []
    }
    
    rgb_dir = f"{episode_path}/{CAMERA}_rgb"
    depth_dir = f"{episode_path}/{CAMERA}_depth"
    mask_dir = f"{episode_path}/{CAMERA}_mask"
    
    # FIX: Use numeric sorting instead of string sorting
    def numeric_sort(filename):
        """Extract numeric part from filename for proper sorting."""
        import re
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    if os.path.exists(rgb_dir):
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')], key=numeric_sort)
        image_data['rgb'] = [f"{rgb_dir}/{f}" for f in rgb_files]
    
    if os.path.exists(depth_dir):
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')], key=numeric_sort)
        image_data['depth'] = [f"{depth_dir}/{f}" for f in depth_files]
    
    if os.path.exists(mask_dir):
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')], key=numeric_sort)
        image_data['mask'] = [f"{mask_dir}/{f}" for f in mask_files]
    
    return demo, image_data

def decode_mask_image(mask_path):
    """
    Decodes RLBench RGB masks into unique integer IDs.
    RLBench/PyRep encodes Object IDs using: ID = R + (G * 256) + (B * 65536)
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found at {mask_path}")
    mask_img = PILImage.open(mask_path).convert('RGB')
    mask_array = np.array(mask_img).astype(np.int32)
    decoded_mask = mask_array[:, :, 0] + \
                   (mask_array[:, :, 1] * 256) + \
                   (mask_array[:, :, 2] * 65536)
    return decoded_mask

def calculate_object_position(mask_handles, handle_list):
    """Calculate position as center of mass of all pixels belonging to given handles."""
    if not handle_list:
        return None
    obj_mask = np.isin(mask_handles, handle_list)
    coords = np.where(obj_mask)
    if len(coords[0]) == 0:
        return None
    center_y = int(np.mean(coords[0]))
    center_x = int(np.mean(coords[1]))
    return (center_x, center_y)


def _base_object_type(object_name):
    """Convert object id/name to a generic type string."""
    # Example: cup_1 -> cup, rubbish -> rubbish, robot -> robot
    base = re.sub(r'_\d+$', '', str(object_name))
    return base.replace('_', ' ')


def _rgb_to_color_name(rgb):
    """Map an RGB triplet to a coarse color name."""
    rgb_u8 = np.uint8([[rgb]])
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Low saturation colors first.
    if s < 30:
        if v < 45:
            return "black"
        if v > 210:
            return "white"
        return "gray"

    # Brown is dark orange/yellow.
    if 8 <= h <= 25 and v < 160:
        return "brown"
    if h < 10 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 105:
        return "cyan"
    if h < 135:
        return "blue"
    if h < 160:
        return "purple"
    return "pink"


def estimate_object_colors(
    image_data,
    objects,
    handles_by_object,
    start_frame=0,
    max_frames=30,
    min_pixels=40,
):
    """Estimate a representative color per object from RGB + mask frames."""
    color_info = {}
    if not image_data.get('rgb') or not image_data.get('mask'):
        return color_info

    n_frames = min(len(image_data['rgb']), len(image_data['mask']))
    if start_frame >= n_frames:
        return color_info

    end_frame = min(n_frames, start_frame + max_frames)
    per_object_rgb_samples = {name: [] for name in objects}

    for frame_idx in range(start_frame, end_frame):
        rgb_bgr = cv2.imread(image_data['rgb'][frame_idx])
        if rgb_bgr is None:
            continue
        rgb_frame = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        mask_handles = decode_mask_image(image_data['mask'][frame_idx])

        for name in objects:
            handle_ids = handles_by_object.get(name, [])
            if not handle_ids:
                continue
            obj_mask = np.isin(mask_handles, handle_ids)
            if int(obj_mask.sum()) < min_pixels:
                continue
            pixels = rgb_frame[obj_mask]
            if pixels.size == 0:
                continue
            per_object_rgb_samples[name].append(np.median(pixels, axis=0))

    for name in objects:
        samples = per_object_rgb_samples.get(name, [])
        if not samples:
            continue
        rgb_med = np.median(np.vstack(samples), axis=0)
        rgb_triplet = [int(round(c)) for c in rgb_med.tolist()]
        color_info[name] = {
            "color": _rgb_to_color_name(rgb_triplet),
            "rgb": rgb_triplet,
        }

    return color_info


def build_scene_graph_objects(
    objects,
    image_data=None,
    handles_by_object=None,
    start_frame=0,
):
    """Build scene-graph object metadata with optional color attributes."""
    object_meta = {}
    color_info = {}
    if image_data is not None and handles_by_object is not None:
        color_info = estimate_object_colors(
            image_data=image_data,
            objects=objects,
            handles_by_object=handles_by_object,
            start_frame=start_frame,
        )

    for name in objects:
        base_type = _base_object_type(name)
        meta = {"type": base_type}
        pretty_name = str(name).replace('_', ' ')
        color = color_info.get(name, {}).get("color")
        if color:
            meta["attributes"] = {"color": color}
            # Keep instance disambiguation (e.g. "cup 1") in language labels.
            meta["display_name"] = f"{color} {pretty_name}".strip()
        object_meta[name] = meta
    return object_meta

def extract_positions_over_time(image_data, objects, handles_by_object, start_frame=0):
    """Extract positions for all OBJECTS over the episode.
    
    When an object is occluded (not visible in the mask), the last known position
    is used instead of None. This assumes the object's position does not change
    while occluded.
    """
    positions = {name: [] for name in objects}
    positions['frames'] = []
    last_known_position = {name: None for name in objects}

    for i, mask_path in enumerate(image_data['mask']):
        if i < start_frame:
            continue
        mask_handles = decode_mask_image(mask_path)
        for name in objects:
            pos = calculate_object_position(mask_handles, handles_by_object[name])
            # If visible, update last known position; if occluded, use last known position
            if pos is not None:
                last_known_position[name] = pos
                positions[name].append(pos)
            else:
                positions[name].append(last_known_position[name])
        positions['frames'].append(i)

    return positions

# Video creation + object-id mask encoding
#
# Each OBJECTS entry gets a 1-based index. In the mask video, pixels belonging
# to that object are coloured (0, 0, index) in RGB. Background = (0, 0, 0).
# A model reads the blue channel to get the object ID, then looks up
# object_color_map.json to know which object it belongs to.

def build_object_index_map(objects, object_mapping):
    """Build sequential index per OBJECTS entry + per-handle lookup."""
    handle_to_index = {}
    object_info = {}
    for idx_0, name in enumerate(objects):
        idx = idx_0 + 1  # 1-based
        label = f"obj_{idx}"
        handles = sorted([h for h, n in object_mapping.items() if n == name])
        for h in handles:
            handle_to_index[h] = idx
        object_info[label] = {
            'name': name,
            'index': idx,
            'handle_ids': handles,
            'mask_color_rgb': [0, 0, idx],
        }
    return handle_to_index, object_info


# Vivid colour palette for mask display — maps 1-based object indices to
# bright, easily distinguishable RGB colours.
VIVID_PALETTE = [
    [0, 0, 0],          # 0 = background (black)
    [255, 50, 50],      # 1 = red
    [50, 120, 255],     # 2 = blue
    [50, 220, 50],      # 3 = green
    [255, 180, 30],     # 4 = orange
    [180, 50, 255],     # 5 = purple
    [0, 220, 220],      # 6 = cyan
    [255, 105, 180],    # 7 = pink
    [160, 110, 50],     # 8 = brown
    [255, 255, 80],     # 9 = yellow
    [100, 255, 180],    # 10 = mint
]


def mask_id_to_vivid(mask_id_image):
    """Convert a (H, W, 3) mask-ID image (pixel = [0,0,id]) to vivid colours.

    The *encoded* mask stores object index in the blue channel (RGB order) or
    red channel (BGR order from OpenCV).  This function reads the first non-zero
    channel per pixel and maps it through VIVID_PALETTE.

    Args:
        mask_id_image: (H, W, 3) uint8 array with encoded IDs.

    Returns:
        (H, W, 3) uint8 RGB image with vivid colours.
    """
    h, w = mask_id_image.shape[:2]
    # Object index is in whichever channel is non-zero (typically channel 0 or 2)
    idx_map = mask_id_image.max(axis=2).astype(np.int32)
    vivid = np.zeros((h, w, 3), dtype=np.uint8)
    for obj_idx in np.unique(idx_map):
        if obj_idx == 0:
            continue
        color = VIVID_PALETTE[obj_idx % len(VIVID_PALETTE)]
        vivid[idx_map == obj_idx] = color
    return vivid


def create_videos(image_data, objects, object_mapping, handles_by_object,
                  scene_graph, output_dir, start_frame=0):
    """Create overlay + encoded-mask + vivid-mask videos and save object_color_map.json."""
    if not image_data['rgb']:
        print("No RGB images"); return None

    handle_to_index, object_info = build_object_index_map(objects, object_mapping)

    # Save JSON
    json_path = os.path.join(output_dir, 'object_color_map.json')
    with open(json_path, 'w') as f:
        json.dump(object_info, f, indent=2)
    print("✓ object_color_map.json saved")
    for lbl, info in object_info.items():
        print(f"  {lbl}: {info['name']}, handles={info['handle_ids']}, mask=(0,0,{info['index']})")

    first_frame = cv2.imread(image_data['rgb'][start_frame])
    if first_frame is None:
        print("Cannot read first frame"); return None
    h, w = first_frame.shape[:2]

    os.makedirs(output_dir, exist_ok=True)
    overlay_path = os.path.join(output_dir, 'episode_overlay.mp4')
    mask_path = os.path.join(output_dir, 'episode_mask.mp4')

    # Overlay colors per object (for visual video only)
    _palette = [[255,0,0],[0,0,255],[0,200,0],[255,165,0],
                [128,0,128],[0,200,200],[255,105,180],[139,69,19]]
    obj_bgr = {}
    for i, name in enumerate(objects):
        rgb = _palette[i % len(_palette)]
        obj_bgr[name] = [rgb[2], rgb[1], rgb[0]]  # convert to BGR

    # Use imageio with ffmpeg for browser-compatible h264 MP4
    try:
        import imageio
        USE_IO = True
        ow = imageio.get_writer(overlay_path, fps=10, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p'])
        mw = imageio.get_writer(mask_path, fps=10, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p'])
    except Exception:
        USE_IO = False
        cc = cv2.VideoWriter_fourcc(*'mp4v')
        ow = cv2.VideoWriter(overlay_path, cc, 10.0, (w, h))
        mw = cv2.VideoWriter(mask_path, cc, 10.0, (w, h))

    n = 0
    for i, rgb_path in enumerate(image_data['rgb'][start_frame:], start=start_frame):
        frame = cv2.imread(rgb_path)
        if frame is None: continue
        mask_decoded = decode_mask_image(image_data['mask'][i])

        # Overlay frame
        ol = np.zeros_like(frame, dtype=np.uint8)
        for name in objects:
            bgr = obj_bgr[name]
            for hid in handles_by_object[name]:
                ol[mask_decoded == hid] = bgr
        blended = cv2.addWeighted(frame, 0.7, ol, 0.3, 0)

        # Mask frame: (0, 0, index) in RGB — this is the encoded version
        mf = np.zeros_like(frame, dtype=np.uint8)
        for hid, idx in handle_to_index.items():
            mf[mask_decoded == hid] = [idx, 0, 0]  # BGR → RGB = (0, 0, idx)

        # Scene graph overlay
        pos_idx = i - start_frame
        if scene_graph and pos_idx < len(scene_graph['frames']):
            fd = scene_graph['frames'][pos_idx]
            for oid, od in fd['objects'].items():
                if od['visible']:
                    x, y = int(od['position'][0]), int(od['position'][1])
                    cv2.circle(blended, (x, y), 8, (255, 255, 255), 2)
                    cv2.putText(blended, oid, (x+10, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for r in fd['relationships']:
                o1, o2 = r['object1'], r['object2']
                if o1 in fd['objects'] and o2 in fd['objects']:
                    p1 = (int(fd['objects'][o1]['position'][0]), int(fd['objects'][o1]['position'][1]))
                    p2 = (int(fd['objects'][o2]['position'][0]), int(fd['objects'][o2]['position'][1]))
                    cv2.arrowedLine(blended, p1, p2, (0, 255, 255), 2, tipLength=0.1)
                    mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
                    cv2.putText(blended, r['type'], (mx+5, my-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(blended, f"Frame {i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if USE_IO:
            ow.append_data(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            # Write vivid-coloured version for the mask video (easier to see)
            mf_rgb = cv2.cvtColor(mf, cv2.COLOR_BGR2RGB)
            mf_vivid = mask_id_to_vivid(mf_rgb)
            mw.append_data(mf_vivid)
        else:
            ow.write(blended); mw.write(mf)
        n += 1

    if USE_IO: ow.close(); mw.close()
    else: ow.release(); mw.release()

    # Also save encoded mask video (for training pipelines)
    encoded_mask_path = os.path.join(output_dir, 'episode_mask_encoded.mp4')
    try:
        import imageio
        ew = imageio.get_writer(encoded_mask_path, fps=10, codec='libx264',
                                output_params=['-pix_fmt', 'yuv420p'])
        for i, _ in enumerate(image_data['rgb'][start_frame:], start=start_frame):
            mask_decoded = decode_mask_image(image_data['mask'][i])
            mf = np.zeros((h, w, 3), dtype=np.uint8)
            for hid, idx in handle_to_index.items():
                mf[mask_decoded == hid] = [0, 0, idx]  # RGB = (0, 0, idx)
            ew.append_data(mf)
        ew.close()
        print(f"✓ Encoded mask: {encoded_mask_path} ({n} frames) — pixel=(0,0,object_index)")
    except Exception as e:
        print(f"⚠ Could not write encoded mask video: {e}")
        encoded_mask_path = None

    print(f"\n✓ Overlay video: {overlay_path} ({n} frames)")
    print(f"✓ Mask video (vivid): {mask_path} ({n} frames) — for visualization")
    print(f"  Mask encoding in encoded file: pixel = (0, 0, object_index). Background = (0,0,0).")
    return overlay_path, mask_path, json_path


# ============================================================================
# REUSABLE FUNCTIONS — used by both the notebook and CLI inference scripts
# ============================================================================

def load_rlbench_images(variation_dir, episode=0, camera='front'):
    """Load sorted RGB and mask image paths from an RLBench episode directory.

    Args:
        variation_dir: Path to the variation directory
                       (e.g. .../stack_cups/variation0)
        episode:       Episode number (default 0)
        camera:        Camera name (default 'front')

    Returns:
        (rgb_paths, mask_paths) — sorted lists of absolute file paths.
    """
    episode_dir = os.path.join(variation_dir, 'episodes', f'episode{episode}')

    def _num_key(fn):
        nums = re.findall(r'\d+', fn)
        return int(nums[0]) if nums else 0

    rgb_dir = os.path.join(episode_dir, f'{camera}_rgb')
    mask_dir = os.path.join(episode_dir, f'{camera}_mask')

    rgb_paths = []
    if os.path.exists(rgb_dir):
        files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')],
                       key=_num_key)
        rgb_paths = [os.path.join(rgb_dir, f) for f in files]

    mask_paths = []
    if os.path.exists(mask_dir):
        files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')],
                       key=_num_key)
        mask_paths = [os.path.join(mask_dir, f) for f in files]

    return rgb_paths, mask_paths


def create_detection_overlay_video(rgb_paths, frame_results, output_path,
                                   fps=20.0):
    """Create an overlay video: original RGB + bboxes + centre dots + arrows.

    Args:
        rgb_paths:     list of RGB image paths (one per frame).
        frame_results: list of dicts, one per frame.  Each dict must contain:
            'frame_id':       int
            'objects':        {name: {'position': [x,y], 'visible': bool,
                                      'bbox': [x1,y1,x2,y2] (optional),
                                      'score': float (optional)}}
            'relationships':  [{'object1': str, 'object2': str,
                                'type': str, 'score': float (optional)}]
        output_path:   where to write the .mp4 file.
        fps:           frames per second.

    Returns:
        output_path on success, None on failure.
    """
    if not rgb_paths or not frame_results:
        print("No data for overlay video")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    first = cv2.imread(rgb_paths[0])
    if first is None:
        print(f"Cannot read {rgb_paths[0]}")
        return None
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Per-object colour palette (BGR)
    _pal = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 165, 255),
            (128, 0, 128), (200, 200, 0), (180, 105, 255), (19, 69, 139)]
    obj_names = list(frame_results[0]['objects'].keys()) if frame_results else []
    obj_col = {n: _pal[i % len(_pal)] for i, n in enumerate(obj_names)}

    n_written = 0
    for rgb_path, fr in zip(rgb_paths, frame_results):
        frame = cv2.imread(rgb_path)
        if frame is None:
            continue

        # --- bounding boxes + centres ---
        for name, od in fr['objects'].items():
            if not od.get('visible', False):
                continue
            col = obj_col.get(name, (255, 255, 255))

            if 'bbox' in od:
                x1, y1, x2, y2 = [int(v) for v in od['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

            cx, cy = int(od['position'][0]), int(od['position'][1])
            cv2.circle(frame, (cx, cy), 6, col, -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

            label = name
            if 'score' in od:
                label += f" {od['score']:.2f}"
            cv2.putText(frame, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        # --- relationship arrows ---
        for rel in fr.get('relationships', []):
            o1, o2 = rel['object1'], rel['object2']
            objs = fr['objects']
            if (o1 in objs and o2 in objs
                    and objs[o1].get('visible') and objs[o2].get('visible')):
                p1 = (int(objs[o1]['position'][0]),
                      int(objs[o1]['position'][1]))
                p2 = (int(objs[o2]['position'][0]),
                      int(objs[o2]['position'][1]))
                cv2.arrowedLine(frame, p1, p2, (0, 255, 255), 2,
                                tipLength=0.1)
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                rtxt = rel['type']
                if 'score' in rel:
                    rtxt += f" {rel['score']:.2f}"
                cv2.putText(frame, rtxt, (mx + 5, my - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),
                            1, cv2.LINE_AA)

        cv2.putText(frame, f"Frame {fr.get('frame_id', n_written)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        writer.write(frame)
        n_written += 1

    writer.release()
    print(f"Overlay video saved: {output_path} ({n_written} frames)")
    return output_path


def create_rlbench_mask_video(mask_paths, object_mapping, objects, output_path,
                              fps=20.0):
    """Create a mask-ID video from RLBench ground-truth masks.

    Each tracked object is assigned a 1-based index.  In the output video every
    pixel belonging to that object is set to ``(0, 0, index)`` in RGB
    (``(index, 0, 0)`` in BGR as written by OpenCV).  Background = black.

    Args:
        mask_paths:      list of mask-image paths (same order as RGB frames).
        object_mapping:  ``{handle_id (int): object_name (str), ...}``
        objects:         ordered list of object names (defines 1-based index).
        output_path:     where to write the .mp4 file.
        fps:             frames per second.

    Returns:
        output_path on success, None on failure.
    """
    if not mask_paths:
        print("No mask data for mask video")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # handle-ID → 1-based object index
    handle_to_idx = {}
    for idx_0, name in enumerate(objects):
        obj_idx = idx_0 + 1
        for hid, hname in object_mapping.items():
            if hname == name:
                handle_to_idx[int(hid)] = obj_idx

    first_mask = decode_mask_image(mask_paths[0])
    h, w = first_mask.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    n_written = 0
    for mpath in mask_paths:
        decoded = decode_mask_image(mpath)
        mf = np.zeros((h, w, 3), dtype=np.uint8)
        for hid, oidx in handle_to_idx.items():
            mf[decoded == hid] = [oidx, 0, 0]  # BGR channel order
        writer.write(mf)
        n_written += 1

    writer.release()
    print(f"Mask video saved: {output_path} ({n_written} frames)")
    return output_path


def save_relationship_template(scene_graph, gripper_states, output_path):
    """Save relationship template based on gripper state transitions.
    
    This extracts relationship patterns at each gripper transition point,
    allowing reuse across different variations/episodes of the same task.
    
    Args:
        scene_graph: Scene graph dict with 'frames' list
        gripper_states: List of bool (gripper closed states)
        output_path: Where to save the template JSON
    
    Returns:
        Template dict
    """
    template = {
        'description': 'Relationship template based on gripper transitions',
        'transitions': []
    }
    
    # Find all gripper state changes
    gripper_changes = []
    for i in range(1, len(gripper_states)):
        if gripper_states[i] != gripper_states[i-1]:
            gripper_changes.append({
                'frame': i,
                'from_state': 'open' if not gripper_states[i-1] else 'closed',
                'to_state': 'closed' if gripper_states[i] else 'open'
            })
    
    # Build a frame-id -> frame map for robust lookup (frame ids may not be perfectly aligned)
    frames_sorted = sorted(scene_graph.get('frames', []), key=lambda f: f.get('frame_id', -1))
    frame_ids = [f.get('frame_id', -1) for f in frames_sorted]

    # For each transition, capture a goal relationship snapshot for its segment.
    # Segment i spans [transition_i, transition_{i+1}) in frame-id space.
    # We prefer the latest non-empty relationships in that segment so tasks where
    # state changes occur near the segment end are captured correctly.
    for trans_idx, trans in enumerate(gripper_changes):
        start_fid = trans['frame']
        if trans_idx + 1 < len(gripper_changes):
            end_fid = gripper_changes[trans_idx + 1]['frame'] - 1
        else:
            end_fid = frame_ids[-1] if frame_ids else start_fid

        candidate_frames = [
            f for f in frames_sorted
            if start_fid <= f.get('frame_id', -1) <= end_fid
        ]

        if not candidate_frames and frames_sorted:
            # Fallback to nearest available frame >= start, otherwise last frame.
            later = [f for f in frames_sorted if f.get('frame_id', -1) >= start_fid]
            candidate_frames = later[:1] if later else [frames_sorted[-1]]

        selected_frame = None
        for f in reversed(candidate_frames):
            if f.get('relationships'):
                selected_frame = f
                break
        if selected_frame is None and candidate_frames:
            selected_frame = candidate_frames[-1]

        # Fallback: if segment frames are all empty, use nearest prior
        # non-empty frame up to the segment end. This helps tasks where
        # annotation was placed just before the transition frame (often the
        # case when the transition happens at the last frame).
        if (
            selected_frame is not None
            and not selected_frame.get('relationships')
            and frames_sorted
        ):
            prior = [
                f for f in frames_sorted
                if f.get('frame_id', -1) <= end_fid and f.get('relationships')
            ]
            if prior:
                selected_frame = prior[-1]
                print(
                    f"  transition {trans_idx}: segment empty, "
                    f"fallback to prior annotated frame {selected_frame.get('frame_id')}"
                )

        relationships = []
        selected_fid = start_fid
        if selected_frame is not None:
            relationships = [r.copy() for r in selected_frame.get('relationships', [])]
            selected_fid = selected_frame.get('frame_id', start_fid)

        template['transitions'].append({
            'transition_index': trans_idx,
            'from_state': trans['from_state'],
            'to_state': trans['to_state'],
            'relationships': relationships
        })
        print(
            f"  transition {trans_idx}: {trans['from_state']}->{trans['to_state']} "
            f"segment [{start_fid}, {end_fid}] -> frame {selected_fid} "
            f"({len(relationships)} rels)"
        )
    
    # Fallback: if this is a single-transition template and the selected segment
    # was empty, but the scene graph has annotated relationships elsewhere, use
    # the latest annotated relationships to avoid silent empty templates.
    total_rels = sum(len(t.get('relationships', [])) for t in template['transitions'])
    if len(template['transitions']) == 1 and total_rels == 0 and frames_sorted:
        nonempty_anywhere = [f for f in frames_sorted if f.get('relationships')]
        if nonempty_anywhere:
            fallback_rels = [r.copy() for r in nonempty_anywhere[-1].get('relationships', [])]
            template['transitions'][0]['relationships'] = fallback_rels
            total_rels = len(fallback_rels)
            print(
                "  fallback(single-transition): using latest annotated frame "
                f"{nonempty_anywhere[-1].get('frame_id')} ({total_rels} rels)"
            )

    # Guard: avoid silently saving an all-empty template.
    total_rels = sum(len(t.get('relationships', [])) for t in template['transitions'])
    if template['transitions'] and total_rels == 0:
        raise ValueError(
            "Template would be empty (0 relationships across all transitions). "
            "Please annotate at least one relationship before saving."
        )

    # Save template
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"✓ Relationship template saved: {output_path}")
    print(f"  Captured {len(template['transitions'])} gripper transitions")
    
    return template


def apply_relationship_template(
    template_path,
    scene_graph,
    gripper_states,
    *,
    single_transition_apply_from_start=False,
    single_transition_apply_at_end=True,
):
    """Apply a relationship template to a scene graph based on gripper transitions.
    
    Args:
        template_path: Path to template JSON file
        scene_graph: Scene graph dict to modify (in-place)
        gripper_states: List of bool (gripper closed states) for this variation
        single_transition_apply_from_start: If True, when exactly one template
            transition is matched, apply that relationship segment from the
            first frame instead of the matched gripper-change frame. This
            helps tasks where the semantic transition spans most of the
            episode but the gripper change occurs only near the end.
        single_transition_apply_at_end: If True, when template has exactly one
            transition, apply its relationships at the last frame only. This
            is useful for single-goal tasks where final relation should appear
            at the episode end, not at an earlier gripper change.
    
    Returns:
        Modified scene_graph
    """
    # Load template
    with open(template_path, 'r') as f:
        template = json.load(f)

    # Guard: empty templates cause silent no-op; fail fast so users can fix
    # the source annotation/template.
    if template.get('transitions'):
        total_template_rels = sum(len(t.get('relationships', [])) for t in template['transitions'])
        if total_template_rels == 0:
            raise ValueError(
                f"Template has {len(template['transitions'])} transitions but 0 relationships. "
                "Re-create template from an annotated scene graph."
            )
    
    # Find gripper transitions in current data
    gripper_changes = []
    for i in range(1, len(gripper_states)):
        if gripper_states[i] != gripper_states[i-1]:
            gripper_changes.append({
                'frame': i,
                'from_state': 'open' if not gripper_states[i-1] else 'closed',
                'to_state': 'closed' if gripper_states[i] else 'open'
            })
    
    print(f"Template has {len(template['transitions'])} transitions")
    print(f"Current variation has {len(gripper_changes)} transitions")

    # Build a list of (start_frame_id, relationships) segments by matching
    # template transitions to current transitions in order.
    # Segment 0: frame 0 up to first transition → no relationships (empty)
    # Segment N: from transition N frame onwards → template transition N's relationships
    segments = []  # list of (start_frame_id, relationships_list)

    applied_count = 0

    # Special handling for single-transition templates: final relation at end.
    if (
        single_transition_apply_at_end
        and len(template.get('transitions', [])) == 1
        and scene_graph.get('frames')
    ):
        last_frame_id = scene_graph['frames'][-1]['frame_id']
        rels = [r.copy() for r in template['transitions'][0].get('relationships', [])]
        segments.append((last_frame_id, rels))
        applied_count = 1
        print(
            "ℹ Single-transition end mode: applying template relationships "
            f"at last frame {last_frame_id}."
        )
    else:
        for tmpl_trans in template['transitions']:
            trans_idx = tmpl_trans['transition_index']

            if trans_idx >= len(gripper_changes):
                print(f"⚠ No matching transition {trans_idx} in current variation (only {len(gripper_changes)} found)")
                continue

            current_trans = gripper_changes[trans_idx]

            if (current_trans['from_state'] != tmpl_trans['from_state'] or
                    current_trans['to_state'] != tmpl_trans['to_state']):
                print(f"⚠ Transition {trans_idx} type mismatch: "
                      f"template={tmpl_trans['from_state']}->{tmpl_trans['to_state']}, "
                      f"current={current_trans['from_state']}->{current_trans['to_state']}")
                continue

            segments.append((current_trans['frame'], tmpl_trans['relationships']))
            applied_count += 1
            print(f"✓ Matched transition {trans_idx}: "
                  f"{tmpl_trans['from_state']} -> {tmpl_trans['to_state']} "
                  f"at frame {current_trans['frame']}")

    # Sort segments by start frame so we can do a simple forward scan
    segments.sort(key=lambda s: s[0])

    # Edge case: a single semantic transition often corresponds to "start -> goal"
    # for the full episode, while the physical gripper change may happen very late.
    # In that case, applying only from the late frame leaves only a few frames labeled.
    if (
        single_transition_apply_from_start
        and len(template.get('transitions', [])) == 1
        and len(segments) == 1
        and scene_graph.get('frames')
    ):
        first_frame_id = scene_graph['frames'][0]['frame_id']
        rels = segments[0][1]
        segments = [(first_frame_id, rels)]
        print(
            "ℹ Single-transition mode: applying template relationships "
            f"from first frame {first_frame_id}."
        )

    # Walk every frame in the scene graph and assign relationships based on
    # which segment it falls into.  Before the first segment → empty list.
    seg_idx = 0
    current_rels = []  # relationships active before any transition
    for frame in scene_graph['frames']:
        fid = frame['frame_id']
        # Advance to the next segment whose start_frame has been reached
        while seg_idx < len(segments) and fid >= segments[seg_idx][0]:
            current_rels = [r.copy() for r in segments[seg_idx][1]]
            seg_idx += 1
        # Simply set — no merging, no selective removal
        frame['relationships'] = [r.copy() for r in current_rels]

    print(f"\n✓ Applied {applied_count} transitions from template to {len(scene_graph['frames'])} frames")
    return scene_graph
