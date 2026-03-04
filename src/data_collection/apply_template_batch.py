#!/usr/bin/env python3
"""
Batch apply relationship template to multiple task variations.

This script applies a saved relationship template to multiple variations
of a task, automatically creating scene graphs based on gripper transitions.
It also generates visualization videos and object color maps.

Generated outputs per variation:
    - {task}_scene_graph.json       : Scene graph with relationships
    - episode_overlay.mp4            : RGB + mask overlay + scene graph annotations
    - episode_mask.mp4               : Vivid color-coded mask video
    - episode_mask_encoded.mp4       : Encoded mask with object indices
    - object_color_map.json          : Object-to-color mapping

Usage:
    python apply_template_batch.py --task stack_cups --variations 1 2 3 --episode 0

Author: Auto-generated scene graph tool
"""

import argparse
import os
import sys
import json

# Add src to path
sys.path.insert(0, '/workspace/src')
import utils.scene_graph_utils as utils


def process_variation(task_name, variation, episode, dataset_path, camera, template_path, generate_videos=True):
    """Process a single variation using the template.
    
    Args:
        generate_videos: If True, generate visualization videos (default: True)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {task_name} / variation {variation} / episode {episode}")
    print(f"{'='*60}")
    
    # Load task info
    task_path = os.path.join(dataset_path, task_name)
    task_info = utils.load_task_info(task_path)
    
    OBJECTS = task_info['object_names']
    OBJECT_MAPPING = task_info['object_mapping']
    
    # Load variation data
    try:
        demo, image_data = utils.load_episode_data(
            task_name, variation, episode, 
            DATASET_PATH=dataset_path, CAMERA=camera
        )
    except Exception as e:
        print(f"❌ Failed to load variation {variation}: {e}")
        return False
    
    # Extract gripper states
    gripper_states = utils.extract_gripper_states(demo)
    
    # Find gripper change frames
    gripper_change_frames = []
    for i in range(1, len(gripper_states)):
        if gripper_states[i] != gripper_states[i-1]:
            gripper_change_frames.append(i)
    
    print(f"Gripper state changes at frames: {gripper_change_frames}")
    
    # Build handles_by_object
    handles_by_object = {name: [] for name in OBJECTS}
    for handle_id, obj_name in OBJECT_MAPPING.items():
        if obj_name in handles_by_object:
            handles_by_object[obj_name].append(handle_id)
    for name in OBJECTS:
        handles_by_object[name].sort()
    
    # Extract positions
    positions = utils.extract_positions_over_time(
        image_data, OBJECTS, handles_by_object, start_frame=20
    )
    
    # Build scene graph
    scene_graph = {
        'objects': {name: {'type': name} for name in OBJECTS},
        'frames': []
    }
    
    for pos_idx, frame_id in enumerate(positions['frames']):
        frame_data = {
            'frame_id': frame_id,
            'timestamp': frame_id / 10.0,
            'objects': {},
            'relationships': []
        }
        for name in OBJECTS:
            pos = positions[name][pos_idx]
            frame_data['objects'][name] = {
                'position': list(pos) if pos is not None else [0, 0],
                'visible': pos is not None
            }
        if frame_id < len(gripper_states):
            frame_data['gripper_closed'] = gripper_states[frame_id]
        
        scene_graph['frames'].append(frame_data)
    
    # Apply template
    try:
        utils.apply_relationship_template(template_path, scene_graph, gripper_states)
    except Exception as e:
        print(f"❌ Failed to apply template: {e}")
        return False
    
    # Save scene graph
    output_dir = os.path.join(dataset_path, task_name, f'variation{variation}')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'{task_name}_scene_graph.json')
    with open(output_file, 'w') as f:
        json.dump(scene_graph, f, indent=2)
    
    print(f"✓ Scene graph saved: {output_file}")
    
    # Generate videos and object color map
    if generate_videos:
        print("\nGenerating videos...")
        try:
            result = utils.create_videos(
                image_data, OBJECTS, OBJECT_MAPPING, handles_by_object,
                scene_graph, output_dir, start_frame=20
            )
            
            if result:
                overlay_path, mask_path, json_path = result
                print(f"✓ Overlay video: {overlay_path}")
                print(f"✓ Mask video: {mask_path}")
                print(f"✓ Object color map: {json_path}")
            else:
                print("⚠ Failed to generate videos")
                
        except Exception as e:
            print(f"⚠ Error generating videos: {e}")
            # Continue anyway - scene graph is still saved
    else:
        print("(Skipping video generation)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch apply relationship template to multiple variations. "
                    "Generates scene graphs, visualization videos, and object color maps."
    )
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (e.g., stack_cups)')
    parser.add_argument('--variations', type=int, nargs='+', required=True,
                       help='Variation numbers to process (e.g., 1 2 3)')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode number (default: 0)')
    parser.add_argument('--dataset_path', type=str, default='/workspace/datasets/rlbench',
                       help='Path to RLBench dataset')
    parser.add_argument('--camera', type=str, default='front',
                       choices=['front', 'wrist', 'left_shoulder', 'right_shoulder', 'overhead'],
                       help='Camera view to use (default: front)')
    parser.add_argument('--template', type=str, default=None,
                       help='Path to template file (auto-detected if not specified)')
    parser.add_argument('--skip-videos', action='store_true',
                       help='Skip video generation (only create scene graphs)')
    
    args = parser.parse_args()
    
    # Determine template path
    if args.template:
        template_path = args.template
    else:
        template_path = os.path.join(
            args.dataset_path, args.task, 
            f'{args.task}_relationship_template.json'
        )
    
    # Check template exists
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        print(f"\nPlease create a template first:")
        print(f"  1. Manually annotate one variation (e.g., variation 0)")
        print(f"  2. Use 'Save as Template' button in the notebook")
        print(f"  3. Run this script to apply to other variations")
        return 1
    
    print(f"Using template: {template_path}")
    
    # Process each variation
    success_count = 0
    fail_count = 0
    
    for variation in args.variations:
        success = process_variation(
            args.task, variation, args.episode, 
            args.dataset_path, args.camera, template_path,
            generate_videos=not args.skip_videos
        )
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {success_count} variations")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count} variations")
    print(f"\nTotal processed: {success_count + fail_count}")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
