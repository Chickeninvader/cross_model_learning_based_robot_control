#!/usr/bin/env python3
"""
Replay demonstration script for robosuite
Plays back recorded demonstrations from NPZ or HDF5 files

Usage:
    python replay_demo.py --demo-dir ./data/recordings/panda_Lift_20241008_143022
    python replay_demo.py --hdf5 ./data/recordings/demos.hdf5
    python replay_demo.py --demo-dir ./data/recordings/panda_Lift_20241008_143022 --episode 5
    python replay_demo.py --demo-dir ./data/recordings/panda_Lift_20241008_143022 --use-states
"""

import argparse
import numpy as np
import os
import json
import h5py
import time
from pathlib import Path

import sys
import robosuite as suite

# Note: create_env.py is now in the same directory (src/)


def replay_from_episode_dir(episode_dir, use_states=False, playback_speed=1.0):
    """
    Replay a demonstration from an episode directory.

    Args:
        episode_dir: Path to episode directory containing demo.npz and model.xml
        use_states: If True, replay using states (more accurate). If False, replay using actions (default)
        playback_speed: Speed multiplier (1.0 = normal speed)
    """
    print(f"\n{'='*60}")
    print(f"Replaying: {episode_dir}")
    print(f"{'='*60}")

    # Load metadata
    metadata_path = os.path.join(episode_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            print(f"Environment: {metadata.get('environment', 'Unknown')}")
            print(f"Robot: {metadata.get('robot', 'Unknown')}")
            print(f"Steps: {metadata.get('num_steps', 'Unknown')}")
            print(f"Total Reward: {metadata.get('total_reward', 'Unknown'):.3f}")
            print(f"Duration: {metadata.get('duration', 'Unknown'):.2f}s")
            # Show simulation parameters if available
            if 'control_freq' in metadata:
                print(f"Control Frequency: {metadata['control_freq']} Hz")
            if 'pos_sensitivity' in metadata:
                print(f"Position Sensitivity: {metadata['pos_sensitivity']}")
            if 'rot_sensitivity' in metadata:
                print(f"Rotation Sensitivity: {metadata['rot_sensitivity']}")
    else:
        print("Warning: No metadata found")

    # Load demonstration data
    demo_path = os.path.join(episode_dir, "demo.npz")
    if not os.path.exists(demo_path):
        print(f"Error: demo.npz not found in {episode_dir}")
        return

    demo_data = np.load(demo_path)
    states = demo_data["states"]
    actions = demo_data.get("actions", None)
    rewards = demo_data.get("rewards", None)

    print(f"\nLoaded {len(states)} states")

    # Load model XML
    model_path = os.path.join(episode_dir, "model.xml")
    if not os.path.exists(model_path):
        print(f"Error: model.xml not found in {episode_dir}")
        return

    with open(model_path, "r") as f:
        model_xml = f.read()

    # Extract environment info and simulation parameters from metadata
    env_name = metadata.get("environment", "Lift")
    robot_name = metadata.get("robot", "Panda")

    # Get simulation parameters from metadata (with fallback defaults)
    control_freq = metadata.get("control_freq", 50)
    horizon = metadata.get("horizon", 10000)

    # Check if this is a custom environment
    if env_name == "UnifiedCustomEnv":
        # Load environment config from the parent directory
        parent_dir = Path(episode_dir).parent
        env_config_name = parent_dir.name.split('_')[0]  # Extract name from timestamp directory
        config_path = Path(f"data/environments/{env_config_name}.json")

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

            from create_env import UnifiedCustomEnv

            # Create custom environment
            object_configs = config['objects']
            placement_mode = config.get('placement_mode', 'random')
            placement_params = config.get('placement_params', {})

            env = UnifiedCustomEnv(
                robots=config.get('robot', robot_name),
                object_configs=object_configs,
                placement_mode=placement_mode,
                placement_params=placement_params,
                has_renderer=True,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=control_freq,
                horizon=horizon,
            )
            print(f"Loaded custom environment: {env_config_name}")
        else:
            print(f"Warning: Could not find config for {env_config_name}, using Lift environment")
            env = suite.make(
                env_name="Lift",
                robots=robot_name,
                has_renderer=True,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=control_freq,
            )
    else:
        # Standard robosuite environment
        env = suite.make(
            env_name=env_name,
            robots=robot_name,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=control_freq,
        )

    # Calculate frame timing based on control frequency
    frame_time = 1.0 / control_freq  # seconds per frame

    print(f"\nPlayback mode: {'States (Deterministic)' if use_states else 'Actions (Default)'}")
    print(f"Playback speed: {playback_speed}x")
    print(f"{'='*60}\n")

    # Reset environment
    print("Resetting environment...")
    env.reset()

    # Initialize the viewer - this is crucial for visualization
    print("Opening visualization window...")
    # The viewer needs to be explicitly initialized
    if not hasattr(env, 'viewer') or env.viewer is None:
        env.viewer = env._get_viewer()

    # Render to actually open the window
    env.render()

    # Give the viewer time to initialize
    time.sleep(0.5)

    print("Visualization window opened!")
    print("\nStarting playback...")

    # Replay demonstration
    if not use_states and actions is not None:
        # Action-based replay
        print("Replaying using actions...")
        print("\n*** START OF DEMO ***")

        # Set initial state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        for i, action in enumerate(actions):
            env.step(action)
            env.render()

            # Print action breakdown
            if i % 10 == 0 or any(abs(action[:6]) > 0.001):
                # Format: ( dx,  dy,  dz,  dr,  dp,  dy, grip)
                dx, dy, dz = action[0], action[1], action[2]
                dr, dp, dyaw = action[3], action[4], action[5]
                grip = action[6] if len(action) > 6 else 0.0

                # Build description
                desc = []
                if abs(dx) > 0.001:
                    desc.append(f"Translation in x-direction ({dx:+.4f})")
                if abs(dy) > 0.001:
                    desc.append(f"Translation in y-direction ({dy:+.4f})")
                if abs(dz) > 0.001:
                    desc.append(f"Translation in z-direction ({dz:+.4f})")
                if abs(dr) > 0.001:
                    desc.append(f"Rotation in roll/x ({dr:+.4f})")
                if abs(dp) > 0.001:
                    desc.append(f"Rotation in pitch/y ({dp:+.4f})")
                if abs(dyaw) > 0.001:
                    desc.append(f"Rotation in yaw/z ({dyaw:+.4f})")

                if not desc:
                    desc.append("No movement (pause)")

                action_str = f"({dx:+.4f}, {dy:+.4f}, {dz:+.4f}, {dr:+.4f}, {dp:+.4f}, {dyaw:+.4f}, {grip:+.2f})"
                desc_str = " + ".join(desc)
                print(f"Step {i:4d}: {action_str}  <-- {desc_str}")

            # Print progress summary
            if i % 100 == 0 and i > 0:
                reward_str = f", Reward: {rewards[i]:.3f}" if rewards is not None else ""
                print(f"--- Step {i}/{len(actions)}{reward_str} ---")

            # Control playback speed using dynamic timing
            time.sleep(frame_time / playback_speed)

        print("*** END OF DEMO ***\n")

    else:
        # State-based replay (more deterministic)
        print("Replaying using states (deterministic)...")

        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()

            # Print progress
            if i % 50 == 0:
                reward_str = f", Reward: {rewards[i]:.3f}" if rewards is not None and i < len(rewards) else ""
                print(f"Step {i}/{len(states)}{reward_str}")

            # Control playback speed using dynamic timing
            time.sleep(frame_time / playback_speed)

    print(f"\n✓ Replay complete!")

    # Keep window open
    print("\nVisualization window is open.")
    print("Press Enter to close...")
    input()

    env.close()


def replay_from_hdf5(hdf5_path, episode_num=None, use_states=False, playback_speed=1.0):
    """
    Replay a demonstration from an HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        episode_num: Specific episode number to replay (None for random)
        use_states: If True, replay using states (more accurate). If False, replay using actions (default)
        playback_speed: Speed multiplier
    """
    print(f"\n{'='*60}")
    print(f"Loading from HDF5: {hdf5_path}")
    print(f"{'='*60}")

    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found: {hdf5_path}")
        return

    with h5py.File(hdf5_path, "r") as f:
        data_grp = f["data"]

        # Print dataset info
        print(f"Created: {data_grp.attrs.get('date', 'Unknown')} {data_grp.attrs.get('time', 'Unknown')}")
        print(f"Total episodes: {data_grp.attrs.get('num_episodes', len(list(data_grp.keys())))}")

        # List all episodes
        episodes = sorted([key for key in data_grp.keys() if key.startswith("demo_")])

        if not episodes:
            print("Error: No demonstrations found in HDF5 file")
            return

        # Select episode
        if episode_num is not None:
            ep_key = f"demo_{episode_num}"
            if ep_key not in episodes:
                print(f"Error: Episode {episode_num} not found")
                print(f"Available episodes: 0 to {len(episodes) - 1}")
                return
        else:
            import random
            ep_key = random.choice(episodes)
            episode_num = int(ep_key.split("_")[1])

        print(f"\nReplaying episode: {episode_num}")

        ep_grp = data_grp[ep_key]

        # Load episode data
        states = ep_grp["states"][()]
        actions = ep_grp.get("actions", None)
        if actions is not None:
            actions = actions[()]

        model_xml = ep_grp.attrs.get("model_file", None)

        # Print episode info
        print(f"Steps: {ep_grp.attrs.get('num_samples', len(states))}")
        if "environment" in ep_grp.attrs:
            print(f"Environment: {ep_grp.attrs['environment']}")
        if "robot" in ep_grp.attrs:
            print(f"Robot: {ep_grp.attrs['robot']}")
        if "total_reward" in ep_grp.attrs:
            print(f"Total Reward: {ep_grp.attrs['total_reward']:.3f}")
        # Show simulation parameters if available
        if "control_freq" in ep_grp.attrs:
            print(f"Control Frequency: {ep_grp.attrs['control_freq']} Hz")

        # Extract environment info and simulation parameters
        env_name = ep_grp.attrs.get("environment", "Lift")
        robot_name = ep_grp.attrs.get("robot", "Panda")
        control_freq = ep_grp.attrs.get("control_freq", 50)

    # Create environment using recorded parameters
    env = suite.make(
        env_name=env_name,
        robots=robot_name,
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=control_freq,
    )

    # Calculate frame timing based on control frequency
    frame_time = 1.0 / control_freq  # seconds per frame

    print(f"\nPlayback mode: {'States (Deterministic)' if use_states else 'Actions (Default)'}")
    print(f"Playback speed: {playback_speed}x")
    print(f"{'='*60}\n")

    # Reset - don't use reset_from_xml_string as it causes geom name conflicts
    env.reset()

    # Initialize the viewer by rendering once
    env.render()

    # Set initial state if available
    if len(states) > 0:
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

    # Replay demonstration
    if not use_states and actions is not None and len(actions) > 0:
        # Action-based replay
        print("Replaying using actions...")

        for i, action in enumerate(actions):
            env.step(action)
            env.render()

            if i % 50 == 0:
                print(f"Step {i}/{len(actions)}")

            # Control playback speed using dynamic timing
            time.sleep(frame_time / playback_speed)

    else:
        # State-based replay
        print("Replaying using states (deterministic)...")

        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            env.render()

            if i % 50 == 0:
                print(f"Step {i}/{len(states)}")

            # Control playback speed using dynamic timing
            time.sleep(frame_time / playback_speed)

    print(f"\n✓ Replay complete!")

    # Keep window open
    print("\nPress Enter to close...")
    input()

    env.close()


def list_demonstrations(demo_dir):
    """List all available demonstrations in a directory"""
    print(f"\n{'='*60}")
    print(f"Demonstrations in: {demo_dir}")
    print(f"{'='*60}")

    # Check for HDF5 files
    hdf5_files = list(Path(demo_dir).glob("*.hdf5"))
    if hdf5_files:
        print("\nHDF5 files:")
        for hdf5_file in hdf5_files:
            print(f"  - {hdf5_file.name}")

            with h5py.File(hdf5_file, "r") as f:
                data_grp = f["data"]
                num_eps = data_grp.attrs.get("num_episodes", len([k for k in data_grp.keys() if k.startswith("demo_")]))
                print(f"    Episodes: {num_eps}")

    # Check for episode directories
    episode_dirs = sorted([d for d in Path(demo_dir).iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if episode_dirs:
        print(f"\nEpisode directories: {len(episode_dirs)}")
        for ep_dir in episode_dirs[:10]:  # Show first 10
            metadata_path = os.path.join(ep_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    print(f"  - {ep_dir.name}: {metadata.get('num_steps', '?')} steps, "
                          f"reward: {metadata.get('total_reward', '?')}")
            else:
                print(f"  - {ep_dir.name}")

        if len(episode_dirs) > 10:
            print(f"  ... and {len(episode_dirs) - 10} more")

    if not hdf5_files and not episode_dirs:
        print("No demonstrations found!")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Replay robosuite demonstrations")

    # Input source
    parser.add_argument("--demo-dir", type=str,
                        help="Directory containing demonstrations")

    parser.add_argument("--hdf5", type=str,
                        help="HDF5 file containing demonstrations")

    parser.add_argument("--episode", type=int,
                        help="Specific episode number to replay")

    parser.add_argument("--list", action="store_true",
                        help="List available demonstrations")

    # Playback options
    parser.add_argument("--use-states", action="store_true",
                        help="Replay using states (more deterministic, opt-in)")

    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")

    args = parser.parse_args()

    # List mode
    if args.list:
        if args.demo_dir:
            list_demonstrations(args.demo_dir)
        else:
            list_demonstrations("./data/recordings")
        return

    # Replay from HDF5
    if args.hdf5:
        hdf5_path = args.hdf5

        # Check if user provided a directory instead of HDF5 file
        if os.path.isdir(hdf5_path):
            # Look for HDF5 file in the directory
            hdf5_files = list(Path(hdf5_path).glob("*.hdf5"))
            if hdf5_files:
                hdf5_path = str(hdf5_files[0])
                print(f"Found HDF5 file: {hdf5_path}")
            else:
                print(f"Error: No HDF5 file found in {args.hdf5}")
                print("The directory may not contain a demos.hdf5 file yet.")
                print("\nTo create an HDF5 file from recorded episodes:")
                print(f"  python3 -c 'from teleop_panda import create_hdf5_dataset; create_hdf5_dataset(\"{args.hdf5}\")'")
                return

        replay_from_hdf5(
            hdf5_path,
            episode_num=args.episode,
            use_states=args.use_states,
            playback_speed=args.speed
        )

    # Replay from directory
    elif args.demo_dir:
        if args.episode is not None:
            # Replay specific episode
            episode_dir = os.path.join(args.demo_dir, f"episode_{args.episode}")
            if not os.path.exists(episode_dir):
                print(f"Error: Episode directory not found: {episode_dir}")
                return

            replay_from_episode_dir(
                episode_dir,
                use_states=args.use_states,
                playback_speed=args.speed
            )
        else:
            # Find and replay first episode
            episode_dirs = sorted([d for d in Path(args.demo_dir).iterdir()
                                 if d.is_dir() and d.name.startswith("episode_")])

            if not episode_dirs:
                print(f"Error: No episode directories found in {args.demo_dir}")

                # Check for HDF5 files
                hdf5_files = list(Path(args.demo_dir).glob("*.hdf5"))
                if hdf5_files:
                    print(f"\nFound HDF5 file: {hdf5_files[0]}")
                    print(f"Try: python replay_demo.py --hdf5 {hdf5_files[0]}")
                return

            # Replay first episode
            replay_from_episode_dir(
                str(episode_dirs[0]),
                use_states=args.use_states,
                playback_speed=args.speed
            )

    else:
        print("Error: Please specify either --demo-dir or --hdf5")
        print("\nExamples:")
        print("  python replay_demo.py --demo-dir ./data/recordings/panda_Lift_20241008_143022")
        print("  python replay_demo.py --hdf5 ./data/recordings/demos.hdf5")
        print("  python replay_demo.py --demo-dir ./data/recordings/panda_Lift_20241008_143022 --episode 5")
        print("  python replay_demo.py --list --demo-dir ./data/recordings")


if __name__ == "__main__":
    main()
