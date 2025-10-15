#!/usr/bin/env python3
"""
Run Simulations with Custom Environments
Execute automated simulations, teleoperation, and demonstration recording

Usage:
    # Run simulation with a saved environment
    python run_simulation.py --env simple_pick --episodes 10

    # Run with teleoperation (SpaceMouse)
    python run_simulation.py --mode teleop --robot Panda --task Lift

    # Record demonstrations with teleoperation
    python run_simulation.py --mode teleop --robot Panda --task Stack --record --num-episodes 5

    # Run with custom policy
    python run_simulation.py --env mixed_objects --policy random --episodes 20
"""

import argparse
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import signal
import warnings
import h5py

# Note: create_env.py is now in the same directory (src/)

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    shutdown_flag = True
    print("\n\nShutdown signal received. Cleaning up...")

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


def load_env_config(env_name):
    """Load environment configuration"""
    config_path = Path(f"data/environments/{env_name}.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Environment '{env_name}' not found. Run: python src/env_manager.py --list")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def create_environment(config, enable_recording=False, camera_obs=False):
    """Create environment from configuration"""
    from create_env import UnifiedCustomEnv

    # Handle both old and new config formats
    if 'objects' in config:
        # New format
        object_configs = config['objects']
        placement_mode = config.get('placement_mode', 'random')
        placement_params = config.get('placement_params', {})
    else:
        # Old format - convert to new format
        num_objects = config.get('num_objects', 3)
        object_type = config.get('object_type', 'box')
        object_size = config.get('object_size', 0.03)
        colors = config.get('colors', 'default')

        # Create object configs
        object_configs = []
        for i in range(num_objects):
            obj_config = {
                'type': object_type,
                'size': object_size,
                'name': f"{object_type}_{i}",
                'density': 100.0
            }
            # Add color if specified
            if colors != 'default':
                if isinstance(colors, list):
                    obj_config['color'] = colors[i % len(colors)]
            object_configs.append(obj_config)

        placement_mode = 'random'
        placement_range = config.get('placement_range', 0.15)
        placement_params = {
            'x_range': [-placement_range, placement_range],
            'y_range': [-placement_range, placement_range]
        }

    env = UnifiedCustomEnv(
        robots=config['robot'],
        object_configs=object_configs,
        placement_mode=placement_mode,
        placement_params=placement_params,
        has_renderer=True,
        has_offscreen_renderer=enable_recording or camera_obs,
        use_camera_obs=camera_obs,
        control_freq=50,
        horizon=1000,
    )

    return env


class RandomPolicy:
    """Random action policy"""
    def __init__(self, action_dim, scale=0.02):
        self.action_dim = action_dim
        self.scale = scale

    def get_action(self, obs):
        return np.random.randn(self.action_dim) * self.scale


class ScriptedPolicy:
    """Simple scripted policy - reach towards nearest object"""
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.steps = 0

    def get_action(self, obs):
        # Simple sinusoidal motion for demonstration
        self.steps += 1
        action = np.zeros(self.action_dim)

        # Move end-effector in a pattern
        action[0] = 0.01 * np.sin(self.steps * 0.1)  # X
        action[1] = 0.01 * np.cos(self.steps * 0.1)  # Y
        action[2] = 0.005 * np.sin(self.steps * 0.05)  # Z

        return action


def run_episode(env, policy, record=False, show_progress=True):
    """Run a single episode"""
    obs = env.reset()

    episode_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'observations': []
    }

    total_reward = 0.0
    step_count = 0

    while step_count < env.horizon:
        # Get action from policy
        action = policy.get_action(obs)

        # Take step
        obs, reward, done, info = env.step(action)
        env.render()

        # Record data if needed
        if record:
            episode_data['states'].append(env.sim.get_state().flatten())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(obs)

        total_reward += reward
        step_count += 1

        if show_progress and step_count % 50 == 0:
            print(f"  Step {step_count}/{env.horizon}, Reward: {total_reward:.3f}")

        if done:
            break

    episode_data['total_reward'] = total_reward
    episode_data['num_steps'] = step_count

    return episode_data


def save_episode_data(episode_data, output_dir, episode_num):
    """Save episode data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dir = output_dir / f"episode_{episode_num}"
    episode_dir.mkdir(exist_ok=True)

    # Save as npz
    np.savez(
        episode_dir / "data.npz",
        states=np.array(episode_data['states']),
        actions=np.array(episode_data['actions']),
        rewards=np.array(episode_data['rewards'])
    )

    # Save metadata
    metadata = {
        'episode': episode_num,
        'total_reward': episode_data['total_reward'],
        'num_steps': episode_data['num_steps'],
        'timestamp': datetime.now().isoformat()
    }

    with open(episode_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved episode data to {episode_dir}")


def save_demonstration(demo_dir, episode_num, states, actions, rewards, timestamps, env, device=None):
    """Save demonstration data with model XML and simulation parameters"""
    ep_dir = os.path.join(demo_dir, f"episode_{episode_num}")
    os.makedirs(ep_dir, exist_ok=True)

    # Save states and actions
    np.savez_compressed(
        os.path.join(ep_dir, "demo.npz"),
        states=np.array(states),
        actions=np.array(actions),
        rewards=np.array(rewards),
        timestamps=np.array(timestamps)
    )

    # Save model XML
    xml_str = env.sim.model.get_xml()
    with open(os.path.join(ep_dir, "model.xml"), "w") as f:
        f.write(xml_str)

    # Get environment name
    base_env = env.env if hasattr(env, 'env') else env
    env_name = base_env.__class__.__name__

    # Save metadata with simulation parameters
    metadata = {
        "episode_num": episode_num,
        "num_steps": len(actions),
        "total_reward": float(np.sum(rewards)),
        "duration": timestamps[-1] if timestamps else 0,
        "environment": env_name,
        "robot": env.robots[0].name,
        "timestamp": datetime.now().isoformat(),
        # Simulation parameters for accurate replay
        "control_freq": getattr(env, 'control_freq', 50),
        "horizon": getattr(env, 'horizon', 1000),
    }

    # Add device sensitivity parameters if available
    if device is not None:
        metadata["pos_sensitivity"] = getattr(device, 'pos_sensitivity', 1.0)
        metadata["rot_sensitivity"] = getattr(device, 'rot_sensitivity', 1.0)

    with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to: {ep_dir}")


def create_hdf5_dataset(demo_dir, output_file="demos.hdf5"):
    """Combine all demonstrations into a single HDF5 file"""
    print(f"\n{'='*60}")
    print("Creating HDF5 dataset...")
    print(f"{'='*60}")

    hdf5_path = os.path.join(demo_dir, output_file)
    episode_dirs = sorted([d for d in Path(demo_dir).iterdir() if d.is_dir() and d.name.startswith("episode_")])

    if not episode_dirs:
        print("No episodes found!")
        return

    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["date"] = datetime.now().strftime("%Y-%m-%d")
        data_grp.attrs["time"] = datetime.now().strftime("%H:%M:%S")
        data_grp.attrs["num_episodes"] = len(episode_dirs)

        for idx, ep_dir in enumerate(episode_dirs):
            print(f"Processing {ep_dir.name}...")

            demo_data = np.load(os.path.join(ep_dir, "demo.npz"))
            states = demo_data["states"]
            actions = demo_data["actions"]

            with open(os.path.join(ep_dir, "model.xml"), "r") as xml_file:
                model_xml = xml_file.read()

            ep_grp = data_grp.create_group(f"demo_{idx}")
            ep_grp.create_dataset("states", data=states)
            ep_grp.create_dataset("actions", data=actions)
            ep_grp.attrs["model_file"] = model_xml
            ep_grp.attrs["num_samples"] = len(actions)

            if os.path.exists(os.path.join(ep_dir, "metadata.json")):
                with open(os.path.join(ep_dir, "metadata.json"), "r") as meta_file:
                    metadata = json.load(meta_file)
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str)):
                            ep_grp.attrs[key] = value

    print(f"\n✓ HDF5 dataset created: {hdf5_path}")
    print(f"  Total episodes: {len(episode_dirs)}")


def collect_demonstration(env, device, demo_dir, episode_num):
    """Collect a single demonstration episode with SpaceMouse"""
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}")
    print(f"{'='*60}")
    print("Control the robot using SpaceMouse")
    print("  - Move SpaceMouse to control arm position/rotation")
    print("  - Button 1: Toggle gripper")
    print("  - Menu button: End episode")
    print(f"{'='*60}\n")

    states = []
    actions = []
    rewards = []
    timestamps = []

    obs = env.reset()
    env.render()
    device.start_control()

    initial_state = np.array(env.sim.get_state().flatten())
    states.append(initial_state)

    step_count = 0
    start_time = datetime.now()

    import time

    # Calculate frame timing based on control frequency for consistent playback
    control_freq = getattr(env, 'control_freq', 50)
    frame_time = 1.0 / control_freq

    print("\n*** START OF DEMO ***")

    try:
        while True:
            if shutdown_flag:
                print("\n\nEpisode interrupted by shutdown signal")
                return False

            action_dict = device.input2action()

            if action_dict is None:
                print("\nEpisode ended by user")
                break

            robot_name = env.robots[0].name
            action = action_dict[robot_name]

            try:
                obs, reward, done, info = env.step(action)
            except (SystemError, Exception) as e:
                if shutdown_flag:
                    print("\n\nEpisode interrupted during step")
                    return False
                else:
                    print(f"Warning: Step error: {e}")
                    # If episode terminated, break out
                    if "terminated episode" in str(e).lower():
                        print(f"\nEpisode reached maximum steps ({step_count})")
                        break
                    continue

            current_state = np.array(env.sim.get_state().flatten())
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            timestamps.append((datetime.now() - start_time).total_seconds())

            try:
                env.render()
            except:
                pass

            step_count += 1

            # Print action breakdown every 10 steps or when there's significant movement
            if step_count % 10 == 0 or any(abs(action[:6]) > 0.005):
                dx, dy, dz = action[0], action[1], action[2]
                dr, dp, dyaw = action[3], action[4], action[5]
                grip = action[6] if len(action) > 6 else 0.0

                # Build description
                desc = []
                if abs(dx) > 0.001:
                    desc.append(f"Trans X ({dx:+.4f})")
                if abs(dy) > 0.001:
                    desc.append(f"Trans Y ({dy:+.4f})")
                if abs(dz) > 0.001:
                    desc.append(f"Trans Z ({dz:+.4f})")
                if abs(dr) > 0.001:
                    desc.append(f"Roll ({dr:+.4f})")
                if abs(dp) > 0.001:
                    desc.append(f"Pitch ({dp:+.4f})")
                if abs(dyaw) > 0.001:
                    desc.append(f"Yaw ({dyaw:+.4f})")

                if not desc:
                    desc.append("No movement (pause)")

                action_str = f"({dx:+.4f}, {dy:+.4f}, {dz:+.4f}, {dr:+.4f}, {dp:+.4f}, {dyaw:+.4f}, {grip:+.2f})"
                desc_str = " + ".join(desc)
                print(f"Step {step_count:4d}: {action_str}  <-- {desc_str}")

            # Print summary every 100 steps
            if step_count % 100 == 0:
                print(f"--- Progress: {step_count} steps, Reward: {reward:.3f} ---")

            # Check if episode is done (reached horizon)
            if done:
                print(f"\nEpisode completed at step {step_count} (horizon reached)")
                break

            # Sleep using dynamic timing based on control frequency
            time.sleep(frame_time)

    except Exception as e:
        print(f"\n\nEpisode interrupted: {e}")
        return False

    print("*** END OF DEMO ***\n")

    if step_count < 10:
        print("Episode too short, discarding...")
        return False

    print(f"\nSave this demonstration? ({step_count} steps)")
    response = input("(y/n): ").lower()

    if response != 'y':
        print("Demonstration discarded")
        return False

    save_demonstration(demo_dir, episode_num, states, actions, rewards, timestamps, env, device)
    print(f"Demonstration saved: episode_{episode_num}")
    return True


def run_simulation(args):
    """Run complete simulation"""

    # Load environment configuration
    print(f"\n{'='*60}")
    print(f"Running Simulation")
    print(f"{'='*60}")

    config = load_env_config(args.env)
    print(f"Environment: {config.get('description', args.env)}")
    print(f"Robot: {config['robot']}")

    # Handle both old and new config formats
    if 'objects' in config:
        # New format
        num_objects = len(config['objects'])
        object_type = config['objects'][0]['type'] if config['objects'] else 'unknown'
        print(f"Objects: {num_objects} x {object_type}")
    else:
        # Old format
        print(f"Objects: {config['num_objects']} x {config['object_type']}")

    print(f"Episodes: {args.episodes}")
    print(f"Policy: {args.policy}")
    if args.record:
        print(f"Recording: Enabled → {args.output_dir}")
    print(f"{'='*60}\n")

    # Create environment
    env = create_environment(config, enable_recording=args.record, camera_obs=args.camera_obs)

    # Create policy
    if args.policy == "random":
        policy = RandomPolicy(env.robots[0].action_dim)
        print("Using random policy\n")
    elif args.policy == "scripted":
        policy = ScriptedPolicy(env.robots[0].action_dim)
        print("Using scripted policy\n")
    else:
        print(f"Unknown policy: {args.policy}, using random")
        policy = RandomPolicy(env.robots[0].action_dim)

    # Run episodes
    results = []

    try:
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}")

            episode_data = run_episode(
                env,
                policy,
                record=args.record,
                show_progress=args.verbose
            )

            results.append({
                'episode': episode,
                'reward': episode_data['total_reward'],
                'steps': episode_data['num_steps']
            })

            print(f"  Reward: {episode_data['total_reward']:.3f}, Steps: {episode_data['num_steps']}")

            # Save episode data if recording
            if args.record:
                save_episode_data(episode_data, args.output_dir, episode)

            print()

        # Print summary
        print(f"\n{'='*60}")
        print("Simulation Complete")
        print(f"{'='*60}")

        rewards = [r['reward'] for r in results]
        print(f"Total Episodes: {len(results)}")
        print(f"Average Reward: {np.mean(rewards):.3f}")
        print(f"Std Dev: {np.std(rewards):.3f}")
        print(f"Min Reward: {np.min(rewards):.3f}")
        print(f"Max Reward: {np.max(rewards):.3f}")

        if args.record:
            print(f"\nData saved to: {args.output_dir}")
            print(f"Files: {args.episodes} episodes")

        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")

    finally:
        env.close()


def run_teleoperation(args):
    """Run with SpaceMouse teleoperation control"""
    from robosuite.wrappers import VisualizationWrapper
    from custom_spacemouse import CustomSpaceMouse

    print(f"\n{'='*60}")
    print(f"{args.robot} Robot Teleoperation with SpaceMouse")
    print(f"{'='*60}")

    # Check if using custom environment or robosuite task
    if args.env:
        # Custom environment mode
        print(f"Custom Environment: {args.env}")
        config = load_env_config(args.env)
        print(f"Objects: {len(config.get('objects', []))} objects")
        use_custom_env = True
    else:
        # Standard robosuite task mode
        print(f"Task: {args.task}")
        use_custom_env = False

    print(f"Controller: {args.controller}")
    print(f"Recording: {'Yes' if args.record else 'No'}")
    if args.record:
        print(f"Episodes to collect: {args.num_episodes}")
    print(f"{'='*60}\n")

    # Create environment
    # Recording mode: use specified horizon (default 5000 = 100s at 50Hz)
    # Free play mode: use 250000 steps (~83 minutes at 50Hz)
    horizon = args.horizon if args.record else 250000

    if use_custom_env:
        # Create custom environment
        from create_env import UnifiedCustomEnv

        # Handle both old and new config formats
        if 'objects' in config:
            object_configs = config['objects']
            placement_mode = config.get('placement_mode', 'random')
            placement_params = config.get('placement_params', {})
        else:
            # Convert old format to new
            num_objects = config.get('num_objects', 3)
            object_type = config.get('object_type', 'box')
            object_size = config.get('object_size', 0.03)
            colors = config.get('colors', 'default')

            object_configs = []
            for i in range(num_objects):
                obj_config = {
                    'type': object_type,
                    'size': object_size,
                    'name': f"{object_type}_{i}",
                    'density': 100.0
                }
                if colors != 'default' and isinstance(colors, list):
                    obj_config['color'] = colors[i % len(colors)]
                object_configs.append(obj_config)

            placement_mode = 'random'
            placement_range = config.get('placement_range', 0.15)
            placement_params = {
                'x_range': [-placement_range, placement_range],
                'y_range': [-placement_range, placement_range]
            }

        env = UnifiedCustomEnv(
            robots=config.get('robot', args.robot),
            object_configs=object_configs,
            placement_mode=placement_mode,
            placement_params=placement_params,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=50,
            horizon=horizon,
        )

        print(f"Custom environment created: {args.env}")
        print(f"Objects: {', '.join([obj.name for obj in env.objects])}")
    else:
        # Create standard robosuite environment
        import robosuite as suite

        env = suite.make(
            env_name=args.task,
            robots=args.robot,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=50,
            horizon=horizon,
        )

        print(f"Environment created: {args.task}")

    # Add end-effector visualization for all environments
    try:
        env = VisualizationWrapper(env, indicator_configs='default')
        print(f"End-effector visualization enabled (red sphere marker)")
    except Exception as e:
        print(f"Note: Visualization wrapper not available ({e})")
        print(f"Proceeding without end-effector marker")

    print(f"Action space: {env.action_spec}")
    if not args.record:
        print(f"Free play mode: Effectively unlimited steps")
    print()

    # Create SpaceMouse device
    device = CustomSpaceMouse(
        env=env,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity
    )

    # Recording mode
    if args.record:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = args.env if use_custom_env else f"{args.robot}_{args.task}"
        demo_dir = os.path.join(args.demo_dir, f"{env_name}_{timestamp}")
        os.makedirs(demo_dir, exist_ok=True)

        print(f"Demonstrations will be saved to: {demo_dir}\n")

        episode_num = 0
        successful_episodes = 0

        try:
            while successful_episodes < args.num_episodes and not shutdown_flag:
                if shutdown_flag:
                    print("\n\nShutdown requested, stopping collection...")
                    break

                if collect_demonstration(env, device, demo_dir, episode_num):
                    successful_episodes += 1
                    print(f"\nSuccessful episodes: {successful_episodes}/{args.num_episodes}")

                episode_num += 1

                if successful_episodes < args.num_episodes and not shutdown_flag:
                    print("\nStarting next episode in 2 seconds...")
                    import time
                    for _ in range(20):
                        if shutdown_flag:
                            print("\nShutdown requested during wait, stopping...")
                            break
                        time.sleep(0.1)
                    if shutdown_flag:
                        break

        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user")

        # Create HDF5 dataset
        if successful_episodes > 0:
            print("\nCreating HDF5 dataset from collected episodes...")
            create_hdf5_dataset(demo_dir)

        print(f"\n{'='*60}")
        print(f"Collection Summary:")
        print(f"  Total episodes collected: {successful_episodes}")
        print(f"  Saved to: {demo_dir}")
        print(f"{'='*60}")

    # Free play mode
    else:
        print("--- Free Play Mode ---")
        print("Press Ctrl+C to exit\n")

        try:
            obs = env.reset()
            env.render()
            device.start_control()

            step_count = 0

            import time

            while True:
                if shutdown_flag:
                    print("\nShutdown requested, exiting...")
                    break

                action_dict = device.input2action()

                if action_dict is None:
                    print("\nExiting...")
                    break

                robot_name = env.robots[0].name
                action = action_dict[robot_name]
                obs, reward, done, info = env.step(action)
                env.render()

                step_count += 1

                # Sleep to prevent CPU spinning while maintaining responsiveness (100Hz = 10ms)
                time.sleep(0.01)

                # Check if episode is done and reset
                if done:
                    print(f"\nEpisode completed! Steps: {step_count}")
                    print("Resetting environment...\n")
                    obs = env.reset()
                    env.render()
                    step_count = 0

        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt received, exiting...")
        except Exception as e:
            print(f"\n\nError occurred: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    print("\nCleaning up...")
    device.close()
    env.close()
    print("Environment closed")


def main():
    parser = argparse.ArgumentParser(
        description="Run simulations, teleoperation, and demonstration recording",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automated simulation with saved environment
  python run_simulation.py --mode sim --env simple_pick --episodes 10
  
  # Teleoperation with Panda robot
  python run_simulation.py --mode teleop --robot Panda --task Lift
  
  # Record demonstrations with Sawyer
  python run_simulation.py --mode teleop --robot Sawyer --task Stack --record --num-episodes 5
  
  # Simulation with recording
  python run_simulation.py --mode sim --env multi_ball --episodes 5 --record
        """
    )

    # Mode selection
    parser.add_argument("--mode", type=str, default="sim",
                        choices=["sim", "teleop"],
                        help="Mode: 'sim' for automated simulation, 'teleop' for teleoperation (default: sim)")

    # Simulation mode arguments
    parser.add_argument("--env", type=str,
                        help="Name of saved environment configuration (required for sim mode)")

    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run in sim mode (default: 10)")

    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "scripted"],
                        help="Policy to use in sim mode (default: random)")

    # Teleoperation mode arguments
    parser.add_argument("--robot", type=str, default="Panda",
                        choices=["Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e"],
                        help="Robot type for teleop mode (default: Panda)")

    parser.add_argument("--task", type=str, default="Lift",
                        choices=["Lift", "Stack", "PickPlace", "NutAssembly", "Door", "Wipe"],
                        help="Task name for teleop mode (default: Lift)")

    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        choices=["OSC_POSE", "IK_POSE", "JOINT_POSITION"],
                        help="Controller type for teleop mode (default: OSC_POSE)")

    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to collect in teleop recording mode (default: 1)")

    parser.add_argument("--pos-sensitivity", type=float, default=3.0,
                        help="Position sensitivity for SpaceMouse (default: 3.0)")

    parser.add_argument("--rot-sensitivity", type=float, default=2.0,
                        help="Rotation sensitivity for SpaceMouse (default: 2.0)")

    parser.add_argument("--horizon", type=int, default=5000,
                        help="Episode horizon/max steps for recording (default: 5000 = 100s at 50Hz)")

    # Recording options (both modes)
    parser.add_argument("--record", action="store_true",
                        help="Enable recording/data collection")

    parser.add_argument("--output-dir", type=str, default="data/simulations",
                        help="Output directory for sim mode recordings (default: data/simulations)")

    parser.add_argument("--demo-dir", type=str, default="data/recordings",
                        help="Directory for teleop demonstrations (default: data/recordings)")

    parser.add_argument("--camera-obs", action="store_true",
                        help="Enable camera observations")

    # Display options
    parser.add_argument("--verbose", action="store_true",
                        help="Show detailed progress")

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == "sim":
        if not args.env:
            parser.error("--env is required for simulation mode")

        if not Path(f"data/environments/{args.env}.json").exists():
            print(f"Error: Environment '{args.env}' not found")
            print("\nCreate one with:")
            print(f"  python src/create_env.py --save {args.env} --num-objects 5 --type ball")
            return

        run_simulation(args)

    elif args.mode == "teleop":
        run_teleoperation(args)


if __name__ == "__main__":
    main()
