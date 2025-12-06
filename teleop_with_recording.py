#!/usr/bin/env python3
"""
Teleoperation script for robosuite environments with video recording.
Runs 1 episode at 50Hz for specified environments and saves MP4 videos.

Usage:
    python teleop_with_recording.py --environment Lift --robots Panda --device keyboard
    python teleop_with_recording.py --environment Stack --robots Panda --device keyboard
    python teleop_with_recording.py --environment PickPlace --robots Panda --device keyboard
"""

import argparse
import time
import numpy as np
import imageio
import signal
import sys

import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
import robosuite.macros as macros

# Set the image convention to opencv for imageio
macros.IMAGE_CONVENTION = "opencv"

# Global variables for cleanup
writer = None
env = None
device = None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutting down gracefully...")
    try:
        if writer is not None:
            print("Closing video writer...")
            writer.close()
        if device is not None:
            print("Stopping device...")
            device._enabled = False
        if env is not None:
            print("Closing environment...")
            env.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    print("Shutdown complete.")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift", help="Environment name: Lift, Stack, PickPlace, PickPlaceCan, PickPlaceMilk, PickPlaceBread, PickPlaceCereal")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="default", help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--controller", type=str, default=None, help="Choice of controller")
    parser.add_argument("--device", type=str, default="keyboard", help="Input device: keyboard, spacemouse, dualsense, mjgui")
    parser.add_argument("--pos-sensitivity", type=float, default=0.3, help="How much to scale position user inputs (default: 0.3 for slower, smoother control)")
    parser.add_argument("--rot-sensitivity", type=float, default=0.3, help="How much to scale rotation user inputs (default: 0.3 for slower, smoother control)")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency in Hz")
    parser.add_argument("--video-path", type=str, default=None, help="Path to save video (default: <environment>_teleop.mp4)")
    parser.add_argument("--camera", type=str, default="frontview", help="Name of camera to use for recording (frontview, birdview, agentview, sideview)")
    parser.add_argument("--video-height", type=int, default=1088, help="Video height in pixels (default: 1088, divisible by 16 for video encoding)")
    parser.add_argument("--video-width", type=int, default=1920, help="Video width in pixels (default: 1920 for Full HD)")
    parser.add_argument("--video-skip-frame", type=int, default=1, help="Record every Nth frame")
    parser.add_argument("--reverse_xy", type=bool, default=False, help="(DualSense Only)Reverse the effect of the x and y axes")
    args = parser.parse_args()

    # Set default video path if not provided
    if args.video_path is None:
        args.video_path = f"{args.environment}_teleop.mp4"

    # Get controller config
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment with offscreen renderer for video recording
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=args.camera,
        camera_heights=args.video_height,
        camera_widths=args.video_width,
        reward_shaping=True,
        control_freq=args.control_freq,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # Initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from custom_spacemouse import CustomSpaceMouse
        print("\n=== SpaceMouse Setup ===")
        print("Using custom SpaceMouse driver with proper packet handling")
        print("Vendor ID: 1133 (0x046d)")
        print("Product ID: 50731 (0xc62b)")
        device = CustomSpaceMouse(
            env=env,
            vendor_id=1133,  # 0x046d - Logitech/3Dconnexion SpaceMouse Pro
            product_id=50731,  # 0xc62b - SpaceMouse Pro
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
        print("\nSpaceMouse initialized successfully!")
        print("Waiting for device to start sending data...")
        # Give the thread time to start and read some data
        time.sleep(2.0)
        print(f"Current values: x={device.x:.3f}, y={device.y:.3f}, z={device.z:.3f}")
        print("Move the SpaceMouse - you should see the robot arm respond")
        print("=== End Setup ===\n")
    elif args.device == "dualsense":
        from robosuite.devices import DualSense
        device = DualSense(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            reverse_xy=args.reverse_xy,
        )
    elif args.device == "mjgui":
        from robosuite.devices.mjgui import MJGUI
        device = MJGUI(env=env)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard', 'dualsense', 'spacemouse', or 'mjgui'.")

    print(f"\nStarting teleoperation for {args.environment}")
    print(f"Control frequency: {args.control_freq} Hz")
    print(f"Recording video to: {args.video_path}")
    print(f"Camera: {args.camera}")
    print(f"Device: {args.device}")
    print("\nPress ESC to reset and end episode\n")

    # Create video writer with macro_block_size=1 to avoid resizing issues
    writer = imageio.get_writer(args.video_path, fps=args.control_freq, macro_block_size=1)

    # Reset the environment (single episode)
    obs = env.reset()

    # Setup rendering
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    env.render()

    # Initialize variables
    last_grasp = 0
    frame_count = 0

    # Initialize device control
    device.start_control()
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Run single episode loop
    try:
        while True:
            start = time.time()

            # Set active robot
            active_robot = env.robots[device.active_robot]

            # Get the newest action
            input_ac_dict = device.input2action()

            # If action is none, then this is a reset so we should break
            if input_ac_dict is None:
                print("\nEpisode ended by user reset")
                break

            from copy import deepcopy
            action_dict = deepcopy(input_ac_dict)

            # Set arm actions
            for arm in active_robot.arms:
                if isinstance(active_robot.composite_controller, WholeBody):
                    controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                else:
                    controller_input_type = active_robot.part_controllers[arm].input_type

                if controller_input_type == "delta":
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                elif controller_input_type == "absolute":
                    action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                else:
                    raise ValueError

            # Maintain gripper state for each robot but only update the active robot with action
            env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
            env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
            env_action = np.concatenate(env_action)
            for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

            # Step the environment
            obs, reward, done, info = env.step(env_action)
            env.render()

            # Record video frame
            if frame_count % args.video_skip_frame == 0:
                frame = obs[args.camera + "_image"]
                writer.append_data(frame)
                if frame_count % 50 == 0:
                    print(f"Recorded frame #{frame_count}")

            frame_count += 1

            # Limit frame rate
            elapsed = time.time() - start
            diff = 1 / args.control_freq - elapsed
            if diff > 0:
                time.sleep(diff)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nError during teleoperation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            if device is not None:
                device._enabled = False
            if writer is not None:
                writer.close()
                print(f"Video saved to {args.video_path}")
                print(f"Total frames recorded: {frame_count}")
            if env is not None:
                env.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print("Cleanup complete.")
