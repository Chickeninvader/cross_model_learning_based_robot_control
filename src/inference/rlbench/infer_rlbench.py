"""
Run inference in the RLBench simulator.

Uses an EEF-pose action mode (EndEffectorPoseViaPlanning + Discrete gripper)
so the 8-D action vector [x, y, z, qx, qy, qz, qw, gripper] matches the
LeRobot dataset format produced by convert_rlbench_to_lerobot.py.

The script:
  1. Launches RLBench in headless mode.
  2. Resets the requested task + variation.
  3. Runs a policy (dummy / random / LeRobot) for N steps.
  4. Records every observation (all cameras + low-dim) exactly like
     dataset_generator.py so the output can be fed back into the pipeline.

Usage (dummy / random actions):
    python src/inference/rlbench/infer_rlbench.py \
        --task stack_cups --variation 0 \
        --episodes 1 --max_steps 100 \
        --policy dummy \
        --save_path output/rlbench_inference

Usage (LeRobot checkpoint — stub, fill in later):
    python src/inference/rlbench/infer_rlbench.py \
        --task stack_cups --variation 0 \
        --policy lerobot \
        --checkpoint path/to/checkpoint
"""

import argparse
import os
import pickle
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# RLBench imports
# ---------------------------------------------------------------------------
from pyrep.const import RenderMode

import rlbench.backend.task as task_module
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    EndEffectorPoseViaIK,
    JointVelocity,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils as rlbench_utils
from rlbench.backend.const import (
    IMAGE_FORMAT,
    LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER,
    RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER,
    OVERHEAD_RGB_FOLDER, OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER,
    WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER,
    FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER,
    LOW_DIM_PICKLE, DEPTH_SCALE,
)
from rlbench.backend.observation import Observation
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment


# ===================================================================
# Policy interface
# ===================================================================

class Policy(ABC):
    """Base class for inference policies."""

    @abstractmethod
    def reset(self):
        """Called at the beginning of every episode."""

    @abstractmethod
    def predict(self, obs: Observation) -> np.ndarray:
        """Return an 8-D action: [x, y, z, qx, qy, qz, qw, gripper].

        The first 7 values are an *absolute* target EEF pose in world frame.
        The last value is the gripper command (>0.5 → open, <0.5 → close).
        """


class DummyPolicy(Policy):
    """Repeat the current EEF pose (robot stays still, gripper open).

    Works with EEF action modes (ee_planning / ee_ik).
    """

    def reset(self):
        pass

    def predict(self, obs: Observation) -> np.ndarray:
        pose = obs.gripper_pose.copy()          # (7,) xyz + quat
        gripper = np.array([1.0])               # open
        return np.concatenate([pose, gripper])


class JointVelocityDummyPolicy(Policy):
    """Zero joint velocities (robot stays still, gripper open).

    Works with the joint_velocity action mode.  Action dim = 8
    (7 joint velocities + 1 gripper).
    """

    def reset(self):
        pass

    def predict(self, obs: Observation) -> np.ndarray:
        return np.zeros(8, dtype=np.float64)   # 7 joints + gripper open


class RandomPolicy(Policy):
    """Apply small random perturbations around the current EEF pose.

    Works with EEF action modes (ee_planning / ee_ik).
    """

    def __init__(self, pos_std: float = 0.005, rot_std: float = 0.01):
        self.pos_std = pos_std
        self.rot_std = rot_std

    def reset(self):
        pass

    def predict(self, obs: Observation) -> np.ndarray:
        pose = obs.gripper_pose.copy()
        pose[:3] += np.random.normal(0, self.pos_std, size=3)
        # Slightly perturb quaternion and re-normalise
        pose[3:] += np.random.normal(0, self.rot_std, size=4)
        pose[3:] /= np.linalg.norm(pose[3:])
        gripper = np.array([float(np.random.random() > 0.5)])
        return np.concatenate([pose, gripper])


class JointVelocityRandomPolicy(Policy):
    """Random small joint velocity commands.

    Works with the joint_velocity action mode.
    """

    def __init__(self, vel_std: float = 0.1):
        self.vel_std = vel_std

    def reset(self):
        pass

    def predict(self, obs: Observation) -> np.ndarray:
        arm = np.random.normal(0.0, self.vel_std, size=7)
        gripper = np.array([1.0])  # open
        return np.concatenate([arm, gripper])


class LeRobotPolicy(Policy):
    """Placeholder for a trained LeRobot model (e.g. SmolVLA).

    TODO: load the checkpoint, implement pre/post processing, call model.
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        # self.model = ...  # load your LeRobot model here
        print(f"[LeRobotPolicy] checkpoint: {checkpoint_path}  (not loaded yet)")

    def reset(self):
        pass

    def predict(self, obs: Observation) -> np.ndarray:
        # ---- stub: just stay in place ----
        pose = obs.gripper_pose.copy()
        gripper = np.array([float(obs.gripper_open)])
        return np.concatenate([pose, gripper])


# ===================================================================
# Observation recording  (mirrors dataset_generator.save_demo)
# ===================================================================

def _mkdirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_observations(observations: list, save_dir: str):
    """Save a list of Observation objects in the RLBench convention.

    Produces:
        <save_dir>/
            front_rgb/0.png, 1.png, …
            wrist_rgb/…
            front_depth/…   (float→RGB encoded)
            front_mask/…    (uint8)
            … (all 5 cameras × 3 modalities)
            low_dim_obs.pkl
    """

    cam_specs = [
        ("left_shoulder",  LEFT_SHOULDER_RGB_FOLDER,  LEFT_SHOULDER_DEPTH_FOLDER,  LEFT_SHOULDER_MASK_FOLDER),
        ("right_shoulder", RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER),
        ("overhead",       OVERHEAD_RGB_FOLDER,       OVERHEAD_DEPTH_FOLDER,       OVERHEAD_MASK_FOLDER),
        ("wrist",          WRIST_RGB_FOLDER,          WRIST_DEPTH_FOLDER,          WRIST_MASK_FOLDER),
        ("front",          FRONT_RGB_FOLDER,          FRONT_DEPTH_FOLDER,          FRONT_MASK_FOLDER),
    ]

    # Create all directories up-front
    all_dirs = []
    for _, rgb_d, depth_d, mask_d in cam_specs:
        all_dirs.extend([
            os.path.join(save_dir, rgb_d),
            os.path.join(save_dir, depth_d),
            os.path.join(save_dir, mask_d),
        ])
    _mkdirs(*all_dirs)

    for i, obs in enumerate(observations):
        for cam_name, rgb_folder, depth_folder, mask_folder in cam_specs:
            rgb_data   = getattr(obs, f"{cam_name}_rgb", None)
            depth_data = getattr(obs, f"{cam_name}_depth", None)
            mask_data  = getattr(obs, f"{cam_name}_mask", None)

            if rgb_data is not None:
                Image.fromarray(rgb_data).save(
                    os.path.join(save_dir, rgb_folder, IMAGE_FORMAT % i))

            if depth_data is not None:
                depth_img = rlbench_utils.float_array_to_rgb_image(
                    depth_data, scale_factor=DEPTH_SCALE)
                depth_img.save(
                    os.path.join(save_dir, depth_folder, IMAGE_FORMAT % i))

            if mask_data is not None:
                mask_img = Image.fromarray(
                    (mask_data * 255).astype(np.uint8))
                mask_img.save(
                    os.path.join(save_dir, mask_folder, IMAGE_FORMAT % i))

        # Null out images before pickling (same pattern as dataset_generator)
        for cam_name, _, _, _ in cam_specs:
            for attr_suffix in ("_rgb", "_depth", "_point_cloud", "_mask"):
                setattr(obs, f"{cam_name}{attr_suffix}", None)

    # Pickle the low-dim data
    with open(os.path.join(save_dir, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(observations, f)

    print(f"  Saved {len(observations)} observations → {save_dir}")


# ===================================================================
# Main inference loop
# ===================================================================

def build_policy(args) -> Policy:
    is_jv = (args.action_mode == "joint_velocity")
    if args.policy == "dummy":
        return JointVelocityDummyPolicy() if is_jv else DummyPolicy()
    elif args.policy == "random":
        return JointVelocityRandomPolicy() if is_jv else RandomPolicy()
    elif args.policy == "lerobot":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --policy lerobot")
        return LeRobotPolicy(args.checkpoint)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")


def run_inference(args):
    img_size = list(args.image_size)

    # ---- Observation config (record everything) ----
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    if args.renderer == "opengl3":
        for cam in [obs_config.right_shoulder_camera,
                    obs_config.left_shoulder_camera,
                    obs_config.overhead_camera,
                    obs_config.wrist_camera,
                    obs_config.front_camera]:
            cam.render_mode = RenderMode.OPENGL3
    elif args.renderer == "opengl":
        for cam in [obs_config.right_shoulder_camera,
                    obs_config.left_shoulder_camera,
                    obs_config.overhead_camera,
                    obs_config.wrist_camera,
                    obs_config.front_camera]:
            cam.render_mode = RenderMode.OPENGL

    # ---- Action mode ----
    if args.action_mode == "ee_planning":
        arm_mode = EndEffectorPoseViaPlanning(
            absolute_mode=True, collision_checking=False)
    elif args.action_mode == "ee_ik":
        arm_mode = EndEffectorPoseViaIK(
            absolute_mode=True, collision_checking=False)
    elif args.action_mode == "joint_velocity":
        arm_mode = JointVelocity()
    else:
        raise ValueError(f"Unknown action mode: {args.action_mode}")

    action_mode = MoveArmThenGripper(
        arm_action_mode=arm_mode,
        gripper_action_mode=Discrete(),
    )

    # ---- Launch environment ----
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()
    print(f"RLBench environment launched  (image_size={img_size})")

    # ---- Get task ----
    task_class = task_file_to_task_class(args.task)
    task_env = env.get_task(task_class)
    task_env.set_variation(args.variation)
    print(f"Task: {args.task}  variation: {args.variation}")

    # ---- Build policy ----
    policy = build_policy(args)

    # ---- Run episodes ----
    os.makedirs(args.save_path, exist_ok=True)

    for ep_idx in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx}")
        print(f"{'='*60}")

        policy.reset()
        descriptions, initial_obs = task_env.reset()
        print(f"  Descriptions: {descriptions}")

        observations = [initial_obs]
        total_reward = 0.0
        success = False

        for step in range(args.max_steps):
            action = policy.predict(observations[-1])

            # action[:7] = absolute EEF pose,  action[7] = gripper
            try:
                obs, reward, terminate = task_env.step(action)
            except Exception as e:
                print(f"  Step {step}: action failed ({e}), stopping episode.")
                break

            observations.append(obs)
            total_reward += reward

            if reward > 0:
                success = True
            if terminate or success:
                print(f"  Step {step}: task {'succeeded' if success else 'terminated'}!")
                break

            if (step + 1) % 20 == 0:
                print(f"  Step {step}: reward={total_reward:.2f}")

        print(f"  Episode {ep_idx} done — {len(observations)} observations, "
              f"total_reward={total_reward:.2f}, success={success}")

        # ---- Save recorded observations ----
        episode_dir = os.path.join(args.save_path, f"episode{ep_idx}")
        save_observations(observations, episode_dir)

    env.shutdown()
    print(f"\nAll done. Results saved to {args.save_path}")


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference in RLBench and record observations.")

    p.add_argument("--task", type=str, required=True,
                   help="RLBench task name, e.g. 'stack_cups'.")
    p.add_argument("--variation", type=int, default=0,
                   help="Task variation index (default: 0).")
    p.add_argument("--episodes", type=int, default=1,
                   help="Number of episodes to run.")
    p.add_argument("--max_steps", type=int, default=100,
                   help="Max steps per episode before stopping.")

    p.add_argument("--policy", type=str, default="dummy",
                   choices=["dummy", "random", "lerobot"],
                   help="Which policy to use (default: dummy = stay in place).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to LeRobot checkpoint (required if --policy lerobot).")

    p.add_argument("--image_size", nargs=2, type=int, default=[256, 256],
                   help="Image resolution (H W).")
    p.add_argument("--action_mode", type=str, default="joint_velocity",
                   choices=["ee_planning", "ee_ik", "joint_velocity"],
                   help="Action mode (default: joint_velocity).")
    p.add_argument("--renderer", type=str, default="opengl",
                   choices=["opengl", "opengl3"],
                   help="Rendering backend (default: opengl for headless).")
    p.add_argument("--save_path", type=str,
                   default="output/rlbench_inference",
                   help="Where to save recorded observations.")

    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
