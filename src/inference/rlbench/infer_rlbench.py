"""
Run inference in the RLBench simulator.

Uses an EEF-pose action mode (EndEffectorPoseViaPlanning + Discrete gripper)
so the 8-D action vector [x, y, z, qx, qy, qz, qw, gripper] matches the
LeRobot dataset format produced by convert_rlbench_to_lerobot.py.

The script:
  1. Launches RLBench in headless mode.
  2. Resets the requested task + variation.
  3. Runs a policy (dummy / random / LeRobot) for N steps.
    4. Records every observation (front + wrist cameras + low-dim) so the
         output can be fed back into the pipeline.

Usage (dummy / random actions):
    python src/inference/rlbench/infer_rlbench.py \
        --task stack_cups --variation 0 \
        --episodes 1 --max_steps 100 \
        --policy dummy \
        --save_path output/rlbench_inference
        
python src/inference/rlbench/infer_rlbench.py \
    --task stack_cups --variation 0 \
    --action_mode ee_planning \
    --policy lerobot \
    --checkpoint output/lerobot/groot_smoke_stack_cups_variation1_20260303_182119/checkpoints/last/pretrained_model \
    --dataset_root datasets/lerobot/stack_cups_variation1 \
    --task_description "Pick up cup 1."

Usage (LeRobot checkpoint — e.g. GROOT):
    python src/inference/rlbench/infer_rlbench.py \
        --task stack_cups --variation 0 \
        --policy lerobot \
        --checkpoint output/lerobot/groot_smoke_stack_cups_variation1_20260303_182119/checkpoints/last/pretrained_model \
        --dataset_root datasets/lerobot/stack_cups_variation1 \
        --task_description "Pick up cup 1."
"""

import argparse
import json
import os
import pickle
import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import copy
from pathlib import Path

import numpy as np
import torch

# Add project root to sys.path for local imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.rlbench_utils import delta_action_to_absolute, images_to_video
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
        return np.random.rand(8)   # 7 joints + gripper open


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
    """Run a trained LeRobot policy (e.g. GROOT / SmolVLA) in RLBench.

    Loads the checkpoint using LeRobot's factory utilities and runs the full
    inference pipeline:  observation → preprocessor → policy → postprocessor → action.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str,
        task_description: str = "",
        device: str | None = None,
    ):
        # ---- lazy LeRobot imports (keeps non-lerobot paths dependency-free) ----
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference

        self.checkpoint_path = checkpoint_path
        self.task_description = task_description
        self._prepare_obs = prepare_observation_for_inference

        # ---- resolve device ----
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ---- load dataset metadata (for feature spec + stats) ----
        # We need a repo_id for the LeRobotDatasetMetadata constructor.
        # Since we're loading from a local path, we create a dummy repo_id
        # and point `root` at the actual directory.
        dataset_root = Path(dataset_root)
        ds_meta = LeRobotDatasetMetadata(
            repo_id="local/rlbench_dataset",
            root=dataset_root,
        )
        self.ds_features = ds_meta.features

        # ---- load policy config from checkpoint ----
        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        policy_cfg.pretrained_path = checkpoint_path
        policy_cfg.device = str(self.device)

        # ---- build the policy model ----
        self.model = make_policy(policy_cfg, ds_meta=ds_meta)
        self.model.eval()
        print(f"[LeRobotPolicy] Loaded {policy_cfg.type} from {checkpoint_path}")
        print(f"  input_features : {list(policy_cfg.input_features.keys())}")
        print(f"  output_features: {list(policy_cfg.output_features.keys())}")
        print(f"  device         : {self.device}")

        # ---- build pre/post processor pipelines ----
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=checkpoint_path,
            dataset_stats=ds_meta.stats,
        )

        self.use_amp = policy_cfg.use_amp

        # ---- figure out which cameras the policy expects ----
        # e.g.  "observation.images.front_rgb" → "front_rgb"
        # Use policy's input_features, not dataset features, to avoid requiring
        # cameras that were in training data but aren't needed by the model
        self.image_keys = [
            k for k in policy_cfg.input_features.keys()
            if k.startswith("observation.images.")
        ]
        # The state key is always "observation.state"
        self.state_key = "observation.state"

    def reset(self):
        """Reset internal action-chunk cache in the policy."""
        self.model.reset()

    # ------------------------------------------------------------------
    # RLBench Observation  →  LeRobot observation dict  →  action
    # ------------------------------------------------------------------
    def _obs_to_dict(self, obs: Observation) -> dict:
        """Convert an RLBench Observation to the flat dict that LeRobot expects.

        Keys produced (matching the dataset info.json):
            observation.state         – (8,) float32  [x y z qx qy qz qw gripper]
            observation.images.<cam>  – (H, W, 3) uint8  (only cameras the policy needs)
        """
        # state: EEF pose (7) + gripper open flag (1)
        state = np.concatenate([
            obs.gripper_pose.astype(np.float32),            # (7,)
            np.array([float(obs.gripper_open)], dtype=np.float32),  # (1,)
        ])

        obs_dict: dict = {
            self.state_key: state,
        }

        # Map LeRobot image key → RLBench attribute name
        # e.g. "observation.images.front_rgb" → obs.front_rgb
        for img_key in self.image_keys:
            cam_attr = img_key.split("observation.images.")[-1]   # "front_rgb"
            img_data = getattr(obs, cam_attr, None)
            if img_data is not None:
                # RLBench images are uint8 (H, W, 3) — exactly what LeRobot expects
                obs_dict[img_key] = img_data.astype(np.uint8)
            else:
                raise RuntimeError(
                    f"Policy expects image '{img_key}' but RLBench observation "
                    f"has no attribute '{cam_attr}'. Available cameras: "
                    f"{[a for a in dir(obs) if a.endswith('_rgb')]}"
                )

        return obs_dict

    def predict(self, obs: Observation) -> np.ndarray:
        """Full LeRobot inference pipeline: obs → preprocess → model → postprocess → action.

        The model outputs a *delta* EEF action [dx, dy, dz, dqx, dqy, dqz, dqw, gripper].
        This method converts it back to an *absolute* target pose for the RLBench controller.
        """
        obs_dict = self._obs_to_dict(obs)
        obs_dict = copy(obs_dict)

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext(),
        ):
            # numpy → tensor, images /255 + CHW, add batch dim, move to device
            obs_dict = self._prepare_obs(
                obs_dict, self.device,
                task=self.task_description,
                robot_type="rlbench_franka",
            )
            # preprocessor (normalization, GROOT packing, etc.)
            obs_dict = self.preprocessor(obs_dict)

            # policy forward
            action_tensor = self.model.select_action(obs_dict)

            # postprocessor (unnormalization, action unpacking, etc.)
            action_tensor = self.postprocessor(action_tensor)

        # action_tensor: (1, action_dim) or (action_dim,)
        delta_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float64)

        # Convert delta action → absolute target pose for the RLBench controller
        current_state = np.concatenate([
            obs.gripper_pose.astype(np.float64),              # (7,)
            np.array([float(obs.gripper_open)], dtype=np.float64),  # (1,)
        ])
        action = delta_action_to_absolute(delta_action, current_state)
        return action


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
            front_rgb.mp4, wrist_rgb.mp4
            low_dim_obs.pkl
    """

    cam_specs = [
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

    # Encode quick preview videos (RGB only) for front + wrist.
    # Uses the same ffmpeg-based helper as the RLBench→LeRobot converter.
    n_frames = len(observations)
    if n_frames > 0:
        frame_start, frame_end = 0, n_frames - 1
        video_fps = 10
        video_codec = "libopenh264"
        video_pix_fmt = "yuv420p"
        for cam_name, rgb_folder, _, _ in cam_specs:
            rgb_dir = os.path.join(save_dir, rgb_folder)
            video_out = os.path.join(save_dir, f"{rgb_folder}.mp4")
            try:
                images_to_video(
                    rgb_dir,
                    video_out,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    fps=video_fps,
                    codec=video_codec,
                    pix_fmt=video_pix_fmt,
                )
            except Exception as exc:  # nosec: B110
                print(f"  Warning: failed to create video for {rgb_folder}: {exc}")

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
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for --policy lerobot")
        return LeRobotPolicy(
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset_root,
            task_description=args.task_description or "",
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown policy: {args.policy}")


def run_inference(args):
    img_size = list(args.image_size)

    # ---- Observation config (front + wrist cameras only) ----
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)

    for cam in [obs_config.wrist_camera, obs_config.front_camera]:
        # RGB-only recording (matches convert_rlbench_to_lerobot.py expectations)
        cam.set_all(False)
        cam.rgb = True
        cam.depth = False
        cam.mask = False
        cam.point_cloud = False
        cam.image_size = img_size
        cam.depth_in_meters = False
        cam.masks_as_one_channel = False

    if args.renderer == "opengl3":
        for cam in [obs_config.wrist_camera,
                    obs_config.front_camera]:
            cam.render_mode = RenderMode.OPENGL3
    elif args.renderer == "opengl":
        for cam in [obs_config.wrist_camera,
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

            # action[:7] = absolute EEF target pose (delta→absolute already
            # converted inside LeRobotPolicy.predict),  action[7] = gripper
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
    p.add_argument("--max_steps", type=int, default=300,
                   help="Max steps per episode before stopping.")

    p.add_argument("--policy", type=str, default="dummy",
                   choices=["dummy", "random", "lerobot"],
                   help="Which policy to use (default: dummy = stay in place).")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to LeRobot pretrained_model dir (required if --policy lerobot).")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Path to LeRobot dataset root (has meta/ data/ dirs). "
                        "Required for --policy lerobot to get feature spec + stats.")
    p.add_argument("--task_description", type=str, default=None,
                   help="Natural-language task description for the policy "
                        "(e.g. 'Pick up cup 1.').")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device for policy inference (default: auto).")

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
