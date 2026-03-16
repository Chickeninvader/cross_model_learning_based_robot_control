"""
Run inference in the RLBench simulator.

Supports two RLBench action modes that map to LeRobot-style 8-D actions:

- EEF modes (`ee_planning` / `ee_ik`):
    action = [x, y, z, qx, qy, qz, qw, gripper] (absolute target pose)
- Joint mode (`joint_velocity`):
    action = [dq0..dq6, gripper] (joint velocity command)

For `--policy lerobot`, the interpretation depends on `--action_mode`:
- EEF modes: model outputs delta-EEF which is converted to absolute.
- Joint mode: model outputs joint velocities directly.

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
    --checkpoint output/lerobot/groot_smoke_stack_cups_eef_20260303_182119/checkpoints/last/pretrained_model \
    --dataset_root datasets/lerobot/stack_cups_eef \
    --task_description "Pick up cup 1."

Usage (LeRobot checkpoint — e.g. GROOT):
    python src/inference/rlbench/infer_rlbench.py \
        --task stack_cups --variation 0 \
        --policy lerobot \
        --checkpoint output/lerobot/groot_smoke_stack_cups_eef_20260303_182119/checkpoints/last/pretrained_model \
        --dataset_root datasets/lerobot/stack_cups_eef \
        --task_description "Pick up cup 1."

Usage (LeRobot checkpoint — e.g. SmolVLA):
    python src/inference/rlbench/infer_rlbench.py \
        --task put_rubbish_in_bin --variation 0 \
        --action_mode ee_planning \
        --policy lerobot \
        --checkpoint output/lerobot/smolvla_put_rubbish_in_bin_all_20260312_183353 \
        --dataset_root datasets/lerobot_without_prompt/put_rubbish_in_bin_all \
        --task_description "Pick up paper. Release paper, then place paper in trash bin."
"""

import argparse
import json
import os
import re
import sys
import tempfile
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

from src.utils.rlbench_infer_utils import (
    eef_pose_change,
    set_policy_task_description,
    split_task_description_into_steps,
)
from src.utils.rlbench_rollout_io import save_observations
from src.utils.rlbench_utils import delta_action_to_absolute

# ---------------------------------------------------------------------------
# RLBench imports
# ---------------------------------------------------------------------------
from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    EndEffectorPoseViaIK,
    JointVelocity,
)
from rlbench.action_modes.gripper_action_modes import Discrete
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
                """Return an 8-D action.

                Interpretation depends on the RLBench `--action_mode`:

                - `ee_planning` / `ee_ik`:
                    [x, y, z, qx, qy, qz, qw, gripper]  (absolute target EEF pose)
                - `joint_velocity`:
                    [dq0, dq1, dq2, dq3, dq4, dq5, dq6, gripper]  (joint velocity command)

                The last value is the gripper command (>0.5 → open, <0.5 → close).
                """


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
        action_mode: str = "ee_planning",
        rename_map: dict[str, str] | None = None,
        device: str | None = None,
    ):
        # ---- lazy LeRobot imports (keeps non-lerobot paths dependency-free) ----
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference

        self.checkpoint_path = self._resolve_pretrained_model_dir(checkpoint_path)
        self.task_description = task_description
        self.action_mode = action_mode
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
        policy_cfg = self._load_policy_config_compat(PreTrainedConfig, self.checkpoint_path)
        policy_cfg.pretrained_path = self.checkpoint_path
        policy_cfg.device = str(self.device)

        # ---- auto-detect rename_map from checkpoint preprocessor if not provided ----
        # Some checkpoints (e.g., SmolVLA) use camera keys like camera1/2/3 but rely on
        # a saved `rename_observations_processor` to map dataset keys (front/wrist) to those.
        # Passing a truthy rename_map to `make_policy` bypasses strict visual-key validation.
        if rename_map is None and policy_cfg.type == "smolvla":
            rename_map = self._load_rename_map_from_checkpoint(self.checkpoint_path)
        self.rename_map = rename_map

        # ---- build the policy model ----
        self.model = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=self.rename_map)
        self.model.eval()
        print(f"[LeRobotPolicy] Loaded {policy_cfg.type} from {self.checkpoint_path}")
        print(f"  input_features : {list(policy_cfg.input_features.keys())}")
        print(f"  output_features: {list(policy_cfg.output_features.keys())}")
        print(f"  device         : {self.device}")
        if self.rename_map:
            print(f"  rename_map     : {self.rename_map}")

        # ---- build pre/post processor pipelines ----
        # Some checkpoints persist a `device_processor` step with `device: "cuda"`.
        # On CPU-only runs, pipeline construction fails before we can post-adjust
        # step devices. Override at load time so construction is always safe.
        device_step_override = {"device": str(self.device)}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=self.checkpoint_path,
            dataset_stats=ds_meta.stats,
            preprocessor_overrides={"device_processor": device_step_override},
            postprocessor_overrides={"device_processor": device_step_override},
        )

        # Ensure the loaded processor pipelines follow the requested device.
        self._force_device_in_pipeline(self.preprocessor, self.device)
        self._force_device_in_pipeline(self.postprocessor, self.device)

        self.use_amp = policy_cfg.use_amp

        # ---- decide which RLBench cameras to fetch ----
        # RLBench Observation provides `front_rgb` and `wrist_rgb` in this script.
        # Some checkpoints (e.g., SmolVLA) expect different camera keys (camera1/2/3)
        # but ship a `rename_observations_processor` in the saved preprocessor.
        # So we provide RLBench/dataset keys (front/wrist) and let the preprocessor
        # rename them if needed.
        preferred = [
            "observation.images.front_rgb",
            "observation.images.wrist_rgb",
        ]
        self.image_keys = [k for k in preferred if k in self.ds_features]
        if not self.image_keys:
            self.image_keys = sorted(
                k for k in self.ds_features.keys() if k.startswith("observation.images.")
            )
        if not self.image_keys:
            raise RuntimeError(
                "No image keys found to fetch from RLBench. "
                "Expected dataset features to include observation.images.* or at least front_rgb/wrist_rgb."
            )

        # Low-dim state keys: provide whatever the dataset declares.
        # Some joint-space datasets store joints in `observation.state` (SmolVLA convention),
        # while older exports used `observation.joint_state`.
        self.state_keys: list[str] = []
        for k in ["observation.state", "observation.joint_state"]:
            if k in self.ds_features:
                self.state_keys.append(k)
        if not self.state_keys:
            raise RuntimeError(
                "No supported low-dim state keys found in dataset features. "
                "Expected observation.state and/or observation.joint_state."
            )

    def reset(self):
        """Reset internal action-chunk cache in the policy (if present)."""
        reset_fn = getattr(self.model, "reset", None)
        if callable(reset_fn):
            reset_fn()

    @staticmethod
    def _resolve_pretrained_model_dir(path: str) -> str:
        """Resolve a user-provided path to a `pretrained_model/` directory."""
        p = Path(path)

        # 1) Direct path to pretrained_model
        if (p / "config.json").is_file():
            return str(p)

        # 2) Path points to a checkpoint dir that contains pretrained_model/
        if (p / "pretrained_model" / "config.json").is_file():
            return str(p / "pretrained_model")

        # 3) Path points to a run dir that contains checkpoints/
        last = p / "checkpoints" / "last" / "pretrained_model"
        if (last / "config.json").is_file():
            return str(last)

        ckpt_root = p / "checkpoints"
        if ckpt_root.is_dir():
            candidates: list[Path] = []
            for ckpt_dir in ckpt_root.iterdir():
                if not ckpt_dir.is_dir():
                    continue
                pm = ckpt_dir / "pretrained_model"
                if (pm / "config.json").is_file():
                    candidates.append(pm)

            if candidates:
                def _score(pm_dir: Path) -> tuple[int, str]:
                    name = pm_dir.parent.name
                    return (int(name) if name.isdigit() else -1, name)

                candidates.sort(key=_score)
                return str(candidates[-1])

        raise FileNotFoundError(
            f"Could not find a LeRobot pretrained_model directory from: {path}. "
            "Expected one of: <...>/pretrained_model/, <...>/checkpoints/<id>/pretrained_model/, "
            "or <run_dir>/checkpoints/last/pretrained_model/."
        )

    @staticmethod
    def _load_rename_map_from_checkpoint(checkpoint_path: str) -> dict[str, str] | None:
        preproc_path = Path(checkpoint_path) / "policy_preprocessor.json"
        if not preproc_path.is_file():
            return None
        try:
            with open(preproc_path, "r") as f:
                preproc_cfg = json.load(f)

            for step in preproc_cfg.get("steps", []):
                if step.get("registry_name") != "rename_observations_processor":
                    continue
                rename_map = step.get("config", {}).get("rename_map", None)
                if isinstance(rename_map, dict) and rename_map:
                    if all(isinstance(k, str) and isinstance(v, str) for k, v in rename_map.items()):
                        return rename_map
        except Exception as exc:  # nosec: B110
            print(f"[LeRobotPolicy] Warning: failed to read rename_map from {preproc_path}: {exc}")
        return None

    @staticmethod
    def _load_policy_config_compat(PreTrainedConfig, checkpoint_path: str):
        """Load policy config and gracefully handle stale unknown keys.

        Some checkpoints were exported with keys that are no longer present in
        the current LeRobot config dataclass (e.g., `compile_model`, `compile_mode`).
        In that case, strip only invalid keys reported by the parser and retry.
        """
        try:
            return PreTrainedConfig.from_pretrained(checkpoint_path)
        except Exception as exc:
            cfg_path = Path(checkpoint_path) / "config.json"
            if not cfg_path.is_file():
                raise

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            invalid_keys = [key for key in re.findall(r"`([^`]+)`", str(exc)) if key in cfg]
            if not invalid_keys:
                raise

            patched_cfg = dict(cfg)
            removed_keys: list[str] = []
            for key in invalid_keys:
                if key in patched_cfg:
                    patched_cfg.pop(key)
                    removed_keys.append(key)

            if not removed_keys:
                raise

            with tempfile.TemporaryDirectory(prefix="lerobot_cfg_compat_") as tmp_dir:
                tmp_cfg_path = Path(tmp_dir) / "config.json"
                with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                    json.dump(patched_cfg, f)

                try:
                    policy_cfg = PreTrainedConfig.from_pretrained(tmp_dir)
                except Exception:
                    raise exc

            print(
                "[LeRobotPolicy] Warning: removed unsupported config keys "
                f"{removed_keys} while loading {cfg_path}."
            )
            return policy_cfg

    @staticmethod
    def _force_device_in_pipeline(pipeline, device: torch.device) -> None:
        steps = getattr(pipeline, "steps", [])
        for step in steps:
            registry_name = getattr(step.__class__, "_registry_name", None)
            if registry_name != "device_processor":
                continue
            step.device = str(device)
            post_init = getattr(step, "__post_init__", None)
            if callable(post_init):
                post_init()

    # ------------------------------------------------------------------
    # RLBench Observation  →  LeRobot observation dict  →  action
    # ------------------------------------------------------------------
    def _obs_to_dict(self, obs: Observation) -> dict:
        """Convert an RLBench Observation to the flat dict that LeRobot expects.

        Keys produced (matching the dataset info.json):
            observation.state         – (8,) float32  [EEF pose + gripper] OR [joint positions + gripper]
            observation.joint_state   – (8,) float32  [joint_0..joint_6 gripper] (backward-compatible; if present in dataset)
            observation.images.<cam>  – (H, W, 3) uint8  (only cameras the policy needs)
        """
        obs_dict: dict = {}

        if "observation.state" in self.state_keys:
            # Some datasets use `observation.state` for EEF pose; others (e.g., our
            # joint-space RLBench exports) use it for joint positions. Infer which
            # one the dataset expects based on the feature's motor names.
            state_spec = self.ds_features.get("observation.state", {})
            motor_names = state_spec.get("names") or []
            motor_names = [str(n) for n in motor_names]
            state_is_joint = any(n.startswith("joint_") for n in motor_names[:7])

            if state_is_joint:
                joint_pos = getattr(obs, "joint_positions", None)
                if joint_pos is None:
                    raise RuntimeError(
                        "Dataset expects joint positions in observation.state but RLBench observation has no joint_positions."
                    )
                joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
                if joint_pos.shape[0] < 7:
                    raise RuntimeError(
                        f"Expected joint_positions to have 7 values, got shape {joint_pos.shape}."
                    )
                state = np.concatenate([
                    joint_pos[:7],
                    np.array([float(obs.gripper_open)], dtype=np.float32),
                ])
            else:
                state = np.concatenate([
                    np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)[:7],
                    np.array([float(obs.gripper_open)], dtype=np.float32),
                ])
            obs_dict["observation.state"] = state

        if "observation.joint_state" in self.state_keys:
            joint_pos = getattr(obs, "joint_positions", None)
            if joint_pos is None:
                raise RuntimeError(
                    "Dataset expects observation.joint_state but RLBench observation has no joint_positions."
                )
            joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
            if joint_pos.shape[0] < 7:
                raise RuntimeError(
                    f"Expected joint_positions to have 7 values, got shape {joint_pos.shape}."
                )
            joint_state = np.concatenate([
                joint_pos[:7],
                np.array([float(obs.gripper_open)], dtype=np.float32),
            ])
            obs_dict["observation.joint_state"] = joint_state

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
        action_vec = action_tensor.squeeze(0).cpu().numpy().astype(np.float64).reshape(-1)
        if action_vec.shape[0] != 8:
            raise RuntimeError(f"Expected policy to output an 8-D action, got shape {action_vec.shape}.")

        # joint_velocity mode: policy outputs joint velocity commands directly.
        if self.action_mode == "joint_velocity":
            return action_vec

        # EEF modes: policy outputs delta EEF actions; convert to absolute pose.
        current_state = np.concatenate([
            np.asarray(obs.gripper_pose, dtype=np.float64).reshape(-1)[:7],
            np.array([float(obs.gripper_open)], dtype=np.float64),
        ])
        return delta_action_to_absolute(action_vec, current_state)


# ===================================================================
# Main inference loop
# ===================================================================

def build_policy(args) -> Policy:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --policy lerobot")
    if not args.dataset_root:
        raise ValueError("--dataset_root is required for --policy lerobot")
    return LeRobotPolicy(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        task_description=args.task_description or "",
        action_mode=args.action_mode,
        device=args.device,
    )

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

        descriptions, initial_obs = task_env.reset()
        print(f"  Descriptions: {descriptions}")

        # Pick the language instruction source:
        # - prefer explicit --task_description
        # - otherwise fall back to RLBench's sampled description
        episode_task_description = args.task_description
        if episode_task_description is None:
            episode_task_description = descriptions[0] if descriptions else ""

        # Multi-instruction support:
        # If the task description contains multiple sentences/lines, we execute them
        # sequentially for `--max_steps` steps each, without resetting the simulator.
        task_steps = split_task_description_into_steps(episode_task_description)
        if len(task_steps) > 1:
            print(f"  Task steps ({len(task_steps)}): {task_steps}")

        observations = [initial_obs]
        total_reward = 0.0
        success = False

        global_step = 0
        terminate = False
        stall_eps_rot_rad = float(np.deg2rad(args.stall_eps_rot_deg))
        for step_idx, step_task in enumerate(task_steps):
            set_policy_task_description(policy, step_task)
            policy.reset()

            stall_count = 0

            if len(task_steps) > 1:
                print(f"\n  --- Instruction {step_idx + 1}/{len(task_steps)} ---")
                print(f"  {step_task}")

            for phase_step in range(args.max_steps):
                step_id = global_step
                action = policy.predict(observations[-1])

                # action[:7] = absolute EEF target pose (delta→absolute already
                # converted inside LeRobotPolicy.predict),  action[7] = gripper
                try:
                    obs, reward, terminate = task_env.step(action)
                except Exception as e:
                    print(
                        f"  Step {step_id} (phase_step={phase_step}): action failed ({e}), stopping episode."
                    )
                    terminate = True
                    reward = 0.0
                    obs = None

                if obs is not None:
                    observations.append(obs)
                total_reward += float(reward)

                if reward > 0:
                    success = True

                if terminate or success:
                    print(
                        f"  Step {step_id}: task {'succeeded' if success else 'terminated'}!"
                    )
                    global_step += 1
                    break

                # Stall heuristic: if EEF pose doesn't change for N steps,
                # assume the current instruction is complete and move on.
                if obs is not None and args.stall_steps > 0 and len(observations) >= 2:
                    try:
                        pos_delta, rot_delta = eef_pose_change(observations[-2], observations[-1])
                    except Exception:
                        pos_delta, rot_delta = float("inf"), float("inf")

                    if pos_delta < args.stall_eps_pos and rot_delta < stall_eps_rot_rad:
                        stall_count += 1
                    else:
                        stall_count = 0

                    if stall_count >= args.stall_steps:
                        rot_deg = float(np.rad2deg(rot_delta))
                        if step_idx < (len(task_steps) - 1):
                            print(
                                "  Stall detected "
                                f"(Δpos={pos_delta:.6f} m, Δrot={rot_deg:.3f} deg) "
                                f"for {stall_count} steps — advancing to next instruction."
                            )
                            global_step += 1
                            break
                        else:
                            print(
                                "  Stall detected "
                                f"(Δpos={pos_delta:.6f} m, Δrot={rot_deg:.3f} deg) "
                                f"for {stall_count} steps — stopping episode."
                            )
                            terminate = True
                            global_step += 1
                            break

                if ((step_id + 1) % 20) == 0:
                    print(f"  Step {step_id}: reward={total_reward:.2f}")
                global_step += 1

            if terminate or success:
                break

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
                   help="Path to a LeRobot run dir / checkpoint dir / pretrained_model dir (required if --policy lerobot).")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Path to LeRobot dataset root (has meta/ data/ dirs). "
                        "Required for --policy lerobot to get feature spec + stats.")
    p.add_argument("--task_description", type=str, default=None,
                    help=(
                        "Natural-language task description for the policy. "
                        "If you provide multiple sentences/lines (e.g. 'Pick up paper. Release paper, then place paper in trash bin'), "
                        "the script executes them sequentially for --max_steps steps each, without resetting the simulator."
                    ))
    p.add_argument("--device", type=str, default=None,
                   help="Torch device for policy inference (default: auto).")

    p.add_argument("--stall_steps", type=int, default=0,
                   help=(
                        "Stall heuristic (0 disables). If > 0, consider the current instruction complete when the EEF pose "
                        "change stays below thresholds for this many consecutive steps. For multi-instruction prompts, this advances "
                        "to the next instruction; for the last instruction it stops the episode."
                   ))
    p.add_argument("--stall_eps_pos", type=float, default=0.0005,
                   help="Position change threshold in meters for stall detection (default: 5e-4).")
    p.add_argument("--stall_eps_rot_deg", type=float, default=1.0,
                   help="Rotation change threshold in degrees for stall detection (default: 1.0).")

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
