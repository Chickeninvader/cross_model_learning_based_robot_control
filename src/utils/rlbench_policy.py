import json
import re
import tempfile
from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from rlbench.backend.observation import Observation

from src.utils.rlbench_utils import delta_action_to_absolute


class Policy(ABC):
    """Base class for RLBench inference policies."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the beginning of every episode."""

    @abstractmethod
    def predict(self, obs: Observation) -> np.ndarray:
        """Return an 8-D action.

        Interpretation depends on the RLBench action mode:

        - EEF modes (`ee_planning` / `ee_ik`):
            [x, y, z, qx, qy, qz, qw, gripper]  (absolute target EEF pose)
        - `joint_velocity`:
            [dq0, dq1, dq2, dq3, dq4, dq5, dq6, gripper]  (joint velocity command)

        The last value is the gripper command (>0.5 → open, <0.5 → close).
        """


class LeRobotPolicy(Policy):
    """Run a trained LeRobot policy (e.g. GROOT / SmolVLA) in RLBench.

    Loads the checkpoint using LeRobot's factory utilities and runs the full
    inference pipeline: observation → preprocessor → policy → postprocessor → action.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dataset_root: str,
        task_description: str = "",
        action_mode: str = "ee_planning",
        rename_map: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ):
        # Lazy LeRobot imports (keeps non-lerobot paths dependency-free).
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference

        self.checkpoint_path = self._resolve_pretrained_model_dir(checkpoint_path)
        self.task_description = task_description
        self.action_mode = action_mode
        self._prepare_obs = prepare_observation_for_inference

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        dataset_root_path = Path(dataset_root)
        ds_meta = LeRobotDatasetMetadata(
            repo_id="local/rlbench_dataset",
            root=dataset_root_path,
        )
        self.ds_features = ds_meta.features

        policy_cfg = self._load_policy_config_compat(PreTrainedConfig, self.checkpoint_path)
        policy_cfg.pretrained_path = self.checkpoint_path
        policy_cfg.device = str(self.device)

        if rename_map is None and policy_cfg.type == "smolvla":
            rename_map = self._load_rename_map_from_checkpoint(self.checkpoint_path)
        self.rename_map = rename_map

        self.model = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=self.rename_map)
        self.model.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg,
            pretrained_path=self.checkpoint_path,
            dataset_stats=ds_meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        self._force_device_in_pipeline(self.preprocessor, self.device)
        self._force_device_in_pipeline(self.postprocessor, self.device)

        self.use_amp = policy_cfg.use_amp

        preferred = [
            "observation.images.front_rgb",
            "observation.images.wrist_rgb",
        ]
        self.image_keys: List[str] = [k for k in preferred if k in self.ds_features]
        if not self.image_keys:
            self.image_keys = sorted(
                k for k in self.ds_features.keys() if k.startswith("observation.images.")
            )
        if not self.image_keys:
            raise RuntimeError(
                "No image keys found to fetch from RLBench. "
                "Expected dataset features to include observation.images.* or at least front_rgb/wrist_rgb."
            )

        self.state_keys: List[str] = []
        for k in ["observation.state", "observation.joint_state"]:
            if k in self.ds_features:
                self.state_keys.append(k)
        if not self.state_keys:
            raise RuntimeError(
                "No supported low-dim state keys found in dataset features. "
                "Expected observation.state and/or observation.joint_state."
            )

    def reset(self) -> None:
        reset_fn = getattr(self.model, "reset", None)
        if callable(reset_fn):
            reset_fn()

    @staticmethod
    def _resolve_pretrained_model_dir(path: str) -> str:
        """Resolve a user-provided path to a `pretrained_model/` directory."""
        p = Path(path)

        if (p / "config.json").is_file():
            return str(p)

        if (p / "pretrained_model" / "config.json").is_file():
            return str(p / "pretrained_model")

        last = p / "checkpoints" / "last" / "pretrained_model"
        if (last / "config.json").is_file():
            return str(last)

        ckpt_root = p / "checkpoints"
        if ckpt_root.is_dir():
            candidates: List[Path] = []
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
    def _load_rename_map_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, str]]:
        preproc_path = Path(checkpoint_path) / "policy_preprocessor.json"
        if not preproc_path.is_file():
            return None
        try:
            with open(preproc_path, "r", encoding="utf-8") as f:
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
        """Load policy config and gracefully handle stale unknown keys."""
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
            removed_keys: List[str] = []
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

    def _obs_to_dict(self, obs: Observation) -> dict:
        """Convert an RLBench Observation to the flat dict that LeRobot expects."""
        obs_dict: dict = {}

        if "observation.state" in self.state_keys:
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
                state = np.concatenate(
                    [
                        joint_pos[:7],
                        np.array([float(obs.gripper_open)], dtype=np.float32),
                    ]
                )
            else:
                state = np.concatenate(
                    [
                        np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)[:7],
                        np.array([float(obs.gripper_open)], dtype=np.float32),
                    ]
                )
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
            joint_state = np.concatenate(
                [
                    joint_pos[:7],
                    np.array([float(obs.gripper_open)], dtype=np.float32),
                ]
            )
            obs_dict["observation.joint_state"] = joint_state

        for img_key in self.image_keys:
            cam_attr = img_key.split("observation.images.")[-1]
            img_data = getattr(obs, cam_attr, None)
            if img_data is not None:
                obs_dict[img_key] = img_data.astype(np.uint8)
            else:
                raise RuntimeError(
                    f"Policy expects image '{img_key}' but RLBench observation "
                    f"has no attribute '{cam_attr}'."
                )

        return obs_dict

    def predict(self, obs: Observation) -> np.ndarray:
        """Full LeRobot inference pipeline: obs → preprocess → model → postprocess → action."""
        obs_dict = self._obs_to_dict(obs)
        obs_dict = copy(obs_dict)

        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else nullcontext(),
        ):
            obs_dict = self._prepare_obs(
                obs_dict,
                self.device,
                task=self.task_description,
                robot_type="rlbench_franka",
            )
            obs_dict = self.preprocessor(obs_dict)
            action_tensor = self.model.select_action(obs_dict)
            action_tensor = self.postprocessor(action_tensor)

        action_vec = action_tensor.squeeze(0).cpu().numpy().astype(np.float64).reshape(-1)
        if action_vec.shape[0] != 8:
            raise RuntimeError(f"Expected policy to output an 8-D action, got shape {action_vec.shape}.")

        if self.action_mode == "joint_velocity":
            return action_vec

        current_state = np.concatenate(
            [
                np.asarray(obs.gripper_pose, dtype=np.float64).reshape(-1)[:7],
                np.array([float(obs.gripper_open)], dtype=np.float64),
            ]
        )
        return delta_action_to_absolute(action_vec, current_state)

