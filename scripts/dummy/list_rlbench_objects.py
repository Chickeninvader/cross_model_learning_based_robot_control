#!/usr/bin/env python3
"""List RLBench task object names and simulator handles.

Usage:
  python scripts/list_rlbench_objects.py --task put_rubbish_in_bin --variation 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyrep.const import ObjectType

ROOT = Path(__file__).resolve().parents[1]
RLBENCH_ROOT = ROOT / "external" / "RLBench"
if str(RLBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RLBENCH_ROOT))

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment


def _type_name(obj_type: ObjectType) -> str:
    return getattr(obj_type, "name", str(obj_type))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="put_rubbish_in_bin")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--show_gui", action="store_true")
    args = parser.parse_args()

    obs = ObservationConfig()
    obs.set_all(False)
    obs.set_all_low_dim(True)
    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs,
        headless=not args.show_gui,
    )
    env.launch()
    try:
        task_cls = task_file_to_task_class(args.task)
        task_env = env.get_task(task_cls)
        task_env.set_variation(args.variation)
        task_env.reset()

        base = task_env._task.get_base()
        objects = base.get_objects_in_tree(exclude_base=False, first_generation_only=False)
        rows = []
        for ob in objects:
            try:
                rows.append((ob.get_name(), ob.get_handle(), _type_name(ob.get_type())))
            except Exception:
                continue
        rows.sort(key=lambda x: (x[2], x[0]))

        print(f"Task: {args.task} variation: {args.variation}")
        print("name\thandle\ttype")
        for name, handle, tname in rows:
            print(f"{name}\t{handle}\t{tname}")
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
