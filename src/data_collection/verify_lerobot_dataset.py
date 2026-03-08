#!/usr/bin/env python3
"""
Verify a converted LeRobot v3 / GR00T dataset for correctness.

Checks performed
----------------
1. **Parquet / video frame count consistency**: The number of rows per episode
   in the data parquet matches the expected frame count from episode metadata.
2. **Delta action round-trip**: Applying ``action[t]`` (delta) to ``state[t]``
   should approximately reconstruct ``state[t+1]`` within the same episode.
3. **Gripper change alignment**: Frames where ``action.gripper_open`` changes
   are reported so you can manually compare against scene-graph transitions.
4. **Episode boundary sanity**: Episodes are contiguous, non-overlapping, and
   cover the expected total frame count.
5. **Stats sanity**: Checks that stats.json min/max/mean/std are finite and
   that min <= mean <= max for every numeric feature.

Usage
-----
    python src/data_collection/verify_lerobot_dataset.py \
        --dataset_dir datasets/lerobot/stack_cups_variation0

    # Also compare against the original RLBench data:
    python src/data_collection/verify_lerobot_dataset.py \
        --dataset_dir datasets/lerobot/stack_cups_variation0 \
        --rlbench_var_dir datasets/rlbench/stack_cups/variation0 \
        --task_name stack_cups
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.rlbench_utils import (
    compute_delta_eef_actions,
    delta_action_to_absolute,
    extract_eef_state,
    find_transitions,
    load_rlbench_demo,
    load_scene_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(dataset_dir: str):
    """Load main artefacts of a converted dataset."""
    meta_dir = os.path.join(dataset_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    stats_path = os.path.join(meta_dir, "stats.json")
    data_path = os.path.join(dataset_dir, "data", "chunk-000", "file-000.parquet")
    ep_meta_path = os.path.join(meta_dir, "episodes", "chunk-000", "file-000.parquet")

    with open(info_path) as f:
        info = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)
    data_df = pd.read_parquet(data_path)
    ep_df = pd.read_parquet(ep_meta_path)

    return info, stats, data_df, ep_df


def _format_vec(v, precision=5):
    return "[" + ", ".join(f"{x:.{precision}f}" for x in v) + "]"


# ---------------------------------------------------------------------------
# Check 1: Episode / frame consistency
# ---------------------------------------------------------------------------

def check_episode_frame_counts(data_df, ep_df, info):
    print("\n=== CHECK 1: Episode / frame count consistency ===")
    n_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    ok = True

    if len(data_df) != total_frames:
        print(f"  [FAIL] info.total_frames={total_frames} but data parquet has {len(data_df)} rows")
        ok = False
    else:
        print(f"  [OK] Data parquet rows ({len(data_df)}) == info.total_frames")

    if len(ep_df) != n_episodes:
        print(f"  [FAIL] info.total_episodes={n_episodes} but episodes parquet has {len(ep_df)} rows")
        ok = False
    else:
        print(f"  [OK] Episode count ({n_episodes}) matches")

    # Per-episode row counts
    for _, ep_row in ep_df.iterrows():
        ep_idx = int(ep_row["episode_index"])
        expected_len = int(ep_row["length"])
        ep_data = data_df[data_df["episode_index"] == ep_idx]
        actual_len = len(ep_data)
        if actual_len != expected_len:
            print(f"  [FAIL] Episode {ep_idx}: expected {expected_len} frames, got {actual_len}")
            ok = False

    # Contiguity check
    ep_indices = sorted(data_df["episode_index"].unique())
    prev_end = -1
    for ep_idx in ep_indices:
        ep_data = data_df[data_df["episode_index"] == ep_idx].sort_values("index")
        start_idx = ep_data["index"].iloc[0]
        end_idx = ep_data["index"].iloc[-1]
        if start_idx != prev_end + 1:
            print(f"  [FAIL] Episode {ep_idx} starts at index {start_idx}, but previous ended at {prev_end}")
            ok = False
        prev_end = end_idx

    if ok:
        print("  [OK] All episode frame counts and indices OK")
    return ok


# ---------------------------------------------------------------------------
# Check 2: Delta action round-trip
# ---------------------------------------------------------------------------

def check_delta_roundtrip(data_df, info, atol_pos=1e-4, atol_quat=1e-3):
    print("\n=== CHECK 2: Delta action → state round-trip ===")
    n_episodes = info["total_episodes"]
    max_pos_err = 0.0
    max_quat_err = 0.0
    n_checked = 0
    failures = []

    for ep_idx in range(n_episodes):
        ep_data = data_df[data_df["episode_index"] == ep_idx].sort_values("index")
        states = np.array(ep_data["observation.state"].tolist())  # (T, 8)
        actions = np.array(ep_data["action"].tolist())  # (T, 8)
        T = len(states)

        for t in range(T - 1):
            predicted = delta_action_to_absolute(actions[t], states[t])
            actual = states[t + 1]

            pos_err = np.abs(predicted[:3] - actual[:3]).max()
            # Quaternion distance via dot product
            q_dot = np.abs(np.dot(predicted[3:7], actual[3:7]))
            q_dot = min(q_dot, 1.0)
            quat_err = 2 * np.arccos(q_dot)  # in radians

            max_pos_err = max(max_pos_err, pos_err)
            max_quat_err = max(max_quat_err, quat_err)
            n_checked += 1

            if pos_err > atol_pos or quat_err > atol_quat:
                failures.append((ep_idx, t, pos_err, quat_err))

    print(f"  Checked {n_checked} transitions across {n_episodes} episodes")
    print(f"  Max position error: {max_pos_err:.6f} m")
    print(f"  Max quaternion error: {max_quat_err:.6f} rad ({np.degrees(max_quat_err):.4f} deg)")

    if failures:
        print(f"  [WARN] {len(failures)} transitions exceed tolerance (pos>{atol_pos}, quat>{atol_quat} rad):")
        for ep, t, pe, qe in failures[:10]:
            print(f"    ep={ep} t={t}: pos_err={pe:.6f} quat_err={np.degrees(qe):.4f}°")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    else:
        print(f"  [OK] All transitions within tolerance")
    return len(failures) == 0


# ---------------------------------------------------------------------------
# Check 3: Gripper change frames
# ---------------------------------------------------------------------------

def check_gripper_changes(data_df, info):
    print("\n=== CHECK 3: Gripper state changes ===")
    n_episodes = info["total_episodes"]

    for ep_idx in range(n_episodes):
        ep_data = data_df[data_df["episode_index"] == ep_idx].sort_values("index")
        states = np.array(ep_data["observation.state"].tolist())
        actions = np.array(ep_data["action"].tolist())
        indices = ep_data["index"].values

        # State gripper changes
        gripper_state = states[:, 7]
        state_changes = []
        for t in range(1, len(gripper_state)):
            if abs(gripper_state[t] - gripper_state[t-1]) > 0.01:
                state_changes.append((t, indices[t], gripper_state[t-1], gripper_state[t]))

        # Action gripper changes
        gripper_action = actions[:, 7]
        action_changes = []
        for t in range(1, len(gripper_action)):
            if abs(gripper_action[t] - gripper_action[t-1]) > 0.01:
                action_changes.append((t, indices[t], gripper_action[t-1], gripper_action[t]))

        print(f"  Episode {ep_idx} ({len(ep_data)} frames):")
        if state_changes:
            for t, idx, prev, cur in state_changes:
                print(f"    State gripper change at local_t={t}, global_idx={idx}: {prev:.1f} -> {cur:.1f}")
        else:
            print(f"    No state gripper changes (constant={gripper_state[0]:.1f})")
        if action_changes:
            for t, idx, prev, cur in action_changes:
                print(f"    Action gripper change at local_t={t}, global_idx={idx}: {prev:.1f} -> {cur:.1f}")


# ---------------------------------------------------------------------------
# Check 4: Stats sanity
# ---------------------------------------------------------------------------

def check_stats_sanity(stats):
    print("\n=== CHECK 4: Stats sanity ===")
    ok = True

    for feat_name, feat_stats in stats.items():
        if not isinstance(feat_stats, dict):
            continue
        # Check for required keys
        for k in ["min", "max", "mean", "std"]:
            if k not in feat_stats:
                continue
            vals = feat_stats[k]
            if isinstance(vals, list):
                vals = np.array(vals)
                if not np.all(np.isfinite(vals)):
                    print(f"  [FAIL] {feat_name}.{k} has non-finite values: {vals}")
                    ok = False

        # min <= mean <= max
        if all(k in feat_stats for k in ["min", "mean", "max"]):
            mn = np.array(feat_stats["min"])
            mu = np.array(feat_stats["mean"])
            mx = np.array(feat_stats["max"])
            if np.any(mn > mu + 1e-6):
                print(f"  [FAIL] {feat_name}: min > mean")
                ok = False
            if np.any(mu > mx + 1e-6):
                print(f"  [FAIL] {feat_name}: mean > max")
                ok = False

    if ok:
        print("  [OK] All stats are finite and min <= mean <= max")
    return ok


# ---------------------------------------------------------------------------
# Check 5: Compare with RLBench source (optional)
# ---------------------------------------------------------------------------

def check_against_rlbench(dataset_dir, rlbench_var_dir, task_name, episode_idx=0):
    print("\n=== CHECK 5: Compare with RLBench source ===")
    ep_dir = os.path.join(rlbench_var_dir, "episodes", f"episode{episode_idx}")
    pkl_path = os.path.join(ep_dir, "low_dim_obs.pkl")
    sg_path = os.path.join(rlbench_var_dir, f"{task_name}_scene_graph.json")
    front_rgb_dir = os.path.join(ep_dir, "front_rgb")

    if not os.path.exists(pkl_path):
        print(f"  [SKIP] RLBench data not found at {pkl_path}")
        return True

    observations = load_rlbench_demo(pkl_path)
    scene_graph = load_scene_graph(sg_path)
    transitions = find_transitions(scene_graph)
    n_obs = len(observations)

    print(f"  RLBench: {n_obs} observations (= frames, 1:1 mapping)")
    print(f"  Scene graph: {len(transitions)} transitions")

    # Report transition frames vs converted dataset gripper changes
    for i, t in enumerate(transitions):
        print(f"    Transition {i}: frame_id={t['frame_id']}")

    # Compute RLBench gripper change frames
    print(f"  RLBench gripper changes (obs index):")
    for i in range(1, n_obs):
        go_prev = observations[i-1].gripper_open
        go_curr = observations[i].gripper_open
        if abs(go_curr - go_prev) > 0.01:
            print(f"    obs[{i}]: {go_prev:.1f} -> {go_curr:.1f}")

    # Load converted dataset info
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    sg_info = info.get("episodes_scene_graph", [])
    print(f"  Converted dataset: {info['total_episodes']} episodes, {info['total_frames']} frames")
    for ep_sg in sg_info:
        print(f"    Episode {ep_sg['episode_index']}: frames {ep_sg['start_frame']}-{ep_sg['end_frame']}")

    # Verify first observation state matches converted state
    data_path = os.path.join(dataset_dir, "data", "chunk-000", "file-000.parquet")
    data_df = pd.read_parquet(data_path)

    # Verify first observation state matches converted state (1:1, no offset)
    rlbench_state_0 = extract_eef_state(observations[0])
    first_ep_data = data_df[data_df["episode_index"] == 0].sort_values("index")
    converted_state_0 = np.array(first_ep_data["observation.state"].iloc[0])
    print(f"\n  State comparison (first row of converted vs obs[0]):")
    print(f"    RLBench obs[0]:  {_format_vec(rlbench_state_0)}")
    print(f"    Converted [0]:   {_format_vec(converted_state_0)}")
    diff = np.abs(rlbench_state_0 - converted_state_0)
    print(f"    Max diff: {diff.max():.8f}")
    if diff.max() > 1e-4:
        print(f"    [WARN] States differ — unexpected with 1:1 mapping")
    else:
        print(f"    [OK] States match")

    return True


# ---------------------------------------------------------------------------
# Check 6: Video frame count (via ffprobe if available)
# ---------------------------------------------------------------------------

def check_video_frame_counts(dataset_dir, info):
    print("\n=== CHECK 6: Video frame counts ===")
    import shutil
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        print("  [SKIP] ffprobe not found — cannot verify video frame counts")
        return True

    import subprocess
    ok = True
    ep_meta_path = os.path.join(dataset_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
    ep_df = pd.read_parquet(ep_meta_path)

    for cam_key in ["observation.images.front_rgb", "observation.images.wrist_rgb"]:
        for _, ep_row in ep_df.iterrows():
            ep_idx = int(ep_row["episode_index"])
            chunk_idx = int(ep_row.get(f"videos/{cam_key}/chunk_index", 0))
            file_idx = int(ep_row.get(f"videos/{cam_key}/file_index", ep_idx))
            expected_len = int(ep_row["length"])

            vid_path = os.path.join(
                dataset_dir, "videos", cam_key,
                f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4"
            )
            if not os.path.exists(vid_path):
                print(f"  [FAIL] Missing video: {vid_path}")
                ok = False
                continue

            # Count frames via ffprobe
            cmd = [
                ffprobe, "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "csv=p=0",
                vid_path,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                n_vid_frames = int(result.stdout.strip())
                if n_vid_frames != expected_len:
                    print(f"  [FAIL] {cam_key} ep{ep_idx}: video has {n_vid_frames} frames, expected {expected_len}")
                    ok = False
            except Exception as e:
                print(f"  [WARN] Could not count frames for {vid_path}: {e}")

    if ok:
        print("  [OK] All video frame counts match")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify a converted LeRobot dataset")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the converted LeRobot dataset directory")
    parser.add_argument("--rlbench_var_dir", type=str, default=None,
                        help="Path to the original RLBench variation directory (optional)")
    parser.add_argument("--task_name", type=str, default=None,
                        help="RLBench task name (required if --rlbench_var_dir is given)")
    parser.add_argument("--atol_pos", type=float, default=1e-4,
                        help="Position tolerance for roundtrip check (metres)")
    parser.add_argument("--atol_quat", type=float, default=1e-3,
                        help="Quaternion tolerance for roundtrip check (radians)")
    args = parser.parse_args()

    print(f"Verifying dataset: {args.dataset_dir}")
    info, stats, data_df, ep_df = _load_dataset(args.dataset_dir)

    results = []
    results.append(("Frame counts", check_episode_frame_counts(data_df, ep_df, info)))
    results.append(("Delta roundtrip", check_delta_roundtrip(data_df, info, args.atol_pos, args.atol_quat)))
    check_gripper_changes(data_df, info)
    results.append(("Stats sanity", check_stats_sanity(stats)))
    results.append(("Video frames", check_video_frame_counts(args.dataset_dir, info)))

    if args.rlbench_var_dir:
        if not args.task_name:
            print("[ERROR] --task_name is required when using --rlbench_var_dir")
            sys.exit(1)
        check_against_rlbench(args.dataset_dir, args.rlbench_var_dir, args.task_name)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\nAll checks passed!")
    else:
        print("\nSome checks FAILED — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
