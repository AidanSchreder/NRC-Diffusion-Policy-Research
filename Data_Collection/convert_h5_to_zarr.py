"""
python convert_h5_to_zarr.py \
/media/Data/demoaccount2/robot_manipulation/Collected_Data/test_force/train \
/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/test_force

"""

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


REQUIRED_H5_KEYS = ("robot_eef_pose", "start_pose", "action", "timestamp")
OPTIONAL_H5_KEYS = ("fts_info",)
VIDEO_FILENAMES = ("0.mp4", "1.mp4")


def _default_training_root():
    candidate = Path(
        "/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0"
    )
    return candidate if candidate.exists() else None


def _load_replay_buffer_class(training_root):
    training_root = Path(training_root)
    if not training_root.exists():
        raise FileNotFoundError(f"Training repo root does not exist: {training_root}")

    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))

    module = importlib.import_module("Training.common.replay_buffer")
    return module.ReplayBuffer


def _copy_episode_videos(src_split_dir, episode_stem, dst_video_root, episode_idx):
    src_video_dir = src_split_dir / episode_stem
    dst_video_dir = dst_video_root / str(episode_idx)
    dst_video_dir.mkdir(parents=True, exist_ok=True)

    copied_any = False
    for filename in VIDEO_FILENAMES:
        src = src_video_dir / filename
        if src.exists():
            shutil.copy2(src, dst_video_dir / filename)
            copied_any = True
    return copied_any


def _resolve_timestamps(h5_data, h5_path, dt):
    timestamp = np.asarray(h5_data["timestamp"], dtype=np.float64)
    if len(timestamp) == 0:
        raise ValueError(f"{h5_path}: timestamp dataset is empty.")
    return timestamp


def _derive_gripper_state(h5_data, n_steps):
    if "action" in h5_data:
        raw_action = np.asarray(h5_data["action"])
        if raw_action.ndim == 2 and raw_action.shape[0] == n_steps and raw_action.shape[1] >= 7:
            return np.asarray(raw_action[:, -1:], dtype=np.float32)
    return np.zeros((n_steps, 1), dtype=np.float32)


def _build_episode_dict(h5_path, dt):
    with h5py.File(h5_path, "r") as f:
        missing = [key for key in REQUIRED_H5_KEYS if key not in f]
        if missing:
            raise KeyError(f"{h5_path} is missing required dataset(s): {missing}")

        h5_data = {key: np.asarray(f[key][:]) for key in REQUIRED_H5_KEYS}
        for key in OPTIONAL_H5_KEYS:
            if key in f:
                h5_data[key] = np.asarray(f[key][:])

    robot_eef_pose = np.asarray(h5_data["robot_eef_pose"], dtype=np.float32)
    start_pose = np.asarray(h5_data["start_pose"], dtype=np.float32)
    if len(robot_eef_pose) == 0:
        raise ValueError(f"{h5_path}: robot_eef_pose is empty.")
    if len(start_pose) != len(robot_eef_pose):
        raise ValueError(
            f"{h5_path}: start_pose length {len(start_pose)} does not match "
            f"robot_eef_pose length {len(robot_eef_pose)}."
        )

    n_steps = len(robot_eef_pose)
    robot_gripper_state = _derive_gripper_state(h5_data, n_steps)
    timestamp = _resolve_timestamps(h5_data, h5_path, dt=dt)
    if len(timestamp) != n_steps:
        raise ValueError(
            f"{h5_path}: timestamp length {len(timestamp)} does not match "
            f"robot_eef_pose length {n_steps}."
        )

    action = np.asarray(h5_data["action"], dtype=np.float32)
    if action.ndim != 2 or action.shape[0] != n_steps or action.shape[1] != 7:
        raise ValueError(
            f"{h5_path}: action shape {action.shape} is incompatible with "
            f"episode length {n_steps}; expected [T, 7]."
        )

    episode = {
        "robot_eef_pose": robot_eef_pose,
        "robot_gripper_state": robot_gripper_state,
        "start_pose": start_pose,
        "action": action,
        "timestamp": np.asarray(timestamp, dtype=np.float64),
    }

    if "fts_info" in h5_data:
        fts_info = np.asarray(h5_data["fts_info"], dtype=np.float32)
        if len(fts_info) == n_steps:
            episode["fts_info"] = fts_info
        else:
            print(
                f"[WARN] {h5_path}: skipping fts_info because length {len(fts_info)} "
                f"does not match episode length {n_steps}"
            )

    return episode


def convert_split(input_split_dir, output_dir, training_root, dt=None, copy_videos=True):
    input_split_dir = Path(input_split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = output_dir / "videos"
    video_root.mkdir(parents=True, exist_ok=True)

    ReplayBuffer = _load_replay_buffer_class(training_root)
    replay_buffer_path = output_dir / "replay_buffer.zarr"
    replay_buffer = ReplayBuffer.create_from_path(str(replay_buffer_path), mode="a")

    episode_paths = sorted(input_split_dir.glob("episode_*.h5"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.h5 files found in {input_split_dir}")

    for h5_path in episode_paths:
        episode = _build_episode_dict(h5_path, dt=dt)
        replay_buffer.add_episode(episode, compressors="disk")
        episode_idx = replay_buffer.n_episodes - 1

        if copy_videos:
            copied = _copy_episode_videos(
                src_split_dir=input_split_dir,
                episode_stem=h5_path.stem,
                dst_video_root=video_root,
                episode_idx=episode_idx,
            )
            if not copied:
                print(f"[WARN] No videos found for {h5_path.stem} in {input_split_dir / h5_path.stem}")

        print(f"Converted {h5_path.name} -> episode index {episode_idx}")

    print(f"Saved replay buffer to {replay_buffer_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert episode_*.h5 files into the replay_buffer.zarr format used by the training pipeline."
    )
    parser.add_argument(
        "input_split_dir",
        type=Path,
        help="Directory containing episode_*.h5 and episode_xxxxx/0.mp4,1.mp4 folders.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where replay_buffer.zarr and videos/ will be written.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=_default_training_root(),
        help="Path to the training repo root that contains the Training/ package.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Fallback timestep in seconds if timestamps are not stored in the H5 or sidecar NPZ.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=None,
        help="Fallback frequency in Hz if timestamps are not stored in the H5 or sidecar NPZ.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Convert only arrays and do not copy per-episode MP4 files.",
    )
    args = parser.parse_args()

    if args.training_root is None:
        raise ValueError(
            "Could not infer the training repo root. Pass --training-root to the repository "
            "that contains the Training/ package."
        )

    dt = args.dt
    if dt is None and args.hz is not None:
        dt = 1.0 / args.hz

    convert_split(
        input_split_dir=args.input_split_dir,
        output_dir=args.output_dir,
        training_root=args.training_root,
        dt=dt,
        copy_videos=not args.skip_videos,
    )


if __name__ == "__main__":
    main()
