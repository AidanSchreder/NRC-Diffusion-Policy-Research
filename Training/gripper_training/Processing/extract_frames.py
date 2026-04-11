"""
extract_frames.py

One-time preprocessing step. Extracts all frames from episode videos into
individual .png files for fast random access during training.

Run this ONCE before training:
    python extract_frames.py

Output structure:
    frames/
        0/              # episode 0
            0/          # camera 0
                00000.png
                00001.png
                ...
            1/          # camera 1
                00000.png
                ...
        1/              # episode 1
            0/
            1/
        ...

After extraction, update FRAMES_DIR in train_gripper.py to point to this folder.
"""

import os
import cv2
from tqdm import tqdm

data_folder = "wiping_diverse_50_simplified_3"

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_DIR  = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/videos"    # same VIDEO_DIR as in train_gripper.py
FRAMES_DIR = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/frames"    # where extracted frames will be saved
CAMERAS    = [0, 1]
# ─────────────────────────────────────────────────────────────────────────────


def extract_all_frames(video_dir, frames_dir, cameras):
    # Find all episode folders (named 0, 1, 2, ...)
    episode_folders = sorted(
        [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))],
        key=lambda x: int(x)
    )
    num_episodes = len(episode_folders)
    print(f"Found {num_episodes} episodes across cameras {cameras}\n")

    total_frames = 0

    for ep_name in tqdm(episode_folders, desc="Episodes"):
        ep_idx = int(ep_name)

        for cam in cameras:
            video_path = os.path.join(video_dir, ep_name, f"{cam}.mp4")
            out_dir    = os.path.join(frames_dir, str(ep_idx), str(cam))
            os.makedirs(out_dir, exist_ok=True)

            # Skip if already extracted (allows resuming interrupted extraction)
            cap         = cv2.VideoCapture(video_path)
            n_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            existing = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            if existing == n_frames:
                total_frames += n_frames
                continue

            # Extract every frame
            cap         = cv2.VideoCapture(video_path)
            frame_idx   = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_path = os.path.join(out_dir, f"{frame_idx:05d}.png")
                cv2.imwrite(out_path, frame)
                frame_idx += 1

            cap.release()
            total_frames += frame_idx

    print(f"\nExtraction complete.")
    print(f"Total frames saved : {total_frames:,}")
    print(f"Output directory   : {frames_dir}")


if __name__ == "__main__":
    extract_all_frames(VIDEO_DIR, FRAMES_DIR, CAMERAS)