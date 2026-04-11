"""
extract_features.py

Run once before training. Passes all frames through frozen R3M and saves
the resulting feature vectors to disk as numpy arrays.

Output:
    features/
        0/          # episode 0
            0.npy   # camera 0 features, shape (num_frames, 512)
            1.npy   # camera 1 features, shape (num_frames, 512)
        1/
            ...
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from r3m import load_r3m

data_folder = "wiping_diverse_50_simplified_3"

FRAMES_DIR   = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/frames"
FEATURES_DIR = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/features"
CAMERAS      = [0, 1]
BATCH_SIZE   = 64    # how many frames to encode at once
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
])

def extract_features():
    print(f"Using device: {DEVICE}")

    # Load R3M once
    encoder = load_r3m("resnet18").to(DEVICE)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    episode_folders = sorted(
        [d for d in os.listdir(FRAMES_DIR) if os.path.isdir(
            os.path.join(FRAMES_DIR, d))],
        key=lambda x: int(x)
    )

    for ep_name in tqdm(episode_folders, desc="Episodes"):
        ep_idx = int(ep_name)

        for cam in CAMERAS:
            out_path = os.path.join(FEATURES_DIR, str(ep_idx), f"{cam}.npy")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Skip if already extracted
            if os.path.exists(out_path):
                continue

            frame_dir = os.path.join(FRAMES_DIR, str(ep_idx), str(cam))
            frame_files = sorted(
                [f for f in os.listdir(frame_dir) if f.endswith(".png")],
                key=lambda x: int(x.split(".")[0])
            )

            all_features = []

            # Process in batches
            for i in range(0, len(frame_files), BATCH_SIZE):
                batch_files = frame_files[i:i + BATCH_SIZE]
                batch = []

                for fname in batch_files:
                    img = cv2.imread(os.path.join(frame_dir, fname))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_t = transform(Image.fromarray(img))
                    batch.append(img_t)

                batch_tensor = torch.stack(batch).to(DEVICE)  # (B, 3, 224, 224)

                with torch.no_grad():
                    features = encoder(batch_tensor).cpu().numpy()  # (B, 512)

                all_features.append(features)

            all_features = np.concatenate(all_features, axis=0)  # (num_frames, 512)
            np.save(out_path, all_features)

    print(f"\nDone. Features saved to {FEATURES_DIR}")

if __name__ == "__main__":
    extract_features()