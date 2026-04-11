"""
train_gripper_classifier.py

Trains a binary gripper classifier on top of R3M (ResNet18) features.

Data layout:
    replay_buffer.zarr/
        data/
            joint_angles        # (N, 6)
            joint_velocities    # (N, 6)
            ee_pose             # (N, 6 or 7)
            gripper_state       # (N,)
        meta/
            episode_ends        # (num_episodes,) — last global index of each episode

    videos/
        0/
            0.mp4               # camera 0, episode 0
            1.mp4               # camera 1, episode 0
        1/
            0.mp4
            1.mp4
        ...

Frame alignment:
    Robot runs at 10Hz, video captured at 15fps -> ratio = 1.5
    For robot timestep i within an episode:
        video_frame_index = round(i * VIDEO_TO_ROBOT_RATIO)
"""

import os
import math
import zarr
import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from Training.gripper_training.Model.gripper_classifier import GripperClassifier
from tqdm import tqdm


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

data_folder = "wiping_diverse_100"
output_file = "gripper_training_history_mod-1.ckpt"

ZARR_PATH       = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/replay_buffer.zarr"
SAVE_PATH       = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/gripper_training/Checkpoints/{output_file}"
FEATURES_DIR    = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/features"

# Zarr keys
JOINTS_KEY      = "data/robot_joint"
VELOCITIES_KEY  = "data/robot_joint_vel"
EE_POSE_KEY     = "data/robot_eef_pose"
GRIPPER_KEY     = "data/robot_gripper_state"
EPISODE_KEY     = "meta/episode_ends"

# Which cameras to use -- set to [0], [1], or [0, 1] for both
CAMERAS         = [0, 1]

EE_POSE_DIM     = 6          # 6 for euler, 7 for quaternion
PROPRIO_DIM     = 6 + 6 + EE_POSE_DIM
T = 5 #history window (last 5 timesteps)

ROBOT_HZ        = 10
VIDEO_FPS       = 15
VIDEO_TO_ROBOT_RATIO = VIDEO_FPS / ROBOT_HZ   # 1.5

GRIPPER_THRESHOLD = 0.5

BATCH_SIZE      = 128
EPOCHS          = 60
LR              = 1e-4

# slow = 1e-4
# med = 5e-4
# fast = 1e-3
VAL_SPLIT       = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class ZarrVideoGripperDataset(Dataset):

    def __init__(self, zarr_path, features_dir, cameras, gripper_threshold=0.5):
        super().__init__()

        # Load Zarr robot data
        store            = zarr.open(zarr_path, mode="r")
        self.joints      = store[JOINTS_KEY][:]
        self.velocities  = store[VELOCITIES_KEY][:]
        self.ee_pose     = store[EE_POSE_KEY][:]
        gripper_raw      = store[GRIPPER_KEY][:]
        episode_ends     = store[EPISODE_KEY][:]

        self.labels    = (gripper_raw > gripper_threshold).astype(np.float32)
        self.n_samples = len(self.labels)

        # Episode boundary lookup
        self.episode_starts = np.concatenate([[0], episode_ends[:-1]])
        self.global_to_episode = np.zeros(self.n_samples, dtype=np.int32)
        self.global_to_local   = np.zeros(self.n_samples, dtype=np.int32)
        for ep_idx, (start, end) in enumerate(zip(self.episode_starts, episode_ends)):
            start, end = int(start), int(end)
            self.global_to_episode[start:end] = ep_idx
            self.global_to_local[start:end]   = np.arange(end - start)

        # Load ALL R3M features into RAM upfront
        # At 512 floats * 45000 frames * 2 cameras = ~180MB — very manageable
        print("Loading pre-computed R3M features into memory...")
        self.features = {}
        for ep_name in sorted(os.listdir(features_dir), key=lambda x: int(x)):
            ep_idx = int(ep_name)
            self.features[ep_idx] = {}
            for cam in cameras:
                path = os.path.join(features_dir, ep_name, f"{cam}.npy")
                self.features[ep_idx][cam] = np.load(path)  # (num_frames, 512)

        self.cameras      = cameras
        self.features_dir = features_dir

        # Proprioception normalisation
        proprio_all       = np.concatenate(
            [self.joints, self.velocities, self.ee_pose], axis=-1
        )
        self.proprio_mean = proprio_all.mean(axis=0).astype(np.float32)
        self.proprio_std = np.maximum(
            proprio_all.std(axis=0),
            0.01
        ).astype(np.float32)
        #self.proprio_std  = (proprio_all.std(axis=0) + 1e-8).astype(np.float32)

        # Class balance summary
        print(f"  Total timesteps : {self.n_samples}")
        print(f"  Gripper closed  : {self.labels.sum():.0f}  "
              f"({100 * self.labels.mean():.1f}%)")
        print(f"  Gripper open    : {(1 - self.labels).sum():.0f}  "
              f"({100 * (1 - self.labels).mean():.1f}%)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ep_idx     = int(self.global_to_episode[idx])
        local_step = int(self.global_to_local[idx])
        frame_idx  = round(local_step * VIDEO_TO_ROBOT_RATIO)
        ep_start = int(self.episode_starts[ep_idx])

        # Clamp to avoid overshooting the end of the feature array
        cam_features = []
        for cam in self.cameras:
            max_frame = len(self.features[ep_idx][cam]) - 1
            frame_idx_clamped = min(frame_idx, max_frame)   # ← add this
            feat = self.features[ep_idx][cam][frame_idx_clamped]
            cam_features.append(feat)

        image_features = np.concatenate(cam_features, axis=-1)  # (1024,)
        image_features = torch.from_numpy(image_features.astype(np.float32))
        
        # -- Proprioception history window ---------------------------------
        # Clamp at episode start so we never reach into a previous episode
        history_indices = [max(ep_start, idx - i) for i in range(T - 1, -1, -1)]

        proprio_history = []
        for h_idx in history_indices:
            step = np.concatenate([
                self.joints[h_idx],
                self.velocities[h_idx],
                self.ee_pose[h_idx],
            ]).astype(np.float32)
            proprio_history.append(step)

        # Flatten history: (T * PROPRIO_DIM,)
        proprio = torch.from_numpy(
            np.concatenate(proprio_history)
        )
        # Normalise using tiled mean/std to match the flattened shape
        proprio_mean_tiled = np.tile(self.proprio_mean, T)
        proprio_std_tiled  = np.tile(self.proprio_std, T)
        proprio = (proprio - torch.from_numpy(proprio_mean_tiled)) / \
                torch.from_numpy(proprio_std_tiled)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image_features, proprio, label


# -----------------------------------------------------------------------------
# TRAINING UTILITIES
# -----------------------------------------------------------------------------

def compute_pos_weight(dataset):
    """
    pos_weight = n_open / n_closed
    Passed to BCEWithLogitsLoss to penalise mistakes on the minority class more.
    """
    n_pos  = dataset.labels.sum()
    n_neg  = len(dataset.labels) - n_pos
    weight = n_neg / (n_pos + 1e-8)
    print(f"  pos_weight (open/closed ratio): {weight:.3f}")
    return torch.tensor([weight], dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train=True, epoch=0, total_epochs=0):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    mode = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [{mode}]", leave=False)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, proprios, labels in pbar:
            images   = images.to(device)
            proprios = proprios.to(device)
            labels   = labels.to(device).reshape(-1, 1)

            logits = model(images, proprios)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds   = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += len(labels)

            pbar.set_postfix({
                "loss": f"{total_loss / total:.4f}",
                "acc" : f"{correct / total:.3f}",
            })

    return total_loss / total, correct / total


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def train():
    print(f"\nUsing device: {DEVICE}")

    # -- Dataset -----------------------------------------------------------
    print("\nBuilding dataset...")
    full_dataset = ZarrVideoGripperDataset(
        zarr_path=ZARR_PATH,
        features_dir=FEATURES_DIR,
        cameras=CAMERAS,
        gripper_threshold=GRIPPER_THRESHOLD,
    )

    n_val   = math.floor(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {n_train} samples  |  Val: {n_val} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # -- Model -------------------------------------------------------------
    print(f"\nBuilding model with {T} history timesteps...")
    model = GripperClassifier(
        proprio_dim = PROPRIO_DIM,   
        num_cams    = len(CAMERAS),
        T           = T       
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    # -- Loss --------------------------------------------------------------
    print("\nComputing class weights...")
    pos_weight = compute_pos_weight(full_dataset).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -- Optimiser ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    # Halve the learning rate if val loss stops improving for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, verbose=True
    )

    # -- Training loop -----------------------------------------------------
    print(f"\nStarting training for {EPOCHS} epochs...\n")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            train=True, epoch=epoch, total_epochs=EPOCHS
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, DEVICE,
            train=False, epoch=epoch, total_epochs=EPOCHS
        )

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{EPOCHS}  |  "
              f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f}  |  "
              f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}")

        # Save checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "proprio_mean": full_dataset.proprio_mean,
                "proprio_std" : full_dataset.proprio_std,
                "T"           : T,
                "proprio_dim" : PROPRIO_DIM,
                "cameras"     : CAMERAS,
                "num_cams"    : len(CAMERAS),
                "val_loss"    : val_loss,
                "val_acc"     : val_acc,
            }, SAVE_PATH)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()