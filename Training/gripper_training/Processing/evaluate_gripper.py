"""
evaluate_gripper_classifier.py

Offline evaluation of the trained gripper classifier on a collected dataset.
Loads the same Zarr + pre-computed features used for training and runs the
classifier on every timestep, outputting the predicted probability alongside
the ground truth gripper state.

Usage:
    python evaluate_gripper_classifier.py

Output:
    - Per-timestep CSV: episode, timestep, prob, predicted, ground_truth
    - Summary metrics: accuracy, confusion matrix, per-episode accuracy
    - Plots: probability vs ground truth for each episode
"""

import os
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from Training.gripper_training.Model.gripper_classifier import GripperClassifier

# -----------------------------------------------------------------------------
# CONFIG — match these to your train_gripper_classifier.py settings
# -----------------------------------------------------------------------------
data_folder = "wiping_diverse_100"
output_folder = "eval_gripper_training_4"
ckpt_file = "gripper_training_4"

ZARR_PATH       = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/replay_buffer.zarr"
FEATURES_DIR    = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/{data_folder}/features"
CKPT_PATH       = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/gripper_training/Checkpoints/{ckpt_file}.ckpt"
OUTPUT_DIR      = f"/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/gripper_training/Evaluations/{output_folder}"

# Zarr keys
JOINTS_KEY      = "data/robot_joint"
VELOCITIES_KEY  = "data/robot_joint_vel"
EE_POSE_KEY     = "data/robot_eef_pose"
GRIPPER_KEY     = "data/robot_gripper_state"
EPISODE_KEY     = "meta/episode_ends"

GRIPPER_THRESHOLD = 0.5
VIDEO_TO_ROBOT_RATIO = 1.5        # 15fps / 10Hz

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------------------

def load_model(ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = GripperClassifier(
        proprio_dim=ckpt["proprio_dim"],
        num_cams=ckpt["num_cams"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}, val_acc={ckpt['val_acc']:.3f})")

    return model, ckpt["proprio_mean"], ckpt["proprio_std"], ckpt["cameras"]


# -----------------------------------------------------------------------------
# EVALUATE
# -----------------------------------------------------------------------------

def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- Load model --------------------------------------------------------
    model, proprio_mean, proprio_std, cameras = load_model(CKPT_PATH)

    # -- Load Zarr data ----------------------------------------------------
    print("\nLoading dataset...")
    store        = zarr.open(ZARR_PATH, mode="r")
    joints       = store[JOINTS_KEY][:]         # (N, 6)
    velocities   = store[VELOCITIES_KEY][:]     # (N, 6)
    ee_pose      = store[EE_POSE_KEY][:]        # (N, 6)
    gripper_raw = store[GRIPPER_KEY][:].squeeze(-1)       # (N,)
    episode_ends = store[EPISODE_KEY][:]        # (num_episodes,)

    labels       = (gripper_raw > GRIPPER_THRESHOLD).astype(np.float32)
    n_samples    = len(labels)
    n_episodes   = len(episode_ends)

    print(f"  Total timesteps : {n_samples}")
    print(f"  Total episodes  : {n_episodes}")

    # -- Build episode boundaries ------------------------------------------
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    # -- Load R3M features -------------------------------------------------
    print("Loading pre-computed R3M features...")
    features = {}
    for ep_name in sorted(os.listdir(FEATURES_DIR), key=lambda x: int(x)):
        ep_idx = int(ep_name)
        features[ep_idx] = {}
        for cam in cameras:
            path = os.path.join(FEATURES_DIR, ep_name, f"{cam}.npy")
            features[ep_idx][cam] = np.load(path)   # (num_frames, 512)

    # -- Run inference on every timestep -----------------------------------
    print("\nRunning classifier on all timesteps...")
    all_probs      = np.zeros(n_samples, dtype=np.float32)
    all_preds      = np.zeros(n_samples, dtype=np.float32)
    all_labels     = labels
    all_episodes   = np.zeros(n_samples, dtype=np.int32)
    all_local_steps = np.zeros(n_samples, dtype=np.int32)

    with torch.no_grad():
        for ep_idx, (start, end) in enumerate(zip(episode_starts, episode_ends)):
            start, end = int(start), int(end)
            ep_len     = end - start

            for local_step in range(ep_len):
                global_idx = start + local_step
                """
                frame_idx  = min(
                    round(local_step * VIDEO_TO_ROBOT_RATIO),
                    len(features[ep_idx][cameras[0]]) - 1
                )
                """
                # Build image features
                frame_idx = round(local_step * VIDEO_TO_ROBOT_RATIO)
                cam_feats = []
                
                for cam in cameras:
                    max_frame = len(features[ep_idx][cam]) - 1
                    frame_idx_clamped = min(frame_idx, max_frame)
                    feat = features[ep_idx][cam][frame_idx_clamped]
                    cam_feats.append(feat)
                image_features = np.concatenate(cam_feats).astype(np.float32)
                image_t = torch.from_numpy(image_features).unsqueeze(0).to(DEVICE)

                # Build normalised proprioception
                proprio_raw = np.concatenate([
                    joints[global_idx],
                    velocities[global_idx],
                    ee_pose[global_idx],
                ]).astype(np.float32)
                proprio_norm = (proprio_raw - proprio_mean) / proprio_std
                proprio_t    = torch.from_numpy(proprio_norm).unsqueeze(0).to(DEVICE)

                # Forward pass
                logit = model(image_t, proprio_t)
                prob  = torch.sigmoid(logit).item()

                all_probs[global_idx]       = prob
                all_preds[global_idx]       = 1.0 if prob > GRIPPER_THRESHOLD else 0.0
                all_episodes[global_idx]    = ep_idx
                all_local_steps[global_idx] = local_step

            if (ep_idx + 1) % 10 == 0 or ep_idx == n_episodes - 1:
                print(f"  Processed {ep_idx + 1}/{n_episodes} episodes")

    # -- Summary metrics ---------------------------------------------------
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    correct  = (all_preds == all_labels).sum()
    accuracy = correct / n_samples
    print(f"Overall accuracy : {accuracy:.4f} ({correct}/{n_samples})")

    # Confusion matrix
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    print(f"\nConfusion Matrix:")
    print(f"  True closed, pred closed (TP) : {tp}")
    print(f"  True open,   pred open   (TN) : {tn}")
    print(f"  True open,   pred closed (FP) : {fp}  ← false grip")
    print(f"  True closed, pred open   (FN) : {fn}  ← missed grip")

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Per-episode accuracy
    print(f"\nPer-episode accuracy:")
    ep_accuracies = []
    for ep_idx in range(n_episodes):
        mask     = all_episodes == ep_idx
        ep_acc   = (all_preds[mask] == all_labels[mask]).mean()
        ep_accuracies.append(ep_acc)
        print(f"  Episode {ep_idx:03d} : {ep_acc:.4f}  "
              f"({mask.sum()} timesteps)")

    # -- Save CSV ----------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    print(f"\nSaving per-timestep predictions to {csv_path}...")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "local_step", "global_step",
                         "prob", "predicted", "ground_truth", "correct"])
        for i in range(n_samples):
            writer.writerow([
                all_episodes[i],
                all_local_steps[i],
                i,
                f"{all_probs[i]:.4f}",
                int(all_preds[i]),
                int(all_labels[i]),
                int(all_preds[i] == all_labels[i]),
            ])

    # -- Plot per-episode probability vs ground truth ----------------------
    print("Generating plots...")
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for ep_idx in range(n_episodes):
        mask       = all_episodes == ep_idx
        probs_ep   = all_probs[mask]
        labels_ep  = all_labels[mask]
        timesteps  = np.arange(len(probs_ep))

        fig, ax = plt.subplots(figsize=(12, 4))

        # Ground truth as shaded background
        ax.fill_between(timesteps, 0, 1,
                        where=labels_ep == 1,
                        alpha=0.15, color="green", label="GT closed")
        ax.fill_between(timesteps, 0, 1,
                        where=labels_ep == 0,
                        alpha=0.15, color="red", label="GT open")

        # Classifier probability
        ax.plot(timesteps, probs_ep, color="blue", linewidth=1.5,
                label="Classifier prob")

        # Threshold line
        ax.axhline(GRIPPER_THRESHOLD, color="gray", linestyle="--",
                   linewidth=1, label=f"Threshold ({GRIPPER_THRESHOLD})")

        ax.set_xlim(0, len(timesteps) - 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("P(gripper closed)")
        ax.set_title(f"Episode {ep_idx}  |  "
                     f"Accuracy: {ep_accuracies[ep_idx]:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()

        plot_path = os.path.join(plots_dir, f"episode_{ep_idx:03d}.png")
        plt.savefig(plot_path, dpi=100)
        plt.close()

    print(f"\nPlots saved to {plots_dir}")
    print(f"CSV saved to   {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    evaluate()