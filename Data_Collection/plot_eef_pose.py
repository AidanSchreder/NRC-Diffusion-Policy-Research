import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === CONFIG ===
zarr_path = "/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/press_2/replay_buffer.zarr"
skip_frames = 2
save_dir = "visualization_outputs"
os.makedirs(save_dir, exist_ok=True)

POSE_LABELS   = ["x",  "y",  "z",  "rx",  "ry",  "rz"]
POSE_UNITS    = ["m",  "m",  "m",  "rad", "rad", "rad"]
POSE_CHANNELS = list(range(6))

# === LOAD ZARR ===
store = zarr.open(zarr_path, mode="r")
data  = store["data"]
meta  = store["meta"]

robot_eef_pose_all = data["robot_eef_pose"][:]   # (N_total, 6)
timestamp_all      = data["timestamp"][:]         # (N_total,)
episode_ends       = meta["episode_ends"][:]      # (n_episodes,)

episode_starts = np.concatenate([[0], episode_ends[:-1]])
n_episodes = len(episode_ends)
print(f"Loaded {n_episodes} episodes, {len(robot_eef_pose_all)} total steps.")

# === SPLIT INTO EPISODES ===
episodes = []
for ep_idx in range(n_episodes):
    start = int(episode_starts[ep_idx]) + skip_frames
    end   = int(episode_ends[ep_idx])
    pose  = robot_eef_pose_all[start:end]
    ts    = timestamp_all[start:end]
    if len(pose) == 0:
        print(f"Skipping episode {ep_idx} - empty after skip_frames.")
        continue
    ts = ts - ts[0]
    episodes.append((ts, pose))

n_valid = len(episodes)
print(f"{n_valid} valid episodes.")

colors = cm.tab20(np.linspace(0, 1, n_valid))

# === INDIVIDUAL CHANNEL PLOTS ===
for ch in range(6):
    fig, ax = plt.subplots(figsize=(13, 4))

    for ep_idx, (ts, pose) in enumerate(episodes):
        ax.plot(
            ts,
            pose[:, ch],
            lw=0.8,
            alpha=0.75,
            color=colors[ep_idx],
            label=f"ep {ep_idx}",
        )

    ax.set_title(f"{POSE_LABELS[ch]} — all episodes", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(f"{POSE_LABELS[ch]} ({POSE_UNITS[ch]})", fontsize=11)
    ax.grid(True, alpha=0.3)

    if n_valid <= 20:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            fontsize=7,
            ncol=1,
        )

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"eef_{POSE_LABELS[ch]}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #print(f"Saved: {out_path}")
    #plt.show()

# === COMBINED OVERVIEW ===
fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=False)
fig.suptitle("Robot EEF Pose — all channels, all episodes", fontsize=14)

for ch in range(6):
    ax = axes[ch]
    for ep_idx, (ts, pose) in enumerate(episodes):
        ax.plot(ts, pose[:, ch], lw=0.7, alpha=0.7, color=colors[ep_idx])
    ax.set_ylabel(f"{POSE_LABELS[ch]}\n({POSE_UNITS[ch]})", fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(save_dir, "eef_all_channels_overview.png")
#plt.savefig(out_path, dpi=150, bbox_inches="tight")
#print(f"Saved: {out_path}")
plt.show()

# === 3D TRAJECTORY PLOT ===
fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection="3d")

for ep_idx, (ts, pose) in enumerate(episodes):
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
    ax.plot(x, y, z, lw=0.7, alpha=0.8, color=colors[ep_idx], label=f"ep {ep_idx}")
    ax.scatter(x[0],  y[0],  z[0],  c="lightgreen", s=10, alpha=0.9, zorder=5)
    ax.scatter(x[-1], y[-1], z[-1], c="red",        s=10, alpha=0.9, zorder=5)

ax.set_title("EEF TCP Trajectory — all episodes", fontsize=13)
ax.set_xlabel("X (m)", labelpad=8)
ax.set_ylabel("Y (m)", labelpad=8)
ax.set_zlabel("Z (m)", labelpad=8)
ax.view_init(elev=30, azim=125)
plt.tight_layout()
out_path = os.path.join(save_dir, "eef_trajectory_3d.png")
#plt.savefig(out_path, dpi=150, bbox_inches="tight")
#print(f"Saved: {out_path}")
plt.show()