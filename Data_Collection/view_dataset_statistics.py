import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === CONFIG ===
dataset_title = "Force-Torque Sensor Simulation"
zarr_path = "/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/press/replay_buffer.zarr"
skip_frames = 2
save_dir = "visualization_outputs"
os.makedirs(save_dir, exist_ok=True)

# === LOAD ZARR REPLAY BUFFER ===
store = zarr.open(zarr_path, mode="r")
data = store["data"]
meta = store["meta"]

# Required keys
robot_eef_pose_all = data["robot_eef_pose"][:]          # (N_total, 6)
robot_gripper_state_all = data["robot_gripper_state"][:] # (N_total, 1)
action_all = data["action"][:]                           # (N_total, 7)
timestamp_all = data["timestamp"][:]                     # (N_total,)
episode_ends = meta["episode_ends"][:]                   # (n_episodes,) cumulative end indices

# Optional FTS
fts_enabled = "fts_info" in data
fts_all = data["fts_info"][:] if fts_enabled else None  # (N_total, 6) or None

print(f"Loaded {len(episode_ends)} episodes, {len(robot_eef_pose_all)} total steps.")
if fts_enabled:
    print("FTS data present.")

# === SPLIT INTO EPISODES ===
episode_starts = np.concatenate([[0], episode_ends[:-1]])

def get_episode_slice(ep_idx):
    start = int(episode_starts[ep_idx]) + skip_frames
    end = int(episode_ends[ep_idx])
    return slice(start, end)

all_eef_poses   = []
all_gripper     = []
all_timestamps  = []
all_fts         = []

for ep_idx in range(len(episode_ends)):
    sl = get_episode_slice(ep_idx)
    eef = robot_eef_pose_all[sl]
    grip = robot_gripper_state_all[sl]
    ts = timestamp_all[sl]

    if len(eef) == 0:
        print(f"Skipping episode {ep_idx} - empty after skip_frames.")
        continue

    # Normalize timestamps to start at 0 per episode
    ts = ts - ts[0]

    all_eef_poses.append(eef)
    all_gripper.append(grip)
    all_timestamps.append(ts)

    if fts_enabled:
        all_fts.append(fts_all[sl])

n_valid = len(all_eef_poses)
print(f"{n_valid} valid episodes after filtering.")

# === CONCATENATE FOR PLOTTING ===
eef_concat      = np.concatenate(all_eef_poses, axis=0)      # (N, 6)
gripper_concat  = np.concatenate(all_gripper, axis=0)         # (N, 1)
ts_concat       = np.concatenate(all_timestamps, axis=0)      # (N,)

# Build combined state: [x,y,z,rx,ry,rz,gripper] = 7 dims
state_concat = np.concatenate([eef_concat, gripper_concat], axis=1)  # (N, 7)
STATE_LABELS = ["x (m)", "y (m)", "z (m)", "rx (rad)", "ry (rad)", "rz (rad)", "gripper"]

# === PLOT STATE VARIANCE ===
n_states = state_concat.shape[1]
fig, axes = plt.subplots(n_states, 1, figsize=(14, 3 * n_states), sharex=True)
fig.suptitle("Robot EEF State: Rolling Mean ± Std Across All Episodes", fontsize=15)

window = 30
for i in range(n_states):
    series = state_concat[:, i]
    mean = np.convolve(series, np.ones(window) / window, mode="valid")
    std = np.array([
        np.std(series[max(0, j - window):j]) for j in range(window, len(series) + 1)
    ])
    t = ts_concat[window - 1:]

    ax = axes[i]
    ax.plot(t, mean, lw=1.2, label=STATE_LABELS[i])
    ax.fill_between(t, mean - std, mean + std, alpha=0.25)
    ax.set_ylabel(STATE_LABELS[i], fontsize=9)
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(save_dir, "robot_state_variance_all.png"), dpi=150)
print("Saved: robot_state_variance_all.png")
plt.show()

# === PLOT TCP 3D TRAJECTORY ===
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

for ep_idx, eef in enumerate(all_eef_poses):
    tcp = eef[:, :3]  # x, y, z
    tcp_x, tcp_y, tcp_z = tcp.T

    if fts_enabled and ep_idx < len(all_fts):
        force = all_fts[ep_idx][:, :3]
        norms = np.linalg.norm(force, axis=1)
        norms_norm = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
        for i in range(1, len(tcp)):
            n = float(norms_norm[i - 1])
            ax.plot(tcp_x[i-1:i+1], tcp_y[i-1:i+1], tcp_z[i-1:i+1],
                    c=(n, 0, 1 - n), lw=0.6)
    else:
        ax.plot(tcp_x, tcp_y, tcp_z, lw=0.6)

    ax.scatter(tcp[0, 0],  tcp[0, 1],  tcp[0, 2],  c="lightgreen", s=8, alpha=0.9)
    ax.scatter(tcp[-1, 0], tcp[-1, 1], tcp[-1, 2], c="red",        s=8, alpha=0.9)

ax.set_title(dataset_title, fontsize=13)
ax.set_xlabel("X (m)", labelpad=8)
ax.set_ylabel("Y (m)", labelpad=8)
ax.set_zlabel("Z (m)", labelpad=8)
ax.view_init(elev=30, azim=125)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "tcp_trajectory_3d.png"), dpi=150)
print("Saved: tcp_trajectory_3d.png")
plt.show()

# === OPTIONAL: PLOT FTS IF PRESENT ===
if fts_enabled and all_fts:
    fts_concat = np.concatenate(all_fts, axis=0)   # (N, 6)
    FTS_LABELS = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Force-Torque Sensor: Rolling Mean ± Std Across All Episodes", fontsize=15)

    for i in range(6):
        series = fts_concat[:, i]
        mean = np.convolve(series, np.ones(window) / window, mode="valid")
        std = np.array([
            np.std(series[max(0, j - window):j]) for j in range(window, len(series) + 1)
        ])
        t = ts_concat[window - 1:]

        ax = axes[i]
        ax.plot(t, mean, lw=1.2, label=FTS_LABELS[i])
        ax.fill_between(t, mean - std, mean + std, alpha=0.25)
        ax.set_ylabel(FTS_LABELS[i], fontsize=9)
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "fts_variance_all.png"), dpi=150)
    print("Saved: fts_variance_all.png")
    plt.show()