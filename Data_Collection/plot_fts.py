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

FTS_LABELS = ["Fx (N)", "Fy (N)", "Fz (N)", "Tx (Nm)", "Ty (Nm)", "Tz (Nm)"]
FTS_SIGN = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

# === LOAD ZARR ===
store = zarr.open(zarr_path, mode="r")
data  = store["data"]
meta  = store["meta"]

if "fts_info" not in data:
    raise RuntimeError("No fts_info found in replay buffer.")

fts_all       = data["fts_info"][:]       # (N_total, 6)
timestamp_all = data["timestamp"][:]      # (N_total,)
episode_ends  = meta["episode_ends"][:]   # (n_episodes,)

episode_starts = np.concatenate([[0], episode_ends[:-1]])
n_episodes = len(episode_ends)
print(f"Loaded {n_episodes} episodes, {len(fts_all)} total steps.")

# === SPLIT INTO EPISODES ===
episodes = []
for ep_idx in range(n_episodes):
    start = int(episode_starts[ep_idx]) + skip_frames
    end   = int(episode_ends[ep_idx])
    fts   = fts_all[start:end]
    ts    = timestamp_all[start:end]
    if len(fts) == 0:
        print(f"Skipping episode {ep_idx} - empty after skip_frames.")
        continue
    ts = ts - ts[0]   # normalize to start at 0
    fts = fts * FTS_SIGN   # <-- add this line
    episodes.append((ts, fts))

n_valid = len(episodes)
print(f"{n_valid} valid episodes.")



# === PLOT: one figure per FTS channel ===
channel_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
units         = ["N",  "N",  "N",  "Nm", "Nm", "Nm"]

for ch in range(6):
    fig, ax = plt.subplots(figsize=(13, 4))

    for ep_idx, (ts, fts) in enumerate(episodes):
        ax.plot(
            ts,
            fts[:, ch],
            lw=0.8,
            alpha=0.75,
            color = "blue",
        )

    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_title(f"{channel_names[ch]} — all episodes", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(f"{channel_names[ch]} ({units[ch]})", fontsize=11)
    ax.grid(True, alpha=0.3)

    # legend outside to the right — only show if not too many episodes
    if n_valid <= 20:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            fontsize=7,
            ncol=1,
        )

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"fts_{channel_names[ch]}.png")
    #plt.savefig(out_path, dpi=150, bbox_inches="tight")
    #print(f"Saved: {out_path}")
    plt.show()

# === COMBINED OVERVIEW: all 6 channels in one figure ===
fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=False)
fig.suptitle("FTS — all channels, all episodes", fontsize=14)

for ch in range(6):
    ax = axes[ch]
    for ep_idx, (ts, fts) in enumerate(episodes):
        ax.plot(ts, fts[:, ch], lw=0.7, alpha=0.7, color="blue")
    ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.4)
    ax.set_ylabel(f"{channel_names[ch]}\n({units[ch]})", fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(save_dir, "fts_all_channels_overview.png")
#plt.savefig(out_path, dpi=150, bbox_inches="tight")
#print(f"Saved: {out_path}")
plt.show()