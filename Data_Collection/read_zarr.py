# Usage: python read_zarr.py <episode_index>
import zarr
import numpy as np
import sys

ep_idx = int(sys.argv[1]) 
# num_values = int(sys.argv[2]) if len(sys.argv) > 2 else 10  # optional (default=10)

data_folder = "lamp_50" # <---------- CHOOSE DATASET HERE

# --- Open dataset ---
z = zarr.open(f"Collected_Data/{data_folder}/replay_buffer.zarr", mode="r")

data_root = z["data"]
keys = list(data_root.array_keys())

# --- Load episode boundaries ---
episode_ends = z["meta"]["episode_ends"][:]
print(f"Episode Ends: {episode_ends}")
starts = np.concatenate([[0], episode_ends[:-1] + 1])
episodes = list(zip(starts, episode_ends))

# --- Safety checks ---
if ep_idx < 0 or ep_idx >= len(episodes):
    raise ValueError(f"Episode index {ep_idx} out of range (0–{len(episodes)-1})")

s, e = episodes[ep_idx]
length = e - s + 1
print(f"\n🟢 Episode {ep_idx} — indices [{s}:{e}] — length {length} timesteps\n")

# --- Print values for each key ---
for key in keys:
    arr = data_root[key]
    subset = arr[s:e]
    print(f"🔸 {key}: shape={arr.shape}")
    print(subset)
    print()

