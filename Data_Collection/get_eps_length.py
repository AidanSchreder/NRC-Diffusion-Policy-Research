import zarr
import numpy as np

store = zarr.open('/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/wiping_diverse_100/replay_buffer.zarr', mode='r')
episode_ends = store['meta/episode_ends'][:]

episode_lengths = np.diff(np.concatenate([[0], episode_ends]))

print(f"Mean:   {np.mean(episode_lengths):.1f} steps")
print(f"Median: {np.median(episode_lengths):.1f} steps")
print(f"Min:    {np.min(episode_lengths)} steps")
print(f"Max:    {np.max(episode_lengths)} steps")