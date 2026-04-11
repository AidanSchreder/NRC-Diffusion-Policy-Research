import zarr
import numpy as np

def add_progress_to_zarr(dataset_path: str):
    
    store = zarr.open(dataset_path, mode='r+')  # open in read/write mode
    print(store['data/robot_eef_pose'].chunks)
    episode_ends = store['meta/episode_ends'][:]
    n_total = episode_ends[-1]
    
    progress = np.zeros((n_total, 1), dtype=np.float32)
    
    for i in range(len(episode_ends)):
        start = episode_ends[i-1] if i > 0 else 0
        end = episode_ends[i]
        T = end - start
        
        progress[start:end, 0] = np.round(np.arange(T) / (T - 1), decimals=3)
    
    # Write directly to data/progress
    if 'data/progress' in store:
        print("Progress key already exists, overwriting...")
        del store['data/progress']
    
    store.array(
        'data/progress',
        data=progress,
        chunks=(387, 1),
        dtype=np.float32
    )
    
    print(f"Added progress to {dataset_path}")
    print(f"Shape: {store['data/progress'].shape}")
    
    # Sanity check
    for i in range(min(3, len(episode_ends))):
        start = episode_ends[i-1] if i > 0 else 0
        end = episode_ends[i]
        ep_progress = store['data/progress'][start:end, 0]
        print(f"Episode {i}: start={ep_progress[0]:.3f}, end={ep_progress[-1]:.3f}, length={end-start}")

if __name__ == "__main__":
    dataset_path = "/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/wiping_diverse_100/replay_buffer.zarr"
    add_progress_to_zarr(dataset_path)
    