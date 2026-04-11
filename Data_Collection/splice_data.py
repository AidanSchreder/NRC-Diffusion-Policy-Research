import os #for file/directory management
import shutil  #builds ontop of os for managing files
import random
import numpy
import zarr

SOURCE_DIR = r'/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/wiping_diverse_100'
DEST_DIR = r'/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/wiping_diverse_100_spliced'
NUM_DEMOS = 74

def get_demo_indices(source_dir):
    video_dir = os.path.join(source_dir, 'videos') #./Data_Collection/Collected_Data/lamp_50/videos
    indices = [int(name) for name in os.listdir(video_dir) if name.isdigit()] #appends all the episode numbers (eg. 0-49) in an array
    return sorted(indices)

def copy_videos(selected_indices, source_dir, dest_dir):
    for item in os.listdir(source_dir):
        src_item_dir = os.path.join(source_dir, item)
        dst_item_dir = os.path.join(dest_dir, item)
        if os.path.isfile(src_item_dir):
            shutil.copy2(src_item_dir, dst_item_dir)
    src_video_dir = os.path.join(source_dir, 'videos')
    dst_video_dir = os.path.join(dest_dir, 'videos')
    os.makedirs(dst_video_dir, exist_ok=True)
    
    for idx in selected_indices:
            shutil.copytree(os.path.join(src_video_dir, str(idx)), os.path.join(dst_video_dir, str(idx))) #copies entire directory tree from a source path to a destination path

def copy_zarr_data(selected_indices, source_dir, dest_dir):
    src_zarr_dir = os.path.join(source_dir, 'replay_buffer.zarr')
    dst_zarr_dir = os.path.join(dest_dir, 'replay_buffer.zarr')
    os.makedirs(dst_zarr_dir, exist_ok=True)
    for item in os.listdir(src_zarr_dir):
        src_item_dir = os.path.join(src_zarr_dir, item)
        dst_item_dir = os.path.join(dst_zarr_dir, item)
        if os.path.isfile(src_item_dir):
            shutil.copy2(src_item_dir, dst_item_dir)

    #copies EVERYTHING in meta folder and z.group file
    for item in ['z.group', 'meta']:
        src = os.path.join(src_zarr_dir, item)
        dst = os.path.join(dst_zarr_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)

    #copy SELECTED demo data
    src_data_dir = os.path.join(src_zarr_dir, 'data')
    dst_data_dir = os.path.join(dst_zarr_dir, 'data')
    os.makedirs(dst_data_dir, exist_ok=True)
    for group in os.listdir(src_data_dir):
        src_group_dir = os.path.join(src_data_dir, group)
        dst_group_dir = os.path.join(dst_data_dir, group)
        if os.path.isfile(src_group_dir):
            shutil.copy2(src_group_dir, dst_group_dir)
        elif os.path.isdir(src_group_dir):
            os.makedirs(dst_group_dir, exist_ok=True)
            #copy .zarray file
            if os.path.isfile(os.path.join(src_group_dir, '.zarray')):
                shutil.copy2(os.path.join(src_group_dir, '.zarray'), os.path.join(dst_group_dir, '.zarray'))
            # copy selected demo files
            for idx in selected_indices:
                fname = f"{idx}.0"
                src_file = os.path.join(src_group_dir, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, os.path.join(dst_group_dir, fname))

def rename(dest_dir):
    video_dir = os.path.join(dest_dir, 'videos')
    zarr_data_dir = os.path.join(dest_dir, 'replay_buffer.zarr', 'data')

    # --- RENAME VIDEO FOLDERS ---
    video_folders = sorted(
        [d for d in os.listdir(video_dir) if d.isdigit()],
        key=lambda x: int(x)
    )

    # First pass: rename to temp to avoid overwrite
    for i, old_name in enumerate(video_folders):
        old_path = os.path.join(video_dir, old_name)
        temp_path = os.path.join(video_dir, f"temp_{i}")
        os.rename(old_path, temp_path)
        print(f"original index: {old_name}, new index: {i}")

    # Second pass: rename temp -> 0..N-1
    temp_folders = sorted(
        [d for d in os.listdir(video_dir) if d.startswith("temp_")],
        key=lambda x: int(x.split("_")[1])
    )

    for i, temp_name in enumerate(temp_folders):
        old_path = os.path.join(video_dir, temp_name)
        new_path = os.path.join(video_dir, str(i))
        os.rename(old_path, new_path)
    

        # --- RENAME ZARR DATA FILES ---
    for group in os.listdir(zarr_data_dir):
        group_dir = os.path.join(zarr_data_dir, group)

        if not os.path.isdir(group_dir):
            continue

        demo_files = sorted(
            [f for f in os.listdir(group_dir) if f.endswith(".0")],
            key=lambda x: int(x.split(".")[0])
        )

        # temp rename
        for i, old_file in enumerate(demo_files):
            old_path = os.path.join(group_dir, old_file)
            temp_path = os.path.join(group_dir, f"temp_{i}.0")
            os.rename(old_path, temp_path)

        # final rename
        temp_files = sorted(
            [f for f in os.listdir(group_dir) if f.startswith("temp_")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        for i, temp_file in enumerate(temp_files):
            old_path = os.path.join(group_dir, temp_file)
            new_path = os.path.join(group_dir, f"{i}.0")
            os.rename(old_path, new_path)


    print("Renaming complete: 0 →", len(video_folders)-1)



def main():
    indices = get_demo_indices(SOURCE_DIR)
    selected = random.sample(indices, NUM_DEMOS)
    os.makedirs(DEST_DIR, exist_ok=True)
    copy_videos(selected, SOURCE_DIR, DEST_DIR)
    copy_zarr_data(selected, SOURCE_DIR, DEST_DIR)
    rename(DEST_DIR)

    #print(f"Spliced dataset created at {DEST_DIR} with {NUM_DEMOS} demos.")

if __name__ == "__main__":
    main()


"""
02/20/26: WIPING_DIVERSE_100_SPLICED (25 DEMOS)
original index: 2, new index: 0
original index: 4, new index: 1
original index: 8, new index: 2
original index: 9, new index: 3
original index: 10, new index: 4
original index: 14, new index: 5
original index: 15, new index: 6
original index: 16, new index: 7
original index: 25, new index: 8
original index: 32, new index: 9
original index: 40, new index: 10
original index: 43, new index: 11
original index: 53, new index: 12
original index: 58, new index: 13
original index: 64, new index: 14
original index: 65, new index: 15
original index: 74, new index: 16
original index: 78, new index: 17
original index: 83, new index: 18
original index: 91, new index: 19
original index: 94, new index: 20
original index: 95, new index: 21
original index: 96, new index: 22
original index: 97, new index: 23
original index: 99, new index: 24
Renaming complete: 0 → 24
"""