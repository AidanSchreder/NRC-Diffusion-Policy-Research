"""
Usage:
python inference.py -i /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy/Inference/Checkpoints/FILENAME -o /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy/Inference/Logs/LOGNAME
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from Training.real_world.real_env import RealEnv
from Training.common.precise_sleep import precise_wait
from Training.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from Training.common.pytorch_util import dict_apply
from Training.workspace.base_workspace import BaseWorkspace
from Training.policy.base_image_policy import BaseImagePolicy
from Training.common.cv2_util import get_image_transform
from Data_Collection.Haptic_Device_Setup.haply_barebones import HapticVisualizer
from Training.common.precise_sleep import precise_wait
from Training.common import pose_repr_util as utils
from Training.common.pose_repr_util import relative_to_absolute_action, absolute_to_relative_action
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Starting Robot Position
j_init = np.array([256.97,-61.94,133.80,109.75,-76.24,-4.66]) / 180 * np.pi
# j_init = np.array([262.85,-102.50,128.16,246.68,-89.5,-10.81]) / 180 * np.pi

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default="", help='Directory to save recording')
@click.option('--robot_ip', '-ri', default="192.168.0.110", help="UR10e's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=12, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    action_offset = 0

    delta_action = False
    relative_action = True

    # diffusion model
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device('cuda')
    policy.eval().to(device)

    # set inference params
    policy.num_inference_steps = 24 # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    # policy.n_action_steps = 6

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager, \
        RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            j_init=j_init,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager
        ) as env:
            
        viz = HapticVisualizer(display_plot=False)
        cv2.setNumThreads(1)
        # Should be the same as demo
        # realsense exposure
        env.realsense.set_exposure(exposure=80, gain=0)
        # realsense white balance
        env.realsense.set_white_balance(white_balance=5900)

        print("Waiting for realsense")
        time.sleep(1.0)

        print("Warming up policy inference")
        obs = env.get_obs()
        base_pose = obs['robot_eef_pose'][-1]

        obs_rel = dict(obs)
        obs_rel['robot_eef_pose'] = absolute_to_relative_action(obs['robot_eef_pose'], base_pose) # Base Pose = Start Pose during warmup (before episode begins)

        print(f"Env.get_obs(): {obs.keys()}")
        with torch.no_grad():
            policy.reset()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs_rel, shape_meta=cfg.task.shape_meta)
            print(f"obs_dict_np keys: {obs_dict_np.keys()}")
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
            result = policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
            #print(f"Predicted Action: {action}")
            assert action.shape[-1] == 7
            del result

        print('Ready for Inference!')
        while True:
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            # ========== policy control loop ==============
            try:
                # start episode
                policy.reset()
                start_delay = 1.0
                eval_t_start = time.time() + start_delay
                t_start = time.monotonic() + start_delay
                env.start_episode(eval_t_start)
                # wait for 1/30 sec to get the closest frame actually
                # reduces overall latency
                frame_latency = 1/30
                precise_wait(eval_t_start - frame_latency, time_func=time.time)
                print("Started!")
                iter_idx = 0
                term_area_start_timestamp = float('inf')
                perv_target_pose = None
                start_pose = None

                actions = []
                prev_pred = None

                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                    # get obs
                    # print('get_obs')
                    obs = env.get_obs()
                    base_pose = obs['robot_eef_pose'][-1]
                    print(f"Base Pose: {base_pose}")

                    if start_pose is None:
                        start_pose = base_pose
                        print(f"Start Pose: {start_pose}")

                    obs['start_pose'][:] = start_pose

                    obs_rel = dict(obs)
                    # obs_rel['robot_eef_pose'] = absolute_to_relative_action(obs['robot_eef_pose'], start_pose)     

                    rel_window = []
                    for j in range(len(obs['robot_eef_pose'])):
                        rel = absolute_to_relative_action(obs['robot_eef_pose'][j], start_pose)
                        rel_window.append(rel)
                    obs_rel['robot_eef_pose'] = np.stack(rel_window, axis=0)

                    obs_timestamps = obs['timestamp']

                    # run inference
                    with torch.no_grad():
                        s = time.time()
                        obs_dict_np = get_real_obs_dict(
                            env_obs=obs_rel, shape_meta=cfg.task.shape_meta)
                        obs_dict = dict_apply(obs_dict_np, 
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                        result = policy.predict_action(obs_dict)
                        # this action starts from the first obs step
                        action = result['action'][0].detach().to('cpu').numpy()
                        #print(f"ACTION: {result}")
                        #print('Inference latency:', time.time() - s)
                    
                    # convert policy action to env actions
                    if delta_action:
                        # Convert Delta to Absolute Action
                        for i in range(len(action)):
                            action[i, :3] += start_pose[:3]
                            for j in range(0, i):
                                action[i, :3] += action[j, :3]
                            action[i, 3:6] = start_pose[3:6]
                        # Regular Absolute Action
                        if len(target_pose) < 7:
                            target_pose = np.append(target_pose, 0)
                        this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        this_target_poses[:] = target_pose
                        this_target_poses[:,:7] = action

                    elif relative_action:
                        # Convert Relative to Absolute Action
                        print(f"REL ACTION: {action[0]}")

                        for i in range(len(action)):
                            action[i, :6] = relative_to_absolute_action(action[i, :6], start_pose)
                            action[i, 2] *= 1.5
                            action[i, 2] -= 0.08
                            action[i, 3:6] = base_pose[3:6]

                            # dx = action[i, 0]
                            # dy = action[i, 1]
                            # dz = action[i, 2]

                            # max_comp = max(abs(dx), abs(dy), abs(dz))
                            # threshold = 0.1

                            # if max_comp > threshold:
                            #     scale = threshold/max_comp
                            #     action[i, :3] *= scale

                            # action[i, 0] += start_pose[0]
                            # action[i, 1] += start_pose[1]
                            # action[i, 2] += start_pose[2]

                        print(f"ABS ACTION: {action[0]}")

                        # Regular Absolute Action
                        if len(target_pose) < 7:
                            target_pose = np.append(target_pose, 0)
                        this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        this_target_poses[:] = target_pose
                        this_target_poses[:,:7] = action
                        #print(f"ACTION: {this_target_poses[0]}")
                        # print(f"Base Pose: {base_pose}")
                        # print(f"Start Pose: {start_pose} \n")
                        actions.append(this_target_poses)

                    else:
                        if len(target_pose) < 7:
                            target_pose = np.append(target_pose, 0)
                        this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        this_target_poses[:] = target_pose
                        this_target_poses[:,:7] = action
                        this_target_poses[:, 3:6] = start_pose[3:6]

                    # deal with timing
                    # the same step actions are always the target for
                    action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset) * dt + obs_timestamps[-1]
                    action_exec_latency = 0.01
                    curr_time = time.time()
                    is_new = action_timestamps > (curr_time + action_exec_latency)
                    if np.sum(is_new) == 0:
                        # exceeded time budget, still do something
                        this_target_poses = this_target_poses[[-1]]
                        # schedule on next available step
                        next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        action_timestamp = eval_t_start + (next_step_idx) * dt
                        # print('Over budget', action_timestamp - curr_time)
                        action_timestamps = np.array([action_timestamp])
                    else:
                        this_target_poses = this_target_poses[is_new]
                        action_timestamps = action_timestamps[is_new]

                    # clip actions - CHANGES TO 3D
                    # this_target_poses[:,:2] = np.clip(
                    #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                            #gripper_action = int(this_target_poses[-1][-1])
                            #(f"GRIPPER: {gripper_action}")

                    # execute actions
                    env.exec_actions(
                        actions=this_target_poses,
                        timestamps=action_timestamps
                    )

                        # gripper_current_state = env.get_gripper_state()

                        # if gripper_action != gripper_current_state:
                        #     env.send_gripper_command(int(gripper_action * 255))

                    #print(f"Submitted {len(this_target_poses)} steps of actions.")

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    text = 'Episode: {}, Time: {:.1f}'.format(
                        episode_id, time.monotonic() - t_start
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])


                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('s'):
                        # Stop episode
                        # Hand control back to human
                        env.end_episode()
                        print('Stopped.')
                        break

                    # auto termination
                    terminate = False
                    if time.monotonic() - t_start > max_duration:
                        terminate = True
                        print('Terminated by the timeout!')

                    term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                    curr_pose = obs['robot_eef_pose'][-1]
                    dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                    if dist < 0.03:
                        # in termination area
                        curr_timestamp = obs['timestamp'][-1]
                        if term_area_start_timestamp > curr_timestamp:
                            term_area_start_timestamp = curr_timestamp
                        else:
                            term_area_time = curr_timestamp - term_area_start_timestamp
                            if term_area_time > 0.5:
                                terminate = True
                                print('Terminated by the policy!')
                    else:
                        # out of the area
                        term_area_start_timestamp = float('inf')

                    if terminate:
                        env.end_episode()
                        return

                    # wait for execution
                    precise_wait(t_cycle_end - frame_latency)
                    iter_idx += steps_per_inference

            except KeyboardInterrupt:
                print("Interrupted!")
                # stop robot.
                env.end_episode()
                break
            
            print("Stopped.")

# %%
if __name__ == '__main__':
    main()
