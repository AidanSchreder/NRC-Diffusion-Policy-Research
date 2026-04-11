"""
Usage:
python inference_gripper.py -i /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Inference/Checkpoints/obs_epoch=0065-val_loss=0.015.ckpt -o /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Inference/Logs/wiping_gripper_obs_increase_2

"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
from collections import deque
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as T_transforms

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
from Training.common import pose_repr_util as utils
from Training.common.pose_repr_util import relative_to_absolute_action, absolute_to_relative_action
from Training.gripper_training.Model.gripper_classifier import GripperClassifier
from r3m import load_r3m

# Starting Robot Position
j_init = np.array([282.44,-73.93,86.34,258.27,-89.28,8.64]) / 180 * np.pi

# Classifier Config
CLASSIFIER_CKPT   = "/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/gripper_training/Checkpoints/gripper_training_3.ckpt"
# /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/gripper_training/Checkpoints/gripper_training_3.ckpt
GRIPPER_THRESHOLD = 0.03    # same threshold that worked during teleoperation testing
CONFIRM_STEPS     = 3
TEMPORAL_WINDOW   = 5
CAMERAS           = [0, 1]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda')


# -----------------------------------------------------------------------------
# GRIPPER CONTROLLER
# -----------------------------------------------------------------------------

class GripperController:
    """
    Wraps the trained GripperClassifier with:
      - Live R3M feature extraction per camera
      - Proprioception history buffer (temporal context)
      - Hysteresis to prevent rapid state changes
    """

    def __init__(self, model, proprio_mean, proprio_std, cameras, threshold=0.5, 
                 #T=5, 
                 confirm_steps=3):
        self.model         = model
        self.proprio_mean  = proprio_mean
        self.proprio_std   = proprio_std
        self.cameras       = cameras
        #self.T             = T
        self.threshold     = threshold
        self.confirm_steps = confirm_steps

        self.current_state = 0
        self.pending_state = None
        self.confirm_count = 0

        # Rolling proprioception buffer
        #self.proprio_buffer = deque(maxlen=T)

        # R3M encoder (frozen)
        print("  Loading R3M encoder for inference...")
        self.encoder = load_r3m("resnet18").to(DEVICE)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.transform = T_transforms.Compose([
            T_transforms.Resize((224, 224)),
            T_transforms.ToTensor(),
            T_transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])

    def reset(self):
        """Call at the start of each episode to clear temporal state."""
        #self.proprio_buffer.clear()
        self.current_state = 0
        self.pending_state = None
        self.confirm_count = 0

    """
    def step(self, obs, proprio_raw):
        
        # Update rolling buffer
        self.proprio_buffer.append(proprio_raw.copy())
        while len(self.proprio_buffer) < self.T:
            self.proprio_buffer.appendleft(np.zeros_like(proprio_raw))
        
        # Build flattened history: (T * PROPRIO_DIM,)
        history      = np.concatenate(list(self.proprio_buffer))
        mean_tiled   = np.tile(self.proprio_mean, self.T)
        std_tiled    = np.tile(self.proprio_std,  self.T)
        std_tiled = np.maximum(std_tiled, 0.01)
        history_norm = (history - mean_tiled) / std_tiled
        proprio_t    = torch.from_numpy(
            history_norm.astype(np.float32)
        ).unsqueeze(0).to(DEVICE)                               # (1, T*PROPRIO_DIM)

        


        # Extract R3M features per camera
        cam_features = []
        for cam in self.cameras:
            frame_np = obs[f'camera_{cam}'][-1].astype(np.uint8)  # (H, W, C)
            frame_t  = self.transform(Image.fromarray(frame_np))
            frame_t  = frame_t.unsqueeze(0).to(DEVICE)            # (1, 3, 224, 224)
            with torch.no_grad():
                feat = self.encoder(frame_t).view(1, -1)           # (1, 512)
            cam_features.append(feat)
        image_features = torch.cat(cam_features, dim=-1)           # (1, 1024)


        # Forward pass
        with torch.no_grad():
            logit = self.model(image_features, proprio_t)
            prob  = torch.sigmoid(logit).item()

        # Hysteresis
        predicted_state = 1 if prob > self.threshold else 0

        if predicted_state == self.current_state:
            self.pending_state = None
            self.confirm_count = 0
        else:
            if predicted_state == self.pending_state:
                self.confirm_count += 1
            else:
                self.pending_state = predicted_state
                self.confirm_count = 1

            if self.confirm_count >= self.confirm_steps:
                self.current_state = self.pending_state
                self.confirm_count = 0
                print(f"  [Gripper] -> "
                      f"{'CLOSE' if self.current_state else 'OPEN'} "
                      f"(prob={prob:.3f})")

        return self.current_state, prob
    
    
    """
 
    
    def step(self, obs, proprio_raw):

        # Normalize current proprio directly (no history window)
        std_safe = np.maximum(self.proprio_std, 0.01)
        proprio_norm = (proprio_raw - self.proprio_mean) / std_safe
        proprio_t = torch.from_numpy(
            proprio_norm.astype(np.float32)
        ).unsqueeze(0).to(DEVICE)                                   # (1, PROPRIO_DIM)

        # Extract R3M features per camera
        cam_features = []
        for cam in self.cameras:
            frame_np = obs[f'camera_{cam}'][-1].astype(np.uint8)   # (H, W, C)
            frame_t  = self.transform(Image.fromarray(frame_np))
            frame_t  = frame_t.unsqueeze(0).to(DEVICE)             # (1, 3, 224, 224)
            with torch.no_grad():
                feat = self.encoder(frame_t).view(1, -1)            # (1, 512)
            cam_features.append(feat)
        image_features = torch.cat(cam_features, dim=-1)            # (1, 1024)

        # Forward pass
        with torch.no_grad():
            logit = self.model(image_features, proprio_t)
            prob  = torch.sigmoid(logit).item()

        # Hysteresis
        predicted_state = 1 if prob > self.threshold else 0

        if predicted_state == self.current_state:
            self.pending_state = None
            self.confirm_count = 0
        else:
            if predicted_state == self.pending_state:
                self.confirm_count += 1
            else:
                self.pending_state = predicted_state
                self.confirm_count = 1

            if self.confirm_count >= self.confirm_steps:
                self.current_state = self.pending_state
                self.confirm_count = 0
                print(f"  [Gripper] -> "
                    f"{'CLOSE' if self.current_state else 'OPEN'} "
                    f"(prob={prob:.3f})")

        return self.current_state, prob


# -----------------------------------------------------------------------------
# CLASSIFIER LOADING
# -----------------------------------------------------------------------------

def load_gripper_classifier(ckpt_path):
    print(f"Loading gripper classifier from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = GripperClassifier(
        proprio_dim = ckpt["proprio_dim"],
        num_cams    = ckpt["num_cams"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Loaded (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.3f})")

    return (
        model,
        ckpt["proprio_mean"],
        ckpt["proprio_std"],
        ckpt.get("cameras", CAMERAS),
        #ckpt.get("T", TEMPORAL_WINDOW),
    )


# -----------------------------------------------------------------------------
# LATENCY TRACKER (unchanged)
# -----------------------------------------------------------------------------

class LatencyTracker:
    def __init__(self, maxlen: int = 100):
        self._values = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        val = float(latency)
        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float:
        return self.max_latency

    def percentile(self, q: float) -> float:
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float:
        return self.percentile(0.95)


OmegaConf.register_new_resolver("eval", eval, replace=True)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', default="", help='Directory to save recording')
@click.option('--robot_ip', '-ri', default="192.168.0.110")
@click.option('--match_dataset', '-m', default=None)
@click.option('--match_episode', '-me', default=None, type=int)
@click.option('--vis_camera_idx', default=0, type=int)
@click.option('--init_joints', '-j', is_flag=True, default=True)
@click.option('--steps_per_inference', '-si', default=6, type=int)
@click.option('--max_duration', '-md', default=120)
@click.option('--frequency', '-f', default=10, type=float)
@click.option('--command_latency', '-cl', default=0.01, type=float)
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints,
    steps_per_inference, max_duration,
    frequency, command_latency):

    # -- Match dataset (unchanged) -----------------------------------------
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

    # -- Load diffusion policy (unchanged) ---------------------------------
    ckpt_path = input
    payload   = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg       = payload['cfg']
    cls       = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    action_offset   = 0
    delta_action    = False
    relative_action = True

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval().to(DEVICE)
    policy.num_inference_steps = 30
    policy.n_action_steps      = 10
    OUTPUT_GAIN = 1.00

    # -- Load gripper classifier -------------------------------------------
    """
    clf_model, proprio_mean, proprio_std, clf_cameras, T = 
        load_gripper_classifier(CLASSIFIER_CKPT)
    
    """

    
    clf_model, proprio_mean, proprio_std, clf_cameras = \
        load_gripper_classifier(CLASSIFIER_CKPT)

    gripper_ctrl = GripperController(
        model         = clf_model,
        proprio_mean  = proprio_mean,
        proprio_std   = proprio_std,
        cameras       = clf_cameras,
        #T             = T,
        threshold     = GRIPPER_THRESHOLD,
        confirm_steps = CONFIRM_STEPS,
    )

    # -- Setup (unchanged) -------------------------------------------------
    dt          = 1 / frequency
    obs_res     = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps:", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)

    latency_tracker  = LatencyTracker()
    H, D             = policy.horizon, policy.action_dim
    time_per_step    = dt
    prev_raw_chunk   = None
    chunk_idx        = 0
    rtc_schedule     = "exp"
    rtc_max_guidance = 5.0
    debug            = True

    with SharedMemoryManager() as shm_manager, \
        RealEnv(
            output_dir          = output,
            robot_ip            = robot_ip,
            frequency           = frequency,
            n_obs_steps         = n_obs_steps,
            obs_image_resolution= obs_res,
            obs_float32         = True,
            init_joints         = init_joints,
            j_init              = j_init,
            enable_multi_cam_vis= True,
            record_raw_video    = True,
            thread_per_video    = 3,
            video_crf           = 21,
            shm_manager         = shm_manager
        ) as env:

        viz = HapticVisualizer(display_plot=False)
        cv2.setNumThreads(1)
        env.realsense.set_exposure(exposure=80, gain=0)
        env.realsense.set_white_balance(white_balance=5900)

        print("Waiting for realsense")
        time.sleep(1.0)

        # Warmup inference
        print("Warming up policy inference")
        obs       = env.get_obs()
        base_pose = obs['robot_eef_pose'][-1]
        obs_rel   = dict(obs)
        obs_rel['robot_eef_pose'] = absolute_to_relative_action(
            obs['robot_eef_pose'], base_pose)

        print(f"Env.get_obs(): {obs.keys()}")
        with torch.no_grad():
            policy.reset()
            obs_dict_np = get_real_obs_dict(
                env_obs=obs_rel, shape_meta=cfg.task.shape_meta)
            print(f"obs_dict_np keys: {obs_dict_np.keys()}")
            obs_dict = dict_apply(obs_dict_np,
                lambda x: torch.from_numpy(x).unsqueeze(0).to(DEVICE))
            result = policy.predict_action(obs_dict)
            action = result['action'][0].detach().to('cpu').numpy()
            assert action.shape[-1] == 6
            del result

        print('Ready for Inference!')
        while True:
            state       = env.get_robot_state()
            target_pose = state['TargetTCPPose']

            try:
                # -- Episode start -----------------------------------------
                policy.reset()
                gripper_ctrl.reset()    # clear history buffer each episode

                warmup_rtc                = False
                start_delay               = 1.0
                eval_t_start              = time.time() + start_delay
                t_start                   = time.monotonic() + start_delay
                env.start_episode(eval_t_start)
                frame_latency             = 1 / 30
                precise_wait(eval_t_start - frame_latency, time_func=time.time)
                print("Started!")

                iter_idx                  = 0
                term_area_start_timestamp = float('inf')
                start_pose                = None
                actions                   = []

                while True:
                    t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                    # -- Observation ---------------------------------------
                    obs       = env.get_obs()
                    base_pose = obs['robot_eef_pose'][-1]

                    if start_pose is None:
                        start_pose = base_pose

                    obs['start_pose'][:] = start_pose
                    obs_rel = dict(obs)

                    rel_window = []
                    for j in range(len(obs['robot_eef_pose'])):
                        rel = absolute_to_relative_action(
                            obs['robot_eef_pose'][j], start_pose)
                        rel_window.append(rel)
                    obs_rel['robot_eef_pose'] = np.stack(rel_window, axis=0)

                    obs_timestamps = obs['timestamp']
                    s_min          = min(steps_per_inference, H)

                    # -- Policy inference (unchanged) ----------------------
                    with torch.no_grad():
                        obs_dict_np = get_real_obs_dict(
                            env_obs=obs_rel, shape_meta=cfg.task.shape_meta)
                        obs_dict = dict_apply(obs_dict_np,
                            lambda x: torch.from_numpy(x).unsqueeze(0).to(DEVICE))

                        is_first_chunk = not warmup_rtc
                        max_latency    = latency_tracker.max() or 0.0

                        if max_latency > 0.0:
                            d_forecast_raw = int(np.ceil(max_latency / time_per_step))
                        else:
                            d_forecast_raw = s_min

                        s_horizon     = int(min(max(s_min, d_forecast_raw), H))
                        d_forecast    = int(min(max(0, d_forecast_raw), s_horizon))
                        prefix_attn_h = max(0, H - s_horizon)
                        infer_start   = time.time()

                        if not warmup_rtc:
                            base_result    = policy.predict_action_flow(obs_dict)
                            prev_raw_chunk = base_result['action'][0].detach().cpu().numpy()
                            warmup_rtc     = True
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += steps_per_inference
                            continue

                        prev_chunk_tensor = torch.from_numpy(
                            prev_raw_chunk).unsqueeze(0).to(DEVICE)

                        result = policy.realtime_action(
                            obs_dict,
                            prev_action_chunk         = prev_chunk_tensor,
                            inference_delay           = d_forecast,
                            prefix_attention_horizon  = prefix_attn_h,
                            prefix_attention_schedule = rtc_schedule,
                            max_guidance_weight       = rtc_max_guidance,
                            n_steps                   = policy.num_inference_steps
                        )

                        curr_raw_chunk = result['action_pred'][0].detach().cpu().numpy()
                        inference_time = time.time() - infer_start
                        latency_tracker.add(inference_time)

                        real_delay_steps = int(np.ceil(inference_time / time_per_step))
                        real_delay_steps = max(0, min(real_delay_steps, s_horizon))
                        if is_first_chunk:
                            real_delay_steps = 0

                        curr_action_world = curr_raw_chunk

                        if debug:
                            with torch.no_grad():
                                naive = policy.predict_action(obs_dict)['action'][0].cpu().numpy()
                            diff = np.linalg.norm(
                                curr_raw_chunk[:(policy.n_action_steps - 1)] -
                                naive[:(policy.n_action_steps - 1)], axis=1)
                            print(
                                f"[RTC DEBUG] "
                                f"lat={inference_time*1000:.1f}ms | "
                                f"d_forecast={d_forecast} | "
                                f"real_delay={real_delay_steps} | "
                                f"mean|D|={diff.mean():.4f} | "
                                f"max|D|={diff.max():.4f}"
                            )

                        H_world          = min(curr_action_world.shape[0], H)
                        s_horizon        = min(s_horizon, H_world)
                        real_delay_steps = min(real_delay_steps, s_horizon)
                        this_chunk_world = curr_action_world[real_delay_steps:s_horizon]

                        if debug:
                            print(
                                f"[RTC DROP] s_horizon={s_horizon}, "
                                f"dropped={real_delay_steps}, "
                                f"exec={s_horizon - real_delay_steps}"
                            )

                        if this_chunk_world.shape[0] == 0:
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += steps_per_inference
                            continue

                        if s_horizon < H:
                            shifted        = curr_raw_chunk[s_horizon:]
                            pad            = np.zeros((s_horizon, D), dtype=curr_raw_chunk.dtype)
                            prev_raw_chunk = np.concatenate([shifted, pad], axis=0)
                        else:
                            prev_raw_chunk = np.zeros_like(curr_raw_chunk)

                        chunk_idx += 1
                        action = this_chunk_world

                    # -- Convert action to env format (unchanged) ----------
                    if delta_action:
                        for i in range(len(action)):
                            action[i, :3] += start_pose[:3]
                            for j in range(0, i):
                                action[i, :3] += action[j, :3]
                            action[i, 3:6] = start_pose[3:6]
                        if len(target_pose) < 7:
                            target_pose = np.append(target_pose, 0)
                        this_target_poses        = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        this_target_poses[:]     = target_pose
                        this_target_poses[:, :6] = action

                    elif relative_action:
                        for i in range(len(action)):
                            action[i, :6] = relative_to_absolute_action(action[i, :6], start_pose)
                            action[i, :3] *= OUTPUT_GAIN
                            action[i, 3:6] = base_pose[3:6]

                        # Build target poses with 7 columns (6 arm + 1 gripper)
                        # Gripper column is left as zero here — classifier fills it in below
                        if len(target_pose) < 7:
                            target_pose = np.append(target_pose, 0)
                        this_target_poses        = np.zeros((len(action), 7), dtype=np.float64)
                        this_target_poses[:]     = target_pose
                        this_target_poses[:, :6] = action     # only 6 dims from policy
                        actions.append(this_target_poses)

                    else:
                        if len(target_pose) < 7:
                            target_pose = np.append(target_pose, 0)
                        this_target_poses           = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                        this_target_poses[:]        = target_pose
                        this_target_poses[:, :7]    = action
                        this_target_poses[:, 3:6]   = start_pose[3:6]

                    # ── Classifier gripper override ────────────────────────
                    # Build proprioception from the most recent observation
                    proprio_raw = np.concatenate([
                        obs['robot_joint'][-1],      # (6,) joint angles
                        obs['robot_joint_vel'][-1],  # (6,) joint velocities
                        obs['robot_eef_pose'][-1],   # (6,) EE pose
                    ]).astype(np.float32)

                    gripper_state, prob = gripper_ctrl.step(obs, proprio_raw)

                    # Override the gripper dimension (column 6) in every
                    # action in the chunk — diffusion policy's gripper
                    # prediction is ignored entirely
                    this_target_poses[:, 6] = float(gripper_state)

                    # -- Timing (unchanged) --------------------------------
                    action_timestamps   = (
                        np.arange(len(action), dtype=np.float64) + action_offset
                    ) * dt + obs_timestamps[-1]
                    action_exec_latency = 0.01
                    curr_time           = time.time()
                    is_new              = action_timestamps > (curr_time + action_exec_latency)

                    if np.sum(is_new) == 0:
                        this_target_poses = this_target_poses[[-1]]
                        next_step_idx     = int(np.ceil((curr_time - eval_t_start) / dt))
                        action_timestamp  = eval_t_start + (next_step_idx) * dt
                        action_timestamps = np.array([action_timestamp])
                    else:
                        this_target_poses = this_target_poses[is_new]
                        action_timestamps = action_timestamps[is_new]

                    # -- Execute -------------------------------------------
                    env.exec_actions(
                        actions    = this_target_poses,
                        timestamps = action_timestamps
                    )

                    # -- Visualise -----------------------------------------
                    episode_id = env.replay_buffer.n_episodes
                    vis_img    = obs[f'camera_{vis_camera_idx}'][-1]
                    text       = 'Episode: {}, Time: {:.1f}'.format(
                        episode_id, time.monotonic() - t_start)
                    cv2.putText(vis_img, text, (10, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, color=(255, 255, 255))

                    # Classifier confidence overlay
                    grip_text = "Gripper: {} ({:.2f})".format(
                        'CLOSE' if gripper_state else 'OPEN', prob)
                    
                    print(grip_text)
                    """
                    cv2.putText(vis_img, grip_text, (10, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1,
                        color=(0, 255, 0) if gripper_state else (0, 0, 255))
                    
                    """

                    cv2.imshow('default', vis_img[..., ::-1])

                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('s'):
                        env.end_episode()
                        print('Stopped.')
                        break

                    # -- Auto termination (unchanged) ----------------------
                    terminate = False
                    if time.monotonic() - t_start > max_duration:
                        terminate = True
                        print('Terminated by the timeout!')

                    term_pose = np.array([
                        3.40948500e-01,  2.17721816e-01,  4.59076878e-02,
                        2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                    curr_pose = obs['robot_eef_pose'][-1]
                    dist      = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)

                    if dist < 0.03:
                        curr_timestamp = obs['timestamp'][-1]
                        if term_area_start_timestamp > curr_timestamp:
                            term_area_start_timestamp = curr_timestamp
                        else:
                            term_area_time = curr_timestamp - term_area_start_timestamp
                            if term_area_time > 0.5:
                                terminate = True
                                print('Terminated by the policy!')
                    else:
                        term_area_start_timestamp = float('inf')

                    if terminate:
                        env.end_episode()
                        break

                    precise_wait(t_cycle_end - frame_latency)
                    iter_idx += steps_per_inference

            except KeyboardInterrupt:
                print("Interrupted!")
                env.end_episode()
                return

            print("Stopped.")


# %%
if __name__ == '__main__':
    main()