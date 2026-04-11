# python collect_haptic_gripper.py -o /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/gripper2

import time
import click
import cv2
import torch
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
import torchvision.transforms as T_transforms
from PIL import Image
from r3m import load_r3m
from collections import deque

from Training.real_world.real_env import RealEnv
from Training.common.precise_sleep import precise_wait
from Haptic_Device_Setup.haply_barebones import HapticVisualizer
from Training.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from Training.gripper_training.Model.gripper_classifier import GripperClassifier

# ── Starting Robot Position ───────────────────────────────────────────────────
j_init = np.array([282.44,-73.93,86.34,258.27,-89.28,8.64]) / 180 * np.pi

# ── Classifier Config ─────────────────────────────────────────────────────────
CLASSIFIER_CKPT = "/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Training/gripper_training/Checkpoints/gripper_training_history_mod-1.ckpt"
CAMERAS         = [0, 1]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Hysteresis: classifier must be confident for this many consecutive steps
# before the gripper state actually changes — prevents rapid flickering
CONFIRM_STEPS   = 3
GRIPPER_THRESHOLD = 0.04
T = 5

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

"""
# ── Gripper Controller with Hysteresis ───────────────────────────────────────
class GripperController:
    def __init__(self, model, proprio_mean, proprio_std, cameras, threshold=0.5, confirm_steps=3):
        self.model         = model
        self.proprio_mean  = proprio_mean
        self.proprio_std   = proprio_std
        self.cameras       = cameras
        self.threshold     = threshold
        self.confirm_steps = confirm_steps

        self.current_state = 0
        self.pending_state = None
        self.confirm_count = 0

        # Load R3M encoder for live feature extraction
        print("Loading R3M encoder for inference...")
        self.encoder = load_r3m("resnet18").to(DEVICE)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.transform = T_transforms.Compose([
            T_transforms.Resize((224, 224)),
            T_transforms.ToTensor(),
            T_transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])

    def step(self, obs, proprio_raw):
        # -- Extract R3M features per camera ------------------------------
        cam_features = []
        for cam in self.cameras:
            frame_np = obs[f'camera_{cam}'][-1]                    # (H, W, C)
            frame_t  = self.transform(Image.fromarray(frame_np))   # (3, 224, 224)
            frame_t  = frame_t.unsqueeze(0).to(DEVICE)             # (1, 3, 224, 224)
            with torch.no_grad():
                feat = self.encoder(frame_t).view(1, -1)           # (1, 512)
            cam_features.append(feat)

        # Concatenate cameras → (1, 1024)
        image_features = torch.cat(cam_features, dim=-1)

        # -- Normalise proprioception -------------------------------------
        proprio_norm = (proprio_raw - self.proprio_mean) / self.proprio_std
        proprio_t    = torch.from_numpy(
            proprio_norm.astype(np.float32)
        ).unsqueeze(0).to(DEVICE)                                  # (1, 18)

        # -- Forward pass through projection head only -------------------
        with torch.no_grad():
            logit = self.model(image_features, proprio_t)
            prob  = torch.sigmoid(logit).item()

        # -- Hysteresis (unchanged) ---------------------------------------
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
                print(f"  [Gripper] → "
                      f"{'CLOSE' if self.current_state else 'OPEN'} "
                      f"(prob={prob:.3f})")

        return self.current_state, prob


"""


class GripperController:
    def __init__(self, model, proprio_mean, proprio_std, cameras, T=5,
                 threshold=0.5, confirm_steps=3):
        self.model         = model
        self.proprio_mean  = proprio_mean   # (PROPRIO_DIM,) single step
        self.proprio_std   = proprio_std
        self.cameras       = cameras
        self.T             = T
        self.threshold     = threshold
        self.confirm_steps = confirm_steps

        self.current_state = 0
        self.pending_state = None
        self.confirm_count = 0

        # Rolling buffer — initialised empty, pads with zeros at episode start
        self.proprio_buffer = deque(maxlen=T)

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
        self.proprio_buffer.clear()
        self.current_state = 0
        self.pending_state = None
        self.confirm_count = 0

    def step(self, obs, proprio_raw):
        # -- Update rolling buffer -------------------------------------
        self.proprio_buffer.append(proprio_raw.copy())

        # Pad with zeros if buffer not yet full (start of episode)
        while len(self.proprio_buffer) < self.T:
            self.proprio_buffer.appendleft(np.zeros_like(proprio_raw))

        # -- Build history vector (T * PROPRIO_DIM,) -------------------
        history      = np.concatenate(list(self.proprio_buffer))
        mean_tiled   = np.tile(self.proprio_mean, self.T)
        std_tiled    = np.tile(self.proprio_std,  self.T)
        std_tiled = np.maximum(std_tiled, 0.01)
        history_norm = (history - mean_tiled) / std_tiled
        proprio_t    = torch.from_numpy(
            history_norm.astype(np.float32)
        ).unsqueeze(0).to(DEVICE)                           # (1, T*PROPRIO_DIM)

        # -- R3M features (unchanged) ----------------------------------
        cam_features = []
        for cam in self.cameras:
            frame_np = obs[f'camera_{cam}'][-1]
            frame_t  = self.transform(Image.fromarray(frame_np))
            frame_t  = frame_t.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = self.encoder(frame_t).view(1, -1)    # (1, 512)
            cam_features.append(feat)
        image_features = torch.cat(cam_features, dim=-1)    # (1, 1024)

        # -- Forward pass ----------------------------------------------
        with torch.no_grad():
            logit = self.model(image_features, proprio_t)
            prob  = torch.sigmoid(logit).item()

        # -- Hysteresis (unchanged) ------------------------------------
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
                print(f"  [Gripper] → "
                      f"{'CLOSE' if self.current_state else 'OPEN'} "
                      f"(prob={prob:.3f})")
        print(f"prob={prob:.4f} logit={logit.item():.4f}"
              f"proprio_norm_range=[{history_norm.min():.2f}, {history_norm.max():.2f}]")

        return self.current_state, prob




# ── Classifier Loading ───────────────────────────────────────────────────────

def load_classifier(ckpt_path):
    print(f"Loading gripper classifier from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = GripperClassifier(
        proprio_dim=ckpt["proprio_dim"],
        num_cams=ckpt["num_cams"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    proprio_mean = ckpt["proprio_mean"]
    proprio_std  = ckpt["proprio_std"]

    print("Classifier loaded successfully.")
    return model, proprio_mean, proprio_std


# ── Main ─────────────────────────────────────────────────────────────────────

@click.command()
@click.option('--output', '-o', default="/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy/Data_Collection/Collected_Data/temp_location", help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="192.168.0.110", help="UR10e's IP address")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Initialize robot joint configuration at start.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command and execution (s).")

def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1.0 / frequency

    # -- Load classifier ---------------------------------------------------
    model, proprio_mean, proprio_std = load_classifier(CLASSIFIER_CKPT)
    gripper_ctrl = GripperController(
        model         = model,
        proprio_mean  = proprio_mean,
        proprio_std   = proprio_std,
        cameras       = CAMERAS,       
        threshold     = GRIPPER_THRESHOLD,
        T             = T,
        confirm_steps = CONFIRM_STEPS,
    )

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                obs_image_resolution=(480, 640),
                frequency=frequency,
                init_joints=init_joints,
                j_init=j_init,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager
            ) as env:

            viz = HapticVisualizer(display_plot=False)
            cv2.setNumThreads(1)
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)
            print('Ready!')
            print("C = Start, S = Save and Stop, Backspace = Delete Episode, Q = Quit")
            print("[Gripper control handed to classifier]")

            target_pose   = env.get_robot_state()['TargetTCPPose'].copy()
            t_start       = time.monotonic()
            iter_idx      = 0
            is_recording  = False
            stop          = False

            try:
                while viz.running and not stop:
                    t_cycle_end    = t_start + (iter_idx + 1) * dt
                    t_sample       = t_cycle_end - command_latency
                    t_command_time = t_cycle_end + dt

                    obs = env.get_obs()

                    if env.start_pose is None:
                        env.start_pose = obs['robot_eef_pose'][-1]
                        print(f"Start Pose: {env.start_pose}")

                    # -- Key handling (unchanged) --------------------------
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            stop = True
                        elif key_stroke == KeyCode(char='c'):
                            env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                            #gripper_ctrl.reset()
                            key_counter.clear()
                            is_recording = True
                            print('Recording!')
                        elif key_stroke == KeyCode(char='s'):
                            env.end_episode()
                            key_counter.clear()
                            is_recording = False
                            print('Stopped.')
                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                                is_recording = False

                    # -- Visualisation (unchanged) -------------------------
                    vis        = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()
                    episode_id = env.replay_buffer.n_episodes
                    text       = f'Episode: {episode_id}'
                    if is_recording:
                        text += ' REC'
                    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('default', vis)
                    cv2.pollKey()

                    precise_wait(t_sample)

                    # -- Haptic device: arm only ---------------------------
                    action    = viz.get_action(mode='euler', seq='xyz', degrees=False)
                    pos_delta = action[:3]
                    rot_delta = action[3:6]
                    # action[-1] (haptic gripper) is intentionally ignored here

                    if is_recording and (np.linalg.norm(pos_delta) > 1e-5 or np.linalg.norm(rot_delta) > 1e-5):
                        # -- Arm control (unchanged) -----------------------
                        target_pose[:3] += pos_delta
                        current_rot      = st.Rotation.from_rotvec(target_pose[3:])
                        delta_rot        = st.Rotation.from_rotvec(rot_delta)
                        new_rot          = delta_rot * current_rot
                        target_pose[3:]  = new_rot.as_rotvec()

                        # -- Classifier gripper control -------------------
                        # Build proprioception vector from current observation
                        # obs arrays are (T, dim) — take the most recent timestep
                        proprio_raw = np.concatenate([
                            obs['robot_joint'][-1],          # (6,) joint angles
                            obs['robot_joint_vel'][-1],      # (6,) joint velocities
                            obs['robot_eef_pose'][-1],       # (6,) EE pose
                        ]).astype(np.float32)

                        gripper_state, prob = gripper_ctrl.step(obs, proprio_raw)

                        # Overlay classifier confidence on visualisation
                        conf_text = f"Gripper: {'CLOSE' if gripper_state else 'OPEN'} ({prob:.2f})"
                        print(conf_text)
                        cv2.putText(vis, conf_text, (10, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 0) if gripper_state else (0, 0, 255), 2)

                        # -- Send arm + classifier gripper as one action --
                        action_with_gripper = np.concatenate(
                            [target_pose[:6], [gripper_state]]
                        ).astype(np.float32)

                        t_command_time = time.time() + dt
                        stage          = 0
                        env.exec_actions(
                            actions    = [action_with_gripper],
                            timestamps = [t_command_time],
                            stages     = [stage]
                        )

                        # -- Send gripper command directly -----------------
                        gripper_current_state = env.get_gripper_state()
                        if gripper_state != gripper_current_state:
                            env.send_gripper_command(int(gripper_state * 255))

                    precise_wait(t_cycle_end)
                    iter_idx += 1

            except KeyboardInterrupt:
                print("Interrupted. Shutting down.")
            finally:
                if is_recording:
                    env.end_episode()
                    print('Recording stopped.')
                viz.stop()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()