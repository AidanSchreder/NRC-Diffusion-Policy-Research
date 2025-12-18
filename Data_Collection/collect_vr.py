# python collect_vr.py -o /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy/Data_Collection/Collected_Data/taskName_numOfEpisodes

import time
import click
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from Training.real_world.real_env import RealEnv
from Training.common.precise_sleep import precise_wait
from Haptic_Device_Setup.haply_barebones import HapticVisualizer
from Data_Collection.Haptic_Device_Setup.BooleanGripperCartesianAction import VRPolicy
from Training.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# Starting Robot Position
# j_init = np.array([256.97,-61.94,133.80,109.75,-76.24,-4.66]) / 180 * np.pi
j_init = np.array([262.85,-87.14,111.61,246.68,-89.5,-10.81]) / 180 * np.pi

POS_GAIN = 1
ROT_GAIN = 0.3
STILL_THRESHOLD = 5e-4

def quat_inverse(q):
    w, x, y, z = q
    n2 = w*w + x*x + y*y + z*z
    return np.array([w, -x, -y, -z]) / (n2 if n2 > 0 else 1.0)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _fmt3(v):
    return f"[{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}]"

@click.command()
@click.option('--output', '-o', default="/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy/Data_Collection/Collected_Data/temp_location", help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="192.168.0.110", help="UR10e's IP address")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Initialize robot joint configuration at start.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command and execution (s).")

def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1.0 / frequency

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

            controller = VRPolicy(right_controller=True)
            print("Waiting for VR controller to be ready...")
            cv2.setNumThreads(1)
            env.realsense.set_exposure(exposure=90, gain=0)
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)
            print('Ready!')
            print("C = Start, S = Save and Stop, Backspace = Delete Episode, Q = Quit")

            target_pose = env.get_robot_state()['TargetTCPPose'].copy()  # [x, y, z, rx, ry, rz]
            t_start = time.monotonic()
            iter_idx = 0
            is_recording = False
            stop = False

            prev_pos = None
            prev_quat = None
            just_enabled = False  # <-- edge flag

            try:
                while not stop:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_time = t_cycle_end + dt

                    obs = env.get_obs()

                    if env.start_pose is None:
                        env.start_pose = obs['robot_eef_pose'][-1]
                        print(f"Start Pose: {env.start_pose}")

                    # handle key presses
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            stop = True
                            break
                        elif key_stroke == KeyCode(char='c'):
                            # Start recording
                            env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                            key_counter.clear()
                            is_recording = True
                            print('Recording!')
                        elif key_stroke == KeyCode(char='s'):
                            # Stop recording
                            env.end_episode()
                            key_counter.clear()
                            is_recording = False
                            print('Stopped.')
                        elif key_stroke == Key.backspace:
                            # Delete the most recent recorded episode
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                                is_recording = False

                    vis = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()
                    episode_id = env.replay_buffer.n_episodes
                    text = f'Episode: {episode_id}'
                    if is_recording:
                        text += ' REC'
                    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow('default', vis)
                    cv2.pollKey() 

                    precise_wait(t_sample)

                    # Read current frame
                    controller._process_reading()
                    vr_state = controller.vr_state
                    info = controller.get_info()

                    # -------- BLOCK WHILE DISABLED --------
                    if not info.get("movement_enabled", False):
                        # Stay here until enabled, printing DISABLED status
                        while True:
                            controller._process_reading()
                            vr_state = controller.vr_state
                            info = controller.get_info()
                            g = vr_state.get('gripper', 0.0)
                            # print(f"[Δpos: [0.0000 0.0000 0.0000], Δrot: [0.0000 0.0000 0.0000], gripper: {g:.4f}, DISABLED]")
                            if info.get("movement_enabled", False):
                                # we will resync once on the next loop body
                                just_enabled = True
                                break
                            time.sleep(0.01)
                        # after enabling, fall through

                    # Refresh once after unblock
                    controller._process_reading()
                    vr_state = controller.vr_state
                    curr_pos = np.array(vr_state['pos'], dtype=float)
                    curr_quat = np.array(vr_state['quat'], dtype=float)  # [w,x,y,z]

                    # -------- RESYNC ONCE ON ENABLE EDGE --------
                    if just_enabled or prev_pos is None or prev_quat is None:
                        prev_pos = curr_pos
                        prev_quat = curr_quat
                        just_enabled = False
                        # print(f"[Δpos: [0.0000 0.0000 0.0000], Δrot: [0.0000 0.0000 0.0000], "
                            # f"gripper: {vr_state.get('gripper', 0.0):.4f}, ENABLED (RESYNC)]")
                        # skip sending this tick to avoid snap
                        t_next = time.perf_counter()
                        time.sleep(0.01)
                        continue

                    # -------- NORMAL DELTAS --------
                    delta_pos = curr_pos - prev_pos
                    dq = quat_multiply(curr_quat, quat_inverse(prev_quat))
                    n = np.linalg.norm(dq)
                    if n > 0:
                        dq = dq / n
                    try:
                        delta_euler = R.from_quat(dq[[1, 2, 3, 0]]).as_euler('xyz', degrees=False)
                    except Exception:
                        delta_euler = np.zeros(3, dtype=float)

                    # Threshold check (still print so you can see loop health)
                    if np.all(np.abs(np.concatenate([delta_pos, delta_euler])) < STILL_THRESHOLD):
                        # print(f"[Δpos: {_fmt3(np.zeros(3))}, Δrot: {_fmt3(np.zeros(3))}, "
                        #     f"gripper: {vr_state.get('gripper', 0.0):.4f}, ENABLED]")
                        prev_pos = curr_pos
                        prev_quat = curr_quat
                        time.sleep(0.01)
                        # pacing
                        t_next += dt
                        sleep_for = t_next - time.perf_counter()
                        if sleep_for > 0: time.sleep(sleep_for)
                        else: t_next = time.perf_counter()
                        continue

                    pos_delta_scaled = delta_pos * POS_GAIN
                    rot_delta_scaled = delta_euler * ROT_GAIN
                    gripper_val = vr_state.get('gripper', 0.0)

                    # print(f"[Δpos: {_fmt3(pos_delta_scaled)}, Δrot: {_fmt3(rot_delta_scaled)}, "
                        # f"gripper: {gripper_val:.4f}, ENABLED]")

                    # Send action
                    action = np.concatenate([pos_delta_scaled, rot_delta_scaled, [gripper_val]])

                    pos_delta = action[:3]
                    rot_delta = action[3:6]
                    gripper_state = float(action[-1])

                    if is_recording and (np.linalg.norm(pos_delta) > 1e-5 or np.linalg.norm(rot_delta) > 1e-5):
                        # Update position
                        target_pose[:3] += pos_delta

                        # Update rotation via proper composition
                        current_rot = st.Rotation.from_rotvec(target_pose[3:])
                        delta_rot = st.Rotation.from_rotvec(rot_delta)
                        new_rot = delta_rot * current_rot
                        target_pose[3:] = new_rot.as_rotvec()

                        action_with_gripper = np.concatenate([target_pose[:6], [gripper_state]]).astype(np.float32)

                        # Log pose for debugging
                        # print("Sending target_pose:", action_with_gripper)
                        
                        # Send action
                        t_command_time = time.time() + dt
                        stage = 0  # or key_counter[Key.space] if you're handling phases
                        env.exec_actions(
                            actions=[action_with_gripper],
                            timestamps=[t_command_time],
                            stages=[stage]
                        )

                        gripper_current_state = env.get_gripper_state()

                        if gripper_state != gripper_current_state:
                            env.send_gripper_command(int(gripper_state * 255))

                    prev_pos = curr_pos
                    prev_quat = curr_quat

                    precise_wait(t_cycle_end)
                    iter_idx += 1

            except KeyboardInterrupt:
                print("Interrupted. Shutting down.")
            finally:
                if is_recording:
                    env.end_episode()
                    print('Recording stopped.')
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
