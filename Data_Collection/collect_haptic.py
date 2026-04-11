# python collect_haptic.py -o /home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy_2.0/Data_Collection/Collected_Data/press_2

"""
check camera connection:
realsense-viewer

"""


import time
import click
import cv2
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st

from Training.real_world.real_env import RealEnv
from Training.common.precise_sleep import precise_wait
from Haptic_Device_Setup.haply_barebones import HapticVisualizer
from Training.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# Starting Robot Position

#lamp task
#j_init = np.array([256.97,-61.94,133.80,109.75,-76.24,-4.66]) / 180 * np.pi

#wiping task
#j_init = np.array([262.85,-87.14,111.61,246.68,-89.5,-10.81]) / 180 * np.pi

#press
j_init = np.array([268.77,-76.17,93.10,255.32,-89.55,-5.22]) / 180 * np.pi



#new wiping experiment
#j_init = np.array([282.44,-73.93,86.34,258.27,-89.28,8.64]) / 180 * np.pi





#ceneter
#j_init = np.array([286.23,-79.18,92.93,256.84,-89.22,12.43]) / 180 * np.pi

#simplifeid wiping
#j_init = np.array([284.78,-72.73,101.10,243.87,-89.68,10.07]) / 180 * np.pi

@click.command()
@click.option('--output', '-o', default="/home/demoaccount/Data/demoaccount2/Fall_2025/NRC_Diffusion_Policy/Data_Collection/Collected_Data/temp_location", help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="192.168.0.110", help="UR10e's IP address")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Initialize robot joint configuration at start.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command and execution (s).")
@click.option('--fts_ip', default="192.168.1.11", help="Force-torque sensor IP address.")
@click.option('--fts_port', default=10547, type=int, help="Force-torque sensor UDP port.")
@click.option('--force_feedback/--no-force_feedback', default=True, help="Enable Haply force feedback using live FTS data.")

def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency,
         fts_ip, fts_port, force_feedback):
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
                enable_force_torque=True,
                fts_ip=fts_ip,
                fts_port=fts_port,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager
            ) as env:

            viz = HapticVisualizer(
                display_plot=False,
                haptic_feedback=force_feedback,
                force_reader=env,
            )
            cv2.setNumThreads(1)
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)
            time.sleep(1.0)
            print('Ready!')
            print("C = Start, S = Save and Stop, Backspace = Delete Episode, Q = Quit")

            target_pose = env.get_robot_state()['TargetTCPPose'].copy()  # [x, y, z, rx, ry, rz]
            t_start = time.monotonic()
            iter_idx = 0
            is_recording = False
            stop = False

            try:
                while viz.running and not stop:
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

                    action = viz.get_action(mode='euler', seq='xyz', degrees=False)
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
