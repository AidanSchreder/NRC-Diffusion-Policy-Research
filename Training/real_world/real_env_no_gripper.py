from typing import Optional
from collections import deque
import logging
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from Training.real_world.rtde_interpolation_controller import RTDEInterpolationController
from Training.real_world.multi_realsense import MultiRealsense, SingleRealsense
from Training.real_world.video_recorder import VideoRecorder
from Training.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from Training.real_world.multi_camera_visualizer import MultiCameraVisualizer
from Training.common.replay_buffer import ReplayBuffer
from Training.common.cv2_util import (
    get_image_transform, optimal_row_cols)

from Data_Collection.Haptic_Device_Setup import robotiq_gripper

#FTS
from pathlib import Path

# FTS Force-torque sensor SDK imports used for live FT observations during inference.
from Training.real_world.mae_sdk_sensureal.mae_fts_sdk.src import mae_fts_sdk as mae
from Training.real_world.mae_sdk_sensureal.mae_sdk.src.mae_sdk import log_utils

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}
#FTS ForceTorqueSensor class --> establish connection to sensor
class ForceTorqueSensor:
    def __init__(self, fts_ip="192.168.1.11", fts_port=10547, log_force=False):
        self.AVAILABLE_FTS_COMMANDS = {str(key): key for key in mae.FTS_COMMAND_RESPONSE.keys()}
        self.logger = log_utils.get_logger(f"{Path(__file__).stem}", log_level=logging.WARNING)
        self.communication_interface = mae.UdpCommunication(
            ip_address=fts_ip,
            port=fts_port,
            timeout_sec=1,
            log_level=logging.WARNING,
        )
        self.log_force=log_force
        self.communication_interface.connect()
        time.sleep(1)
        # Zero the sensor before use
        self.send_command("FtsCommand.TRANSDUCER_SET")
        time.sleep(1)
        self.send_command("FtsCommand.BIAS_SET")
        time.sleep(1)
        self.send_command("FtsCommand.SETTINGS_SAVE")
        time.sleep(1)
        
    def send_command(self, new_command):
        command = self.AVAILABLE_FTS_COMMANDS[new_command]
        self.communication_interface.send_request(command)
        if self.log_force:
            print(f"Sent: {command}.")
        elif new_command != "FtsCommand.STREAM_FT_START":
            print(f"Sent: {command}.")
        # Some Commands don't have a response, hence response_type can be None
        response_type = mae.FTS_COMMAND_RESPONSE[command]
    
        if response_type is not None:
            byte_stream = self.communication_interface.waits_response_bytes()
            if byte_stream is None:
                self.logger.error(f"No Response. Timed-out.")
    
            response = response_type(byte_stream)
            if self.log_force:
                print(f"Received: {response}.")

            return response.ft_sample
        
    def stop(self):
        self.communication_interface.disconnect()

#FTS Reuse the same Rodrigues rotation logic from collection-time FT preprocessing.
def _rodrigues_rotate(vec, axis, angle):
    return (
        (1 - math.cos(angle)) * np.dot(vec, axis) * axis
        + vec * math.cos(angle)
        + np.cross(axis, vec) * math.sin(angle)
    )

class RealEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(640,480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset=0,
            init_joints=True,
            j_init=None,
            
            #FTS Optional live force-torque sensor connection for inference-time fts_info.
            enable_force_torque=False,
            fts_ip="192.168.1.11",
            fts_port=10547,

            # video capture params
            video_capture_fps=15,
            video_capture_resolution=(640,480),
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(640,480),
            # shared memory
            shm_manager=None
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            if 'color' not in data:
                print("[Warning] Missing 'color' in data. Skipping transform.")
                return data
            data['color'] = color_transform(data['color'])
            return data

        
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )
        def vis_transform(data):
            if 'color' not in data:
                print("[Warning] Missing 'color' in vis_transform. Skipping.")
                return data
            data['color'] = vis_color_transform(data['color'])
            return data


        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
            )
        
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])
        
        
        if not init_joints:
            j_init = None

        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125, # UR5 CB3 RTDE
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=3,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
            )
        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None

        # Gripper Implementation: ---------------------------------------------------------
        self.gripper_current_state = 0.0
        self.gripper_history = []
        
        self.gripper = robotiq_gripper.RobotiqGripper()
        self.gripper.connect(robot_ip, 63352)
        self.gripper.activate()
        time.sleep(1)
        # ---------------------------------------------------------------------------------
        
        # FTS Initialize the live FT sensor only when the loaded policy expects fts_info observations.
        self.enable_force_torque = enable_force_torque
        self.ft_sensor = None
        self.last_fts_info = np.zeros((6,), dtype=np.float32)
        self.ft_buffer = deque(maxlen=max(max_obs_buffer_size * 4, n_obs_steps * 8, 64))
        self.ft_timestamp_buffer = deque(maxlen=max(max_obs_buffer_size * 4, n_obs_steps * 8, 64))
        if self.enable_force_torque:
            self.ft_sensor = ForceTorqueSensor(
                fts_ip=fts_ip,
                fts_port=fts_port,
                log_force=False
            )

        self.start_pose = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready
    
    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
            
        #FTS Close the FT sensor connection when inference stops.
        if self.ft_sensor is not None:
            self.ft_sensor.stop()
            self.ft_sensor = None
        if wait:
            self.stop_wait()
        

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # # === Add gripper state ===
        # robot_obs["robot_gripper_state"] = np.full(
        #     (len(this_idxs), 1), 
        #     self.gripper_current_state, 
        #     dtype=np.float32
        # )

        num_samples = robot_obs_raw["robot_eef_pose"].shape[0]

        current_start_pose = self.start_pose
        if current_start_pose is None:
            current_start_pose = np.asarray(
                robot_obs_raw["robot_eef_pose"][this_idxs[-1]],
                dtype=np.float32
            )
        start_pose_arr = np.broadcast_to(
            np.asarray(current_start_pose, dtype=np.float32),
            (num_samples, 6)
        ).copy()
        robot_obs['start_pose'] = start_pose_arr
        robot_obs_raw['start_pose'] = start_pose_arr
        
        # FTS Keep a short history of transformed FT samples so the observation
        # window contains per-timestep values instead of a broadcast copy.
        if self.ft_sensor is not None:
            try:
                latest_fts_info = self.get_fts_observation(
                    tcp_pose=robot_obs["robot_eef_pose"][-1]
                )
                self._push_fts_sample(
                    fts_info=latest_fts_info,
                    timestamp=float(robot_timestamps[this_idxs[-1]])
                )
            except Exception:
                latest_fts_info = self.last_fts_info

            robot_obs_raw['fts_info'] = self._align_fts_samples(robot_timestamps)
            robot_obs['fts_info'] = robot_obs_raw['fts_info'][this_idxs]

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        # print(f"New Actions: {new_actions}")
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints

        for i in range(len(new_actions)):
            pose = new_actions[i]
            if len(pose) == 6:
                pose = np.append(pose, 1.0)  # pad dummy gripper value
            self.robot.schedule_waypoint(
                pose=pose,
                target_time=new_timestamps[i]
            )
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    def _push_fts_sample(self, fts_info: np.ndarray, timestamp: float) -> None:
        fts_arr = np.asarray(fts_info, dtype=np.float32).reshape(6)
        self.last_fts_info = fts_arr
        self.ft_buffer.append(fts_arr.copy())
        self.ft_timestamp_buffer.append(float(timestamp))

    def _align_fts_samples(self, target_timestamps: np.ndarray) -> np.ndarray:
        target_timestamps = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
        if len(target_timestamps) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        if len(self.ft_buffer) == 0:
            return np.broadcast_to(self.last_fts_info, (len(target_timestamps), 6)).astype(np.float32).copy()

        ft_timestamps = np.asarray(self.ft_timestamp_buffer, dtype=np.float64)
        ft_values = np.asarray(self.ft_buffer, dtype=np.float32)
        aligned = np.empty((len(target_timestamps), 6), dtype=np.float32)
        for i, timestamp in enumerate(target_timestamps):
            idx = np.searchsorted(ft_timestamps, timestamp, side='right') - 1
            if idx < 0:
                idx = 0
            aligned[i] = ft_values[idx]
        return aligned
    
    # FTS Match the collection pipeline's FT preprocessing so inference-time fts_info has the same semantics as training data.
    def get_fts_observation(self, tcp_pose: np.ndarray) -> np.ndarray:
        """
        Match the collection pipeline's force-torque preprocessing so
        inference receives the same 6D representation used for training.
        """
        # Read a raw 6D FT sample from the live sensor using the same command path as collection.
        ft_vec_raw = np.asarray(
            self.ft_sensor.send_command("FtsCommand.STREAM_FT_START"),
            dtype=np.float32
        ).reshape(-1)

        if ft_vec_raw.shape[0] < 6:
            raise ValueError(f"Expected 6D FT sample, got shape {ft_vec_raw.shape}")

        force_vec_raw = ft_vec_raw[:3]
        fx, fy, fz = force_vec_raw
        # Remap raw sensor force axes into the intermediate TCP-aligned convention from collection.
        force_vec_tcp_raw = np.array([fy, -fx, fz], dtype=np.float32)
        torque_vec = ft_vec_raw[3:6].astype(np.float32)

        eef_rot_vec = np.asarray(tcp_pose[3:6], dtype=np.float32)
        eef_rot_mag = float(np.linalg.norm(eef_rot_vec))

        if eef_rot_mag < 1e-8:
            eef_rot_matrix = np.eye(3, dtype=np.float32)
        else:
            eef_rot_axis = eef_rot_vec / eef_rot_mag
            x_axis = np.array([1, 0, 0], dtype=np.float32)
            y_axis = np.array([0, 1, 0], dtype=np.float32)
            z_axis = np.array([0, 0, 1], dtype=np.float32)

            # Rotate the force vector into the current TCP frame using the robot end-effector orientation.
            eef_x_axis = _rodrigues_rotate(x_axis, eef_rot_axis, eef_rot_mag)
            eef_y_axis = _rodrigues_rotate(y_axis, eef_rot_axis, eef_rot_mag)
            eef_z_axis = _rodrigues_rotate(z_axis, eef_rot_axis, eef_rot_mag)
            eef_rot_matrix = np.column_stack([eef_x_axis, eef_y_axis, eef_z_axis]).astype(np.float32)

        force_vec_tcp = eef_rot_matrix @ force_vec_tcp_raw
        fx_tcp, fy_tcp, fz_tcp = force_vec_tcp
        # Apply the final axis remap used by collection before concatenating force and torque.
        force_vec = np.array([-fy_tcp, fx_tcp, fz_tcp], dtype=np.float32)

        return np.concatenate((force_vec, torque_vec), axis=0).astype(np.float32)
    
    # recording API
    def start_episode(self, start_time=None):
        """
        Setup cameras and create save path.
        Creates accumulators for observations, actions, and stages at a fixed frequency.

        Parameters
        ----------
        start_time : float, optional
            The timestamp (in seconds) marking when the episode begins.
            If None, the current system time is used.

        Returns
        -------
        None
            Does not return a value.
        """
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready
        if self.start_pose is None:
            self.start_pose = np.asarray(
                self.robot.get_state()["ActualTCPPose"],
                dtype=np.float32
            )

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        """
        Stop recording and save accumulated episode data.
        Saves the complete episode into the replay buffer.

        Returns
        -------
        None
            This function does not return a value.
        """
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None
            self.start_pose = None

    def drop_episode(self):
        print(f"Before drop: {self.replay_buffer.n_episodes}")
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
        print(f"After drop: {self.replay_buffer.n_episodes}")

    def send_gripper_command(self, gripper_value, speed=70, force=50):
        self.gripper.move(gripper_value, speed, force)
        
        try:
          self.gripper_current_state = float(self.get_gripper_state())
        except Exception:
          self.gripper_current_state = 1.0 if gripper_value > 0.5 else 0.0

    def get_gripper_state(self):
        gripper_state = 0.0 if self.gripper.is_open() else 1.0
        return gripper_state
