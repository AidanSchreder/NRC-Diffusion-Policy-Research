import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

import sys
from Data_Collection.Haptic_Device_Setup.teleop_controls.haply_reader.reader import HapticReader

class HapticVisualizer:
    def __init__(self, display_plot=True, gripper_open=1.0, gripper_closed=0.0,
                 gripper_steps=5, haptic_feedback=False, force_reader=None):
        # State
        self.running = True
        self.calibrated = False
        self.haply_to_global = np.eye(4)
        self.pose = np.zeros(3)
        self.orient = np.eye(3)
        self.quat = np.array([0,0,0,1])

        # Deltas
        self.prev_pose = self.pose.copy()
        self.prev_euler = np.zeros(3)
        self.prev_quat = self.quat.copy()

        # Gripper
        self.gripper_open = float(gripper_open)
        self.gripper_closed = float(gripper_closed)
        self.gripper_steps = int(gripper_steps)
        self.gripper_state = self.gripper_open
        self.gripper_target = self.gripper_open
        self.gripper_counter = 0

        # Toggle active
        self.active = False
        self.prev_button_a = False
        self.prev_button_b = False

        # Config
        self.scalar_xyz = np.array([1.5,1.5,1.5])
        self.scalar_rot = np.array([.5,.5,-.5])
        self.negate_angles = True

        # Buttons
        self.buttons = {}

        # Views
        self.mode = 0
        self.view_idx = 0
        self.views = [(0,270),(30,45),(90,0),(0,0),(0,90)]

        # Keyboard listener
        listener = keyboard.Listener(on_press=self.on_press)
        listener.daemon = True
        listener.start()

        # Haply reader + update thread
        self.reader = HapticReader(
            haptic_feedback=haptic_feedback,
            force_reader=force_reader
        )
        threading.Thread(target=self._update_loop, daemon=True).start()

        # Plot
        self.display_plot = bool(display_plot)
        self.fig = None
        if self.display_plot:
            plt.ion()
            self._init_plot()

    def interpret_rotation(self, quat, mode):
        if mode == 0:
            Rm = R.from_quat(quat).as_matrix()[:, [2, 0, 1]]
            return Rm, R.from_matrix(Rm).as_euler('zyx', True)
        return np.eye(3), np.zeros(3)

    def on_press(self, key):
        c = getattr(key, 'char', None)
        if c == 'a':
            if not self.calibrated:
                T = np.eye(4); T[:3, 3] = self.pose
                self.haply_to_global = np.linalg.inv(T)
                self.calibrated = True
            else:
                self.mode = (self.mode + 1) % len(self.views)
        elif c == 'z' and self.display_plot and hasattr(self, 'ax'):
            self.view_idx = (self.view_idx + 1) % len(self.views)
            elev, azim = self.views[self.view_idx]
            self.ax.view_init(elev=elev, azim=azim)
        elif key == keyboard.Key.esc:
            self.stop()
            if self.display_plot and self.fig is not None:
                plt.close(self.fig)

    def _update_loop(self):
        while self.running:
            poses, buttons = self.reader.get_device_state()
            self.buttons = buttons or {}

            # Active toggle (A) — rising edge
            a_pressed = bool(self.buttons.get('a', False))
            if a_pressed and not self.prev_button_a:
                self.active = not self.active
            self.prev_button_a = a_pressed


            # Gripper toggle (B) — rising edge
            b_pressed = bool(self.buttons.get('b', False))
            if b_pressed and not self.prev_button_b:
                # flip target between open and closed
                self.gripper_target = self.gripper_open #if self.gripper_target == self.gripper_open else self.gripper_open
                self.gripper_counter = self.gripper_steps
            self.prev_button_b = b_pressed

            # Pose/orient update
            if poses:
                self.quat = poses['orientation']
                pos = poses['position']
                rot, _ = self.interpret_rotation(self.quat, self.mode)
                T = np.eye(4); T[:3, 3] = pos; T[:3, :3] = rot
                if self.calibrated:
                    T = self.haply_to_global @ T
                self.pose = T[:3, 3]
                self.orient = T[:3, :3]

            # # Gripper smoothing
            # if self.gripper_counter > 0:
            #     step = (self.gripper_target - self.gripper_state) / self.gripper_counter
            #     self.gripper_state += step
            #     self.gripper_counter -= 1

            time.sleep(0.05)
    
    """
    def _update_loop(self):
        while self.running:
            poses, buttons = self.reader.get_device_state()
            self.buttons = buttons or {}

            # Active toggle (A) — rising edge
            self.active = True
            #a_pressed = bool(self.buttons.get('a', False))
            #if a_pressed and not self.prev_button_a:
            #    self.active = not self.active
            #self.prev_button_a = a_pressed

            # Gripper toggle (B) — rising edge
            #b_pressed = bool(self.buttons.get('b', False))

            self.gripper_target = self.gripper_closed
            
            #if b_pressed and not self.prev_button_b:
                # flip target between open and closed
            #    self.gripper_target = self.gripper_closed if self.gripper_target == self.gripper_open else self.gripper_open
            #    self.gripper_counter = self.gripper_steps
            #self.prev_button_b = b_pressed

            # Pose/orient update
            if poses:
                self.quat = poses['orientation']
                pos = poses['position']
                rot, _ = self.interpret_rotation(self.quat, self.mode)
                T = np.eye(4); T[:3, 3] = pos; T[:3, :3] = rot
                if self.calibrated:
                    T = self.haply_to_global @ T
                self.pose = T[:3, 3]
                self.orient = T[:3, :3]

            # # Gripper smoothing
            # if self.gripper_counter > 0:
            #     step = (self.gripper_target - self.gripper_state) / self.gripper_counter
            #     self.gripper_state += step
            #     self.gripper_counter -= 1

            time.sleep(0.05) 
    
    """



    def _init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set(xlim=(-0.2, 0.2), ylim=(-0.2, 0.2), zlim=(0, 0.2),
                    xlabel='X', ylabel='Y', zlabel='Z')
        elev, azim = self.views[self.view_idx]
        self.ax.view_init(elev=elev, azim=azim)

        xx, yy = np.meshgrid(np.linspace(-0.2, 0.2, 10), np.linspace(-0.2, 0.2, 10))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.5)

        r, h = 0.01, 0.05
        th = np.linspace(0, 2*np.pi, 20)
        zc = np.array([0, -h])
        tg, zcg = np.meshgrid(th, zc)
        self.xc = r*np.cos(tg); self.yc = r*np.sin(tg); self.zg = zcg

        o = self.pose
        self.pt, = self.ax.plot([o[0]], [o[1]], [o[2]], 'ko')
        self.lines = []
        for i, style in enumerate(['r-', 'g-', 'b-']):
            end = o + self.orient[:, i]*0.05
            ln, = self.ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], style)
            self.lines.append(ln)

        pts = np.vstack([self.xc.flatten(), self.yc.flatten(), self.zg.flatten()])
        pw = (self.orient @ pts).T + o
        X = pw[:, 0].reshape(self.zg.shape); Y = pw[:, 1].reshape(self.zg.shape); Z = pw[:, 2].reshape(self.zg.shape)
        self.cyl = self.ax.plot_surface(X, Y, Z, alpha=0.7)
        self.ani = FuncAnimation(self.fig, self._animate, interval=50)
        self.fig.canvas.draw()

    def _animate(self, _):
        if not self.display_plot:
            return
        o = self.pose
        ends = [o + self.orient[:, i]*0.05 for i in range(3)]
        self.pt.set_data([o[0]], [o[1]]); self.pt.set_3d_properties([o[2]])
        for ln, end in zip(self.lines, ends):
            ln.set_data([o[0], end[0]], [o[1], end[1]]); ln.set_3d_properties([o[2], end[2]])
        if self.cyl: self.cyl.remove()
        pts = np.vstack([self.xc.flatten(), self.yc.flatten(), self.zg.flatten()])
        pw = (self.orient @ pts).T + o
        X = pw[:, 0].reshape(self.zg.shape); Y = pw[:, 1].reshape(self.zg.shape); Z = pw[:, 2].reshape(self.zg.shape)
        self.cyl = self.ax.plot_surface(X, Y, Z, alpha=0.7)
        self.fig.canvas.draw_idle()
        return [self.pt] + self.lines + [self.cyl]

    def get_action(self, mode='euler', seq='zyx', degrees=False):
        dp = np.zeros(3); da = np.zeros(3); dq = np.zeros(4)
        if self.active:
            dp = self.pose - self.prev_pose
            if mode == 'quat':
                cq = self.quat.copy()
                dq = (R.from_quat(cq) * R.from_quat(self.prev_quat).inv()).as_quat()
            else:
                ce = R.from_matrix(self.orient).as_euler(seq, degrees)
                da = (ce - self.prev_euler) * self.scalar_rot
                if self.negate_angles:
                    da = -da

        dp = dp * self.scalar_xyz
        dp = np.array([dp[1], -dp[0], dp[2]])  # swap XY

        self.prev_pose = self.pose.copy()
        self.prev_quat = self.quat.copy()
        self.prev_euler = R.from_matrix(self.orient).as_euler(seq, degrees)

        # Use current smoothed gripper_state
        g = float(self.gripper_target)

        if mode == 'quat':
            return np.concatenate([dp, dq, [g]])
        else:
            # print(np.concatenate([dp, da, [g]]))
            return np.concatenate([dp, da, [g]])
        

    def stop(self):
        self.running = False
        try:
            self.reader.stop()
        except Exception:
            pass
