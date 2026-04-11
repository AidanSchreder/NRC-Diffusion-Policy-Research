import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

import sys
from Data_Collection.Haptic_Device_Setup.teleop_controls.haply_reader.reader import HapticReader