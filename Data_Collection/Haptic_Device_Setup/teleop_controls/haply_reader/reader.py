import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from misc.subprocess_utils import run_threaded_command
import threading
import time
import orjson
from websockets.sync.client import connect  # Requires websockets >= 12

class HapticReader:
    def __init__(self, uri='ws://localhost:10001', force={"x": 0, "y": 0, "z": 0}):
        self.uri = uri
        self.force = force
        self.poses = {}
        self.buttons = {}
        self.running = True
        self._lock = threading.Lock()
        run_threaded_command(self.connect_and_read)

    def connect_and_read(self):
        first_message = True
        inverse3_device_id = None

        try:
            with connect(self.uri) as ws:
                while self.running:
                    response = ws.recv()
                    data = orjson.loads(response)
                    inverse3_devices = data.get("inverse3", [])
                    verse_grip_devices = data.get("wireless_verse_grip", [])

                    inverse3_data = inverse3_devices[0] if inverse3_devices else {}
                    verse_grip_data = verse_grip_devices[0] if verse_grip_devices else {}

                    if first_message:
                        first_message = False
                        if not inverse3_data:
                            print("No Inverse3 device found.")
                            break
                        if not verse_grip_data:
                            print("No Wireless Verse Grip device found.")
                        inverse3_device_id = inverse3_data.get("device_id")
                        handedness = inverse3_data.get("config", {}).get("handedness")
                        print(f"Inverse3 device ID: {inverse3_device_id}, Handedness: {handedness}")

                    position_raw = inverse3_data["state"].get("cursor_position", {})
                    velocity_raw = inverse3_data["state"].get("cursor_velocity", {})
                    buttons = verse_grip_data.get("state", {}).get("buttons", {})
                    orientation_raw = verse_grip_data.get("state", {}).get("orientation", {})

                    position = [
                        position_raw.get("x", 0.0),
                        position_raw.get("y", 0.0),
                        position_raw.get("z", 0.0)
                    ]
                    orientation = [
                        orientation_raw.get("x", 0.0),
                        orientation_raw.get("y", 0.0),
                        orientation_raw.get("z", 0.0),
                        orientation_raw.get("w", 1.0)
                    ]
                    velocity = [
                        velocity_raw.get("x", 0.0),
                        velocity_raw.get("y", 0.0),
                        velocity_raw.get("z", 0.0)
                    ]

                    with self._lock:
                        self.poses = {
                            "position": position,
                            "velocity": velocity,
                            "orientation": orientation
                        }
                        self.buttons = buttons

                    request_msg = {
                        "inverse3": [
                            {
                                "device_id": inverse3_device_id,
                                "commands": {
                                    "set_cursor_force": {
                                        "values": self.force
                                    }
                                }
                            }
                        ]
                    }

                    ws.send(orjson.dumps(request_msg))

        except Exception as e:
            print(f"WebSocket connection error: {e}")

    def stop(self):
        self.running = False

    def get_device_state(self):
        with self._lock:
            return self.poses, self.buttons

def main():
    haptic_reader = HapticReader()
    # Start the threaded WebSocket reader
    try:
        while haptic_reader.running:
            poses, buttons = haptic_reader.get_device_state()
            print("Poses:", poses)
            print("Buttons:", buttons)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping...")
        haptic_reader.stop()

if __name__ == '__main__':
    main()
