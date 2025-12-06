"""
Custom SpaceMouse driver for SpaceMouse Pro that fixes data packet reading issues.
Based on robosuite's SpaceMouse class but with proper handling for product ID 0xc62b (50731).
"""

import threading
import time
import numpy as np
from robosuite.devices import SpaceMouse
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

def to_int16(y1, y2):
    """Convert two 8 bit bytes to a signed 16 bit integer."""
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x

def scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """Normalize raw HID readings to target range."""
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x

def convert(b1, b2):
    """Converts SpaceMouse message to commands."""
    return scale_to_control(to_int16(b1, b2))


class CustomSpaceMouse(SpaceMouse):
    """
    Custom SpaceMouse driver that properly handles SpaceMouse Pro (0xc62b).
    This fixes the IndexError by correctly parsing the device's data packets.
    """

    def __init__(self, env, vendor_id=1133, product_id=50731, pos_sensitivity=1.0, rot_sensitivity=1.0):
        print(f"Initializing CustomSpaceMouse with vendor_id={vendor_id}, product_id={product_id}")

        # Initialize parent class but we'll override the thread
        # Call Device.__init__ directly to avoid SpaceMouse's thread start
        from robosuite.devices import Device
        Device.__init__(self, env)

        print("Opening SpaceMouse device")
        self.vendor_id = vendor_id
        self.product_id = product_id

        import hid
        self.device = hid.device()
        try:
            self.device.open(self.vendor_id, self.product_id)
        except OSError as e:
            ROBOSUITE_DEFAULT_LOGGER.warning(
                "Failed to open SpaceMouse device. "
                "Consider killing other processes that may be using the device such as 3DconnexionHelper (killall 3DconnexionHelper)"
            )
            raise

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()

        self.single_click_and_hold = False
        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False

        # Launch our custom listener thread
        self.thread = threading.Thread(target=self.run_custom)
        self.thread.daemon = True
        self.thread.start()

        # Keyboard listener for aux controls
        from pynput.keyboard import Listener
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        print("CustomSpaceMouse initialization complete!")

    def run_custom(self):
        """
        Custom listener method that properly handles SpaceMouse Pro data packets.
        This fixes the IndexError by checking packet length before accessing indices.
        """
        t_last_click = -1
        packet_count = 0
        error_count = 0

        print("SpaceMouse listener thread started")

        while True:
            try:
                # Check if disabled
                if not self._enabled and error_count > 0:
                    print("SpaceMouse disabled, stopping thread")
                    break

                # Read data from device - try different packet sizes
                d = self.device.read(13)

                if d is not None and len(d) > 0:
                    packet_count += 1

                    # Debug: print first few packets
                    if packet_count <= 5:
                        print(f"Packet {packet_count}: len={len(d)}, type={d[0]}, data={d[:min(13, len(d))]}")

                    if not self._enabled:
                        continue

                    # Handle different packet types based on first byte
                    if d[0] == 1:  # 6-DOF sensor data
                        # Check if we have enough data
                        if len(d) >= 7:
                            # Position data (bytes 1-6)
                            self.y = convert(d[1], d[2])
                            self.x = convert(d[3], d[4])
                            self.z = convert(d[5], d[6]) * -1.0

                            # Check if rotation data is in same packet (13 bytes total)
                            if len(d) >= 13:
                                # All 6-DOF in one packet (newer models)
                                self.roll = convert(d[7], d[8])
                                self.pitch = convert(d[9], d[10])
                                self.yaw = convert(d[11], d[12])
                            else:
                                # Rotation comes in separate packet (older models)
                                # Keep previous rotation values
                                pass

                            self._control = [
                                self.x,
                                self.y,
                                self.z,
                                self.roll,
                                self.pitch,
                                self.yaw,
                            ]

                            # Debug: print control values periodically
                            if packet_count % 50 == 0:
                                print(f"Control: pos=({self.x:.3f}, {self.y:.3f}, {self.z:.3f}), "
                                      f"rot=({self.roll:.3f}, {self.pitch:.3f}, {self.yaw:.3f})")

                    elif d[0] == 2:  # Rotation data (separate packet for older models)
                        if len(d) >= 7:
                            self.roll = convert(d[1], d[2])
                            self.pitch = convert(d[3], d[4])
                            self.yaw = convert(d[5], d[6])

                            self._control = [
                                self.x,
                                self.y,
                                self.z,
                                self.roll,
                                self.pitch,
                                self.yaw,
                            ]

                    elif d[0] == 3:  # Button data
                        if len(d) >= 2:
                            # Left button (grasp)
                            if d[1] == 1:
                                t_click = time.time()
                                elapsed_time = t_click - t_last_click
                                t_last_click = t_click
                                self.single_click_and_hold = True
                                print("Left button pressed - closing gripper")

                            # Release left button
                            if d[1] == 0:
                                self.single_click_and_hold = False
                                print("Left button released - opening gripper")

                            # Right button (reset)
                            if d[1] == 2:
                                print("Right button pressed - resetting")
                                self._reset_state = 1
                                self._enabled = False
                                self._reset_internal_state()

            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"Error in SpaceMouse thread: {e}")
                    import traceback
                    traceback.print_exc()
                if error_count > 100:
                    print("Too many errors, stopping SpaceMouse thread")
                    break
