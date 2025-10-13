#!/usr/bin/env python3
"""
Custom SpaceMouse device with remapped buttons
Button 1 on SpaceMouse (button 5 in code) toggles gripper
"""

import numpy as np

try:
    import hid
except ImportError:
    print("ERROR: hidapi not installed. Install with: pip install hidapi")
    raise

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix
import threading


class CustomSpaceMouse(Device):
    """
    A customized SpaceMouse device driver with remapped button controls and axes.

    Axis mapping (swapped for easier control):
    - Translation: Y, X, Z (X and Y swapped)
    - Rotation: Pitch, Roll, Yaw (Roll and Pitch swapped)

    Button mapping:
    - Button 1 on SpaceMouse (button 5 in code): Toggle gripper
    - Button 0 (Menu button): Reset simulation
    - Keyboard 'b': Toggle base mode
    - Keyboard 's': Switch active arm
    - Keyboard '=': Switch active robot
    """

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0):
        """
        Initialize the custom SpaceMouse device.

        Args:
            env: Environment instance
            pos_sensitivity (float): Position control sensitivity
            rot_sensitivity (float): Rotation control sensitivity
        """
        super().__init__(env)

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # SpaceMouse vendor and product IDs
        # Support both 3Dconnexion and Logitech vendor IDs (Logitech owns 3Dconnexion)
        self.vendor_product_pairs = [
            (0x256F, 0xC62E),  # 3Dconnexion SpaceMouse Pro
            (0x256F, 0xC62F),  # 3Dconnexion SpaceMouse Pro Wireless
            (0x256F, 0xC631),  # 3Dconnexion SpaceMouse Pro Wireless (receiver)
            (0x256F, 0xC632),  # 3Dconnexion SpaceMouse Pro
            (0x256F, 0xC633),  # 3Dconnexion SpaceMouse Enterprise
            (0x046D, 0xC62B),  # Logitech 3Dconnexion SpaceMouse Pro
            (0x046D, 0xC626),  # Logitech 3Dconnexion SpaceMouse
            (0x046D, 0xC628),  # Logitech 3Dconnexion SpaceMouse Pro Wireless
        ]

        # Control state
        self.single_click_and_hold = False  # Gripper state
        self.rotation = np.array([0.0, 0.0, 0.0])
        self.translation = np.array([0.0, 0.0, 0.0])
        self.reset_state = 0

        # Thread for reading device input
        self.device = None
        self._enabled = False
        self._control_thread = None

        # Gripper toggle state
        self.gripper_closed = False
        self.button_pressed = False  # Track button state for toggle

        # Connect to device
        self._connect()

        print(f"CustomSpaceMouse initialized")
        print(f"  Position sensitivity: {pos_sensitivity}")
        print(f"  Rotation sensitivity: {rot_sensitivity}")
        print(f"  Axis mapping: X/Y swapped, Roll/Pitch swapped")
        print(f"  Button 1 (physical) / Button 5 (code): Toggle gripper")

    def _connect(self):
        """Connect to the SpaceMouse device"""
        # Try to find a SpaceMouse device
        for vendor_id, product_id in self.vendor_product_pairs:
            try:
                self.device = hid.device()
                self.device.open(vendor_id, product_id)
                self.device.set_nonblocking(True)

                manufacturer = self.device.get_manufacturer_string()
                product = self.device.get_product_string()
                print(f"Connected to: {manufacturer} {product}")
                print(f"  Vendor ID: 0x{vendor_id:04X}")
                print(f"  Product ID: 0x{product_id:04X}")
                return

            except Exception as e:
                continue

        # If we get here, no device was found
        raise RuntimeError(
            "No SpaceMouse detected!\n"
            "Please make sure your SpaceMouse is connected.\n"
            "On Linux, you may need to run: sudo systemctl start spacenavd"
        )

    def _read_device_data(self):
        """Read raw data from the device"""
        while self._enabled:
            try:
                data = self.device.read(64)
                if data:
                    self._process_device_data(data)
            except Exception as e:
                print(f"Error reading device: {e}")
                break

    def _process_device_data(self, data):
        """
        Process raw device data and extract motion and button states.

        Data format:
        - data[0] = packet type
        - For motion: data[1-6] contain x, y, z translation and rotation
        - For buttons: data[1] contains button state
        """
        if data[0] == 1:  # Translation data
            # Extract translation values with X and Y swapped (y, x, z)
            self.translation = np.array([
                -self._to_int16(data[3], data[4]),  # Y (swapped and inverted: pull up = arm up)
                self._to_int16(data[1], data[2]),   # X (swapped from position 0)
                -self._to_int16(data[5], data[6])   # Z (inverted: push down = arm down)
            ]) * self.pos_sensitivity * 0.0001

        elif data[0] == 2:  # Rotation data
            # Extract rotation values with roll and pitch swapped (pitch, roll, yaw)
            self.rotation = np.array([
                self._to_int16(data[3], data[4]),  # Pitch (swapped from position 1)
                self._to_int16(data[1], data[2]),  # Roll (swapped from position 0)
                self._to_int16(data[5], data[6])   # Yaw (unchanged)
            ]) * self.rot_sensitivity * 0.0001

        elif data[0] == 3:  # Button data
            # Button data is encoded as bit flags in data[2]
            # Menu button uses data[1] = 1
            # Buttons 1-4 use bit flags in data[2]: 16, 32, 64, 128
            button_state = data[1]
            button_flags = data[2]

            # Decode which button was pressed
            if button_state == 1 and button_flags == 0:
                # Menu button pressed - trigger episode end (for recording mode)
                if not self.button_pressed:
                    self.reset_state = 1  # Signal to end episode
                    self.button_pressed = True
                    print("[Menu button] Ending episode...")

            elif button_flags == 16:
                # Button 1 - Toggle gripper
                if not self.button_pressed:
                    self.gripper_closed = not self.gripper_closed
                    self.single_click_and_hold = self.gripper_closed
                    self.button_pressed = True
                    print(f"Gripper {'CLOSED' if self.gripper_closed else 'OPENED'}")

            elif button_flags == 32:
                # Button 2 - No action
                if not self.button_pressed:
                    self.button_pressed = True

            elif button_flags == 64:
                # Button 3 - No action
                if not self.button_pressed:
                    self.button_pressed = True

            elif button_flags == 128:
                # Button 4 - No action
                if not self.button_pressed:
                    self.button_pressed = True

            elif button_state == 0 and button_flags == 0:
                # All buttons released
                self.button_pressed = False

    @staticmethod
    def _to_int16(low_byte, high_byte):
        """Convert two bytes to signed 16-bit integer"""
        value = (high_byte << 8) | low_byte
        if value > 32767:
            value -= 65536
        return value

    def start_control(self):
        """Start the control thread"""
        self._enabled = True
        self._control_thread = threading.Thread(target=self._read_device_data, daemon=True)
        self._control_thread.start()

    def get_controller_state(self):
        """
        Get the current state of the SpaceMouse.

        Returns:
            dict: Dictionary containing translation, rotation, gripper state, and reset flag
        """
        return {
            "translation": self.translation.copy(),
            "rotation": self.rotation.copy(),
            "gripper": 1.0 if self.single_click_and_hold else -1.0,
            "reset": self.reset_state,
        }

    @staticmethod
    def _axis_angle_to_rotation_matrix(axis_angle):
        """Convert axis-angle representation to rotation matrix"""
        angle = np.linalg.norm(axis_angle)
        if angle < 1e-6:
            return np.eye(3)

        axis = axis_angle / angle
        return rotation_matrix(angle, axis)[:3, :3]

    def input2action(self, device_type=None, robot_num=0):
        """
        Convert SpaceMouse input to robot action.

        Args:
            device_type: Not used, kept for compatibility
            robot_num (int): Which robot to control (for multi-robot setups)

        Returns:
            dict: Action dictionary for the robot, or None if reset is triggered
        """
        # Check for reset
        if self.reset_state == 1:
            self.reset_state = 0
            return None

        # Get current state
        state = self.get_controller_state()

        # Build action based on robot configuration
        dpos = state["translation"]  # Position delta
        rotation = state["rotation"]  # Rotation delta
        grasp = state["gripper"]  # Gripper state

        # Convert rotation to delta orientation
        drotation = self._axis_angle_to_rotation_matrix(rotation)

        # Create action for single arm robot
        action_dict = {}

        # For multi-arm environments, determine active robot
        active_robot = self.env.robots[robot_num] if robot_num < len(self.env.robots) else self.env.robots[0]
        robot_name = active_robot.name

        # Build action array: [dy, dx, dz, dpitch, droll, dyaw, gripper]
        # Note: dpos already has swapped axes (Y, X, Z)
        # Note: rotation already has swapped axes (Pitch, Roll, Yaw)
        action = np.concatenate([
            dpos,
            rotation,
            [grasp]
        ])

        action_dict[robot_name] = action

        return action_dict

    def close(self):
        """Close the device connection"""
        self._enabled = False
        if self._control_thread is not None:
            self._control_thread.join(timeout=1.0)
        if self.device is not None:
            self.device.close()
        print("SpaceMouse closed")


if __name__ == "__main__":
    """Test the custom SpaceMouse device"""
    print("Testing Custom SpaceMouse...")
    print("This test will read SpaceMouse input for 10 seconds")
    print("Move the SpaceMouse and press buttons to test")

    import time

    # Create a mock environment for testing
    class MockEnv:
        class MockRobot:
            def __init__(self):
                self.name = "test_robot"
                self.arms = ["right"]

        def __init__(self):
            self.robots = [self.MockRobot()]

    mock_env = MockEnv()

    try:
        # Create device
        device = CustomSpaceMouse(mock_env, pos_sensitivity=1.0, rot_sensitivity=1.0)
        device.start_control()

        print("\nReading input... (Press Ctrl+C to stop)")

        start_time = time.time()
        while time.time() - start_time < 10:
            state = device.get_controller_state()

            # Print non-zero values
            if np.any(np.abs(state["translation"]) > 0.001) or \
               np.any(np.abs(state["rotation"]) > 0.001) or \
               state["gripper"] != -1.0:

                print(f"\rPos: [{state['translation'][0]:6.3f}, {state['translation'][1]:6.3f}, {state['translation'][2]:6.3f}] "
                      f"Rot: [{state['rotation'][0]:6.3f}, {state['rotation'][1]:6.3f}, {state['rotation'][2]:6.3f}] "
                      f"Gripper: {state['gripper']:4.1f}", end="")

            time.sleep(0.01)

        print("\nTest complete!")
        device.close()

    except KeyboardInterrupt:
        print("\nTest interrupted")
        device.close()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
