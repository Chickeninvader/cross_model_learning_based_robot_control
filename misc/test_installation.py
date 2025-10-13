#!/usr/bin/env python3
"""
Test script to verify robosuite installation
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")

    try:
        import robosuite
        print(f"✓ robosuite {robosuite.__version__}")
    except ImportError as e:
        print(f"✗ robosuite: {e}")
        return False

    try:
        import mujoco
        print(f"✓ mujoco")
    except ImportError as e:
        print(f"✗ mujoco: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False

    try:
        import h5py
        print(f"✓ h5py {h5py.__version__}")
    except ImportError as e:
        print(f"✗ h5py: {e}")
        return False

    try:
        import hid
        print(f"✓ hidapi (SpaceMouse support)")
    except ImportError:
        print(f"⚠ hidapi not installed (SpaceMouse will not work)")

    return True

def test_environment_creation():
    """Test creating a simple environment"""
    print("\nTesting environment creation...")

    try:
        import robosuite as suite

        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
        )

        print("✓ Environment created successfully")

        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successful (obs keys: {list(obs.keys())[:3]}...)")

        # Test step
        import numpy as np
        action_dim = env.robots[0].action_dim
        action = np.zeros(action_dim)
        obs, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spacemouse():
    """Test SpaceMouse connection"""
    print("\nTesting SpaceMouse connection...")

    try:
        import hid

        # SpaceMouse vendor and product IDs
        # Support both 3Dconnexion and Logitech vendor IDs
        VENDOR_IDS = [0x256F, 0x046D]  # 3Dconnexion, Logitech

        # Common SpaceMouse product IDs
        PRODUCT_IDS = {
            0xC62E: "SpaceMouse Pro",
            0xC62F: "SpaceMouse Pro Wireless",
            0xC631: "SpaceMouse Pro Wireless",
            0xC632: "SpaceMouse Pro",
            0xC633: "SpaceMouse Enterprise",
            0xC635: "SpaceMouse Compact",
            0xC636: "SpaceMouse Module",
            0xC62B: "3Dconnexion SpaceMouse Pro",
            0xC626: "3Dconnexion SpaceMouse",
            0xC628: "3Dconnexion SpaceMouse Pro Wireless",
        }

        # List all HID devices
        devices = hid.enumerate()
        spacemouse_found = False

        for device in devices:
            if device['vendor_id'] in VENDOR_IDS:
                product_name = PRODUCT_IDS.get(device['product_id'], f"Unknown (0x{device['product_id']:04X})")
                print(f"✓ Found: {product_name}")
                print(f"  Vendor ID: 0x{device['vendor_id']:04X}")
                print(f"  Product ID: 0x{device['product_id']:04X}")
                spacemouse_found = True

        if not spacemouse_found:
            print("⚠ No SpaceMouse device detected")
            print("  Make sure your SpaceMouse is connected")
            print("  On Linux, you may need to run spacenavd daemon")
            return False

        return True

    except ImportError:
        print("⚠ hidapi not installed - cannot detect SpaceMouse")
        return False
    except Exception as e:
        print(f"⚠ SpaceMouse detection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Robosuite Installation Test")
    print("=" * 50)

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test environment creation
    if not test_environment_creation():
        success = False

    # Test SpaceMouse (non-critical)
    test_spacemouse()

    print("\n" + "=" * 50)
    if success:
        print("✓ All critical tests passed!")
        print("=" * 50)
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
