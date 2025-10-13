#!/bin/bash

# Robosuite Installation Script
# This script installs robosuite and all required dependencies

set -e  # Exit on error

echo "========================================="
echo "Robosuite Installation Script"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install pip3 first."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv robosuite_env
    source robosuite_env/bin/activate
    echo "Virtual environment activated."
fi

echo ""
echo "Installing base dependencies..."
pip3 install --upgrade pip

# Install numpy first (needed for other packages)
pip3 install numpy

# Install MuJoCo
echo ""
echo "Installing MuJoCo..."
pip3 install mujoco

# Install robosuite
echo ""
echo "Installing robosuite..."
pip3 install robosuite

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip3 install h5py  # For demonstration storage
pip3 install imageio  # For video recording
pip3 install imageio-ffmpeg  # For video encoding

# Install SpaceMouse support (Linux/macOS specific)
echo ""
echo "Installing SpaceMouse support..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux detected. Installing spacenav dependencies..."
    # On Linux, may need spacenav driver
    echo "Note: You may need to install spacenavd:"
    echo "  Ubuntu/Debian: sudo apt-get install spacenavd"
    echo "  Fedora: sudo dnf install spacenavd"
    pip3 install hidapi || echo "Warning: hidapi installation failed. SpaceMouse may not work."
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected. Installing hidapi..."
    # Uninstall hid if it exists (conflicts with hidapi)
    pip3 uninstall -y hid 2>/dev/null || true
    pip3 install hidapi
    echo "Note: Please install 3Dconnexion driver from:"
    echo "  https://www.3dconnexion.com/service/drivers.html"
else
    echo "Unknown OS. Attempting hidapi installation..."
    pip3 install hidapi || echo "Warning: hidapi installation failed."
fi

# Install robosuite-models for additional assets
echo ""
echo "Installing robosuite-models..."
pip3 install robosuite-models==1.0.0

# Verify installation
echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="

python3 << EOF
try:
    import robosuite
    print(f"✓ robosuite version: {robosuite.__version__}")

    import mujoco
    print(f"✓ mujoco installed successfully")

    import h5py
    print(f"✓ h5py version: {h5py.__version__}")

    import numpy
    print(f"✓ numpy version: {numpy.__version__}")

    try:
        import hid
        print(f"✓ hidapi installed successfully")
    except ImportError:
        print("⚠ hidapi not installed (SpaceMouse may not work)")

    print("\n✓ Installation completed successfully!")

except ImportError as e:
    print(f"✗ Error: {e}")
    print("Installation may have failed. Please check the error messages above.")
    exit(1)
EOF

echo ""
echo "========================================="
echo "Installation Summary"
echo "========================================="
echo "Robosuite has been installed successfully!"
echo ""
echo "Next steps:"
echo "1. If you created a virtual environment, activate it with:"
echo "   source robosuite_env/bin/activate"
echo ""
echo "2. Connect your SpaceMouse Pro device"
echo ""
echo "3. Run the test scripts to verify everything works:"
echo "   python3 test_installation.py"
echo ""
echo "For SpaceMouse on Linux, make sure spacenavd is running:"
echo "   sudo systemctl start spacenavd"
echo "   sudo systemctl enable spacenavd"
echo ""
echo "========================================="
