#!/bin/bash
# Setup script to fix SpaceMouse permissions

echo "Creating udev rules for SpaceMouse..."

cat << 'EOF' | sudo tee /etc/udev/rules.d/90-spacemouse.rules
# 3Dconnexion Space Mouse Pro
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62b", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62b", MODE="0666", GROUP="plugdev"
# Additional 3Dconnexion devices
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666", GROUP="plugdev"
# Logitech 3D devices
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", MODE="0666", GROUP="plugdev"
EOF

echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "Done! Now unplug and replug your SpaceMouse, or reboot."
echo "After that, you can run without sudo:"
echo "python3 teleop_with_recording.py --environment Lift --robots Panda --device spacemouse --control-freq 50 --video-path Lift_teleop.mp4"
