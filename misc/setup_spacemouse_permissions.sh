#!/bin/bash

# Setup SpaceMouse permissions for Linux
# This script creates a udev rule to allow non-root access to SpaceMouse devices

echo "Setting up SpaceMouse permissions..."

# Create udev rule file
UDEV_RULE="/etc/udev/rules.d/90-spacemouse.rules"

echo "Creating udev rule: $UDEV_RULE"

# Create the rule (supports both 3Dconnexion and Logitech vendor IDs)
sudo tee $UDEV_RULE > /dev/null << 'EOF'
# 3Dconnexion SpaceMouse devices
# 3Dconnexion vendor ID
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="input", ATTRS{idVendor}=="256f", MODE="0666", GROUP="plugdev"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", MODE="0666", GROUP="plugdev"

# Logitech vendor ID (Logitech owns 3Dconnexion)
SUBSYSTEM=="usb", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62*", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="input", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62*", MODE="0666", GROUP="plugdev"
KERNEL=="hidraw*", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62*", MODE="0666", GROUP="plugdev"
EOF

echo "Udev rule created successfully"

# Reload udev rules
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Done!"
echo ""
echo "Please unplug and replug your SpaceMouse for changes to take effect."
echo "Alternatively, run: sudo udevadm trigger"
