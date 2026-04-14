#!/bin/bash
# Fix permissions for Mac launch files
echo "Fixing permissions for Mac launch files..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Remove extended attributes
xattr -cr mac_launch.command 2>/dev/null
xattr -cr mac_launch.sh 2>/dev/null

# Set executable permissions
chmod 755 mac_launch.command
chmod 755 mac_launch.sh

echo "✓ Permissions fixed!"
echo ""
echo "You can now run:"
echo "  - Double-click mac_launch.command"
echo "  - Or run: bash mac_launch.sh"
