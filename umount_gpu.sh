#!/bin/bash
#===============================================================================
# Unmount GPU Server
#===============================================================================
# Safely unmount the SSHFS connection to your GPU cluster
#
# Usage:
#   ./umount_gpu.sh
#===============================================================================

set -e

MOUNT_POINT="$HOME/Projects/batman/gpu-server"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "==============================================="
echo "Unmounting GPU Server"
echo "==============================================="

# Check if mounted
if ! mount | grep -q "$MOUNT_POINT"; then
    echo -e "${YELLOW}⚠ Not currently mounted${NC}"
    exit 0
fi

echo "Unmounting $MOUNT_POINT..."

# Try to unmount
umount "$MOUNT_POINT" 2>/dev/null || diskutil unmount "$MOUNT_POINT" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully unmounted${NC}"
else
    echo -e "${RED}✗ Unmount failed, trying force unmount...${NC}"
    umount -f "$MOUNT_POINT" 2>/dev/null || diskutil unmount force "$MOUNT_POINT"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Force unmount successful${NC}"
    else
        echo -e "${RED}✗ Could not unmount${NC}"
        echo "You may need to close apps using the mount point"
        exit 1
    fi
fi

echo ""
echo "Remount with: ./mount_gpu.sh"
