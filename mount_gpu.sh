#!/bin/bash
#===============================================================================
# Mount GPU Server via SSHFS
#===============================================================================
# Quick access to your GPU cluster files without rsync
#
# Usage:
#   ./mount_gpu.sh
#
# After mounting:
#   - Access files in ./gpu-server/
#   - Drag & drop videos in Finder
#   - View results without manual sync
#
# Unmount:
#   ./umount_gpu.sh
#===============================================================================

set -e

MOUNT_POINT="$HOME/Projects/batman/gpu-server"
REMOTE_USER="youngjin"  # Change this to your username
REMOTE_HOST="xlogin.comp.nus.edu.sg"
REMOTE_PATH="/home/y/youngjin/batman"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "==============================================="
echo "Mounting GPU Server via SSHFS"
echo "==============================================="

# Check if already mounted
if mount | grep -q "$MOUNT_POINT"; then
    echo -e "${YELLOW}⚠ Already mounted at: $MOUNT_POINT${NC}"
    echo "Use './umount_gpu.sh' to unmount first"
    exit 0
fi

# Create mount point if it doesn't exist
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating mount point: $MOUNT_POINT"
    mkdir -p "$MOUNT_POINT"
fi

# Mount with optimal options
echo "Mounting ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}..."
echo ""

sshfs "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "$MOUNT_POINT" \
    -o auto_cache \
    -o reconnect \
    -o defer_permissions \
    -o noappledouble \
    -o volname=GPU-Server

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully mounted!${NC}"
    echo ""
    echo "Access your GPU server at:"
    echo "  $MOUNT_POINT"
    echo ""
    echo "Quick links:"
    echo "  - Runs:      $MOUNT_POINT/runs/"
    echo "  - Results:   $MOUNT_POINT/inference_results/"
    echo "  - Datasets:  $MOUNT_POINT/datasets/"
    echo ""
    echo "Unmount with: ./umount_gpu.sh"
else
    echo -e "${RED}✗ Mount failed${NC}"
    echo "Check your SSH connection and credentials"
    exit 1
fi
