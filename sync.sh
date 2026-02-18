#!/bin/bash
# Sync batman project data between local and GPU server.
#
# Usage:
#   ./sync.sh <from> <to> <relative_path> [--dry-run]
#
# Examples:
#   ./sync.sh local gpu data/projects/One
#   ./sync.sh gpu local data/projects/One
#   ./sync.sh local gpu runs --dry-run
#
# Machines: local, gpu

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_LOCAL="$SCRIPT_DIR"
ROOT_GPU="youngjin@xlogin.comp.nus.edu.sg:/home/y/youngjin/batman"

CONTROL_PATH="/tmp/batman_sync_ssh_%r@%h:%p"
SSH_OPTS="-o ControlPath=$CONTROL_PATH"

get_root() {
    case "$1" in
        local) echo "$ROOT_LOCAL" ;;
        gpu)   echo "$ROOT_GPU" ;;
        *)     echo "" ;;
    esac
}

# Ensure SSH master is up (one password prompt)
ensure_ssh() {
    local host="youngjin@xlogin.comp.nus.edu.sg"
    if ! ssh -O check $SSH_OPTS "$host" 2>/dev/null; then
        echo "Opening SSH connection..."
        ssh -M -f -N -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" -o ControlPersist=10m "$host"
    fi
}

# Parse args
FROM=""
TO=""
REL_PATH=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        *)
            if [[ -z "$FROM" ]]; then FROM="$1"
            elif [[ -z "$TO" ]]; then TO="$1"
            elif [[ -z "$REL_PATH" ]]; then REL_PATH="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$FROM" || -z "$TO" || -z "$REL_PATH" ]]; then
    echo "Usage: $0 <from> <to> <relative_path> [--dry-run]"
    echo ""
    echo "  from / to:  local  or  gpu"
    echo "  path:      e.g.  data/projects/One  or  runs"
    echo ""
    echo "Examples:"
    echo "  $0 local gpu data/projects/One"
    echo "  $0 gpu local data/projects/One"
    exit 1
fi

SRC_ROOT=$(get_root "$FROM")
DST_ROOT=$(get_root "$TO")
[[ -z "$SRC_ROOT" ]] && { echo "Error: unknown source '$FROM'"; exit 1; }
[[ -z "$DST_ROOT" ]] && { echo "Error: unknown destination '$TO'"; exit 1; }
[[ "$FROM" == "$TO" ]] && { echo "Error: source and destination must differ"; exit 1; }

if [[ "$FROM" != "local" || "$TO" != "local" ]]; then
    ensure_ssh
fi

# Paths: directory sync with trailing slashes
SRC="${SRC_ROOT}/${REL_PATH}/"
DST="${DST_ROOT}/${REL_PATH}/"

if [[ "$TO" == "local" ]]; then
    mkdir -p "$ROOT_LOCAL/$REL_PATH"
fi

echo "Syncing: $FROM -> $TO"
echo "  $REL_PATH"
echo ""
rsync -avz --progress -e "ssh $SSH_OPTS" $DRY_RUN "$SRC" "$DST"
echo ""
echo "Done."
