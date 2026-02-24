#!/bin/bash
#===============================================================================
# Run RF-DETR Inference from Local Mac (with auto-sync)
#===============================================================================
#
# Local wrapper that submits an inference job to the GPU cluster via SSH,
# streams the job output, waits for completion, and syncs results locally.
#
# Accepts the same arguments as submit_inference.sh.
#
# Usage:
#   ./run_inference.sh --project data/projects/One --run rfdetr_run_1
#   ./run_inference.sh --project data/projects/One --latest
#   ./run_inference.sh --project data/projects/One --latest --no-sync
#   ./run_inference.sh --project data/projects/One --run my_run --track --frame-interval 5
#
# Requires:
#   - SSH access to the GPU cluster
#   - SSHFS mount at gpu-server/ (run ./mount_gpu.sh) for result syncing
#
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#-------------------------------------------------------------------------------
# SSH / Remote Configuration
#-------------------------------------------------------------------------------
SSH_USER="youngjin"
SSH_HOST="xlogin.comp.nus.edu.sg"
SSH_DEST="${SSH_USER}@${SSH_HOST}"
REMOTE_DIR="/home/y/youngjin/batman"
MOUNT_POINT="$SCRIPT_DIR/gpu-server"
CONTROL_PATH="/tmp/batman_run_ssh_%r@%h:%p"
SSH_OPTS="-o ControlPath=$CONTROL_PATH -o ConnectTimeout=10"

#-------------------------------------------------------------------------------
# Default Configuration (mirrors submit_inference.sh)
#-------------------------------------------------------------------------------
GPU_TYPE="a100-80"
PARTITION=""
TIME="04:00:00"
RUN=""
LATEST=false
PROJECT=""
VIDEO=""
TEST_ONLY=false
MODEL="base"
CONFIDENCE=0.5
FRAME_INTERVAL=1
TRACK=false
NO_KALMAN=false
NO_OPTIMIZE=false
NO_VIDEO=false
TRACK_THRESH=0.25
TRACK_BUFFER=30
MATCH_THRESH=0.8
DRY_RUN=false
NO_SYNC=false
NO_PUSH=false
EXTRA_ARGS=""

POLL_INTERVAL=15

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Local inference runner. Submits to GPU cluster via SSH, waits for"
    echo "completion, and syncs results to your local project directory."
    echo ""
    echo "Required:"
    echo "  --project=PATH, -p   Path to Batman project"
    echo ""
    echo "Run Selection (one required):"
    echo "  --run=NAME           Training run name from project/runs/"
    echo "  --latest             Use the most recent run"
    echo ""
    echo "Video Selection (optional):"
    echo "  --video=ID           Specific video source_key(s) (space-separated)"
    echo "  --test-only          Only run on test-only videos"
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE           GPU type (default: h100-96)"
    echo "  --time=HH:MM:SS     Time limit (default: 04:00:00)"
    echo ""
    echo "Inference Options:"
    echo "  --model=SIZE         Model size: base, large (default: base)"
    echo "  --confidence=N       Confidence threshold (default: 0.5)"
    echo "  --no-optimize        Skip model optimization"
    echo ""
    echo "Video Options:"
    echo "  --frame-interval=N   Run inference every N frames (default: 1)"
    echo "  --track              Enable ByteTrack tracking"
    echo "  --no-kalman          Disable Kalman prediction on non-keyframes"
    echo "  --no-video           Don't save annotated output video"
    echo "  --track-thresh=N     ByteTrack detection threshold (default: 0.25)"
    echo "  --track-buffer=N     Frames to keep lost tracks (default: 30)"
    echo "  --match-thresh=N     IoU threshold for matching (default: 0.8)"
    echo ""
    echo "Sync Options:"
    echo "  --no-sync            Submit and wait, but skip syncing results locally"
    echo "  --no-push            Skip pushing manual_data + project.json to GPU before job"
    echo ""
    echo "Other:"
    echo "  --dry-run            Show generated SLURM script without submitting"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --project data/projects/One --latest"
    echo "  $0 --project data/projects/One --run rfdetr_run_1 --track"
    echo "  $0 --project data/projects/One --latest --no-sync"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu=*)           GPU_TYPE="${1#*=}"; shift ;;
        --gpu)             GPU_TYPE="$2"; shift 2 ;;
        --partition=*)     PARTITION="${1#*=}"; shift ;;
        --partition)       PARTITION="$2"; shift 2 ;;
        --time=*)          TIME="${1#*=}"; shift ;;
        --time)            TIME="$2"; shift 2 ;;
        --run=*|-r=*)      RUN="${1#*=}"; shift ;;
        --run|-r)          RUN="$2"; shift 2 ;;
        --latest)          LATEST=true; shift ;;
        --project=*|-p=*)  PROJECT="${1#*=}"; shift ;;
        --project|-p)      PROJECT="$2"; shift 2 ;;
        --video=*|-v=*)    VIDEO="${1#*=}"; shift ;;
        --video|-v)        VIDEO="$2"; shift 2 ;;
        --test-only)       TEST_ONLY=true; shift ;;
        --model=*)         MODEL="${1#*=}"; shift ;;
        --model)           MODEL="$2"; shift 2 ;;
        --confidence=*|-t=*) CONFIDENCE="${1#*=}"; shift ;;
        --confidence|-t)   CONFIDENCE="$2"; shift 2 ;;
        --frame-interval=*|-n=*) FRAME_INTERVAL="${1#*=}"; shift ;;
        --frame-interval|-n) FRAME_INTERVAL="$2"; shift 2 ;;
        --track)           TRACK=true; shift ;;
        --no-kalman)       NO_KALMAN=true; shift ;;
        --no-optimize)     NO_OPTIMIZE=true; shift ;;
        --no-video)        NO_VIDEO=true; shift ;;
        --track-thresh=*)  TRACK_THRESH="${1#*=}"; shift ;;
        --track-thresh)    TRACK_THRESH="$2"; shift 2 ;;
        --track-buffer=*)  TRACK_BUFFER="${1#*=}"; shift ;;
        --track-buffer)    TRACK_BUFFER="$2"; shift 2 ;;
        --match-thresh=*)  MATCH_THRESH="${1#*=}"; shift ;;
        --match-thresh)    MATCH_THRESH="$2"; shift 2 ;;
        --no-sync)         NO_SYNC=true; shift ;;
        --no-push)         NO_PUSH=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         show_help ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [ -z "$PROJECT" ]; then
    echo "Error: --project is required"
    exit 1
fi

if [ -z "$RUN" ] && [ "$LATEST" = false ]; then
    echo "Error: Must specify --run or --latest"
    exit 1
fi

#-------------------------------------------------------------------------------
# GPU Configuration
#-------------------------------------------------------------------------------
case $GPU_TYPE in
    h200)       PARTITION="${PARTITION:-gpu}";       GRES="gpu:h200:1" ;;
    h100-96)    PARTITION="${PARTITION:-gpu-long}";   GRES="gpu:h100-96:1" ;;
    h100-47)    PARTITION="${PARTITION:-gpu-long}";   GRES="gpu:h100-47:1" ;;
    a100-80)    PARTITION="${PARTITION:-gpu-long}";   GRES="gpu:a100-80:1" ;;
    a100-40)    PARTITION="${PARTITION:-gpu-long}";   GRES="gpu:a100-40:1" ;;
    nv)         PARTITION="${PARTITION:-gpu-short}";  GRES="gpu:nv:1" ;;
    *)          echo "Unknown GPU type: $GPU_TYPE"; exit 1 ;;
esac

#-------------------------------------------------------------------------------
# Check SSHFS Mount (skip if --no-sync or --dry-run)
#-------------------------------------------------------------------------------
if [ "$NO_SYNC" = false ] && [ "$DRY_RUN" = false ]; then
    if ! mount | grep -q "$MOUNT_POINT"; then
        echo "Error: SSHFS mount not found at gpu-server/"
        echo "  Run:  ./mount_gpu.sh"
        echo "  Or:   --no-sync to skip result syncing"
        exit 1
    fi
fi

#-------------------------------------------------------------------------------
# Build Python Arguments
#-------------------------------------------------------------------------------
RUN_ARG=""
if [ -n "$RUN" ]; then
    RUN_ARG="--run ${RUN}"
elif [ "$LATEST" = true ]; then
    RUN_ARG="--latest"
fi

VIDEO_ARG=""
if [ -n "$VIDEO" ]; then
    VIDEO_ARG="--video ${VIDEO}"
fi

TRACK_ARGS=""
if [ "$TRACK" = true ]; then
    TRACK_ARGS="--track --track-thresh ${TRACK_THRESH} --track-buffer ${TRACK_BUFFER} --match-thresh ${MATCH_THRESH}"
fi

OPT_FLAGS=""
[ "$NO_KALMAN" = true ] && OPT_FLAGS="$OPT_FLAGS --no-kalman"
[ "$NO_OPTIMIZE" = true ] && OPT_FLAGS="$OPT_FLAGS --no-optimize"
[ "$NO_VIDEO" = true ] && OPT_FLAGS="$OPT_FLAGS --no-video"
[ "$TEST_ONLY" = true ] && OPT_FLAGS="$OPT_FLAGS --test-only"

#-------------------------------------------------------------------------------
# Generate SLURM Script (identical to submit_inference.sh)
#-------------------------------------------------------------------------------
SLURM_SCRIPT=$(mktemp /tmp/inference_slurm_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=rfdetr-inference
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=${TIME}
#SBATCH --output=logs/slurm_%j_inference.out
#SBATCH --error=logs/slurm_%j_inference.err

TOTAL_START=\$(date +%s)

echo "============================================================"
echo "RF-DETR Inference Job (Project-Centric)"
echo "============================================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$SLURMD_NODENAME"
echo "GPU:          ${GPU_TYPE}"
echo "Project:      ${PROJECT}"
echo "Started:      \$(date)"
echo "============================================================"
echo ""

cd ~/batman
source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Starting inference..."
echo ""

INFERENCE_START=\$(date +%s)

python3 -m cli.inference \\
    --project ${PROJECT} \\
    ${RUN_ARG} \\
    ${VIDEO_ARG} \\
    --model ${MODEL} \\
    --confidence ${CONFIDENCE} \\
    --frame-interval ${FRAME_INTERVAL} \\
    ${TRACK_ARGS} \\
    ${OPT_FLAGS} \\
    --device cuda \\
    ${EXTRA_ARGS}

INFERENCE_END=\$(date +%s)
INFERENCE_ELAPSED=\$((INFERENCE_END - INFERENCE_START))
TOTAL_END=\$(date +%s)
TOTAL_ELAPSED=\$((TOTAL_END - TOTAL_START))
SETUP_ELAPSED=\$((INFERENCE_START - TOTAL_START))

echo ""
echo "============================================================"
echo "Timing Summary"
echo "============================================================"
echo "Setup time:     \${SETUP_ELAPSED}s"
echo "Inference time: \${INFERENCE_ELAPSED}s"
echo "Total time:     \${TOTAL_ELAPSED}s"
echo "============================================================"
echo "Inference complete! Results under: ${PROJECT}/inference/"
echo "Finished: \$(date)"
echo "============================================================"
SLURM_EOF

#-------------------------------------------------------------------------------
# Display Summary
#-------------------------------------------------------------------------------
echo "============================================================"
echo "RF-DETR Inference (Local Runner)"
echo "============================================================"
echo "GPU:            ${GPU_TYPE}"
echo "Partition:      ${PARTITION}"
echo "Time limit:     ${TIME}"
echo "Project:        ${PROJECT}"
echo "Run:            ${RUN_ARG}"
echo "Video:          ${VIDEO:-all project videos}"
echo "Test-only:      ${TEST_ONLY}"
echo "Frame interval: ${FRAME_INTERVAL}"
echo "Tracking:       ${TRACK}"
echo "Pre-push:       $([ "$NO_PUSH" = true ] && echo "off" || echo "on")"
echo "Auto-sync:      $([ "$NO_SYNC" = true ] && echo "off" || echo "on")"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN - Generated SLURM script:"
    echo "============================================================"
    cat "$SLURM_SCRIPT"
    echo "============================================================"
    rm -f "$SLURM_SCRIPT"
    exit 0
fi

#-------------------------------------------------------------------------------
# SSH Helpers (with reconnect/retry logic)
#-------------------------------------------------------------------------------
MAX_SSH_RETRIES=5
SSH_RETRY_DELAY=10

ensure_ssh() {
    if ! ssh -O check $SSH_OPTS "$SSH_DEST" 2>/dev/null; then
        echo "  Reconnecting SSH to $SSH_HOST..."
        ssh -O exit $SSH_OPTS "$SSH_DEST" 2>/dev/null || true
        ssh -M -f -N -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" \
            -o ControlPersist=30m -o ServerAliveInterval=60 \
            -o ServerAliveCountMax=3 "$SSH_DEST"
    fi
}

remote() {
    ssh $SSH_OPTS "$SSH_DEST" "$@"
}

remote_retry() {
    local attempt=0
    while [ $attempt -lt $MAX_SSH_RETRIES ]; do
        if remote "$@" 2>/dev/null; then
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $attempt -lt $MAX_SSH_RETRIES ]; then
            sleep "$SSH_RETRY_DELAY"
            ensure_ssh 2>/dev/null || true
        fi
    done
    return 1
}

start_streaming() {
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null && wait "$TAIL_PID" 2>/dev/null || true
    [ -n "$ERR_PID" ] && kill "$ERR_PID" 2>/dev/null && wait "$ERR_PID" 2>/dev/null || true
    ensure_ssh 2>/dev/null || true
    ssh $SSH_OPTS "$SSH_DEST" \
        "tail -f '$LOG_REMOTE' 2>/dev/null" 2>/dev/null &
    TAIL_PID=$!
    ssh $SSH_OPTS "$SSH_DEST" \
        "tail -f '$ERR_REMOTE' 2>/dev/null" 2>/dev/null \
        | sed 's/^/[stderr] /' >&2 &
    ERR_PID=$!
}

sync_with_retry() {
    local src="$1" dst="$2"
    shift 2
    local attempt=0
    while [ $attempt -lt $MAX_SSH_RETRIES ]; do
        if rsync "$@" "$src" "$dst"; then
            return 0
        fi
        attempt=$((attempt + 1))
        if [ $attempt -lt $MAX_SSH_RETRIES ]; then
            echo "  Sync failed (attempt $attempt/$MAX_SSH_RETRIES), retrying in ${SSH_RETRY_DELAY}s..."
            sleep "$SSH_RETRY_DELAY"
        fi
    done
    echo "  Warning: Sync failed after $MAX_SSH_RETRIES attempts."
    return 1
}

ensure_ssh

#-------------------------------------------------------------------------------
# Pre-sync: push manual_data, frames metadata, labels, and project.json to GPU
#-------------------------------------------------------------------------------
if [ "$NO_PUSH" = false ]; then
    echo ""
    echo "Pushing project data to GPU..."
    LOCAL_PROJECT="$SCRIPT_DIR/$PROJECT"
    REMOTE_PROJECT="$SSH_DEST:$REMOTE_DIR/$PROJECT"

    if [ -d "$LOCAL_PROJECT/manual_data" ]; then
        rsync -az --progress -e "ssh $SSH_OPTS" \
            "$LOCAL_PROJECT/manual_data/" "$REMOTE_PROJECT/manual_data/"
    fi

    # Sync manual data frame metadata (frames/manual_data*/frames.json)
    for frames_dir in "$LOCAL_PROJECT"/frames/manual_data*; do
        if [ -d "$frames_dir" ]; then
            dir_name=$(basename "$frames_dir")
            rsync -az -e "ssh $SSH_OPTS" \
                "$frames_dir/" "$REMOTE_PROJECT/frames/$dir_name/"
        fi
    done

    # Sync labels (annotations)
    if [ -d "$LOCAL_PROJECT/labels/current" ]; then
        rsync -az -e "ssh $SSH_OPTS" \
            "$LOCAL_PROJECT/labels/current/" "$REMOTE_PROJECT/labels/current/"
    fi

    if [ -f "$LOCAL_PROJECT/project.json" ]; then
        rsync -az -e "ssh $SSH_OPTS" \
            "$LOCAL_PROJECT/project.json" "$REMOTE_PROJECT/project.json"
    fi
    echo "Project data synced."
    echo ""
fi

#-------------------------------------------------------------------------------
# Upload & Submit
#-------------------------------------------------------------------------------
REMOTE_SCRIPT="/tmp/$(basename "$SLURM_SCRIPT")"
scp -q $SSH_OPTS "$SLURM_SCRIPT" "$SSH_DEST:$REMOTE_SCRIPT"
rm -f "$SLURM_SCRIPT"

JOB_OUTPUT=$(remote "cd $REMOTE_DIR && mkdir -p logs && sbatch $REMOTE_SCRIPT" 2>&1)
JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/ {print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    echo "$JOB_OUTPUT"
    remote "rm -f $REMOTE_SCRIPT" 2>/dev/null
    exit 1
fi

echo ""
echo "Job submitted: $JOB_ID"

#-------------------------------------------------------------------------------
# Cleanup Trap
#-------------------------------------------------------------------------------
TAIL_PID=""
ERR_PID=""
cleanup() {
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null && wait "$TAIL_PID" 2>/dev/null || true
    [ -n "$ERR_PID" ] && kill "$ERR_PID" 2>/dev/null && wait "$ERR_PID" 2>/dev/null || true
    remote "rm -f $REMOTE_SCRIPT" 2>/dev/null || true
}
trap cleanup EXIT

#-------------------------------------------------------------------------------
# Stream Log & Wait for Completion
#-------------------------------------------------------------------------------
LOG_REMOTE="$REMOTE_DIR/logs/slurm_${JOB_ID}_inference.out"
ERR_REMOTE="$REMOTE_DIR/logs/slurm_${JOB_ID}_inference.err"

echo "Streaming output (log: gpu-server/logs/slurm_${JOB_ID}_inference.out)"
echo ""

# Wait for job to start (log file created) with queue status updates
WAITED=0
while true; do
    if remote "test -f '$LOG_REMOTE'" 2>/dev/null; then
        break
    fi
    QUEUE_INFO=$(remote "squeue -j $JOB_ID -h -o '%T %r'" 2>/dev/null) || QUEUE_INFO=""
    if [ -n "$QUEUE_INFO" ]; then
        JOB_REASON=$(echo "$QUEUE_INFO" | awk '{print $1, "(" $2 ")"}')
        printf "\r  Waiting for job to start... %s  " "$JOB_REASON"
    elif [ $WAITED -gt 0 ]; then
        printf "\r  Waiting for job to start... (checking)  "
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done
if [ "$WAITED" -gt 0 ]; then
    printf "\r  Job started!%-40s\n" ""
fi

# Start streaming stdout + stderr
start_streaming

# Poll squeue until job finishes (resilient to SSH drops)
SSH_FAIL_COUNT=0
while true; do
    sleep "$POLL_INTERVAL"

    if SQUEUE_OUT=$(remote "squeue -j $JOB_ID -h" 2>/dev/null); then
        SSH_FAIL_COUNT=0
        echo "$SQUEUE_OUT" | grep -q "$JOB_ID" || break
        # Restart log streams if they died
        kill -0 "$TAIL_PID" 2>/dev/null || start_streaming
    else
        SSH_FAIL_COUNT=$((SSH_FAIL_COUNT + 1))
        if [ $SSH_FAIL_COUNT -ge 3 ]; then
            echo ""
            echo "  Connection lost. Reconnecting..."
            if ensure_ssh 2>/dev/null; then
                SSH_FAIL_COUNT=0
                start_streaming
            else
                echo "  Retry failed, will keep trying..."
            fi
        fi
    fi
done

# Let tail flush remaining output
sleep 3
kill "$TAIL_PID" 2>/dev/null; wait "$TAIL_PID" 2>/dev/null || true
TAIL_PID=""
kill "$ERR_PID" 2>/dev/null; wait "$ERR_PID" 2>/dev/null || true
ERR_PID=""

#-------------------------------------------------------------------------------
# Check Job Result (with retry)
#-------------------------------------------------------------------------------
JOB_STATE=$(remote_retry "sacct -j $JOB_ID --format=State --noheader -P | head -1 | tr -d ' '") || JOB_STATE="UNKNOWN"

echo ""

if [[ "$JOB_STATE" == "FAILED" || "$JOB_STATE" == "CANCELLED" || "$JOB_STATE" == "TIMEOUT" || "$JOB_STATE" == "OUT_OF_MEMORY" ]]; then
    echo "============================================================"
    echo "Job $JOB_ID finished with state: $JOB_STATE"
    echo "Error log: gpu-server/logs/slurm_${JOB_ID}_inference.err"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "Job $JOB_ID completed! (state: ${JOB_STATE})"
echo "============================================================"

#-------------------------------------------------------------------------------
# Sync Results (with retry)
#-------------------------------------------------------------------------------
if [ "$NO_SYNC" = true ]; then
    echo ""
    echo "Sync skipped (--no-sync). Results available via SSHFS at:"
    echo "  gpu-server/$PROJECT/inference/"
    exit 0
fi

echo ""
echo "Syncing inference results..."

SRC="$MOUNT_POINT/$PROJECT/inference/"
DST="$SCRIPT_DIR/$PROJECT/inference/"

if [ ! -d "$SRC" ]; then
    echo "Warning: No inference directory at $SRC"
    echo "Results may be available via SSHFS at: gpu-server/$PROJECT/inference/"
    exit 0
fi

mkdir -p "$DST"
sync_with_retry "$SRC" "$DST" -a --progress \
    || echo "  Results available via SSHFS at: gpu-server/$PROJECT/inference/"

echo ""
echo "============================================================"
echo "Results synced to: $PROJECT/inference/"
echo "============================================================"
