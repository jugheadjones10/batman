#!/bin/bash
#===============================================================================
# Submit RF-DETR Inference Job to SLURM (Project-Centric)
#===============================================================================
#
# All inference is project-centric: --project is required.
# Results are saved under {project}/inference/{run_name}/{video_id}/.
#
# Usage:
#   ./submit_inference.sh --project data/projects/CraneHook --run rfdetr_run_1
#   ./submit_inference.sh --project data/projects/CraneHook --latest
#   ./submit_inference.sh --project data/projects/CraneHook --run my_run --video video_2
#   ./submit_inference.sh --project data/projects/CraneHook --run my_run --test-only
#   ./submit_inference.sh --project data/projects/CraneHook --run my_run --track --frame-interval 5
#
# GPU Options:
#   --gpu=TYPE    GPU type (h200, h100-96, h100-47, a100-80, a100-40, nv)
#                 Default: a100-40 (inference doesn't need large GPU)
#
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# Default Configuration
#-------------------------------------------------------------------------------
GPU_TYPE="a100-40"
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
EXTRA_ARGS=""

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
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
    echo "  --gpu=TYPE           GPU type (default: a100-40)"
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
    echo "Other:"
    echo "  --dry-run            Show generated script without submitting"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --project data/projects/CraneHook --run rfdetr_run_1"
    echo "  $0 --project data/projects/CraneHook --latest --track --frame-interval 5"
    echo "  $0 --project data/projects/CraneHook --run my_run --test-only"
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
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         show_help ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Validate: --project is required
if [ -z "$PROJECT" ]; then
    echo "Error: --project is required"
    exit 1
fi

# Validate: must have one of --run or --latest
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
# Create SLURM Script
#-------------------------------------------------------------------------------
mkdir -p logs
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
# Submit or Display
#-------------------------------------------------------------------------------
echo "============================================================"
echo "RF-DETR Inference Job (Project-Centric)"
echo "============================================================"
echo "GPU:          ${GPU_TYPE}"
echo "Partition:    ${PARTITION}"
echo "Time limit:   ${TIME}"
echo "Project:      ${PROJECT}"
echo "Run:          ${RUN_ARG}"
echo "Video:        ${VIDEO:-all project videos}"
echo "Test-only:    ${TEST_ONLY}"
echo "Frame interval: ${FRAME_INTERVAL}"
echo "Tracking:     ${TRACK}"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN - Generated SLURM script:"
    echo "============================================================"
    cat "$SLURM_SCRIPT"
    echo "============================================================"
else
    JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
    echo ""
    echo "Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/slurm_${JOB_ID}_inference.out"
    echo "  tail -f logs/slurm_${JOB_ID}_inference.err"
fi

rm -f "$SLURM_SCRIPT"
