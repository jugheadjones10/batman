#!/bin/bash
#===============================================================================
# Submit RF-DETR Inference Job to SLURM
#===============================================================================
#
# Usage:
#   ./submit_inference.sh --run rfdetr_h200_20260120_105925 --project data/projects/CraneHook --input video.mp4
#   ./submit_inference.sh --latest --project data/projects/CraneHook --input video.mp4
#   ./submit_inference.sh --checkpoint runs/my_run/best.pth --input images/*.jpg
#   ./submit_inference.sh --run my_run -p data/projects/MyProject --input video.mp4 --frame-interval 5 --track
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
CHECKPOINT=""
RUN=""
LATEST=false
PROJECT=""
INPUT_FILES=""
OUTPUT_DIR=""
MODEL="base"
CONFIDENCE=0.5
FRAME_INTERVAL=1
TRACK=false
NO_KALMAN=false
NO_OPTIMIZE=false
TRACK_THRESH=0.25
TRACK_BUFFER=30
MATCH_THRESH=0.8
DRY_RUN=false
CLASSES=""
EXTRA_ARGS=""

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Model Selection (one required):"
    echo "  --run=NAME           Run name (auto-finds checkpoint in runs/<name>/)"
    echo "  --latest             Use the most recent run in runs/"
    echo "  --checkpoint=PATH    Explicit path to model checkpoint"
    echo ""
    echo "Class Names (recommended):"
    echo "  --project=PATH, -p   Load class names from a Batman project"
    echo "  --classes=NAMES      Manually specify class names (space-separated)"
    echo ""
    echo "Required:"
    echo "  --input=FILES        Input image(s) or video file(s)"
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE           GPU type (default: a100-40)"
    echo "  --time=HH:MM:SS      Time limit (default: 04:00:00)"
    echo ""
    echo "Inference Options:"
    echo "  --output=PATH        Output directory (default: inference_results/<timestamp>)"
    echo "  --model=SIZE         Model size: base, large (default: base)"
    echo "  --confidence=N       Confidence threshold (default: 0.5)"
    echo "  --no-optimize        Skip model optimization (faster startup)"
    echo ""
    echo "Video Options:"
    echo "  --frame-interval=N   Run inference every N frames (default: 1)"
    echo "  --track              Enable ByteTrack tracking"
    echo "  --no-kalman          Disable Kalman prediction on non-keyframes"
    echo "  --track-thresh=N     ByteTrack detection threshold (default: 0.25)"
    echo "  --track-buffer=N     Frames to keep lost tracks (default: 30)"
    echo "  --match-thresh=N     IoU threshold for matching (default: 0.8)"
    echo ""
    echo "Other:"
    echo "  --dry-run            Show generated script without submitting"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  # Use run name + project for classes"
    echo "  $0 --run rfdetr_h200_20260120_105925 --project data/projects/CraneHook --input video.mp4"
    echo ""
    echo "  # Use latest run with tracking"
    echo "  $0 --latest -p data/projects/CraneHook --input video.mp4 --frame-interval 5 --track"
    echo ""
    echo "  # Explicit checkpoint"
    echo "  $0 --checkpoint runs/run1/best.pth --classes crane_hook --input video.mp4"
    exit 0
}

for arg in "$@"; do
    case $arg in
        --gpu=*)           GPU_TYPE="${arg#*=}" ;;
        --partition=*)     PARTITION="${arg#*=}" ;;
        --time=*)          TIME="${arg#*=}" ;;
        --checkpoint=*)    CHECKPOINT="${arg#*=}" ;;
        --run=*)           RUN="${arg#*=}" ;;
        --latest)          LATEST=true ;;
        --project=*|-p=*)  PROJECT="${arg#*=}" ;;
        -p)                shift; PROJECT="$1" ;;
        --input=*)         INPUT_FILES="${arg#*=}" ;;
        --output=*)        OUTPUT_DIR="${arg#*=}" ;;
        --model=*)         MODEL="${arg#*=}" ;;
        --confidence=*)    CONFIDENCE="${arg#*=}" ;;
        --frame-interval=*) FRAME_INTERVAL="${arg#*=}" ;;
        --track)           TRACK=true ;;
        --no-kalman)       NO_KALMAN=true ;;
        --no-optimize)     NO_OPTIMIZE=true ;;
        --track-thresh=*)  TRACK_THRESH="${arg#*=}" ;;
        --track-buffer=*)  TRACK_BUFFER="${arg#*=}" ;;
        --match-thresh=*)  MATCH_THRESH="${arg#*=}" ;;
        --classes=*)       CLASSES="${arg#*=}" ;;
        --dry-run)         DRY_RUN=true ;;
        --help|-h)         show_help ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# Validate: must have one of --checkpoint, --run, or --latest
if [ -z "$CHECKPOINT" ] && [ -z "$RUN" ] && [ "$LATEST" = false ]; then
    echo "Error: Must specify one of --checkpoint, --run, or --latest"
    exit 1
fi

if [ -z "$INPUT_FILES" ]; then
    echo "Error: --input is required"
    exit 1
fi

# Generate output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="inference_results/${TIMESTAMP}"
fi

#-------------------------------------------------------------------------------
# GPU Configuration
#-------------------------------------------------------------------------------
case $GPU_TYPE in
    h200)
        PARTITION="${PARTITION:-gpu}"
        GRES="gpu:h200:1"
        ;;
    h100-96)
        PARTITION="${PARTITION:-gpu-long}"
        GRES="gpu:h100-96:1"
        ;;
    h100-47)
        PARTITION="${PARTITION:-gpu-long}"
        GRES="gpu:h100-47:1"
        ;;
    a100-80)
        PARTITION="${PARTITION:-gpu-long}"
        GRES="gpu:a100-80:1"
        ;;
    a100-40)
        PARTITION="${PARTITION:-gpu-long}"
        GRES="gpu:a100-40:1"
        ;;
    nv)
        PARTITION="${PARTITION:-gpu-short}"
        GRES="gpu:nv:1"
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE"
        exit 1
        ;;
esac

#-------------------------------------------------------------------------------
# Build Model Selection Argument
#-------------------------------------------------------------------------------
MODEL_ARG=""
if [ -n "$CHECKPOINT" ]; then
    MODEL_ARG="--checkpoint ${CHECKPOINT}"
elif [ -n "$RUN" ]; then
    MODEL_ARG="--run ${RUN}"
elif [ "$LATEST" = true ]; then
    MODEL_ARG="--latest"
fi

#-------------------------------------------------------------------------------
# Build Class Names Argument
#-------------------------------------------------------------------------------
CLASS_ARG=""
if [ -n "$PROJECT" ]; then
    CLASS_ARG="--project ${PROJECT}"
elif [ -n "$CLASSES" ]; then
    CLASS_ARG="--classes ${CLASSES}"
fi

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

echo "============================================================"
echo "RF-DETR Inference Job"
echo "============================================================"
echo "Job ID:       \$SLURM_JOB_ID"
echo "Node:         \$SLURMD_NODENAME"
echo "GPU:          ${GPU_TYPE}"
echo "Started:      \$(date)"
echo "============================================================"
echo ""

# Navigate to project directory
cd ~/batman

# Activate virtual environment
source .venv/bin/activate

# Prevent OpenBLAS threading issues
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Inference Configuration:"
echo "  Model:          ${MODEL_ARG}"
echo "  Project:        ${PROJECT:-none}"
echo "  Input:          ${INPUT_FILES}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Model size:     RF-DETR ${MODEL}"
echo "  Confidence:     ${CONFIDENCE}"
echo "  Frame interval: ${FRAME_INTERVAL}"
echo "  Tracking:       ${TRACK}"
echo "  Kalman:         $([ "$NO_KALMAN" = true ] && echo "disabled" || echo "enabled")"
echo "  Optimization:   $([ "$NO_OPTIMIZE" = true ] && echo "disabled" || echo "enabled")"
echo ""

SLURM_EOF

# Build tracking arguments
TRACK_ARGS=""
if [ "$TRACK" = true ]; then
    TRACK_ARGS="--track --track-thresh ${TRACK_THRESH} --track-buffer ${TRACK_BUFFER} --match-thresh ${MATCH_THRESH}"
fi

# Build optional flags
OPT_FLAGS=""
if [ "$NO_KALMAN" = true ]; then
    OPT_FLAGS="$OPT_FLAGS --no-kalman"
fi
if [ "$NO_OPTIMIZE" = true ]; then
    OPT_FLAGS="$OPT_FLAGS --no-optimize"
fi

# Add the inference command
cat >> "$SLURM_SCRIPT" << EOF
echo "Starting inference..."

python3 -m cli.inference \\
    ${MODEL_ARG} \\
    ${CLASS_ARG} \\
    --input ${INPUT_FILES} \\
    --output ${OUTPUT_DIR} \\
    --model ${MODEL} \\
    --confidence ${CONFIDENCE} \\
    --frame-interval ${FRAME_INTERVAL} \\
    ${TRACK_ARGS} \\
    ${OPT_FLAGS} \\
    --device cuda \\
    ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo "Inference complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Finished: \$(date)"
echo "============================================================"
EOF

#-------------------------------------------------------------------------------
# Submit or Display
#-------------------------------------------------------------------------------
echo "============================================================"
echo "RF-DETR Inference Job"
echo "============================================================"
echo "GPU:          ${GPU_TYPE}"
echo "Partition:    ${PARTITION}"
echo "Time limit:   ${TIME}"
echo "Model:        ${MODEL_ARG}"
echo "Project:      ${PROJECT:-none}"
echo "Input:        ${INPUT_FILES}"
echo "Output:       ${OUTPUT_DIR}"
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
    # Submit job
    JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
    echo ""
    echo "Job submitted: $JOB_ID"
    echo ""
    echo "Monitor with:"
    echo "  squeue -j $JOB_ID"
    echo "  tail -f logs/slurm_${JOB_ID}_inference.out"
    echo "  tail -f logs/slurm_${JOB_ID}_inference.err"
fi

# Cleanup
rm -f "$SLURM_SCRIPT"
