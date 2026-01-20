#!/bin/bash
#===============================================================================
# Submit RF-DETR Inference Job to SLURM
#===============================================================================
#
# Usage:
#   ./submit_inference.sh --checkpoint runs/my_run/best.pth --input video.mp4
#   ./submit_inference.sh --checkpoint runs/my_run/best.pth --input images/*.jpg
#   ./submit_inference.sh --checkpoint runs/my_run/best.pth --input video.mp4 --frame-interval 5 --track
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
INPUT_FILES=""
OUTPUT_DIR=""
MODEL="base"
CONFIDENCE=0.5
FRAME_INTERVAL=1
TRACK=false
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
    echo "  --checkpoint=PATH    Path to trained model checkpoint"
    echo "  --input=FILES        Input image(s) or video file(s)"
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE           GPU type (default: a100-40)"
    echo "  --time=HH:MM:SS      Time limit (default: 04:00:00)"
    echo ""
    echo "Inference Options:"
    echo "  --output=PATH        Output directory (default: inference_results)"
    echo "  --model=SIZE         Model size: base, large (default: base)"
    echo "  --confidence=N       Confidence threshold (default: 0.5)"
    echo ""
    echo "Video Options:"
    echo "  --frame-interval=N   Run inference every N frames (default: 1)"
    echo "  --track              Enable ByteTrack tracking"
    echo "  --track-thresh=N     ByteTrack detection threshold (default: 0.25)"
    echo "  --track-buffer=N     Frames to keep lost tracks (default: 30)"
    echo "  --match-thresh=N     IoU threshold for matching (default: 0.8)"
    echo ""
    echo "Other:"
    echo "  --dry-run            Show generated script without submitting"
    echo "  --help               Show this help"
    echo ""
    echo "Examples:"
    echo "  # Single video"
    echo "  $0 --checkpoint runs/run1/best.pth --input video.mp4"
    echo ""
    echo "  # Video with tracking, every 5 frames"
    echo "  $0 --checkpoint runs/run1/best.pth --input video.mp4 --frame-interval 5 --track"
    echo ""
    echo "  # Batch images"
    echo "  $0 --checkpoint runs/run1/best.pth --input 'images/*.jpg'"
    exit 0
}

for arg in "$@"; do
    case $arg in
        --gpu=*)           GPU_TYPE="${arg#*=}" ;;
        --partition=*)     PARTITION="${arg#*=}" ;;
        --time=*)          TIME="${arg#*=}" ;;
        --checkpoint=*)    CHECKPOINT="${arg#*=}" ;;
        --input=*)         INPUT_FILES="${arg#*=}" ;;
        --output=*)        OUTPUT_DIR="${arg#*=}" ;;
        --model=*)         MODEL="${arg#*=}" ;;
        --confidence=*)    CONFIDENCE="${arg#*=}" ;;
        --frame-interval=*) FRAME_INTERVAL="${arg#*=}" ;;
        --track)           TRACK=true ;;
        --track-thresh=*)  TRACK_THRESH="${arg#*=}" ;;
        --track-buffer=*)  TRACK_BUFFER="${arg#*=}" ;;
        --match-thresh=*)  MATCH_THRESH="${arg#*=}" ;;
        --dry-run)         DRY_RUN=true ;;
        --help|-h)         show_help ;;
        *)                 EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
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

# Show GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Inference Configuration:"
echo "  Checkpoint:     ${CHECKPOINT}"
echo "  Input:          ${INPUT_FILES}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Model:          RF-DETR ${MODEL}"
echo "  Confidence:     ${CONFIDENCE}"
echo "  Frame interval: ${FRAME_INTERVAL}"
echo "  Tracking:       ${TRACK}"
echo ""

SLURM_EOF

# Build tracking arguments
TRACK_ARGS=""
if [ "$TRACK" = true ]; then
    TRACK_ARGS="--track --track-thresh ${TRACK_THRESH} --track-buffer ${TRACK_BUFFER} --match-thresh ${MATCH_THRESH}"
fi

# Add the inference command
cat >> "$SLURM_SCRIPT" << EOF
echo "Starting inference..."

python3 -m cli.inference \\
    --checkpoint ${CHECKPOINT} \\
    --input ${INPUT_FILES} \\
    --output ${OUTPUT_DIR} \\
    --model ${MODEL} \\
    --confidence ${CONFIDENCE} \\
    --frame-interval ${FRAME_INTERVAL} \\
    ${TRACK_ARGS} \\
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
echo "Checkpoint:   ${CHECKPOINT}"
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
