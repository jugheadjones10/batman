#!/bin/bash
#===============================================================================
# Submit RF-DETR Training Job to SLURM
#===============================================================================
#
# Usage:
#   ./submit_train.sh                              # Use defaults (H200)
#   ./submit_train.sh --gpu=a100-40                # Use A100 40GB
#   ./submit_train.sh --gpu=h100 --epochs=100      # H100 with 100 epochs
#   ./submit_train.sh --dry-run                    # Show what would be submitted
#
# GPU Options:
#   h200       - NVIDIA H200 (default, best performance)
#   h100-96    - NVIDIA H100 96GB
#   h100-47    - NVIDIA H100 47GB  
#   a100-80    - NVIDIA A100 80GB
#   a100-40    - NVIDIA A100 40GB
#   v100       - NVIDIA V100
#   titanrtx   - NVIDIA Titan RTX
#   t4         - NVIDIA Tesla T4
#
#===============================================================================

set -e

#-------------------------------------------------------------------------------
# Default Configuration
#-------------------------------------------------------------------------------
GPU_TYPE="h100-96"  # Default to H100-96 (available on gpu-long)
PARTITION=""        # Auto-set based on GPU type
EPOCHS=50
BATCH_SIZE=""  # Will be set based on GPU if empty
IMAGE_SIZE=640
LR="1e-4"
PATIENCE=10
PROJECT_DIR="data/projects/Test"
OUTPUT_DATASET="datasets/rfdetr_coco"
OUTPUT_DIR=""
MODEL="base"
TIME="24:00:00"
DRY_RUN=false
PREPARE_ONLY=false
NUM_GPUS=1
EXTRA_ARGS=""

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE          GPU type: h200, h100-96, h100-47, a100-80, a100-40, v100, titanrtx, t4"
    echo "  --num-gpus=N        Number of GPUs (default: 1)"
    echo ""
    echo "Training Options:"
    echo "  --project=PATH      Project directory (default: data/projects/Test)"
    echo "  --epochs=N          Training epochs (default: 50)"
    echo "  --batch-size=N      Batch size (auto-set based on GPU if not specified)"
    echo "  --image-size=N      Image size (default: 640)"
    echo "  --lr=RATE           Learning rate (default: 1e-4)"
    echo "  --patience=N        Early stopping patience (default: 10)"
    echo "  --model=SIZE        Model size: base, large (default: base)"
    echo "  --output-dir=PATH   Output directory for run"
    echo ""
    echo "SLURM Options:"
    echo "  --partition=NAME    SLURM partition (auto-detected if not set)"
    echo "  --time=HH:MM:SS     Time limit (default: 24:00:00)"
    echo ""
    echo "Other:"
    echo "  --prepare-only      Only prepare dataset, don't train"
    echo "  --dry-run           Show generated script without submitting"
    echo "  --help              Show this help"
    exit 0
}

for arg in "$@"; do
    case $arg in
        --gpu=*)        GPU_TYPE="${arg#*=}" ;;
        --num-gpus=*)   NUM_GPUS="${arg#*=}" ;;
        --partition=*)  PARTITION="${arg#*=}" ;;
        --epochs=*)     EPOCHS="${arg#*=}" ;;
        --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
        --image-size=*) IMAGE_SIZE="${arg#*=}" ;;
        --lr=*)         LR="${arg#*=}" ;;
        --patience=*)   PATIENCE="${arg#*=}" ;;
        --project=*)    PROJECT_DIR="${arg#*=}" ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
        --model=*)      MODEL="${arg#*=}" ;;
        --time=*)       TIME="${arg#*=}" ;;
        --prepare-only) PREPARE_ONLY=true ;;
        --dry-run)      DRY_RUN=true ;;
        --help|-h)      show_help ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

#-------------------------------------------------------------------------------
# Map GPU Type to SLURM Configuration
#-------------------------------------------------------------------------------
# Note: H200 is only on 'gpu' partition (3h limit), others available on 'gpu-long' (3d limit)
case $GPU_TYPE in
    h200)
        SLURM_GRES="gpu:h200-141:${NUM_GPUS}"
        DEFAULT_BATCH=16
        MEM="256G"
        DEFAULT_PARTITION="gpu"  # H200 only on gpu partition (3h limit!)
        MAX_TIME="3:00:00"
        echo "⚠️  Warning: H200 only available on 'gpu' partition with 3-hour limit!"
        echo "   For longer training, use --gpu=h100-96 or --gpu=h100-47"
        ;;
    h100-96|h100)
        SLURM_GRES="gpu:h100-96:${NUM_GPUS}"
        DEFAULT_BATCH=16
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    h100-47)
        SLURM_GRES="gpu:h100-47:${NUM_GPUS}"
        DEFAULT_BATCH=12
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    a100-80)
        SLURM_GRES="gpu:a100-80:${NUM_GPUS}"
        DEFAULT_BATCH=12
        MEM="128G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    a100-40|a100)
        SLURM_GRES="gpu:a100-40:${NUM_GPUS}"
        DEFAULT_BATCH=8
        MEM="64G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    nv|v100|titanv|titanrtx|t4)
        SLURM_GRES="gpu:nv:${NUM_GPUS}"
        DEFAULT_BATCH=4
        MEM="32G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        ;;
    *)
        echo "Error: Unknown GPU type: $GPU_TYPE"
        echo ""
        echo "Available GPUs:"
        echo "  h100-96  - H100 96GB (recommended for long training)"
        echo "  h100-47  - H100 47GB"
        echo "  a100-80  - A100 80GB"
        echo "  a100-40  - A100 40GB"
        echo "  h200     - H200 (3h limit only!)"
        echo "  nv       - V100/Titan/T4"
        exit 1
        ;;
esac

# Set partition (auto or override)
if [ -z "$PARTITION" ]; then
    PARTITION=$DEFAULT_PARTITION
fi

# Warn if time exceeds partition limit
if [ "$PARTITION" = "gpu" ] && [ "$TIME" != "3:00:00" ]; then
    echo "⚠️  Adjusting time to 3:00:00 (gpu partition limit)"
    TIME="3:00:00"
fi

# Set batch size
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$DEFAULT_BATCH
fi

# Generate output directory
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="runs/rfdetr_${GPU_TYPE}_${TIMESTAMP}"
fi

# Generate job name
JOB_NAME="rfdetr-${MODEL}-${GPU_TYPE}"

#-------------------------------------------------------------------------------
# Create Logs Directory
#-------------------------------------------------------------------------------
mkdir -p logs

#-------------------------------------------------------------------------------
# Generate SLURM Script
#-------------------------------------------------------------------------------
SLURM_SCRIPT=$(mktemp /tmp/slurm_rfdetr_XXXXXX.sh)

cat > "$SLURM_SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/slurm_%j_${JOB_NAME}.out
#SBATCH --error=logs/slurm_%j_${JOB_NAME}.err
#SBATCH --time=${TIME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${SLURM_GRES}
#SBATCH --mem=${MEM}
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e0425887@u.nus.edu
SLURM_EOF

cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'

#===============================================================================
# Job Execution
#===============================================================================

echo "============================================================"
echo "RF-DETR Training Job"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Started:       $(date)"
echo "Working Dir:   $(pwd)"
echo "============================================================"

# Change to project directory
cd ~/batman || { echo "Error: ~/batman not found"; exit 1; }

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Using system Python 3.12 (no activation needed)
# If you set up a venv later, uncomment:
# source .venv/bin/activate

# Print Python info
echo "Python: $(which python3) ($(python3 --version 2>&1))"
echo ""

SLURM_EOF

# Add the training command
if [ "$PREPARE_ONLY" = true ]; then
    cat >> "$SLURM_SCRIPT" << EOF

echo "Preparing dataset only..."
python3 finetune_rfdetr.py \\
    --project ${PROJECT_DIR} \\
    --output-dataset ${OUTPUT_DATASET} \\
    --prepare-only \\
    ${EXTRA_ARGS}

EOF
else
    cat >> "$SLURM_SCRIPT" << EOF

echo "Training Configuration:"
echo "  Project:     ${PROJECT_DIR}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Model:       RF-DETR ${MODEL}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch Size:  ${BATCH_SIZE}"
echo "  Image Size:  ${IMAGE_SIZE}"
echo "  LR:          ${LR}"
echo "  Patience:    ${PATIENCE}"
echo ""

echo "Starting training..."
python3 finetune_rfdetr.py \\
    --project ${PROJECT_DIR} \\
    --output-dataset ${OUTPUT_DATASET} \\
    --output-dir ${OUTPUT_DIR} \\
    --model ${MODEL} \\
    --epochs ${EPOCHS} \\
    --batch-size ${BATCH_SIZE} \\
    --image-size ${IMAGE_SIZE} \\
    --lr ${LR} \\
    --patience ${PATIENCE} \\
    --device cuda \\
    --num-workers 8 \\
    ${EXTRA_ARGS}

EOF
fi

cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
SLURM_EOF

#-------------------------------------------------------------------------------
# Submit or Display
#-------------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SLURM Job Configuration"
echo "============================================================"
echo "GPU:          ${GPU_TYPE} (${NUM_GPUS}x)"
echo "Partition:    ${PARTITION}"
echo "GRES:         ${SLURM_GRES}"
echo "Memory:       ${MEM}"
echo "Time:         ${TIME}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "=== Generated SLURM Script (dry run) ==="
    echo ""
    cat "$SLURM_SCRIPT"
    echo ""
    echo "=== End of Script ==="
    rm "$SLURM_SCRIPT"
else
    echo "Submitting job..."
    JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        echo ""
        echo "✓ Job submitted successfully!"
        echo "  Job ID: $JOB_ID"
        echo ""
        echo "Useful commands:"
        echo "  squeue -j $JOB_ID              # Check job status"
        echo "  scancel $JOB_ID                # Cancel job"
        echo "  tail -f logs/slurm_${JOB_ID}_${JOB_NAME}.out  # Watch output"
        echo "  tail -f logs/slurm_${JOB_ID}_${JOB_NAME}.err  # Watch errors"
        echo ""
        
        # Save the script for reference
        cp "$SLURM_SCRIPT" "logs/submitted_${JOB_ID}.sh"
        echo "Script saved to: logs/submitted_${JOB_ID}.sh"
    else
        echo "Error: Failed to submit job"
        cat "$SLURM_SCRIPT"
        rm "$SLURM_SCRIPT"
        exit 1
    fi
    
    rm "$SLURM_SCRIPT"
fi
