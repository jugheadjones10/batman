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
PROJECT_DIR="data/projects/One"
# Where to write the prepared COCO dataset (train/val/test splits); used as input for training
# Empty = auto: ${PROJECT_DIR}/exports/coco
OUTPUT_DATASET=""
# Where to write the training run (checkpoints, logs). Empty = auto: runs/rfdetr_<gpu>_<timestamp>
OUTPUT_DIR=""
# Model size: nano (~3M) | small (~10M) | base (~28M) | medium (~48M) | large (~76M). base = balanced speed/accuracy
MODEL="base"
TIME="24:00:00"
DRY_RUN=false
PREPARE_ONLY=false
NUM_GPUS=1
FILTER_CLASSES=""
MAX_FRAMES_PER_CLASS=""
INFER_AFTER=false
INFER_TEST_ONLY=false
EXTRA_ARGS=""

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE          GPU type (see below)"
    echo "  --num-gpus=N        Number of GPUs (default: 1)"
    echo ""
    echo "Available GPUs:"
    echo "  h200       NVIDIA H200 141GB    (max 4 per node, 3h limit)"
    echo "  h100-96    NVIDIA H100 96GB     (max 2 per node)"
    echo "  h100-47    NVIDIA H100 47GB     (max 4 per node)"
    echo "  a100-80    NVIDIA A100 80GB     (max 1 per node)"
    echo "  a100-40    NVIDIA A100 40GB     (max 2 per node)"
    echo "  nv         V100/Titan/T4        (max 2 per node)"
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
    echo "  --filter-classes=NAMES  Only train on specific classes (pipe-separated)"
    echo "                          Example: --filter-classes='crane hook|crane-hook'"
    echo "  --max-frames-per-class=N  Cap frames per class (random sample, deterministic)"
    echo ""
    echo "SLURM Options:"
    echo "  --partition=NAME    SLURM partition (auto-detected if not set)"
    echo "  --time=HH:MM:SS     Time limit (default: 24:00:00)"
    echo ""
    echo "Post-training Inference:"
    echo "  --infer-after       Run inference on project videos after training"
    echo "  --infer-test-only   With --infer-after, only run on test-only videos"
    echo ""
    echo "Other:"
    echo "  --prepare-only      Only prepare dataset, don't train"
    echo "  --dry-run           Show generated script without submitting"
    echo "  --help              Show this help"
    exit 0
}

while [[ $# -gt 0 ]]; do
    arg="$1"
    case $arg in
        --gpu=*)        GPU_TYPE="${arg#*=}"; shift ;;
        --num-gpus=*)   NUM_GPUS="${arg#*=}"; shift ;;
        --partition=*)  PARTITION="${arg#*=}"; shift ;;
        --epochs=*)     EPOCHS="${arg#*=}"; shift ;;
        --batch-size=*) BATCH_SIZE="${arg#*=}"; shift ;;
        --image-size=*) IMAGE_SIZE="${arg#*=}"; shift ;;
        --lr=*)         LR="${arg#*=}"; shift ;;
        --patience=*)   PATIENCE="${arg#*=}"; shift ;;
        --project=*)    PROJECT_DIR="${arg#*=}"; shift ;;
        --project)      PROJECT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${arg#*=}"; shift ;;
        --model=*)      MODEL="${arg#*=}"; shift ;;
        --time=*)       TIME="${arg#*=}"; shift ;;
        --filter-classes=*) FILTER_CLASSES="${arg#*=}"; shift ;;
        --max-frames-per-class=*) MAX_FRAMES_PER_CLASS="${arg#*=}"; shift ;;
        --prepare-only) PREPARE_ONLY=true; shift ;;
        --infer-after)  INFER_AFTER=true; shift ;;
        --infer-test-only) INFER_TEST_ONLY=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      show_help ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $arg"; shift ;;
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
        MAX_GPUS=4
        echo "⚠️  Warning: H200 only available on 'gpu' partition with 3-hour limit!"
        echo "   For longer training, use --gpu=h100-96 or --gpu=h100-47"
        ;;
    h100-96|h100)
        SLURM_GRES="gpu:h100-96:${NUM_GPUS}"
        DEFAULT_BATCH=16
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        MAX_GPUS=2
        ;;
    h100-47)
        SLURM_GRES="gpu:h100-47:${NUM_GPUS}"
        DEFAULT_BATCH=12
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        MAX_GPUS=4
        ;;
    a100-80)
        SLURM_GRES="gpu:a100-80:${NUM_GPUS}"
        DEFAULT_BATCH=12
        MEM="128G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        MAX_GPUS=1
        ;;
    a100-40|a100)
        SLURM_GRES="gpu:a100-40:${NUM_GPUS}"
        DEFAULT_BATCH=8
        MEM="64G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        MAX_GPUS=2
        ;;
    nv|v100|titanv|titanrtx|t4)
        SLURM_GRES="gpu:nv:${NUM_GPUS}"
        DEFAULT_BATCH=4
        MEM="32G"
        DEFAULT_PARTITION="gpu-long"
        MAX_TIME="3-00:00:00"
        MAX_GPUS=2
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

# Validate number of GPUs
if [ "$NUM_GPUS" -gt "$MAX_GPUS" ]; then
    echo "Error: Requested $NUM_GPUS GPUs but $GPU_TYPE only supports max $MAX_GPUS per node"
    exit 1
fi

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

# Generate output directory (under project by default)
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${PROJECT_DIR}/runs/rfdetr_${GPU_TYPE}_${TIMESTAMP}"
fi

# Generate dataset output directory (under project by default)
if [ -z "$OUTPUT_DATASET" ]; then
    OUTPUT_DATASET="${PROJECT_DIR}/exports/coco"
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

# Number of GPUs for this job
NUM_GPUS=${NUM_GPUS}
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
echo "GPUs:          $NUM_GPUS"
echo "Started:       $(date)"
echo "Working Dir:   $(pwd)"
echo "============================================================"

# Change to project directory
cd ~/batman || { echo "Error: ~/batman not found"; exit 1; }

# Use project venv so dependencies (loguru, torch, etc.) are available.
# On the cluster, create once:  python3 -m venv .venv && source .venv/bin/activate && pip install -e .
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "Using project venv: $(which python3)"
else
  echo "No .venv found. On the cluster: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  echo "Then resubmit."
fi

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=$((12355 + RANDOM % 1000))  # Random port to avoid conflicts
export WORLD_SIZE=$NUM_GPUS
export RANK=0
export LOCAL_RANK=0

echo "Distributed config: WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# Print Python info
echo "Python: $(which python3) ($(python3 --version 2>&1))"
echo ""

SLURM_EOF

# Build filter-classes argument if specified
# Uses pipe '|' delimiter for class names that may contain spaces
FILTER_ARG=""
if [ -n "${FILTER_CLASSES}" ]; then
    FILTER_ARG="--filter-classes \"${FILTER_CLASSES}\""
fi

MAX_FRAMES_ARG=""
if [ -n "${MAX_FRAMES_PER_CLASS}" ]; then
    MAX_FRAMES_ARG="--max-frames-per-class ${MAX_FRAMES_PER_CLASS}"
fi

# Add the training command
if [ "$PREPARE_ONLY" = true ]; then
    cat >> "$SLURM_SCRIPT" << EOF

echo "Preparing dataset only..."
echo "  Filter classes: ${FILTER_CLASSES:-all}"
python3 -m cli.train \\
    --project ${PROJECT_DIR} \\
    --output-dataset ${OUTPUT_DATASET} \\
    --prepare-only \\
    ${FILTER_ARG} \\
    ${MAX_FRAMES_ARG} \\
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

# Display project classes
echo "Project Classes:"
PROJECT_CLASSES=\$(python3 -c "import json; c=json.load(open('${PROJECT_DIR}/project.json'))['classes']; print('  ' + '\\n  '.join(c))" 2>/dev/null || echo "  (unable to read)")
echo "\$PROJECT_CLASSES"
echo ""
FILTER_DISPLAY="${FILTER_CLASSES:-all classes}"
echo "Training on:   \$FILTER_DISPLAY"
echo ""

echo "Starting training..."

EOF

    # Add the training command with proper quoting
    if [ "$NUM_GPUS" -gt 1 ]; then
        cat >> "$SLURM_SCRIPT" << EOF
# Use torchrun for multi-GPU, regular python for single-GPU
echo "Using torchrun for multi-GPU training (\$NUM_GPUS GPUs)..."
torchrun --nproc_per_node=\$NUM_GPUS --master_port=\$MASTER_PORT \\
    -m cli.train \\
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
    ${FILTER_ARG} \\
    ${MAX_FRAMES_ARG} \\
    ${EXTRA_ARGS}

EOF
    else
        cat >> "$SLURM_SCRIPT" << EOF
echo "Using single-GPU training..."
python3 -m cli.train \\
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
    ${FILTER_ARG} \\
    ${MAX_FRAMES_ARG} \\
    ${EXTRA_ARGS}

EOF
    fi
fi

# Add post-training inference if requested
if [ "$INFER_AFTER" = true ] && [ "$PREPARE_ONLY" != true ]; then
    INFER_CLI_FLAGS="--latest --device cuda"
    if [ "$INFER_TEST_ONLY" = true ]; then
        INFER_CLI_FLAGS="$INFER_CLI_FLAGS --test-only"
    fi

    cat >> "$SLURM_SCRIPT" << EOF
TRAIN_EXIT=\$?

if [ \$TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "Training failed (exit code \$TRAIN_EXIT), skipping inference"
    echo "============================================================"
    echo "Job completed at: \$(date)"
    echo "Exit code: \$TRAIN_EXIT"
    echo "============================================================"
    exit \$TRAIN_EXIT
fi

echo ""
echo "============================================================"
echo "Post-Training Inference"
echo "============================================================"
echo ""

python3 -m cli.inference \\
    --project ${PROJECT_DIR} \\
    ${INFER_CLI_FLAGS}

EXIT_CODE=\$?
EOF
else
    cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'
EXIT_CODE=$?
SLURM_EOF
fi

cat >> "$SLURM_SCRIPT" << 'SLURM_EOF'

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
if [ -n "${MAX_FRAMES_PER_CLASS}" ]; then
echo "Max frames/class: ${MAX_FRAMES_PER_CLASS}"
fi
echo "Output Dir:   ${OUTPUT_DIR}"
if [ "$INFER_AFTER" = true ]; then
    if [ "$INFER_TEST_ONLY" = true ]; then
        echo "Infer After:  yes (test-only videos)"
    else
        echo "Infer After:  yes (all videos)"
    fi
fi
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
