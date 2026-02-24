#!/bin/bash
#===============================================================================
# Run RF-DETR Training from Local Mac (with auto-sync)
#===============================================================================
#
# Local wrapper that submits a training job to the GPU cluster via SSH,
# streams the job output, waits for completion, and syncs lightweight
# results (JSON metadata only, not .pth checkpoints) locally.
#
# Accepts the same arguments as submit_train.sh.
#
# Usage:
#   ./run_training.sh --project data/projects/One
#   ./run_training.sh --project data/projects/One --gpu=h100-96 --epochs=50
#   ./run_training.sh --project data/projects/One --sources=manual_data --label=manual-only
#   ./run_training.sh --project data/projects/One --infer-after --infer-test-only
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
# Default Configuration (mirrors submit_train.sh)
#-------------------------------------------------------------------------------
GPU_TYPE="h100-96"
PARTITION=""
EPOCHS=50
BATCH_SIZE=""
IMAGE_SIZE=640
LR="1e-4"
PATIENCE=10
PROJECT_DIR="data/projects/One"
OUTPUT_DATASET=""
OUTPUT_DIR=""
MODEL="base"
TIME="24:00:00"
DRY_RUN=false
NO_SYNC=false
NO_PUSH=false
PREPARE_ONLY=false
NUM_GPUS=1
FILTER_CLASSES=""
MAX_FRAMES_PER_CLASS=""
SOURCES=""
MANUAL_SPLIT_STRATEGY=""
MANUAL_DATASETS=""
EXCLUDE_MANUAL_DATASETS=""
INFER_AFTER=false
INFER_TEST_ONLY=false
LABEL=""
EXTRA_ARGS=""

POLL_INTERVAL=30

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Local training runner. Pushes project data to GPU, submits training"
    echo "via SSH, waits for completion, and syncs results locally."
    echo ""
    echo "GPU Options:"
    echo "  --gpu=TYPE          GPU type (default: h100-96)"
    echo "  --num-gpus=N        Number of GPUs (default: 1)"
    echo ""
    echo "Training Options:"
    echo "  --project=PATH      Project directory (default: data/projects/One)"
    echo "  --epochs=N          Training epochs (default: 50)"
    echo "  --batch-size=N      Batch size (auto-set based on GPU if not specified)"
    echo "  --image-size=N      Image size (default: 640)"
    echo "  --lr=RATE           Learning rate (default: 1e-4)"
    echo "  --patience=N        Early stopping patience (default: 10)"
    echo "  --model=SIZE        Model size: base, large (default: base)"
    echo "  --label=NAME        Label appended to run name"
    echo "  --output-dir=PATH   Output directory for run (overrides auto-naming)"
    echo "  --filter-classes=NAMES  Only train on specific classes (pipe-separated)"
    echo "  --max-frames-per-class=N  Cap frames per class"
    echo "  --sources=TYPES     Data sources: manual_data, imports (comma-separated)"
    echo "  --manual-split=STRATEGY  How to split manual data"
    echo "  --manual-datasets=NAMES  Include only these manual subdatasets"
    echo "  --exclude-manual-datasets=NAMES  Exclude these manual subdatasets"
    echo ""
    echo "SLURM Options:"
    echo "  --partition=NAME    SLURM partition (auto-detected if not set)"
    echo "  --time=HH:MM:SS    Time limit (default: 24:00:00)"
    echo ""
    echo "Post-training Inference:"
    echo "  --infer-after       Run inference on project videos after training"
    echo "  --infer-test-only   With --infer-after, only run on test-only videos"
    echo ""
    echo "Sync Options:"
    echo "  --no-sync           Skip syncing results locally after completion"
    echo "  --no-push           Skip pushing manual_data + project.json to GPU before job"
    echo ""
    echo "Other:"
    echo "  --prepare-only      Only prepare dataset, don't train"
    echo "  --dry-run           Show generated SLURM script without submitting"
    echo "  --help              Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --project data/projects/One"
    echo "  $0 --project data/projects/One --gpu=h100-96 --epochs=100 --label=v2"
    echo "  $0 --project data/projects/One --sources=manual_data --label=manual-only"
    echo "  $0 --project data/projects/One --infer-after --infer-test-only"
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
        --label=*)      LABEL="${arg#*=}"; shift ;;
        --model=*)      MODEL="${arg#*=}"; shift ;;
        --time=*)       TIME="${arg#*=}"; shift ;;
        --filter-classes=*) FILTER_CLASSES="${arg#*=}"; shift ;;
        --max-frames-per-class=*) MAX_FRAMES_PER_CLASS="${arg#*=}"; shift ;;
        --sources=*)        SOURCES="${arg#*=}"; shift ;;
        --manual-split=*)   MANUAL_SPLIT_STRATEGY="${arg#*=}"; shift ;;
        --manual-datasets=*) MANUAL_DATASETS="${arg#*=}"; shift ;;
        --exclude-manual-datasets=*) EXCLUDE_MANUAL_DATASETS="${arg#*=}"; shift ;;
        --prepare-only) PREPARE_ONLY=true; shift ;;
        --infer-after)  INFER_AFTER=true; shift ;;
        --infer-test-only) INFER_TEST_ONLY=true; shift ;;
        --no-sync)      NO_SYNC=true; shift ;;
        --no-push)      NO_PUSH=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --help|-h)      show_help ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $arg"; shift ;;
    esac
done

#-------------------------------------------------------------------------------
# Map GPU Type to SLURM Configuration
#-------------------------------------------------------------------------------
case $GPU_TYPE in
    h200)
        SLURM_GRES="gpu:h200-141:${NUM_GPUS}"
        DEFAULT_BATCH=16
        MEM="256G"
        DEFAULT_PARTITION="gpu"
        MAX_GPUS=4
        echo "Warning: H200 only available on 'gpu' partition with 3-hour limit!"
        echo "  For longer training, use --gpu=h100-96 or --gpu=h100-47"
        ;;
    h100-96|h100)
        SLURM_GRES="gpu:h100-96:${NUM_GPUS}"
        DEFAULT_BATCH=16
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_GPUS=2
        ;;
    h100-47)
        SLURM_GRES="gpu:h100-47:${NUM_GPUS}"
        DEFAULT_BATCH=12
        MEM="256G"
        DEFAULT_PARTITION="gpu-long"
        MAX_GPUS=4
        ;;
    a100-80)
        SLURM_GRES="gpu:a100-80:${NUM_GPUS}"
        DEFAULT_BATCH=12
        MEM="128G"
        DEFAULT_PARTITION="gpu-long"
        MAX_GPUS=1
        ;;
    a100-40|a100)
        SLURM_GRES="gpu:a100-40:${NUM_GPUS}"
        DEFAULT_BATCH=8
        MEM="64G"
        DEFAULT_PARTITION="gpu-long"
        MAX_GPUS=2
        ;;
    nv|v100|titanv|titanrtx|t4)
        SLURM_GRES="gpu:nv:${NUM_GPUS}"
        DEFAULT_BATCH=4
        MEM="32G"
        DEFAULT_PARTITION="gpu-long"
        MAX_GPUS=2
        ;;
    *)
        echo "Error: Unknown GPU type: $GPU_TYPE"
        exit 1
        ;;
esac

if [ "$NUM_GPUS" -gt "$MAX_GPUS" ]; then
    echo "Error: Requested $NUM_GPUS GPUs but $GPU_TYPE only supports max $MAX_GPUS per node"
    exit 1
fi

if [ -z "$PARTITION" ]; then
    PARTITION=$DEFAULT_PARTITION
fi

if [ "$PARTITION" = "gpu" ] && [ "$TIME" != "3:00:00" ]; then
    echo "Warning: Adjusting time to 3:00:00 (gpu partition limit)"
    TIME="3:00:00"
fi

if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$DEFAULT_BATCH
fi

if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="rfdetr_${GPU_TYPE}_${TIMESTAMP}"
    if [ -n "$LABEL" ]; then
        RUN_NAME="${RUN_NAME}_${LABEL}"
    fi
    OUTPUT_DIR="${PROJECT_DIR}/runs/${RUN_NAME}"
fi

if [ -z "$OUTPUT_DATASET" ]; then
    OUTPUT_DATASET="${PROJECT_DIR}/exports/coco"
fi

JOB_NAME="rfdetr-${MODEL}-${GPU_TYPE}"

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

if [ -n "${MANUAL_DATASETS}" ] && [ -n "${EXCLUDE_MANUAL_DATASETS}" ]; then
    echo "Error: --manual-datasets and --exclude-manual-datasets are mutually exclusive"
    exit 1
fi

#-------------------------------------------------------------------------------
# Build CLI Arguments
#-------------------------------------------------------------------------------
FILTER_ARG=""
if [ -n "${FILTER_CLASSES}" ]; then
    FILTER_ARG="--filter-classes \"${FILTER_CLASSES}\""
fi

MAX_FRAMES_ARG=""
if [ -n "${MAX_FRAMES_PER_CLASS}" ]; then
    MAX_FRAMES_ARG="--max-frames-per-class ${MAX_FRAMES_PER_CLASS}"
fi

SOURCES_ARG=""
if [ -n "${SOURCES}" ]; then
    SOURCES_ARG="--sources ${SOURCES}"
fi

MANUAL_SPLIT_ARG=""
if [ -n "${MANUAL_SPLIT_STRATEGY}" ]; then
    MANUAL_SPLIT_ARG="--manual-split-strategy ${MANUAL_SPLIT_STRATEGY}"
fi

MANUAL_DS_ARG=""
if [ -n "${MANUAL_DATASETS}" ]; then
    MANUAL_DS_ARG="--manual-datasets ${MANUAL_DATASETS}"
fi
if [ -n "${EXCLUDE_MANUAL_DATASETS}" ]; then
    MANUAL_DS_ARG="--exclude-manual-datasets ${EXCLUDE_MANUAL_DATASETS}"
fi

#-------------------------------------------------------------------------------
# Generate SLURM Script (identical to submit_train.sh)
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

cd ~/batman || { echo "Error: ~/batman not found"; exit 1; }

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
  echo "Using project venv: $(which python3)"
else
  echo "No .venv found. On the cluster: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  echo "Then resubmit."
fi

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

export MASTER_ADDR=localhost
export MASTER_PORT=$((12355 + RANDOM % 1000))
export WORLD_SIZE=$NUM_GPUS
export RANK=0
export LOCAL_RANK=0

echo "Distributed config: WORLD_SIZE=$WORLD_SIZE, MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

echo "Python: $(which python3) ($(python3 --version 2>&1))"
echo ""

SLURM_EOF

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
    ${SOURCES_ARG} \\
    ${MANUAL_SPLIT_ARG} \\
    ${MANUAL_DS_ARG} \\
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

echo "Project Classes:"
PROJECT_CLASSES=\$(python3 -c "import json; c=json.load(open('${PROJECT_DIR}/project.json'))['classes']; print('  ' + '\\n  '.join(c))" 2>/dev/null || echo "  (unable to read)")
echo "\$PROJECT_CLASSES"
echo ""
FILTER_DISPLAY="${FILTER_CLASSES:-all classes}"
echo "Training on:   \$FILTER_DISPLAY"
echo "  Sources:     ${SOURCES:-all}"
echo "  Manual split: ${MANUAL_SPLIT_STRATEGY:-train_only}"
echo ""

echo "Starting training..."

EOF

    if [ "$NUM_GPUS" -gt 1 ]; then
        cat >> "$SLURM_SCRIPT" << EOF
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
    ${SOURCES_ARG} \\
    ${MANUAL_SPLIT_ARG} \\
    ${MANUAL_DS_ARG} \\
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
    ${SOURCES_ARG} \\
    ${MANUAL_SPLIT_ARG} \\
    ${MANUAL_DS_ARG} \\
    ${EXTRA_ARGS}

EOF
    fi
fi

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
# Display Summary
#-------------------------------------------------------------------------------
echo "============================================================"
echo "RF-DETR Training (Local Runner)"
echo "============================================================"
echo "GPU:            ${GPU_TYPE} (${NUM_GPUS}x)"
echo "Partition:      ${PARTITION}"
echo "Memory:         ${MEM}"
echo "Time limit:     ${TIME}"
echo "Project:        ${PROJECT_DIR}"
echo "Output:         ${OUTPUT_DIR}"
echo "Model:          RF-DETR ${MODEL}"
echo "Epochs:         ${EPOCHS}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Image Size:     ${IMAGE_SIZE}"
echo "LR:             ${LR}"
echo "Patience:       ${PATIENCE}"
if [ -n "${FILTER_CLASSES}" ]; then
echo "Filter classes: ${FILTER_CLASSES}"
fi
if [ -n "${SOURCES}" ]; then
echo "Sources:        ${SOURCES}"
fi
if [ -n "${MANUAL_SPLIT_STRATEGY}" ]; then
echo "Manual split:   ${MANUAL_SPLIT_STRATEGY}"
fi
if [ -n "${MANUAL_DATASETS}" ]; then
echo "Manual DS:      ${MANUAL_DATASETS} (include)"
fi
if [ -n "${EXCLUDE_MANUAL_DATASETS}" ]; then
echo "Manual DS:      ${EXCLUDE_MANUAL_DATASETS} (exclude)"
fi
if [ "$INFER_AFTER" = true ]; then
    if [ "$INFER_TEST_ONLY" = true ]; then
        echo "Infer after:    yes (test-only videos)"
    else
        echo "Infer after:    yes (all videos)"
    fi
fi
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
# SSH Helpers
#-------------------------------------------------------------------------------
ensure_ssh() {
    if ! ssh -O check $SSH_OPTS "$SSH_DEST" 2>/dev/null; then
        echo "Opening SSH connection to $SSH_HOST..."
        ssh -M -f -N -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" \
            -o ControlPersist=30m -o ServerAliveInterval=60 "$SSH_DEST"
    fi
}

remote() {
    ssh $SSH_OPTS "$SSH_DEST" "$@"
}

ensure_ssh

#-------------------------------------------------------------------------------
# Pre-sync: push manual_data + project.json to GPU
#-------------------------------------------------------------------------------
if [ "$NO_PUSH" = false ]; then
    echo ""
    echo "Pushing project data to GPU..."
    LOCAL_PROJECT="$SCRIPT_DIR/$PROJECT_DIR"
    REMOTE_PROJECT="$SSH_DEST:$REMOTE_DIR/$PROJECT_DIR"

    if [ -d "$LOCAL_PROJECT/manual_data" ]; then
        rsync -az --progress -e "ssh $SSH_OPTS" \
            "$LOCAL_PROJECT/manual_data/" "$REMOTE_PROJECT/manual_data/"
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
    if [ -n "$TAIL_PID" ]; then
        kill "$TAIL_PID" 2>/dev/null
        wait "$TAIL_PID" 2>/dev/null
    fi
    if [ -n "$ERR_PID" ]; then
        kill "$ERR_PID" 2>/dev/null
        wait "$ERR_PID" 2>/dev/null
    fi
    remote "rm -f $REMOTE_SCRIPT" 2>/dev/null
}
trap cleanup EXIT

#-------------------------------------------------------------------------------
# Stream Log & Wait for Completion
#-------------------------------------------------------------------------------
LOG_REMOTE="$REMOTE_DIR/logs/slurm_${JOB_ID}_${JOB_NAME}.out"
ERR_REMOTE="$REMOTE_DIR/logs/slurm_${JOB_ID}_${JOB_NAME}.err"

echo "Streaming output (log: gpu-server/logs/slurm_${JOB_ID}_${JOB_NAME}.out)"
echo ""

ssh $SSH_OPTS "$SSH_DEST" \
    "while [ ! -f '$LOG_REMOTE' ]; do sleep 2; done; tail -f '$LOG_REMOTE'" 2>/dev/null &
TAIL_PID=$!

ssh $SSH_OPTS "$SSH_DEST" \
    "while [ ! -f '$ERR_REMOTE' ]; do sleep 2; done; tail -f '$ERR_REMOTE'" 2>/dev/null \
    | sed 's/^/[stderr] /' >&2 &
ERR_PID=$!

while remote "squeue -j $JOB_ID -h 2>/dev/null" 2>/dev/null | grep -q "$JOB_ID"; do
    sleep "$POLL_INTERVAL"
done

sleep 3
kill "$TAIL_PID" 2>/dev/null; wait "$TAIL_PID" 2>/dev/null
TAIL_PID=""
kill "$ERR_PID" 2>/dev/null; wait "$ERR_PID" 2>/dev/null
ERR_PID=""

#-------------------------------------------------------------------------------
# Check Job Result
#-------------------------------------------------------------------------------
JOB_STATE=$(remote "sacct -j $JOB_ID --format=State --noheader -P 2>/dev/null | head -1 | tr -d ' '" 2>/dev/null || echo "UNKNOWN")

echo ""

if [[ "$JOB_STATE" == "FAILED" || "$JOB_STATE" == "CANCELLED" || "$JOB_STATE" == "TIMEOUT" || "$JOB_STATE" == "OUT_OF_MEMORY" ]]; then
    echo "============================================================"
    echo "Job $JOB_ID finished with state: $JOB_STATE"
    echo "Error log: gpu-server/logs/slurm_${JOB_ID}_${JOB_NAME}.err"
    echo "============================================================"
    exit 1
fi

echo "============================================================"
echo "Job $JOB_ID completed! (state: ${JOB_STATE})"
echo "============================================================"

#-------------------------------------------------------------------------------
# Sync Results (JSON metadata only, no .pth checkpoints)
#-------------------------------------------------------------------------------
if [ "$NO_SYNC" = true ]; then
    echo ""
    echo "Sync skipped (--no-sync). Results available via SSHFS at:"
    echo "  gpu-server/$OUTPUT_DIR/"
    exit 0
fi

echo ""
echo "Syncing training results (JSON metadata only)..."

RUN_SRC="$MOUNT_POINT/$OUTPUT_DIR"
RUN_DST="$SCRIPT_DIR/$OUTPUT_DIR"

if [ -d "$RUN_SRC" ]; then
    mkdir -p "$RUN_DST"
    rsync -a --include='*.json' --exclude='*' "$RUN_SRC/" "$RUN_DST/"
    echo "  Synced JSON files to: $OUTPUT_DIR/"
else
    echo "  Warning: Run directory not found at $RUN_SRC"
fi

if [ "$INFER_AFTER" = true ]; then
    echo ""
    echo "Syncing inference results..."
    INFER_SRC="$MOUNT_POINT/$PROJECT_DIR/inference/"
    INFER_DST="$SCRIPT_DIR/$PROJECT_DIR/inference/"

    if [ -d "$INFER_SRC" ]; then
        mkdir -p "$INFER_DST"
        rsync -a --progress "$INFER_SRC" "$INFER_DST"
        echo "  Synced inference to: $PROJECT_DIR/inference/"
    else
        echo "  Warning: No inference directory at $INFER_SRC"
    fi
fi

echo ""
echo "============================================================"
echo "Done! Training metadata synced to: $OUTPUT_DIR/"
if [ "$INFER_AFTER" = true ]; then
echo "Inference results synced to: $PROJECT_DIR/inference/"
fi
echo "============================================================"
