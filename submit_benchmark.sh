#!/bin/bash
#
# Submit RF-DETR latency benchmark jobs for multiple GPUs
#
# Usage:
#   ./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus h200,a100-80,a100-40
#   ./submit_benchmark.sh --latest --gpus h200,h100-96,a100-80,a100-40
#   ./submit_benchmark.sh --checkpoint runs/my_run/best.pth --gpus all
#

set -e

#-------------------------------------------------------------------------------
# Default Values
#-------------------------------------------------------------------------------
CHECKPOINT=""
RUN=""
LATEST=false
MODEL="base"
IMAGE_SIZE=640
WARMUP=10
RUNS=100
NO_OPTIMIZE=false
GPU_TYPES=""
TIME="00:30:00"
VIDEO="crane_hook_1_short.mp4"  # Default video for realistic benchmark
NO_VIDEO=false
CREATE_LATENCY_VIDEO=false

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --run)
            RUN="$2"
            shift 2
            ;;
        --latest)
            LATEST=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --no-optimize)
            NO_OPTIMIZE=true
            shift
            ;;
        --gpus)
            GPU_TYPES="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --video)
            VIDEO="$2"
            shift 2
            ;;
        --no-video)
            NO_VIDEO=true
            shift
            ;;
        --create-latency-video)
            CREATE_LATENCY_VIDEO=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Submit RF-DETR latency benchmark jobs for multiple GPUs"
            echo ""
            echo "Model Selection (required, mutually exclusive):"
            echo "  --checkpoint PATH    Path to checkpoint file"
            echo "  --run NAME          Run name (e.g., rfdetr_h200_20260120_105925)"
            echo "  --latest            Use the latest run"
            echo ""
            echo "Model Configuration:"
            echo "  --model SIZE        Model size: base or large (default: base)"
            echo "  --image-size SIZE   Image size for inference (default: 640)"
            echo "  --no-optimize       Disable model optimization"
            echo ""
            echo "Benchmark Configuration:"
            echo "  --warmup N          Number of warmup runs (default: 10)"
            echo "  --runs N            Number of benchmark runs (default: 100)"
            echo "  --video FILE        Video file for realistic benchmark (default: crane_hook_1_short.mp4)"
            echo "  --no-video          Use synthetic dummy images instead of video"
            echo "  --create-latency-video  Create side-by-side latency visualization video (requires video)"
            echo ""
            echo "GPU Selection (required):"
            echo "  --gpus TYPES        Comma-separated GPU types or 'all'"
            echo "                      Options: h200,h100-96,h100-47,a100-80,a100-40,nv"
            echo "                      Example: --gpus h200,a100-80,a100-40"
            echo "                      Example: --gpus all"
            echo ""
            echo "SLURM Options:"
            echo "  --time LIMIT        Time limit (default: 00:30:00)"
            echo ""
            echo "Examples:"
            echo "  $0 --run rfdetr_h200_20260120_105925 --gpus h200,a100-80"
            echo "  $0 --latest --gpus all"
            echo "  $0 --checkpoint runs/my_run/best.pth --gpus h200,h100-96,a100-80,a100-40"
            echo "  $0 --run my_run --gpus h200 --create-latency-video"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Validate Arguments
#-------------------------------------------------------------------------------
MODEL_SELECTION_COUNT=0
[[ -n "$CHECKPOINT" ]] && ((MODEL_SELECTION_COUNT++)) || true
[[ -n "$RUN" ]] && ((MODEL_SELECTION_COUNT++)) || true
[[ "$LATEST" == true ]] && ((MODEL_SELECTION_COUNT++)) || true

if [ $MODEL_SELECTION_COUNT -eq 0 ]; then
    echo "Error: Must specify one of --checkpoint, --run, or --latest"
    echo "Use -h or --help for usage information"
    exit 1
fi

if [ $MODEL_SELECTION_COUNT -gt 1 ]; then
    echo "Error: Can only specify one of --checkpoint, --run, or --latest"
    exit 1
fi

if [ -z "$GPU_TYPES" ]; then
    echo "Error: Must specify --gpus"
    echo "Use -h or --help for usage information"
    exit 1
fi

#-------------------------------------------------------------------------------
# Expand GPU Types
#-------------------------------------------------------------------------------
if [ "$GPU_TYPES" == "all" ]; then
    GPU_ARRAY=(h200 h100-96 h100-47 a100-80 a100-40 nv)
else
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_TYPES"
fi

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
# Build Optional Arguments
#-------------------------------------------------------------------------------
OPTIONAL_ARGS=""
OPTIONAL_ARGS="${OPTIONAL_ARGS} --model ${MODEL}"
OPTIONAL_ARGS="${OPTIONAL_ARGS} --image-size ${IMAGE_SIZE}"
OPTIONAL_ARGS="${OPTIONAL_ARGS} --warmup ${WARMUP}"
OPTIONAL_ARGS="${OPTIONAL_ARGS} --runs ${RUNS}"

if [ "$NO_OPTIMIZE" = true ]; then
    OPTIONAL_ARGS="${OPTIONAL_ARGS} --no-optimize"
fi

# Video benchmark (default) or synthetic
if [ "$NO_VIDEO" = true ]; then
    VIDEO_ARG=""
    BENCHMARK_TYPE="synthetic"
else
    VIDEO_ARG="--video ${VIDEO}"
    BENCHMARK_TYPE="video (${VIDEO})"
fi

# Add latency video creation flag if requested
if [ "$CREATE_LATENCY_VIDEO" = true ]; then
    if [ "$NO_VIDEO" = true ]; then
        echo "Warning: --create-latency-video requires video input (--video), ignoring flag"
        CREATE_LATENCY_VIDEO=false
    else
        OPTIONAL_ARGS="${OPTIONAL_ARGS} --create-latency-video"
    fi
fi

#-------------------------------------------------------------------------------
# Create Output Directory with Timestamp
#-------------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "============================================================"
echo "RF-DETR Latency Benchmark Suite"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "GPU types: ${GPU_ARRAY[*]}"
echo "Model: ${MODEL}"
echo "Benchmark type: ${BENCHMARK_TYPE}"
echo "Warmup runs: ${WARMUP}"
echo "Test runs: ${RUNS}"
echo "Create latency video: ${CREATE_LATENCY_VIDEO}"
echo "============================================================"
echo ""

#-------------------------------------------------------------------------------
# Submit Jobs for Each GPU
#-------------------------------------------------------------------------------
JOB_IDS=()

for GPU_TYPE in "${GPU_ARRAY[@]}"; do
    echo "Submitting benchmark for GPU: ${GPU_TYPE}"

    # Configure GPU-specific SLURM settings
    case $GPU_TYPE in
        h200)
            PARTITION="gpu"
            GRES="gpu:h200:1"
            ;;
        h100-96)
            PARTITION="gpu-long"
            GRES="gpu:h100-96:1"
            ;;
        h100-47)
            PARTITION="gpu-long"
            GRES="gpu:h100-47:1"
            ;;
        a100-80)
            PARTITION="gpu-long"
            GRES="gpu:a100-80:1"
            ;;
        a100-40)
            PARTITION="gpu-long"
            GRES="gpu:a100-40:1"
            ;;
        nv)
            PARTITION="gpu-short"
            GRES="gpu:nv:1"
            ;;
        *)
            echo "  Error: Unknown GPU type: $GPU_TYPE"
            continue
            ;;
    esac

    # Create SLURM script
    SLURM_SCRIPT=$(mktemp /tmp/benchmark_slurm_XXXXXX.sh)

    cat > "$SLURM_SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=rfdetr-bench-${GPU_TYPE}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=${TIME}
#SBATCH --output=logs/slurm_%j_benchmark_${GPU_TYPE}.out
#SBATCH --error=logs/slurm_%j_benchmark_${GPU_TYPE}.err

echo "============================================================"
echo "RF-DETR Latency Benchmark - ${GPU_TYPE}"
echo "============================================================"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Hostname: \${HOSTNAME}"
echo "GPU Type: ${GPU_TYPE}"
echo "Partition: ${PARTITION}"
echo "Start Time: \$(date)"
echo "============================================================"
echo ""

# Navigate to project directory
cd ~/batman

# Activate virtual environment
source .venv/bin/activate

# Record start time
START_TIME=\$(date +%s)

# Run benchmark
echo "Running benchmark..."
python3 -m cli.benchmark_latency ${MODEL_ARG} ${OPTIONAL_ARGS} ${VIDEO_ARG} \\
    --output ${OUTPUT_DIR}/${GPU_TYPE}

# Record end time
END_TIME=\$(date +%s)
ELAPSED=\$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Benchmark Complete"
echo "============================================================"
echo "Total Time: \$(printf '%02d:%02d:%02d' \$((ELAPSED/3600)) \$((ELAPSED%3600/60)) \$((ELAPSED%60)))"
echo "Results saved to: ${OUTPUT_DIR}/${GPU_TYPE}/"
echo "============================================================"
SLURM_EOF

    # Submit job
    JOB_OUTPUT=$(sbatch "$SLURM_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+')
    JOB_IDS+=("$JOB_ID")

    echo "  Job ID: $JOB_ID"
    echo "  Log: logs/slurm_${JOB_ID}_benchmark_${GPU_TYPE}.out"
    echo ""

    # Clean up temp script
    rm -f "$SLURM_SCRIPT"

    # Brief delay to avoid overwhelming the scheduler
    sleep 0.5
done

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo "============================================================"
echo "All benchmark jobs submitted!"
echo "============================================================"
echo "Job IDs: ${JOB_IDS[*]}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Useful commands:"
echo "  # Check status of all jobs"
echo "  squeue -u \$USER"
echo "  watch -n 2 squeue -u \$USER"
echo ""
echo "  # Monitor stdout logs (one per GPU)"
for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    GPU="${GPU_ARRAY[$i]}"
    echo "  tail -f logs/slurm_${JOB_ID}_benchmark_${GPU}.out  # ${GPU}"
done
echo ""
echo "  # Monitor stderr logs (errors)"
for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    GPU="${GPU_ARRAY[$i]}"
    echo "  tail -f logs/slurm_${JOB_ID}_benchmark_${GPU}.err  # ${GPU}"
done
echo ""
echo "  # Monitor all stdout for this run"
# Build list of log files for current run
CURRENT_OUT_LOGS=""
CURRENT_ERR_LOGS=""
for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    GPU="${GPU_ARRAY[$i]}"
    CURRENT_OUT_LOGS="${CURRENT_OUT_LOGS} logs/slurm_${JOB_ID}_benchmark_${GPU}.out"
    CURRENT_ERR_LOGS="${CURRENT_ERR_LOGS} logs/slurm_${JOB_ID}_benchmark_${GPU}.err"
done
echo "  tail -f${CURRENT_OUT_LOGS}"
echo ""
echo "  # Monitor all stderr for this run (errors)"
echo "  tail -f${CURRENT_ERR_LOGS}"
echo ""
echo "  # Monitor all logs for this run (stdout + stderr)"
echo "  tail -f${CURRENT_OUT_LOGS}${CURRENT_ERR_LOGS}"
echo ""
echo "  # Cancel all jobs if needed"
echo "  scancel ${JOB_IDS[*]}"
echo ""
echo "When complete, compare results with:"
echo "  python -m cli.compare_latency ${OUTPUT_DIR}"
echo "============================================================"

# Save job info with monitoring commands
cat > "${OUTPUT_DIR}/job_info.txt" << EOF
Benchmark Suite
===============
Timestamp: ${TIMESTAMP}
GPU Types: ${GPU_ARRAY[*]}
Job IDs: ${JOB_IDS[*]}

Model Selection: ${MODEL_ARG}
Model Size: ${MODEL}
Benchmark Type: ${BENCHMARK_TYPE}
Warmup Runs: ${WARMUP}
Test Runs: ${RUNS}

Monitoring Commands
===================
# Check job status
squeue -j $(IFS=,; echo "${JOB_IDS[*]}")

# Monitor all stdout for this run
tail -f${CURRENT_OUT_LOGS}

# Monitor all stderr for this run (errors)
tail -f${CURRENT_ERR_LOGS}

# Monitor all logs for this run (stdout + stderr)
tail -f${CURRENT_OUT_LOGS}${CURRENT_ERR_LOGS}

# Individual stdout logs:
EOF

# Add individual log commands
for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    GPU="${GPU_ARRAY[$i]}"
    echo "tail -f logs/slurm_${JOB_ID}_benchmark_${GPU}.out  # ${GPU} stdout" >> "${OUTPUT_DIR}/job_info.txt"
done

echo "" >> "${OUTPUT_DIR}/job_info.txt"
echo "# Individual stderr logs:" >> "${OUTPUT_DIR}/job_info.txt"
for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    GPU="${GPU_ARRAY[$i]}"
    echo "tail -f logs/slurm_${JOB_ID}_benchmark_${GPU}.err  # ${GPU} stderr" >> "${OUTPUT_DIR}/job_info.txt"
done

# Save log file lists for monitor.sh
echo "${CURRENT_OUT_LOGS}" > "${OUTPUT_DIR}/.out_logs"
echo "${CURRENT_ERR_LOGS}" > "${OUTPUT_DIR}/.err_logs"

# Create a helper script for monitoring
cat > "${OUTPUT_DIR}/monitor.sh" << 'MONITOR_EOF'
#!/bin/bash
# Quick monitoring helper for this benchmark suite

# Read log file lists for this specific run
OUT_LOGS=$(cat .out_logs 2>/dev/null | sed 's|^|../../|g; s| | ../../|g')
ERR_LOGS=$(cat .err_logs 2>/dev/null | sed 's|^|../../|g; s| | ../../|g')

case "${1:-status}" in
    status)
        echo "Checking job status..."
        squeue -j $(cat job_info.txt | grep "Job IDs:" | cut -d: -f2 | tr ' ' ',') 2>/dev/null || echo "No jobs running"
        ;;
    logs|out)
        echo "Tailing stdout logs for this run (Ctrl+C to exit)..."
        eval "tail -f $OUT_LOGS"
        ;;
    err|errors)
        echo "Tailing stderr logs for this run (Ctrl+C to exit)..."
        eval "tail -f $ERR_LOGS"
        ;;
    all)
        echo "Tailing all logs for this run - stdout + stderr (Ctrl+C to exit)..."
        eval "tail -f $OUT_LOGS $ERR_LOGS"
        ;;
    results)
        echo "Checking for completed results..."
        for dir in */; do
            if [ -f "$dir/benchmark_results.json" ]; then
                gpu=$(basename "$dir")
                echo "✓ $gpu - Complete"
            else
                gpu=$(basename "$dir")
                [ "$gpu" != "*/" ] && echo "○ $gpu - Pending/Running"
            fi
        done
        ;;
    compare)
        echo "Comparing results..."
        RESULTS_DIR="$(pwd)"
        cd ../.. && python -m cli.compare_latency "$RESULTS_DIR"
        ;;
    *)
        echo "Usage: ./monitor.sh [command]"
        echo ""
        echo "Commands:"
        echo "  status   - Check job status (default)"
        echo "  logs     - Tail stdout logs for this run"
        echo "  out      - Tail stdout logs (alias)"
        echo "  err      - Tail stderr logs (errors)"
        echo "  all      - Tail both stdout + stderr"
        echo "  results  - Check which GPUs have completed"
        echo "  compare  - Compare results"
        ;;
esac
MONITOR_EOF

chmod +x "${OUTPUT_DIR}/monitor.sh"

echo ""
echo "Helper script created: ${OUTPUT_DIR}/monitor.sh"
echo "  cd ${OUTPUT_DIR} && ./monitor.sh logs    # Tail all logs"
echo "  cd ${OUTPUT_DIR} && ./monitor.sh status  # Check job status"
echo "  cd ${OUTPUT_DIR} && ./monitor.sh results # Check completion"
echo "============================================================"
