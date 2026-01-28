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
echo "Image size: ${IMAGE_SIZE}"
echo "Warmup runs: ${WARMUP}"
echo "Test runs: ${RUNS}"
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
python3 -m cli.benchmark_latency ${MODEL_ARG} ${OPTIONAL_ARGS} \\
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
echo "  # Monitor all logs at once (stdout + stderr)"
echo "  tail -f logs/slurm_*_benchmark_*.out logs/slurm_*_benchmark_*.err"
echo ""
echo "  # Monitor all stdout only"
echo "  tail -f logs/slurm_*_benchmark_*.out"
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
Image Size: ${IMAGE_SIZE}
Warmup Runs: ${WARMUP}
Test Runs: ${RUNS}

Monitoring Commands
===================
# Check job status
squeue -j $(IFS=,; echo "${JOB_IDS[*]}")

# Monitor all logs (stdout + stderr)
tail -f logs/slurm_*_benchmark_*.out logs/slurm_*_benchmark_*.err

# Monitor stdout only
tail -f logs/slurm_*_benchmark_*.out

# Monitor stderr only (errors)
tail -f logs/slurm_*_benchmark_*.err

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

# Create a helper script for monitoring
cat > "${OUTPUT_DIR}/monitor.sh" << 'MONITOR_EOF'
#!/bin/bash
# Quick monitoring helper for this benchmark suite

case "${1:-status}" in
    status)
        echo "Checking job status..."
        squeue -j $(cat job_info.txt | grep "Job IDs:" | cut -d: -f2 | tr ' ' ',') 2>/dev/null || echo "No jobs running"
        ;;
    logs|out)
        echo "Tailing stdout logs (Ctrl+C to exit)..."
        tail -f ../../logs/slurm_*_benchmark_*.out
        ;;
    err|errors)
        echo "Tailing stderr logs (Ctrl+C to exit)..."
        tail -f ../../logs/slurm_*_benchmark_*.err
        ;;
    all)
        echo "Tailing all logs - stdout + stderr (Ctrl+C to exit)..."
        tail -f ../../logs/slurm_*_benchmark_*.out ../../logs/slurm_*_benchmark_*.err
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
        echo "  logs     - Tail stdout logs"
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
