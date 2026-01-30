#!/bin/bash
#===============================================================================
# Run All Class Imbalance Experiments on SLURM
#===============================================================================
#
# Usage:
#   ./experiments/run_all.sh              # Submit all experiments
#   ./experiments/run_all.sh --dry-run    # Preview without submitting
#   ./experiments/run_all.sh --local      # Run locally (no SLURM)
#
#===============================================================================

set -e

cd "$(dirname "$0")/.."  # Change to project root

#-------------------------------------------------------------------------------
# Configuration (NUS SoC Compute Cluster)
#-------------------------------------------------------------------------------
# See: https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/
#
# Available GPUs:
#   h100-96  - H100 96GB (xgpi0-9, 2 per node)
#   h100-47  - H100 47GB MIG (xgpi10-19, 4 per node)
#   a100-80  - A100 80GB (xgph0-9)
#   a100-40  - A100 40GB (xgpg0-9, xgph10-19 MIG)
#   nv       - Older GPUs (V100, Titan V, Titan RTX, T4)
#
PARTITION="gpu-long"
GPU_TYPE="h100-96"  # H100 96GB
CPUS_PER_TASK=8
MEM_GB=64
TIMEOUT_MIN=1440  # 24 hours
ARRAY_PARALLELISM=4

# Experiments to run
EXPERIMENTS="exp_person_25,exp_person_50,exp_person_75,exp_person_100"

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------
DRY_RUN=false
LOCAL_MODE=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            echo "DRY RUN MODE - showing config only"
            ;;
        --local)
            LOCAL_MODE=true
            echo "LOCAL MODE - running without SLURM"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Preview config without running"
            echo "  --local      Run locally without SLURM"
            echo "  --help       Show this help"
            exit 0
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Run Experiments
#-------------------------------------------------------------------------------
echo "============================================================"
echo "Class Imbalance Experiments"
echo "============================================================"
echo "Experiments: $EXPERIMENTS"
echo "Partition:   $PARTITION"
echo "GPU Type:    $GPU_TYPE"
echo "Timeout:     $TIMEOUT_MIN minutes"
echo "============================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    # Dry run - show config for first experiment only
    echo "Showing resolved config for exp_person_25..."
    echo ""
    python experiments/train_experiment.py \
        experiment=exp_person_25 \
        --cfg job
    
    echo ""
    echo "============================================================"
    echo "Would submit these experiments: $EXPERIMENTS"
    echo "============================================================"
    exit 0
fi

if [ "$LOCAL_MODE" = true ]; then
    # Run locally without SLURM (one at a time)
    echo "Running locally (no SLURM)..."
    python experiments/train_experiment.py --multirun \
        experiment=$EXPERIMENTS \
        hydra/launcher=basic
else
    # Run on SLURM (NUS SoC Compute Cluster)
    # Uses --gres format for GPU allocation: gpu:<type>:<count>
    echo "Submitting to SLURM..."
    
    python experiments/train_experiment.py --multirun \
        experiment=$EXPERIMENTS \
        hydra/launcher=submitit_slurm \
        hydra.launcher.partition=$PARTITION \
        hydra.launcher.cpus_per_task=$CPUS_PER_TASK \
        hydra.launcher.mem_gb=$MEM_GB \
        hydra.launcher.timeout_min=$TIMEOUT_MIN \
        hydra.launcher.array_parallelism=$ARRAY_PARALLELISM \
        "hydra.launcher.gres=gpu:${GPU_TYPE}:1"
fi

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo "============================================================"
    echo "Jobs submitted!"
    echo ""
    echo "Monitor with:"
    echo "  squeue -u \$USER"
    echo ""
    echo "View logs:"
    echo "  tail -f experiments/multirun/*/exp_person_*/train_experiment.log"
    echo ""
    echo "Collect results after completion:"
    echo "  python experiments/collect_results.py --latest"
    echo "============================================================"
fi
