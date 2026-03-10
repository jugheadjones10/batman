#!/bin/bash
# Quick trials: only the new configs we haven't tested yet
set -u

cd "$(dirname "$0")/.."
IMAGE="data/projects/Crane hook + Person/frames/video_1/video_1_000041.jpg"
PROMPTS="crane hook,person"
MIMALLOC="$HOME/.local/lib/libmimalloc.so"
TIMEOUT=120

run_trial() {
    local name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "TRIAL: $name"
    echo "=========================================="
    timeout "$TIMEOUT" env "$@" \
        SAM_IMAGE="$IMAGE" SAM_PROMPTS="$PROMPTS" SAM_MODEL="sam3.pt" \
        uv run python tests/sam_trial.py 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo ">>> RESULT: OK"
    elif [ $rc -eq 124 ]; then
        echo ">>> RESULT: TIMEOUT (>${TIMEOUT}s)"
    elif [ $rc -eq 134 ] || [ $rc -eq 137 ] || [ $rc -eq 139 ]; then
        echo ">>> RESULT: CRASHED (signal $((rc - 128)))"
    else
        echo ">>> RESULT: FAILED (exit $rc)"
    fi
    echo ""
    return $rc
}

RESULTS=()

# mimalloc + GPU (half=0)
run_trial "mimalloc + half=0 + auto" LD_PRELOAD="$MIMALLOC" SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("mimalloc+half0+auto:$?")

# mimalloc + GPU (half=1)
run_trial "mimalloc + half=1 + auto" LD_PRELOAD="$MIMALLOC" SAM_HALF=1 SAM_DEVICE=auto
RESULTS+=("mimalloc+half1+auto:$?")

# mimalloc + device=0 explicit
run_trial "mimalloc + half=0 + device=0" LD_PRELOAD="$MIMALLOC" SAM_HALF=0 SAM_DEVICE=0
RESULTS+=("mimalloc+half0+dev0:$?")

# CUDA_LAUNCH_BLOCKING
run_trial "CUDA_LAUNCH_BLOCKING + half=0" CUDA_LAUNCH_BLOCKING=1 SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("clb+half0+auto:$?")

# MALLOC_ARENA_MAX=1
run_trial "MALLOC_ARENA_MAX=1 + half=0" MALLOC_ARENA_MAX=1 SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("arena1+half0+auto:$?")

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    name="${r%%:*}"
    rc="${r##*:}"
    if [ "$rc" -eq 0 ]; then
        echo "  OK      $name"
    elif [ "$rc" -eq 124 ]; then
        echo "  TIMEOUT $name"
    elif [ "$rc" -eq 134 ] || [ "$rc" -eq 137 ] || [ "$rc" -eq 139 ]; then
        echo "  CRASH   $name (signal $((rc - 128)))"
    else
        echo "  FAIL    $name (exit $rc)"
    fi
done
