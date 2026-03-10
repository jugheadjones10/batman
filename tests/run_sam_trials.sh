#!/bin/bash
# Test SAM3 with different allocator/config combos. Each trial is a fresh process.
set -u

cd "$(dirname "$0")/.."
IMAGE="data/projects/Crane hook + Person/frames/video_1/video_1_000041.jpg"
PROMPTS="crane hook,person"
JEMALLOC="$HOME/.local/lib/libjemalloc.so"
TIMEOUT=120

run_trial() {
    local name="$1"
    shift
    echo ""
    echo "=========================================="
    echo "TRIAL: $name"
    echo "=========================================="
    # Run with timeout, capture exit code
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

echo "jemalloc: $JEMALLOC (exists: $([ -f "$JEMALLOC" ] && echo yes || echo no))"
echo "image: $IMAGE"
echo ""

RESULTS=()

# Trial 1: jemalloc + half=0
if [ -f "$JEMALLOC" ]; then
    run_trial "jemalloc + half=0 + auto" LD_PRELOAD="$JEMALLOC" SAM_HALF=0 SAM_DEVICE=auto
    RESULTS+=("jemalloc+half0+auto:$?")
fi

# Trial 2: jemalloc + half=1
if [ -f "$JEMALLOC" ]; then
    run_trial "jemalloc + half=1 + auto" LD_PRELOAD="$JEMALLOC" SAM_HALF=1 SAM_DEVICE=auto
    RESULTS+=("jemalloc+half1+auto:$?")
fi

# Trial 3: tcache=0 + half=0
run_trial "tcache=0 + half=0 + auto" GLIBC_TUNABLES=glibc.malloc.tcache_count=0 SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("tcache0+half0+auto:$?")

# Trial 4: tcache=0 + half=1
run_trial "tcache=0 + half=1 + auto" GLIBC_TUNABLES=glibc.malloc.tcache_count=0 SAM_HALF=1 SAM_DEVICE=auto
RESULTS+=("tcache0+half1+auto:$?")

# Trial 5: default glibc + half=0
run_trial "default + half=0 + auto" SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("default+half0+auto:$?")

# Trial 6: default glibc + half=1
run_trial "default + half=1 + auto" SAM_HALF=1 SAM_DEVICE=auto
RESULTS+=("default+half1+auto:$?")

# Trial 7: mimalloc + half=0 + auto (GPU)
MIMALLOC="$HOME/.local/lib/libmimalloc.so"
if [ -f "$MIMALLOC" ]; then
    run_trial "mimalloc + half=0 + auto" LD_PRELOAD="$MIMALLOC" SAM_HALF=0 SAM_DEVICE=auto
    RESULTS+=("mimalloc+half0+auto:$?")
fi

# Trial 8: mimalloc + half=1 + auto (GPU)
if [ -f "$MIMALLOC" ]; then
    run_trial "mimalloc + half=1 + auto" LD_PRELOAD="$MIMALLOC" SAM_HALF=1 SAM_DEVICE=auto
    RESULTS+=("mimalloc+half1+auto:$?")
fi

# Trial 9: CUDA_LAUNCH_BLOCKING + half=0 + auto
run_trial "CUDA_LAUNCH_BLOCKING + half=0" CUDA_LAUNCH_BLOCKING=1 SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("clb+half0+auto:$?")

# Trial 10: MALLOC_ARENA_MAX=1 + half=0
run_trial "MALLOC_ARENA_MAX=1 + half=0" MALLOC_ARENA_MAX=1 SAM_HALF=0 SAM_DEVICE=auto
RESULTS+=("arena1+half0+auto:$?")

# Trial 11: jemalloc + half=0 + cpu (known working baseline)
if [ -f "$JEMALLOC" ]; then
    run_trial "jemalloc + half=0 + cpu" LD_PRELOAD="$JEMALLOC" SAM_HALF=0 SAM_DEVICE=cpu
    RESULTS+=("jemalloc+half0+cpu:$?")
fi

# Trial 12: mimalloc + half=0 + device=0 (explicit GPU)
if [ -f "$MIMALLOC" ]; then
    run_trial "mimalloc + half=0 + device=0" LD_PRELOAD="$MIMALLOC" SAM_HALF=0 SAM_DEVICE=0
    RESULTS+=("mimalloc+half0+dev0:$?")
fi

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
