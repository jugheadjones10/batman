# Troubleshooting

## SAM3 auto-label: heap corruption on WSL2

When running SAM3 auto-labeling on **WSL2 with a CUDA GPU**, the backend process can crash with one of these glibc errors:

```
free(): double free detected in tcache 2
malloc_consolidate(): unaligned fastbin chunk detected
corrupted double-linked list
```

### Root cause

There are **three** interacting problems:

1. **glibc heap corruption with CUDA on WSL2.**
   PyTorch's CUDA memory management triggers a bug in glibc's malloc (specifically the tcache and fastbin paths) when running inside WSL2. This happens during the first `set_image()` / `setup_model()` call, when the model weights are moved to the GPU. It is reproducible with PyTorch 2.9+cu128, Ultralytics 8.3.x, and glibc 2.39 on WSL2 kernel 6.6.x. The bug does **not** occur on native Linux or on CPU.

2. **`LD_PRELOAD` breaks CUDA auto-detection.**
   The standard fix for glibc malloc bugs is to preload an alternative allocator (jemalloc, mimalloc). However, any `LD_PRELOAD` causes `torch.cuda.is_available()` to return `False` inside the child process, so `device=auto` fails with _"Invalid CUDA 'device=auto'"_. The GPU is still usable -- you just need to pass an explicit device like `device=0`.

3. **Ultralytics prints its banner to stdout.**
   On the first inference call, Ultralytics prints `Ultralytics 8.3.x 🚀 ... CUDA:0 (...)` to stdout regardless of `verbose=False`. If the worker uses stdout for JSON-line IPC, the parent reads the banner as the response, fails to parse it as JSON, and reports a "crash" that never actually happened.

### How Batman solves it

By default (`BATMAN_SAM_IN_PROCESS=0`), SAM3 runs in a **subprocess worker** (`backend/app/services/sam_worker.py`) with three mitigations:

| Problem | Fix |
|---------|-----|
| glibc heap corruption | Worker starts with `LD_PRELOAD=~/.local/lib/libmimalloc.so` (Microsoft's mimalloc allocator replaces glibc malloc) |
| CUDA auto-detect broken | Parent resolves the device **before** spawning the worker (where `torch.cuda.is_available()` still works) and passes `BATMAN_SAM_DEVICE=0` explicitly |
| Ultralytics stdout pollution | Worker duplicates the real stdout fd, then redirects `sys.stdout` to stderr. JSON protocol writes go to the saved fd; Ultralytics prints harmlessly to stderr |

If the worker still crashes (e.g. on a different system), the API server stays up and the job is marked failed with _"SAM worker crashed or timed out; please retry auto-label"_.

### Setup: building mimalloc

mimalloc must be built from source (no sudo required):

```bash
cd /tmp
git clone --depth 1 https://github.com/microsoft/mimalloc.git
cd mimalloc && mkdir -p out/release && cd out/release

# cmake can be installed via: uv pip install cmake
CMAKE=$(python -c "import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")
$CMAKE ../.. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc)
make install
```

The worker looks for `~/.local/lib/libmimalloc.so` automatically. If not found, it falls back to `~/.local/lib/libjemalloc.so`, then to no preload (with a warning).

### Configuration reference

| Env var | Default | Description |
|---------|---------|-------------|
| `BATMAN_SAM_DEVICE` | `auto` | Device for SAM3. `auto` resolves to `0` (GPU) or `cpu`. Use `cpu` to avoid all CUDA-related crashes (slower). |
| `BATMAN_SAM_IN_PROCESS` | `0` | Set to `1` to run SAM3 inside the API process (no subprocess, no mimalloc). Faster but a crash kills the server. |
| `BATMAN_SAM_MODEL_PATH` | `./sam3.pt` | Path to the SAM3 model file. |

### What we tried (and why it didn't work)

| Approach | Result |
|----------|--------|
| Main-thread model load (no thread pool) | Still crashes -- corruption is in CUDA inference, not threading |
| `asyncio.run_in_executor` for model load | Double-free -- PyTorch + threads + CUDA = bad |
| `GLIBC_TUNABLES=glibc.malloc.tcache_count=0` | Different crash: `malloc_consolidate(): unaligned fastbin chunk detected` |
| `MALLOC_ARENA_MAX=1` | Still `double free detected in tcache 2` |
| `CUDA_LAUNCH_BLOCKING=1` | Still crashes |
| `LD_PRELOAD=libjemalloc.so` | CUDA becomes invisible (`torch.cuda.is_available()=False`), and jemalloc with `device=auto` fails |
| `LD_PRELOAD=libmimalloc.so` + `device=auto` | Same CUDA invisible issue |
| `LD_PRELOAD=libmimalloc.so` + `device=0` | **Works** -- mimalloc avoids malloc corruption, explicit device bypasses broken auto-detect |
| `half=True` vs `half=False` | No effect on the crash (both crash with glibc, both work with mimalloc) |
| Proxy timeout increase (Vite) | Helps with slow first requests but doesn't fix the underlying crash |

### Debugging tips

- The worker writes diagnostic info to stderr: device, model path, and `LD_PRELOAD` value. Check the backend logs.
- To test the worker in isolation:

```bash
echo '{"image_path": "path/to/image.jpg", "class_prompts": ["person"]}' | \
  LD_PRELOAD=$HOME/.local/lib/libmimalloc.so \
  BATMAN_SAM_DEVICE=0 \
  BATMAN_SAM_MODEL_PATH=sam3.pt \
  uv run python -m backend.app.services.sam_worker
```

- To run the full trial harness across allocator/device combos: `bash tests/run_sam_trials.sh`

### Related links

- [Ultralytics #21176](https://github.com/ultralytics/ultralytics/issues/21176) -- double free on WSL2
- [PyTorch #50137](https://github.com/pytorch/pytorch/issues/50137) -- double free with CUDA
- [glibc malloc tcache improvements (2025)](https://sourceware.org/pipermail/glibc-cvs/2025q2/088463.html)
