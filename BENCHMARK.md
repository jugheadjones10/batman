# GPU Latency Benchmarking Guide

This guide explains how to benchmark RF-DETR inference latency across different GPUs to measure real-time performance capabilities.

## Overview

The benchmarking system measures:
- **Per-frame latency**: Mean, P50, P95, P99 percentiles
- **Throughput**: Frames per second (FPS)
- **Real-time capability**: Whether GPU can maintain 30 FPS or 60 FPS video processing
- **Latency visualization**: Side-by-side videos showing real-time vs. delayed inference output

## Quick Start

### 1. Run Benchmark on Multiple GPUs

By default, the benchmark uses real video frames from `crane_hook_1_short.mp4` for realistic latency measurements:

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus all
```

Or specify specific GPUs:

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus h200,a100-80,a100-40
```

Use a different video:

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus h200,a100-80 --video my_video.mp4
```

Use synthetic dummy images (for pure GPU comparison without video overhead):

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus h200,a100-80 --no-video
```

Create side-by-side latency visualization video:

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus h200 --create-latency-video
```

This generates a video showing the original feed alongside inference results with realistic latency delays, simulating what real-time processing would look like.

### 2. Monitor Jobs

The script creates a helper script in your results directory for easy monitoring:

```bash
# Navigate to your results directory
cd benchmark_results/20260128_120000/

# Quick commands via helper script
./monitor.sh logs      # Tail all logs (Ctrl+C to exit)
./monitor.sh status    # Check job status
./monitor.sh results   # See which GPUs completed
./monitor.sh compare   # Compare results when done
```

Or use SLURM commands directly:

```bash
# Check all your jobs
squeue -u $USER
watch -n 2 squeue -u $USER  # Auto-refresh every 2 seconds

# Tail all benchmark logs
tail -f logs/slurm_*_benchmark_*.out

# Tail specific GPU log (job ID shown when submitted)
tail -f logs/slurm_12345_benchmark_h200.out
```

### 3. Compare Results

Once all jobs complete, generate a comparison report:

```bash
python -m cli.compare_latency benchmark_results/20260128_120000/
```

Save as markdown:

```bash
python -m cli.compare_latency benchmark_results/20260128_120000/ --output comparison.md
```

## Detailed Usage

### Single GPU Benchmark

Run benchmark on a single GPU manually:

```bash
# On the GPU server
python -m cli.benchmark_latency --run rfdetr_h200_20260120_105925
```

With custom parameters:

```bash
python -m cli.benchmark_latency \
    --run rfdetr_h200_20260120_105925 \
    --model base \
    --image-size 640 \
    --warmup 10 \
    --runs 100
```

### Multi-GPU Benchmark Options

```bash
./submit_benchmark.sh [OPTIONS]

Model Selection (required, mutually exclusive):
  --checkpoint PATH    Path to checkpoint file
  --run NAME          Run name
  --latest            Use the latest run

Model Configuration:
  --model SIZE        Model size: base or large (default: base)
  --image-size SIZE   Image size for synthetic benchmark (default: 640)
  --no-optimize       Disable model optimization

Benchmark Configuration:
  --warmup N          Number of warmup runs (default: 10)
  --runs N            Number of benchmark runs (default: 100)
  --video FILE        Video file for realistic benchmark (default: crane_hook_1_short.mp4)
  --no-video          Use synthetic dummy images instead of video
  --create-latency-video  Create side-by-side latency visualization video (requires video)

GPU Selection (required):
  --gpus TYPES        Comma-separated GPU types or 'all'
                      Options: h200,h100-96,h100-47,a100-80,a100-40,nv

SLURM Options:
  --time LIMIT        Time limit (default: 00:30:00)
```

### Examples

Benchmark all GPUs with default settings:

```bash
./submit_benchmark.sh --latest --gpus all
```

Benchmark specific GPUs with more test runs:

```bash
./submit_benchmark.sh --run my_run --gpus h200,a100-80 --runs 200
```

Test both base and large models:

```bash
./submit_benchmark.sh --run my_run --gpus h200,a100-80 --model base
./submit_benchmark.sh --run my_run --gpus h200,a100-80 --model large
```

## Benchmark Modes

### Video Benchmark (Default)

Uses real video frames for realistic latency measurements:
- Includes actual frame content (varying complexity)
- Tests real-world inference performance
- Default video: `crane_hook_1_short.mp4`

### Synthetic Benchmark (`--no-video`)

Uses random dummy images for pure GPU comparison:
- Consistent, reproducible results
- Isolates GPU compute performance
- No video I/O overhead
- Good for comparing GPUs under identical conditions

### Latency Visualization (`--create-latency-video`)

Creates a side-by-side comparison video showing real-time vs. delayed output:
- **Left side**: Original video playing at real-time speed
- **Right side**: Inference results delayed by actual per-frame latency
- **Purpose**: Visualize what real-time processing would look like with live video feed
- **Requirements**: Must be used with video input (not compatible with `--no-video`)
- **Output files**:
  - `detected_latency.mp4`: Inference video with realistic latency delays
  - `sidebyside_latency.mp4`: Side-by-side comparison video
  - `frames/`: Directory with annotated frames (used for video generation)

This feature helps you understand the practical impact of inference latency by showing exactly how the output would lag behind a live feed.

## Understanding Results

### Latency Metrics

- **Mean**: Average latency across all test runs
- **P50 (Median)**: 50% of inferences complete within this time
- **P95**: 95% of inferences complete within this time
- **P99**: 99% of inferences complete within this time (important for real-time)
- **Max**: Worst-case latency

### Real-Time Thresholds

For real-time video processing:
- **30 FPS**: Requires P99 < 33.33ms (1000ms / 30 frames)
- **60 FPS**: Requires P99 < 16.67ms (1000ms / 60 frames)

We use **P99 instead of mean** because:
- Real-time systems need to handle worst-case scenarios
- P99 ensures 99% of frames meet the latency requirement
- Occasional slow frames (1%) won't disrupt video playback

### Example Output

```
========================================
GPU LATENCY COMPARISON
========================================
Benchmark Mode: video
Video: crane_hook_1_short.mp4 (1920x1080 @ 30.0 FPS)

GPU Type        GPU Name                  Mean       P50        P95        P99        FPS      30fps    60fps   
------------------------------------------------------------------------------------------------------------------------
h200            NVIDIA H200               12.3ms     11.8ms     15.2ms     18.1ms     81.3     ✓        ✓       
a100-80         NVIDIA A100 80GB          15.4ms     14.9ms     19.1ms     22.3ms     64.9     ✓        ✗       
a100-40         NVIDIA A100 40GB          16.8ms     16.2ms     20.8ms     24.5ms     59.5     ✓        ✗       
```

## Directory Structure

```
benchmark_results/
└── 20260128_120000/          # Timestamp of benchmark suite
    ├── job_info.txt          # Job metadata
    ├── h200/                 # GPU-specific results
    │   ├── benchmark_results.json
    │   ├── frames/           # Annotated frames (if --create-latency-video used)
    │   │   ├── frame_00000.jpg
    │   │   ├── frame_00001.jpg
    │   │   └── ...
    │   ├── detected_latency.mp4      # Latency-delayed inference video
    │   └── sidebyside_latency.mp4    # Side-by-side comparison video
    ├── a100-80/
    │   └── benchmark_results.json
    └── a100-40/
        └── benchmark_results.json
```

## Result Format

Each `benchmark_results.json` contains:

```json
{
  "timestamp": "2026-01-28T12:00:00",
  "hostname": "node123",
  "checkpoint": "runs/rfdetr_h200_20260120_105925/best.pth",
  "model_size": "base",
  "benchmark_mode": "video",
  "optimized": true,
  "gpu_info": {
    "available": true,
    "name": "NVIDIA H200",
    "memory_gb": 141.0,
    "device": "cuda"
  },
  "video_info": {
    "path": "crane_hook_1_short.mp4",
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "total_frames": 150
  },
  "benchmark_config": {
    "warmup_runs": 10,
    "test_runs": 100
  },
  "metrics": {
    "mean_ms": 12.3,
    "std_ms": 2.1,
    "min_ms": 10.1,
    "max_ms": 22.5,
    "p50_ms": 11.8,
    "p95_ms": 15.2,
    "p99_ms": 18.1,
    "fps": 81.3
  },
  "realtime_capable": {
    "30fps": true,
    "60fps": true,
    "native_fps": true,
    "native_fps_value": 30.0
  },
  "per_frame_results": [
    {"frame_idx": 0, "inference_time_ms": 12.1},
    {"frame_idx": 1, "inference_time_ms": 11.8},
    {"frame_idx": 2, "inference_time_ms": 13.2}
  ]
}
```

Notes:
- For synthetic benchmarks (`--no-video`), `video_info` is replaced with `image_size` and `native_fps` fields are not present.
- `per_frame_results` contains per-frame timing data, used for latency visualization.

## Tips

1. **Run warmup**: Always use warmup runs to avoid cold-start effects from GPU initialization and CUDA kernel compilation.

2. **Sufficient test runs**: Use at least 100 test runs to get statistically significant percentile values.

3. **Consistent conditions**: Run all benchmarks with the same model checkpoint and parameters for fair comparison.

4. **Check GPU load**: Ensure GPUs aren't running other workloads during benchmarking.

5. **Model optimization**: The `--no-optimize` flag disables model optimization, useful for debugging but slower.

6. **Latency visualization**: Use `--create-latency-video` to see the practical impact of latency:
   - Shows exactly how delayed the output would be in a real-time scenario
   - Helps identify if inference is fast enough for your use case
   - Makes P99 latency metrics more tangible and understandable
   - Note: This increases benchmark time and storage (saves annotated frames)

## Troubleshooting

### Jobs fail immediately

Check SLURM logs in `logs/slurm_*_benchmark_*.err`:

```bash
cat logs/slurm_*_benchmark_*.err
```

Common issues:
- Virtual environment not activated
- Missing dependencies
- Invalid checkpoint path

### Results show CPU instead of GPU

Ensure CUDA is available and visible in the job:

```bash
# Add to SLURM script for debugging
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

### P99 latency is much higher than mean

This is normal! P99 captures worst-case scenarios:
- Occasional GPU thermal throttling
- Operating system interrupts
- Memory allocation overhead

Focus on P99 for real-time decisions, not mean.

## Next Steps

After benchmarking:

1. **Analyze results**: Identify which GPUs meet your real-time requirements
2. **Test with real videos**: Run full inference on actual videos to measure end-to-end latency
3. **Consider tracking overhead**: ByteTrack and Kalman prediction add processing time
4. **Optimize frame intervals**: Use `--frame-interval` in inference to skip frames if needed
