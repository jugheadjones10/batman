# Benchmark Latency CLI

Measure RF-DETR inference latency with detailed statistics and real-time capability analysis.

## Overview

The benchmark CLI provides:

- **Latency statistics**: Mean, median, percentiles (P50, P95, P99)
- **Throughput metrics**: FPS and frames per second
- **Real-time capability**: Checks if model runs faster than video framerate
- **Visualization**: Optional latency-delayed videos
- **Benchmark modes**: Synthetic images or real video frames

## Basic Usage

```bash
# Benchmark with synthetic images
python -m cli.benchmark_latency --run my_training_run

# Benchmark with real video
python -m cli.benchmark_latency --run my_training_run --video test.mp4
```

## Command Builder

<div class="command-builder-widget" data-tool="benchmark_latency" data-params='[
  {"name": "checkpoint", "type": "path", "required": false, "description": "Path to checkpoint file", "group": "Model"},
  {"name": "run", "type": "text", "required": false, "description": "Run name", "group": "Model"},
  {"name": "latest", "type": "flag", "description": "Use latest run", "group": "Model"},
  {"name": "model", "type": "choice", "choices": ["base", "large"], "default": "base", "description": "Model architecture", "group": "Model"},
  {"name": "image-size", "type": "number", "default": 640, "min": 320, "max": 1280, "step": 32, "description": "Image size", "group": "Model"},
  {"name": "no-optimize", "type": "flag", "description": "Disable model optimization", "group": "Model"},
  {"name": "video", "type": "path", "description": "Video file for benchmark", "group": "Benchmark"},
  {"name": "use-video", "type": "flag", "description": "Use default video (crane_hook_1_short.mp4)", "group": "Benchmark"},
  {"name": "warmup", "type": "number", "default": 10, "min": 0, "description": "Warmup runs", "group": "Benchmark"},
  {"name": "runs", "type": "number", "default": 100, "min": 1, "description": "Test runs", "group": "Benchmark"},
  {"name": "output", "type": "path", "default": "benchmark_results/", "description": "Output directory", "group": "Output"},
  {"name": "create-latency-video", "type": "flag", "description": "Create latency visualization video", "group": "Output"}
]'></div>

## Parameters

### Model Selection (Choose One)

#### `--checkpoint PATH`

Path to model checkpoint file.

```bash
--checkpoint runs/my_run/best.pth
```

#### `--run NAME`

Run name to auto-find checkpoint in `runs/<name>/`.

```bash
--run rfdetr_h100_20260120_105925
```

#### `--latest`

Use the most recent training run.

### Model Configuration

#### `--model SIZE`

Model architecture size.

- **Choices**: `base`, `large`
- **Default**: `base`

#### `--image-size N`

Image size for synthetic benchmark.

- **Default**: `640`
- **Common values**: `512`, `640`, `800`, `1024`

#### `--no-optimize`

Disable model optimization (use if encountering errors).

### Benchmark Configuration

#### `--video PATH`

Video file path for realistic benchmark.

```bash
--video path/to/test_video.mp4
```

Uses video frames instead of synthetic dummy images.

#### `--use-video`

Use default video (`crane_hook_1_short.mp4`).

#### `--warmup N`

Number of warmup runs before measuring.

- **Default**: `10`
- **Purpose**: Stabilize GPU and model state

#### `--runs N`

Number of test runs to measure.

- **Default**: `100`
- **Higher values**: More accurate statistics

### Output

#### `--output PATH` or `-o PATH`

Output directory for benchmark results.

- **Default**: `benchmark_results/`

#### `--create-latency-video`

Create latency-delayed visualization video (requires `--video`).

## Output

### Benchmark Results File

Creates `benchmark_results.json`:

```json
{
  "gpu": "NVIDIA H100",
  "model": "base",
  "image_size": 640,
  "runs": 100,
  "warmup": 10,
  "latency_ms": {
    "mean": 8.5,
    "median": 8.3,
    "std": 0.4,
    "min": 7.9,
    "max": 10.2,
    "p50": 8.3,
    "p95": 9.1,
    "p99": 9.8
  },
  "throughput": {
    "fps": 117.6,
    "frames_per_second": 117.6
  },
  "real_time_capable": {
    "30fps": true,
    "60fps": true,
    "120fps": false
  },
  "frame_latencies": [8.1, 8.3, 8.5, ...]
}
```

### Console Output

```
=== Benchmark Results ===
GPU: NVIDIA H100
Model: base (640x640)
Runs: 100 (warmup: 10)

Latency (ms):
  Mean:   8.5 ± 0.4
  Median: 8.3
  Min:    7.9
  Max:    10.2
  P50:    8.3
  P95:    9.1
  P99:    9.8

Throughput:
  FPS: 117.6

Real-time capable:
  ✓ 30 FPS (33.3 ms/frame)
  ✓ 60 FPS (16.7 ms/frame)
  ✗ 120 FPS (8.3 ms/frame)
```

### Latency Video

If `--create-latency-video` is used:

- `detected_latency.mp4`: Frames appear when inference completes
- `sidebyside_latency.mp4`: Original vs latency-delayed side-by-side

## Examples

### Example 1: Quick Benchmark

Benchmark with synthetic images:

```bash
python -m cli.benchmark_latency --run my_training_run
```

### Example 2: Realistic Benchmark

Use real video frames:

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --runs 100
```

### Example 3: High-Precision Benchmark

More runs for accurate statistics:

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --warmup 20 \
  --runs 500
```

### Example 4: Create Visualization

Generate latency-delayed video:

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --create-latency-video \
  --output benchmark_results/my_test
```

### Example 5: Large Model Benchmark

Test large model performance:

```bash
python -m cli.benchmark_latency \
  --run my_large_run \
  --model large \
  --video test_video.mp4 \
  --image-size 800
```

### Example 6: Latest Run Quick Test

Benchmark most recent training:

```bash
python -m cli.benchmark_latency \
  --latest \
  --use-video \
  --runs 50
```

## Understanding Results

### Latency Metrics

| Metric           | Description     | When to Use         |
| ---------------- | --------------- | ------------------- |
| **Mean**         | Average latency | Overall performance |
| **Median (P50)** | Middle value    | Typical performance |
| **P95**          | 95th percentile | Near-worst case     |
| **P99**          | 99th percentile | Worst case          |
| **Std Dev**      | Variability     | Consistency         |

### Real-Time Capability

Checks if mean latency meets framerate requirements:

- **30 FPS**: ≤ 33.3 ms/frame
- **60 FPS**: ≤ 16.7 ms/frame
- **120 FPS**: ≤ 8.3 ms/frame

### Throughput

Frames per second the model can process:

```
FPS = 1000 / mean_latency_ms
```

## Benchmarking Best Practices

### 1. Use Real Video Frames

Synthetic images may not reflect real-world performance:

```bash
--video test_video.mp4
```

### 2. Sufficient Warmup

Allow GPU to reach steady state:

```bash
--warmup 20
```

### 3. Adequate Test Runs

More runs = more accurate statistics:

```bash
--runs 200
```

### 4. Match Training Configuration

Use same image size as training:

```bash
--image-size 640  # If trained with --image-size 640
```

### 5. Monitor GPU State

Ensure no other processes are using GPU:

```bash
nvidia-smi
```

## Comparing Performance

### Compare Across GPUs

Use the [Compare Latency CLI](compare-latency.md):

```bash
python -m cli.compare_latency benchmark_results/
```

### Compare Models

Benchmark multiple models:

```bash
# Base model
python -m cli.benchmark_latency --run base_run --model base -o results/base/

# Large model
python -m cli.benchmark_latency --run large_run --model large -o results/large/
```

### Compare Image Sizes

Test different resolutions:

```bash
# 640x640
python -m cli.benchmark_latency --run my_run --image-size 640 -o results/640/

# 800x800
python -m cli.benchmark_latency --run my_run --image-size 800 -o results/800/
```

## Latency Visualization

The `--create-latency-video` option creates two videos:

### 1. Latency-Delayed Video

Shows how the model "sees" the world:

- Frames appear when inference completes
- Reveals actual processing delay
- Useful for real-time system design

### 2. Side-by-Side Comparison

Original video alongside latency-delayed output:

- Left: Real-time original
- Right: Inference-delayed output
- Shows gap between capture and detection

## Related

- **[Compare Latency CLI](compare-latency.md)** - Compare benchmark results
- **[Submit Benchmark Script](../scripts/submit-benchmark.md)** - SLURM benchmarking
- **[Benchmarking Guide](../guides/benchmarking.md)** - Complete guide
- **[Create Latency Video CLI](create-latency-video.md)** - Standalone video creation
