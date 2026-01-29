# Create Latency Video CLI

Create latency-delayed inference videos that show frames appearing when inference completes.

## Overview

This tool creates a visualization that shows how a real-time detector "sees" the world:
- Frames appear only when inference completes
- Reveals actual processing delay
- Useful for understanding real-time performance

## Basic Usage

```bash
python -m cli.create_latency_video \
  --video original_video.mp4 \
  --results benchmark_results.json \
  --frames annotated_frames/ \
  --output latency_video.mp4
```

## Parameters

### `--video PATH` (Required)
Original video file path.

### `--results PATH` (Required)
Path to `benchmark_results.json` from [Benchmark Latency CLI](benchmark-latency.md).

### `--frames PATH` (Required)
Directory containing annotated frames.

### `--output PATH` (Required)
Output path for latency video.

## How It Works

1. **Read latencies**: Load per-frame latencies from `benchmark_results.json`
2. **Compute delays**: Calculate when each frame completes inference
3. **Render video**: Show frames at their completion times

### Example Timeline

```
Frame 0: Captured at 0.000s → Inference done at 0.008s → Show at 0.008s
Frame 1: Captured at 0.033s → Inference done at 0.041s → Show at 0.041s
Frame 2: Captured at 0.067s → Inference done at 0.075s → Show at 0.075s
...
```

The gap between capture and display shows the model's latency.

## Example

### Complete Workflow

```bash
# 1. Run benchmark with frames
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --output benchmark_results/

# 2. Create latency video
python -m cli.create_latency_video \
  --video test_video.mp4 \
  --results benchmark_results/benchmark_results.json \
  --frames benchmark_results/frames/ \
  --output latency_delayed.mp4
```

### Using Benchmark's Built-in Option

Alternatively, use the benchmark CLI's built-in option:

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --create-latency-video
```

This automatically creates:
- `detected_latency.mp4`
- `sidebyside_latency.mp4`

## Use Cases

### 1. Real-Time System Design

Understand actual delay in your system:

```bash
python -m cli.create_latency_video \
  --video robot_camera.mp4 \
  --results benchmark_results.json \
  --frames frames/ \
  --output robot_latency.mp4
```

Watch the video to see if delays are acceptable.

### 2. Model Comparison

Compare latency visually:

```bash
# Base model
python -m cli.create_latency_video \
  --video test.mp4 \
  --results base_results.json \
  --frames base_frames/ \
  --output base_latency.mp4

# Large model
python -m cli.create_latency_video \
  --video test.mp4 \
  --results large_results.json \
  --frames large_frames/ \
  --output large_latency.mp4
```

### 3. Stakeholder Demonstrations

Show non-technical stakeholders the impact of latency:

```bash
python -m cli.create_latency_video \
  --video demo_video.mp4 \
  --results demo_results.json \
  --frames demo_frames/ \
  --output demo_latency.mp4
```

## Related

- **[Benchmark Latency CLI](benchmark-latency.md)** - Run benchmarks
- **[Create Side-by-Side Video CLI](create-sidebyside-video.md)** - Side-by-side comparison
- **[Benchmarking Guide](../guides/benchmarking.md)** - Complete guide
