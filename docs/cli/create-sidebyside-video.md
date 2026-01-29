# Create Side-by-Side Video CLI

Create side-by-side comparison videos showing original feed vs latency-delayed inference.

## Overview

This tool creates a split-screen video:
- **Left**: Real-time original video
- **Right**: Latency-delayed inference output
- Shows the gap between capture and detection

## Basic Usage

```bash
python -m cli.create_sidebyside_video \
  --original original_video.mp4 \
  --latency latency_delayed.mp4 \
  --results benchmark_results.json \
  --output sidebyside.mp4
```

## Parameters

### `--original PATH` (Required)
Original video file path.

### `--latency PATH` (Required)
Latency-delayed inference video from [Create Latency Video CLI](create-latency-video.md).

### `--results PATH` (Required)
Path to `benchmark_results.json` from [Benchmark Latency CLI](benchmark-latency.md).

### `--output PATH` (Required)
Output path for side-by-side video.

## Output

Creates a video with:
- **Left side**: Original video at original framerate
- **Right side**: Latency-delayed inference
- **Text overlay**: Latency statistics

### Example Frame

```
┌──────────────────┬──────────────────┐
│   Original       │   Inference      │
│   Feed           │   (Delayed)      │
│                  │                  │
│  [Live video]    │  [Detected +8ms] │
│                  │                  │
└──────────────────┴──────────────────┘
     Mean: 8.5ms  |  P95: 9.1ms
```

## Example

### Complete Workflow

```bash
# 1. Run benchmark
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

# 3. Create side-by-side comparison
python -m cli.create_sidebyside_video \
  --original test_video.mp4 \
  --latency latency_delayed.mp4 \
  --results benchmark_results/benchmark_results.json \
  --output sidebyside.mp4
```

### Using Benchmark's Built-in Option

Alternatively, use the benchmark CLI's built-in option:

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --create-latency-video
```

This automatically creates `sidebyside_latency.mp4`.

## Use Cases

### 1. Performance Demonstrations

Show clients or stakeholders the effect of latency:

```bash
python -m cli.create_sidebyside_video \
  --original demo.mp4 \
  --latency demo_latency.mp4 \
  --results demo_results.json \
  --output demo_comparison.mp4
```

### 2. Model Comparison

Compare multiple models side-by-side:

```bash
# Create comparison for base model
python -m cli.create_sidebyside_video \
  --original test.mp4 \
  --latency base_latency.mp4 \
  --results base_results.json \
  --output base_comparison.mp4

# Create comparison for large model
python -m cli.create_sidebyside_video \
  --original test.mp4 \
  --latency large_latency.mp4 \
  --results large_results.json \
  --output large_comparison.mp4
```

### 3. Documentation

Create videos for technical documentation or papers:

```bash
python -m cli.create_sidebyside_video \
  --original paper_video.mp4 \
  --latency paper_latency.mp4 \
  --results paper_results.json \
  --output figure_5_latency_comparison.mp4
```

## Related

- **[Benchmark Latency CLI](benchmark-latency.md)** - Run benchmarks
- **[Create Latency Video CLI](create-latency-video.md)** - Create latency videos
- **[Benchmarking Guide](../guides/benchmarking.md)** - Complete guide
