# GPU Latency Benchmarking Guide

This guide explains how to benchmark RF-DETR inference latency across different GPUs to measure real-time performance capabilities.

## Overview

The benchmarking system measures:
- **Per-frame latency**: Mean, P50, P95, P99 percentiles
- **Throughput**: Frames per second (FPS)
- **Real-time capability**: Whether GPU can maintain 30 FPS or 60 FPS video processing

## Quick Start

### 1. Run Benchmark on Multiple GPUs

Submit benchmark jobs for all available GPUs:

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus all
```

Or specify specific GPUs:

```bash
./submit_benchmark.sh --run rfdetr_h200_20260120_105925 --gpus h200,a100-80,a100-40
```

### 2. Monitor Jobs

Check job status:

```bash
squeue -u $USER
watch squeue -u $USER  # Auto-refresh
```

Check logs:

```bash
tail -f logs/slurm_*_benchmark_*.out
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
  --image-size SIZE   Image size (default: 640)
  --no-optimize       Disable model optimization

Benchmark Configuration:
  --warmup N          Number of warmup runs (default: 10)
  --runs N            Number of benchmark runs (default: 100)

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
    │   └── benchmark_results.json
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
  "image_size": 640,
  "optimized": true,
  "gpu_info": {
    "available": true,
    "name": "NVIDIA H200",
    "memory_gb": 141.0,
    "device": "cuda"
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
    "60fps": true
  }
}
```

## Tips

1. **Run warmup**: Always use warmup runs to avoid cold-start effects from GPU initialization and CUDA kernel compilation.

2. **Sufficient test runs**: Use at least 100 test runs to get statistically significant percentile values.

3. **Consistent conditions**: Run all benchmarks with the same model checkpoint and parameters for fair comparison.

4. **Check GPU load**: Ensure GPUs aren't running other workloads during benchmarking.

5. **Model optimization**: The `--no-optimize` flag disables model optimization, useful for debugging but slower.

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
