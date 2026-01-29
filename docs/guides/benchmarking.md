# Benchmarking Guide

Complete guide for measuring and comparing RF-DETR inference performance.

## Overview

Benchmarking workflow:
1. **Plan Benchmarks** - Define goals and scope
2. **Run Benchmarks** - Execute measurements
3. **Analyze Results** - Interpret metrics
4. **Compare Performance** - Evaluate across GPUs
5. **Optimize** - Improve performance

## Why Benchmark?

- **GPU Selection**: Choose cost-effective hardware
- **Deployment Planning**: Verify real-time capability
- **Model Comparison**: Evaluate model sizes
- **Performance Tuning**: Identify bottlenecks
- **Documentation**: Report performance metrics

## Step 1: Plan Benchmarks

### Define Goals

What do you need to know?

- ✅ Can model run at 30/60 FPS?
- ✅ Which GPU is most cost-effective?
- ✅ Base vs Large model performance?
- ✅ Performance at different resolutions?

### Choose Benchmark Mode

#### Synthetic Images

Fast, consistent, hardware-focused:

```bash
python -m cli.benchmark_latency --run my_run
```

**Pros**: Fast, reproducible  
**Cons**: Not representative of real workload

#### Real Video

Realistic workload:

```bash
python -m cli.benchmark_latency --run my_run --video test.mp4
```

**Pros**: Representative of production  
**Cons**: Dependent on video content

### Select GPUs

**Local**:
- Use available GPU

**Cluster**:
- All GPUs: `--gpus all`
- Specific: `--gpus h100-96,a100-80,a100-40`
- Single: `--gpus a100-40`

## Step 2: Run Benchmarks

### Local Benchmark

#### Quick Test

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --runs 50
```

#### Realistic Benchmark

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --warmup 20 \
  --runs 200
```

#### With Visualization

```bash
python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --runs 100 \
  --create-latency-video
```

### Cluster Benchmark

#### All GPUs

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus all \
  --video test_video.mp4 \
  --runs 100
```

#### Specific GPUs

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus h100-96,a100-80,a100-40 \
  --video test_video.mp4 \
  --runs 200
```

#### High Precision

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus all \
  --video test_video.mp4 \
  --warmup 50 \
  --runs 500
```

## Step 3: Analyze Results

### Understanding Metrics

#### Latency Statistics

```
Mean:   8.5 ms   # Average latency
Median: 8.3 ms   # Typical latency
Std:    0.4 ms   # Variability
P95:    9.1 ms   # 95th percentile (near-worst)
P99:    9.8 ms   # 99th percentile (worst-case)
```

**What to look for**:
- Low mean and median = fast typical performance
- Low std dev = consistent performance
- P95/P99 close to median = predictable performance

#### Throughput

```
FPS: 117.6       # Frames per second
```

**Calculation**: `FPS = 1000 / mean_latency_ms`

**Interpretation**:
- FPS ≥ 120: Excellent (120 Hz displays)
- FPS ≥ 60: Great (60 Hz displays, smooth video)
- FPS ≥ 30: Good (standard video)
- FPS < 30: May need optimization

#### Real-Time Capability

```
✓ 30 FPS (33.3 ms/frame)
✓ 60 FPS (16.7 ms/frame)
✗ 120 FPS (8.3 ms/frame)
```

**Meaning**: Can model process frames faster than they're produced?

### Benchmark Results File

```json
{
  "gpu": "NVIDIA H100",
  "model": "base",
  "image_size": 640,
  "runs": 100,
  "latency_ms": {
    "mean": 8.5,
    "median": 8.3,
    "std": 0.4,
    "p95": 9.1,
    "p99": 9.8
  },
  "throughput": {
    "fps": 117.6
  },
  "real_time_capable": {
    "30fps": true,
    "60fps": true
  }
}
```

## Step 4: Compare Performance

### Compare Across GPUs

```bash
python -m cli.compare_latency benchmark_results/20260128_172934/
```

Output:

```
GPU          Mean (ms)  P95 (ms)  FPS    30fps  60fps
--------------------------------------------------------
h100-96      8.5        9.1       117.6  ✓      ✓
a100-80      10.2       11.0      98.0   ✓      ✓
a100-40      12.8       13.8      78.1   ✓      ✓
```

### Save Comparison

```bash
python -m cli.compare_latency \
  benchmark_results/20260128_172934/ \
  -o BENCHMARK.md \
  --format markdown
```

### Interpret Comparisons

**Questions to ask**:
1. Which GPU meets my FPS requirement?
2. Is the performance difference worth the cost?
3. Which GPU has most consistent performance (low P99)?
4. Do I need the fastest GPU or is mid-tier sufficient?

## Step 5: Optimize

### If Performance is Insufficient

#### 1. Try Smaller Model

```bash
# Benchmark large model
python -m cli.benchmark_latency --run large_run --model large

# Benchmark base model
python -m cli.benchmark_latency --run base_run --model base
```

Base model is ~3x faster than large.

#### 2. Reduce Input Resolution

Train with smaller image size:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --image-size 512  # Instead of 640 or 800
```

Benchmark:

```bash
python -m cli.benchmark_latency \
  --run my_run \
  --image-size 512
```

#### 3. Enable Model Optimization

Enabled by default, but verify:

```bash
python -m cli.benchmark_latency --run my_run --video test.mp4
# Optimization is ON by default

# Explicitly disable to compare:
python -m cli.benchmark_latency --run my_run --video test.mp4 --no-optimize
```

#### 4. Use Faster GPU

Compare benchmark results and choose faster GPU.

## Common Benchmarking Scenarios

### Scenario 1: Quick Performance Check

Fast sanity check:

```bash
python -m cli.benchmark_latency --run my_run --runs 50
```

### Scenario 2: Production Validation

Rigorous testing for deployment:

```bash
./submit_benchmark.sh \
  --run my_run \
  --gpus all \
  --video production_video.mp4 \
  --warmup 50 \
  --runs 500 \
  --create-latency-video
```

### Scenario 3: Model Comparison

Compare different model sizes:

```bash
# Base model
./submit_benchmark.sh --run base_run --model base --gpus all

# Large model
./submit_benchmark.sh --run large_run --model large --gpus all

# Compare
python -m cli.compare_latency benchmark_results/base_run/
python -m cli.compare_latency benchmark_results/large_run/
```

### Scenario 4: Resolution Study

Test different input sizes:

```bash
for size in 512 640 800 1024; do
  python -m cli.benchmark_latency \
    --run my_run \
    --image-size $size \
    --runs 100 \
    -o benchmark_results/size_${size}/
done
```

### Scenario 5: GPU Selection

Help choose deployment GPU:

```bash
./submit_benchmark.sh \
  --run my_run \
  --gpus a100-80,a100-40 \
  --video test.mp4 \
  --runs 200

python -m cli.compare_latency benchmark_results/latest/ -o GPU_COMPARISON.md
```

## Latency Visualization

### Create Latency Videos

```bash
python -m cli.benchmark_latency \
  --run my_run \
  --video test.mp4 \
  --create-latency-video
```

Creates:
- `detected_latency.mp4` - Frames appear when inference completes
- `sidebyside_latency.mp4` - Original vs delayed side-by-side

### Interpret Latency Videos

**What to look for**:
- **Delay gap**: Time between original and delayed
- **Stuttering**: Indicates variable latency
- **Smooth delay**: Indicates consistent latency

## Best Practices

### 1. Use Realistic Video

Real video better represents production:

```bash
--video production_sample.mp4
```

### 2. Sufficient Warmup

GPU needs time to reach steady state:

```bash
--warmup 20  # Minimum
--warmup 50  # Better for critical measurements
```

### 3. Enough Runs

More runs = more accurate statistics:

```bash
--runs 100   # Quick test
--runs 200   # Standard
--runs 500   # High precision
```

### 4. Consistent Conditions

- Same video for all GPU comparisons
- Same warmup/runs settings
- Same model and image size
- No other processes on GPU

### 5. Check GPU State

Before benchmarking:

```bash
nvidia-smi  # Ensure no other processes
```

### 6. Document Everything

Save benchmark configs and results:

```bash
./submit_benchmark.sh ... > benchmark_config.txt
python -m cli.compare_latency ... -o BENCHMARK.md
```

## Troubleshooting

### Inconsistent Results

**Solutions**:
1. Increase warmup: `--warmup 50`
2. Increase runs: `--runs 500`
3. Check for other GPU processes
4. Ensure stable GPU clocks

### Unexpected Slow Performance

**Check**:
1. GPU utilization: `nvidia-smi`
2. Thermal throttling
3. Power limits
4. Other processes

### High Variability (Large Std Dev)

**Causes**:
- Insufficient warmup
- GPU thermal throttling
- Competing processes
- Dynamic clocks

**Solutions**:
- Increase warmup
- Check thermal state
- Clear other GPU processes

## Reporting Results

### For Papers/Documentation

Include:
- GPU model and VRAM
- Model size (base/large)
- Input resolution
- Batch size (typically 1 for inference)
- Number of runs
- Mean, P50, P95, P99 latencies
- Throughput (FPS)

### Example Report

```markdown
## Performance Benchmarks

Model: RF-DETR-Base (640x640)
Runs: 500 (warmup: 50)

| GPU | Mean (ms) | P95 (ms) | P99 (ms) | FPS | Real-time |
|-----|-----------|----------|----------|-----|-----------|
| H100 96GB | 8.5 ± 0.4 | 9.1 | 9.8 | 117.6 | 60fps ✓ |
| A100 80GB | 10.2 ± 0.5 | 11.0 | 11.5 | 98.0 | 60fps ✓ |
| A100 40GB | 12.8 ± 0.6 | 13.8 | 14.2 | 78.1 | 60fps ✓ |
```

## Related

- **[Benchmark Latency CLI](../cli/benchmark-latency.md)** - Command reference
- **[Compare Latency CLI](../cli/compare-latency.md)** - Comparison tool
- **[Submit Benchmark Script](../scripts/submit-benchmark.md)** - SLURM benchmarking
- **[Inference Workflow](inference.md)** - Run inference
