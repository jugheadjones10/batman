# Compare Latency CLI

Compare latency benchmark results across multiple GPUs and generate comparison tables.

## Overview

The compare CLI:
- Aggregates benchmark results from multiple runs
- Generates comparison tables with statistics
- Identifies which GPUs meet real-time requirements (30fps, 60fps)
- Supports text and markdown output formats

## Basic Usage

```bash
# Compare all benchmarks in directory
python -m cli.compare_latency benchmark_results/

# Compare specific GPUs
python -m cli.compare_latency benchmark_results/ --gpus h100-96,a100-80,a100-40

# Save to markdown
python -m cli.compare_latency benchmark_results/ -o comparison.md --format markdown
```

## Parameters

### Required

#### `benchmark_dir`
Directory containing benchmark result subdirectories.

```bash
python -m cli.compare_latency benchmark_results/20260128_172934/
```

Expected structure:
```
benchmark_results/20260128_172934/
├── h100-96/
│   └── benchmark_results.json
├── a100-80/
│   └── benchmark_results.json
└── a100-40/
    └── benchmark_results.json
```

### Optional

#### `--gpus TYPES`
Comma-separated GPU types to compare.
- **Default**: All found GPUs

```bash
--gpus h100-96,a100-80,a100-40
```

#### `--output PATH` or `-o PATH`
Save comparison to file.
- **Format auto-detected** from extension (`.md` or `.txt`)

```bash
-o comparison.md
-o comparison.txt
```

#### `--format FORMAT`
Output format.
- **Choices**: `text`, `markdown`
- **Default**: `text`

## Output

### Console Output (Text)

```
========================================================================================================================
GPU LATENCY COMPARISON
========================================================================================================================
Benchmark Mode: video

GPU Type     GPU Name                       Mean      P50       P95       P99       FPS     30fps  60fps
------------------------------------------------------------------------------------------------------------------------
h100-96      NVIDIA H100 96GB               8.5ms     8.3ms     9.1ms     9.8ms     117.6   yes    yes
a100-80      NVIDIA A100 80GB               10.2ms    10.0ms    11.0ms    11.5ms    98.0    yes    yes
a100-40      NVIDIA A100 40GB               12.8ms    12.5ms    13.8ms    14.2ms    78.1    yes    no
========================================================================================================================

DETAILED STATISTICS
------------------------------------------------------------------------------------------------------------------------

H100-96 - NVIDIA H100 96GB
  Latency:
    Mean:   8.50 ms  (+/- 0.40 ms)
    Min:    7.90 ms
    Max:    10.20 ms
    P50:    8.30 ms
    P95:    9.10 ms
    P99:    9.80 ms
  Throughput: 117.6 FPS
  Real-time:
    30 FPS: YES
    60 FPS: YES
```

### Markdown Output

```markdown
# GPU Latency Comparison

## Summary

| GPU Type | GPU Name | Mean | P50 | P95 | P99 | FPS | 30fps | 60fps |
|----------|----------|------|-----|-----|-----|-----|-------|-------|
| h100-96 | NVIDIA H100 96GB | 8.5ms | 8.3ms | 9.1ms | 9.8ms | 117.6 | yes | yes |
| a100-80 | NVIDIA A100 80GB | 10.2ms | 10.0ms | 11.0ms | 11.5ms | 98.0 | yes | yes |
| a100-40 | NVIDIA A100 40GB | 12.8ms | 12.5ms | 13.8ms | 14.2ms | 78.1 | yes | no |
```

## Examples

### Example 1: Compare All GPUs

Compare all benchmarks in directory:

```bash
python -m cli.compare_latency benchmark_results/20260128_172934/
```

### Example 2: Compare Specific GPUs

Compare only H100 and A100 variants:

```bash
python -m cli.compare_latency benchmark_results/20260128_172934/ \
  --gpus h100-96,h100-47,a100-80,a100-40
```

### Example 3: Save to Markdown

Generate markdown comparison for documentation:

```bash
python -m cli.compare_latency benchmark_results/20260128_172934/ \
  --output BENCHMARK.md \
  --format markdown
```

### Example 4: Filter and Save

Compare specific GPUs and save:

```bash
python -m cli.compare_latency benchmark_results/20260128_172934/ \
  --gpus h100-96,a100-80 \
  -o comparison.txt
```

## Understanding Comparisons

### Performance Metrics

| Column | Description | Importance |
|--------|-------------|------------|
| **Mean** | Average latency | Overall performance |
| **P50** | Median latency | Typical performance |
| **P95** | 95th percentile | Near-worst case |
| **P99** | 99th percentile | Worst case |
| **FPS** | Throughput | Processing speed |

### Real-Time Indicators

- **yes** = GPU meets framerate requirement (P99 < threshold)
- **no** = GPU does not meet requirement

Real-time capability is determined by **P99** latency, not mean.

## Use Cases

### 1. GPU Selection

Determine which GPU to use for deployment:

```bash
python -m cli.compare_latency benchmark_results/
```

Look for GPUs meeting your framerate requirement.

### 2. Cost-Performance Analysis

Compare performance against GPU cost:

```bash
# Benchmark on cluster
./submit_benchmark.sh --run my_run --gpus all

# Compare results
python -m cli.compare_latency benchmark_results/latest/
```

### 3. Documentation

Generate markdown for reports:

```bash
python -m cli.compare_latency benchmark_results/ \
  -o BENCHMARK.md \
  --format markdown
```

### 4. Model Comparison

Compare different model sizes:

```bash
# Base model benchmarks
python -m cli.compare_latency benchmark_results/base_model/ \
  -o base_comparison.txt

# Large model benchmarks
python -m cli.compare_latency benchmark_results/large_model/ \
  -o large_comparison.txt
```

## Tips

### 1. Consistent Benchmark Settings

Ensure all benchmarks use same settings:
- Same model size
- Same image size
- Same number of runs

### 2. Sort Results

Results are sorted by mean latency (fastest first).

### 3. Check P99

P99 is important for worst-case scenarios:
- Video stuttering
- Real-time deadlines

### 4. Consider Power Efficiency

Lower-end GPUs may be more cost-effective:
- A100-40 vs A100-80
- Does your app need the fastest GPU?

## Related

- **[Benchmark Latency CLI](benchmark-latency.md)** - Run benchmarks
- **[Submit Benchmark Script](../scripts/submit-benchmark.md)** - SLURM benchmarking
- **[Benchmarking Guide](../guides/benchmarking.md)** - Complete guide
