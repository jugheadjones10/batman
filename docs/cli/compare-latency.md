# Compare Latency CLI

Compare latency benchmark results across multiple GPUs and generate comparison tables.

## Overview

The compare CLI:
- Aggregates benchmark results from multiple runs
- Generates comparison tables with statistics
- Identifies which GPUs meet real-time requirements
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
=== Latency Benchmark Comparison ===

Model: base (640x640)
Runs: 100 per GPU

GPU          Mean (ms)  Median (ms)  P95 (ms)  P99 (ms)  FPS    30fps  60fps  120fps
----------------------------------------------------------------------------------
h100-96      8.5        8.3          9.1       9.8       117.6  ✓      ✓      ✗
a100-80      10.2       10.0         11.0      11.5      98.0   ✓      ✓      ✗
a100-40      12.8       12.5         13.8      14.2      78.1   ✓      ✓      ✗

Real-time capable @ 30fps: h100-96, a100-80, a100-40
Real-time capable @ 60fps: h100-96, a100-80, a100-40
Real-time capable @ 120fps: (none)

Fastest GPU: h100-96 (8.5 ms/frame, 117.6 FPS)
```

### Markdown Output

```markdown
# Latency Benchmark Comparison

**Model**: base (640x640)  
**Runs**: 100 per GPU

| GPU | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | FPS | 30fps | 60fps | 120fps |
|-----|-----------|-------------|----------|----------|-----|-------|-------|--------|
| h100-96 | 8.5 | 8.3 | 9.1 | 9.8 | 117.6 | ✓ | ✓ | ✗ |
| a100-80 | 10.2 | 10.0 | 11.0 | 11.5 | 98.0 | ✓ | ✓ | ✗ |
| a100-40 | 12.8 | 12.5 | 13.8 | 14.2 | 78.1 | ✓ | ✓ | ✗ |

**Real-time capable @ 30fps**: h100-96, a100-80, a100-40  
**Real-time capable @ 60fps**: h100-96, a100-80, a100-40  
**Real-time capable @ 120fps**: (none)

**Fastest GPU**: h100-96 (8.5 ms/frame, 117.6 FPS)
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
| **Median** | Typical latency | Most common case |
| **P95** | 95th percentile | Near-worst case |
| **P99** | 99th percentile | Worst case |
| **FPS** | Throughput | Processing speed |

### Real-Time Indicators

- **✓** = GPU meets framerate requirement
- **✗** = GPU does not meet requirement

### Fastest GPU

Identified by lowest mean latency and highest FPS.

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
