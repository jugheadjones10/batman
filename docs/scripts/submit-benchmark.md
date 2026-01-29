# Submit Benchmark Script

Submit latency benchmarks for RF-DETR models across multiple GPU types via SLURM.

## Basic Usage

```bash
# Benchmark on all GPUs
./submit_benchmark.sh --run my_training_run --gpus all

# Benchmark on specific GPUs
./submit_benchmark.sh --run my_training_run --gpus h100-96,a100-80,a100-40
```

## Command Builder

<div class="command-builder-widget" data-tool="submit_benchmark" data-params='[
  {"name": "checkpoint", "type": "path", "description": "Checkpoint path", "group": "Model"},
  {"name": "run", "type": "text", "description": "Run name", "group": "Model"},
  {"name": "latest", "type": "flag", "description": "Use latest run", "group": "Model"},
  {"name": "model", "type": "choice", "choices": ["base", "large"], "default": "base", "description": "Model size", "group": "Model"},
  {"name": "image-size", "type": "number", "default": 640, "min": 320, "max": 1280, "step": 32, "description": "Image size", "group": "Model"},
  {"name": "no-optimize", "type": "flag", "description": "Disable optimization", "group": "Model"},
  {"name": "gpus", "type": "text", "required": true, "description": "GPU types (comma-separated or all)", "group": "GPU"},
  {"name": "time", "type": "text", "default": "00:30:00", "description": "Time limit per job", "group": "GPU"},
  {"name": "warmup", "type": "number", "default": 10, "min": 0, "description": "Warmup runs", "group": "Benchmark"},
  {"name": "runs", "type": "number", "default": 100, "min": 1, "description": "Test runs", "group": "Benchmark"},
  {"name": "video", "type": "path", "description": "Video file for benchmark", "group": "Benchmark"},
  {"name": "no-video", "type": "flag", "description": "Use synthetic images", "group": "Benchmark"},
  {"name": "create-latency-video", "type": "flag", "description": "Create latency videos", "group": "Output"}
]'></div>

## Parameters

### Model Selection (Choose One)

#### `--checkpoint=PATH`

Path to model checkpoint.

```bash
./submit_benchmark.sh --checkpoint runs/my_run/best.pth --gpus all
```

#### `--run=NAME`

Run name to auto-find checkpoint.

```bash
./submit_benchmark.sh --run rfdetr_h100_20260120_105925 --gpus all
```

#### `--latest`

Use the most recent training run.

```bash
./submit_benchmark.sh --latest --gpus all
```

### Model Configuration

#### `--model=SIZE`

Model architecture size.

- **Choices**: `base`, `large`
- **Default**: `base`

#### `--image-size=N`

Image size for inference.

- **Default**: `640`

#### `--no-optimize`

Disable model optimization.

### GPU Selection (Required)

#### `--gpus=TYPES`

Comma-separated GPU types or `all`.

**Available GPU types**:

- `h200` - H200 141GB
- `h100-96` - H100 96GB
- `h100-47` - H100 47GB
- `a100-80` - A100 80GB
- `a100-40` - A100 40GB
- `nv` - V100/Titan/T4

```bash
# All GPUs
--gpus all

# Specific GPUs
--gpus h100-96,a100-80,a100-40

# Single GPU
--gpus h100-96
```

### SLURM Options

#### `--time=LIMIT`

Time limit per job in format `HH:MM:SS`.

- **Default**: `00:30:00` (30 minutes)

### Benchmark Configuration

#### `--warmup=N`

Number of warmup runs.

- **Default**: `10`

#### `--runs=N`

Number of test runs to measure.

- **Default**: `100`

### Benchmark Mode

#### `--video=PATH`

Video file for realistic benchmark.

- **Default**: `crane_hook_1_short.mp4` (if available)

```bash
./submit_benchmark.sh --run my_run --gpus all --video test_video.mp4
```

#### `--no-video`

Use synthetic dummy images instead of video.

```bash
./submit_benchmark.sh --run my_run --gpus all --no-video
```

### Output

#### `--create-latency-video`

Create latency-delayed visualization videos (requires `--video`).

```bash
./submit_benchmark.sh \
  --run my_run \
  --gpus all \
  --video test.mp4 \
  --create-latency-video
```

Creates:

- `detected_latency.mp4` - Latency-delayed output
- `sidebyside_latency.mp4` - Side-by-side comparison

## Examples

### Example 1: Quick Benchmark on All GPUs

```bash
./submit_benchmark.sh --run my_training_run --gpus all
```

### Example 2: Specific GPUs

Benchmark on H100 and A100 variants:

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus h100-96,h100-47,a100-80,a100-40
```

### Example 3: High-Precision Benchmark

More runs for accurate statistics:

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus all \
  --warmup 20 \
  --runs 500
```

### Example 4: Realistic Video Benchmark

Use real video frames:

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus all \
  --video test_video.mp4 \
  --runs 200
```

### Example 5: With Visualization

Create latency videos:

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus h100-96,a100-40 \
  --video test_video.mp4 \
  --create-latency-video
```

### Example 6: Large Model Benchmark

Test large model performance:

```bash
./submit_benchmark.sh \
  --run my_large_run \
  --model large \
  --gpus all \
  --image-size 800
```

### Example 7: Latest Run

Benchmark most recent training:

```bash
./submit_benchmark.sh --latest --gpus all --video test.mp4
```

### Example 8: Extended Time

Longer time limit for large benchmarks:

```bash
./submit_benchmark.sh \
  --run my_run \
  --gpus all \
  --runs 1000 \
  --time 01:00:00
```

## Output

### Job Submission

```bash
$ ./submit_benchmark.sh --run my_run --gpus h100-96,a100-80

Benchmark Configuration:
  Model: my_run
  GPUs: h100-96, a100-80
  Runs: 100 (warmup: 10)
  Video: crane_hook_1_short.mp4
  Output: benchmark_results/20260128_172934/

Submitting jobs...
  [h100-96] Job 123458 submitted
  [a100-80] Job 123459 submitted

Monitor with:
  squeue -u $USER
  tail -f logs/job_123458.log

Compare results when complete:
  python -m cli.compare_latency benchmark_results/20260128_172934/
```

### Generated Files

```
benchmark_results/20260128_172934/
├── h100-96/
│   ├── benchmark_results.json
│   ├── detected_latency.mp4      (if --create-latency-video)
│   └── sidebyside_latency.mp4    (if --create-latency-video)
├── a100-80/
│   ├── benchmark_results.json
│   ├── detected_latency.mp4
│   └── sidebyside_latency.mp4
└── comparison.md                  (after running compare_latency)

logs/
├── job_123458.log                 # h100-96 log
└── job_123459.log                 # a100-80 log
```

### Monitor Scripts

Each job gets a `monitor.sh` helper:

```bash
# Generated monitor scripts
benchmark_results/20260128_172934/h100-96/monitor.sh
benchmark_results/20260128_172934/a100-80/monitor.sh
```

Run to monitor specific GPU:

```bash
./benchmark_results/20260128_172934/h100-96/monitor.sh
```

## Comparing Results

After jobs complete, compare results:

```bash
# Text output
python -m cli.compare_latency benchmark_results/20260128_172934/

# Markdown output
python -m cli.compare_latency \
  benchmark_results/20260128_172934/ \
  -o BENCHMARK.md \
  --format markdown
```

See [Compare Latency CLI](../cli/compare-latency.md) for details.

## Monitoring

### View All Jobs

```bash
# List your jobs
squeue -u $USER

# Watch jobs
watch -n 5 squeue -u $USER
```

### Monitor Specific GPU

```bash
# Follow log
tail -f logs/job_123458.log

# Use monitor script
./benchmark_results/20260128_172934/h100-96/monitor.sh
```

### Check Progress

```bash
# Count completed jobs
ls benchmark_results/20260128_172934/*/benchmark_results.json | wc -l

# Check for errors
grep -i error logs/job_*.log
```

## Best Practices

### 1. Use Video for Realistic Benchmarks

Real video frames better represent production:

```bash
./submit_benchmark.sh --run my_run --gpus all --video test.mp4
```

### 2. Sufficient Runs

More runs = more accurate statistics:

```bash
--runs 200  # Good for production
--runs 500  # Better for papers/reports
```

### 3. Benchmark All Relevant GPUs

Compare across GPU families:

```bash
--gpus h100-96,a100-80,a100-40
```

### 4. Create Visualizations

Latency videos help communicate performance:

```bash
--create-latency-video
```

### 5. Save Comparison Results

Document results for reference:

```bash
python -m cli.compare_latency \
  benchmark_results/20260128_172934/ \
  -o BENCHMARK.md \
  --format markdown
```

## Use Cases

### 1. GPU Selection

Determine which GPU to deploy on:

```bash
./submit_benchmark.sh --run my_run --gpus all
# Compare price/performance
```

### 2. Model Optimization

Compare base vs large model:

```bash
# Base model
./submit_benchmark.sh --run base_run --model base --gpus all

# Large model
./submit_benchmark.sh --run large_run --model large --gpus all
```

### 3. Real-Time Feasibility

Check if model meets framerate requirements:

```bash
./submit_benchmark.sh --run my_run --gpus a100-40 --video test.mp4
# Check if capable of 30fps/60fps
```

### 4. Documentation

Generate benchmark results for papers/reports:

```bash
./submit_benchmark.sh \
  --run my_run \
  --gpus all \
  --video test.mp4 \
  --runs 500 \
  --create-latency-video
```

## Troubleshooting

### Jobs Pending

Check partition availability:

```bash
sinfo -p h100,a100,gpu,nv
```

### Job Failed

Check logs:

```bash
cat logs/job_*.log | grep -i error
```

### Inconsistent Results

Increase warmup and runs:

```bash
./submit_benchmark.sh --warmup 20 --runs 300 ...
```

### H200 Time Limit

H200 on `gpu` partition limited to 3 hours. Script automatically adjusts.

## Related

- **[Benchmark Latency CLI](../cli/benchmark-latency.md)** - Local benchmarking
- **[Compare Latency CLI](../cli/compare-latency.md)** - Compare results
- **[Benchmarking Guide](../guides/benchmarking.md)** - Complete guide
- **[SLURM Guide](../guides/slurm.md)** - SLURM usage
