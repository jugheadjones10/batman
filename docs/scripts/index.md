# SLURM Scripts Overview

Batman provides shell scripts for submitting training, inference, and benchmarking jobs to SLURM HPC clusters.

## Available Scripts

### 🎓 Training & Inference
- **[submit_train.sh](submit-train.md)** - Submit training jobs to SLURM
- **[submit_inference.sh](submit-inference.md)** - Submit inference jobs to SLURM

### ⚡ Benchmarking
- **[submit_benchmark.sh](submit-benchmark.md)** - Submit benchmark jobs for multiple GPUs

### 🔧 Development
- **[run_dev.sh](run-dev.md)** - Start local development servers

## GPU Types

All SLURM scripts support these GPU types:

| GPU Type | VRAM | Partition | GRES | Use Case |
|----------|------|-----------|------|----------|
| `h200` | 141GB | `gpu` | `gpu:h200:1` | Largest models, biggest batches |
| `h100-96` | 96GB | `h100` | `gpu:h100:1` | Large models, training |
| `h100-47` | 47GB | `h100` | `gpu:h100_47:1` | Medium models |
| `a100-80` | 80GB | `a100` | `gpu:a100_80:1` | General purpose |
| `a100-40` | 40GB | `a100` | `gpu:a100_40:1` | Inference, smaller models |
| `nv` | Varies | `nv` | `gpu:nv:1` | V100/Titan/T4 (legacy) |

## Common Patterns

### Model Selection

All scripts accept one of three model specification methods:

```bash
# Option 1: By checkpoint path
--checkpoint path/to/model.pth

# Option 2: By run name (auto-finds checkpoint)
--run rfdetr_h100_20260120_105925

# Option 3: Use latest run
--latest
```

### GPU Selection

Specify GPU type with `--gpu`:

```bash
# Training (large GPU)
./submit_train.sh --gpu h100-96 ...

# Inference (smaller GPU)
./submit_inference.sh --gpu a100-40 ...

# Benchmarking (all GPUs)
./submit_benchmark.sh --gpus all ...
```

### Output Directories

Scripts auto-generate timestamped output directories:

```bash
# Training
runs/rfdetr_h100_20260128_105030/

# Inference
inference_results/20260128_105030/

# Benchmarking
benchmark_results/20260128_105030/
```

### Dry Run Mode

Preview SLURM scripts without submitting:

```bash
./submit_train.sh --dry-run ...
./submit_inference.sh --dry-run ...
```

## Job Management

### Submit Jobs

```bash
# Submit training
sbatch_id=$(./submit_train.sh --gpu h100-96 --project data/projects/Test)

# Submit inference
sbatch_id=$(./submit_inference.sh --run my_run --input video.mp4 --gpu a100-40)

# Submit benchmarks
./submit_benchmark.sh --run my_run --gpus h100-96,a100-80
```

### Monitor Jobs

```bash
# List your jobs
squeue -u $USER

# Watch job status
watch -n 5 squeue -u $USER

# Check specific job
squeue -j <job_id>
```

### View Logs

```bash
# Follow log output
tail -f logs/job_<job_id>.log

# View completed job
cat logs/job_<job_id>.log
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel by name
scancel --name=rfdetr_training
```

## Workflow Examples

### Workflow 1: Training on Cluster

```bash
# 1. Prepare data locally
python -m cli.importer coco --project data/projects/Test --create --classes person

# 2. Submit training
./submit_train.sh \
  --project data/projects/Test \
  --gpu h100-96 \
  --epochs 50

# 3. Monitor training
tail -f logs/job_*.log

# 4. Check results
ls runs/rfdetr_h100_*/
cat runs/rfdetr_h100_*/results.json
```

### Workflow 2: Inference on Videos

```bash
# 1. Submit inference
./submit_inference.sh \
  --run my_training_run \
  --input "videos/*.mp4" \
  --gpu a100-40 \
  --track

# 2. Monitor progress
tail -f logs/job_*.log

# 3. Download results
ls inference_results/*/
```

### Workflow 3: Multi-GPU Benchmarking

```bash
# 1. Submit benchmarks for all GPUs
./submit_benchmark.sh \
  --run my_training_run \
  --gpus all \
  --video test_video.mp4 \
  --runs 100

# 2. Monitor jobs
squeue -u $USER

# 3. Compare results when complete
python -m cli.compare_latency benchmark_results/latest/ -o BENCHMARK.md
```

## Resource Allocation

### Default Batch Sizes

Scripts auto-configure batch sizes:

| GPU | Default Batch Size |
|-----|-------------------|
| H200/H100-96 | 16 |
| H100-47 | 12 |
| A100-80 | 12 |
| A100-40 | 8 |
| NV | 4 |

Override with `--batch-size`:

```bash
./submit_train.sh --batch-size 32 ...
```

### Time Limits

Default time limits:

| Job Type | Default | Max |
|----------|---------|-----|
| Training | 24 hours | Unlimited |
| Inference | 4 hours | Unlimited |
| Benchmark | 30 minutes | Unlimited |

Override with `--time`:

```bash
./submit_train.sh --time 48:00:00 ...
```

### Multi-GPU Training

Use `--num-gpus` for distributed training:

```bash
./submit_train.sh \
  --gpu h100-96 \
  --num-gpus 4 \
  --batch-size 64
```

## Tips & Best Practices

### 1. Use Dry Run First

Preview scripts before submitting:

```bash
./submit_train.sh --dry-run --project ... --gpu h100-96
```

### 2. Monitor Resource Usage

Check GPU utilization during jobs:

```bash
# SSH to compute node
ssh <node_name>

# Check GPU usage
nvidia-smi -l 1
```

### 3. Choose Appropriate GPUs

- **Training**: H100-96, H100-47, A100-80
- **Inference**: A100-40 (cost-effective)
- **Benchmarking**: All types for comparison

### 4. Name Your Runs

Use descriptive run names:

```bash
./submit_train.sh \
  --project data/projects/CraneHook \
  --output-dir runs/crane_hook_v1 \
  --gpu h100-96
```

### 5. Check Partition Availability

```bash
# Check partition status
sinfo

# Check available GPUs
sinfo -p h100,a100,gpu -o "%P %a %l %D %N %G"
```

## Troubleshooting

### Job Pending

```bash
# Check reason
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.20R"
```

Common reasons:
- `Resources`: Waiting for GPU availability
- `Priority`: Lower priority job
- `QOSMaxJobsPerUser`: Too many jobs running

### Job Failed

```bash
# Check logs
cat logs/job_<job_id>.log

# Common issues:
# - Out of memory → Reduce batch size
# - File not found → Check paths
# - Module errors → Check environment
```

### H200 Time Limit

H200 on `gpu` partition has 3-hour limit:

```bash
# Script automatically adjusts time limit
./submit_train.sh --gpu h200 ...  # Max 3 hours

# Use H100-96 for longer jobs
./submit_train.sh --gpu h100-96 ...  # Up to 24+ hours
```

## Environment Setup

### Required Modules

Scripts automatically load:
```bash
module load cuda/12.1
module load python/3.11
```

### Python Environment

Scripts use `uv` for dependency management:
```bash
uv sync
```

## Related

- **[Submit Training](submit-train.md)** - Training job details
- **[Submit Inference](submit-inference.md)** - Inference job details
- **[Submit Benchmark](submit-benchmark.md)** - Benchmark job details
- **[SLURM Usage Guide](../guides/slurm.md)** - Complete SLURM guide
