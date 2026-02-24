# SLURM Scripts Overview

Batman provides shell scripts for submitting training, inference, and benchmarking jobs to SLURM HPC clusters.

## Available Scripts

### Training & Inference
- **[submit_train.sh](submit-train.md)** - Submit training jobs to SLURM
- **[run_training.sh](run-training.md)** - Run training from local Mac with auto-sync
- **[submit_inference.sh](submit-inference.md)** - Submit inference jobs to SLURM
- **[run_inference.sh](run-inference.md)** - Run inference from local Mac with auto-sync

### Benchmarking
- **[submit_benchmark.sh](submit-benchmark.md)** - Submit benchmark jobs for multiple GPUs

### Development
- **[run_dev.sh](run-dev.md)** - Start local development servers

## GPU Types

All SLURM scripts support these GPU types:

| GPU Type | VRAM | Partition | GRES | Max/Node | Use Case |
|----------|------|-----------|------|----------|----------|
| `h200` | 141GB | `gpu` | `gpu:h200-141:N` | 4 | Largest models (3h limit) |
| `h100-96` | 96GB | `gpu-long` | `gpu:h100-96:N` | 2 | Large models, training |
| `h100-47` | 47GB | `gpu-long` | `gpu:h100-47:N` | 4 | Medium models |
| `a100-80` | 80GB | `gpu-long` | `gpu:a100-80:N` | 1 | General purpose |
| `a100-40` | 40GB | `gpu-long` | `gpu:a100-40:N` | 2 | Smaller models |
| `nv` | Varies | `gpu-long` | `gpu:nv:N` | 2 | V100/Titan/T4 (legacy) |

## Common Patterns

### Model Selection

Training uses `--project` to specify input data. Inference and benchmarking use one of these model specification methods:

```bash
# Option 1: By run name (auto-finds checkpoint in project)
--run rfdetr_h100_20260120_105925

# Option 2: Use latest run
--latest

# Option 3: By checkpoint path (benchmark only)
--checkpoint path/to/model.pth
```

### GPU Selection

Specify GPU type with `--gpu`:

```bash
# Training (large GPU)
./submit_train.sh --gpu h100-96 ...

# Inference
./submit_inference.sh --gpu h100-96 ...

# Benchmarking (all GPUs)
./submit_benchmark.sh --gpus all ...
```

### Output Directories

Scripts auto-generate timestamped output directories:

```bash
# Training (under project directory)
{project}/runs/rfdetr_h100-96_20260128_105030/

# Inference (under project directory)
{project}/inference/{run_name}/{video_id}/{timestamp}/

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
./submit_train.sh --gpu h100-96 --project data/projects/MyProject

# Submit inference
./submit_inference.sh --project data/projects/MyProject --run my_run

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
# Training logs
tail -f logs/slurm_<job_id>_rfdetr-base-h100-96.out

# Inference logs
tail -f logs/slurm_<job_id>_inference.out

# Benchmark logs
tail -f logs/slurm_<job_id>_benchmark_<gpu>.out
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

### Workflow 1: Training on Cluster (from Local Mac)

```bash
# Run from your Mac -- pushes data, trains, and syncs results automatically
./run_training.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50

# JSON metadata is synced to data/projects/MyProject/runs/ when done
# Checkpoints remain on GPU, accessible via gpu-server/
```

Or from the cluster directly:

```bash
# 1. Prepare data locally
python -m cli.importer coco --project data/projects/MyProject --create --classes person

# 2. Submit training
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50

# 3. Monitor training
squeue -u $USER
tail -f logs/slurm_*_rfdetr-*.out

# 4. Check results
ls data/projects/MyProject/runs/rfdetr_h100-96_*/
```

### Workflow 2: Inference on Videos (from Local Mac)

```bash
# Run from your Mac -- submits, waits, and syncs results automatically
./run_inference.sh \
  --project data/projects/MyProject \
  --run my_training_run \
  --gpu h100-96 \
  --track

# Results are synced to data/projects/MyProject/inference/ when done
```

Or from the cluster directly:

```bash
# 1. Submit inference
./submit_inference.sh \
  --project data/projects/MyProject \
  --run my_training_run \
  --gpu h100-96 \
  --track

# 2. Monitor progress
tail -f logs/slurm_*_inference.out

# 3. Manually copy results from SSHFS mount
cp -r gpu-server/data/projects/MyProject/inference/ data/projects/MyProject/inference/
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
| Training | 24 hours | 3 days (`gpu-long`), 3 hours (`gpu`) |
| Inference | 4 hours | 3 days (`gpu-long`), 3 hours (`gpu`) |
| Benchmark | 30 minutes | 3 days (`gpu-long`), 3 hours (`gpu`) |

Override with `--time`:

```bash
./submit_train.sh --time 48:00:00 ...
```

### Multi-GPU Training

Use `--num-gpus` for distributed training:

```bash
./submit_train.sh \
  --gpu h100-96 \
  --num-gpus 2 \
  --batch-size 16
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
- **Inference**: H100-96 or A100-40 (cost-effective)
- **Benchmarking**: All types for comparison

### 4. Name Your Runs

Use descriptive labels:

```bash
./submit_train.sh \
  --project data/projects/CraneHook \
  --label v1-base \
  --gpu h100-96
```

### 5. Check Partition Availability

```bash
# Check partition status
sinfo

# Check available GPUs
sinfo -p gpu,gpu-long -o "%P %a %l %D %N %G"
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
cat logs/slurm_<job_id>_*.out

# Common issues:
# - Out of memory: Reduce batch size
# - File not found: Check paths
# - Module errors: Check environment
```

### H200 Time Limit

H200 on `gpu` partition has 3-hour limit:

```bash
# Script automatically adjusts time limit
./submit_train.sh --gpu h200 ...  # Max 3 hours

# Use H100-96 for longer jobs
./submit_train.sh --gpu h100-96 ...  # Up to 3 days
```

## Environment Setup

Scripts activate the Python virtual environment from the project root:

```bash
cd ~/batman
source .venv/bin/activate
```

No `module load` commands are needed -- all dependencies are in the virtual environment.

## Related

- **[Submit Training](submit-train.md)** - Training job details
- **[Run Training (Local)](run-training.md)** - Local training runner with auto-sync
- **[Submit Inference](submit-inference.md)** - Inference job details
- **[Run Inference (Local)](run-inference.md)** - Local inference runner with auto-sync
- **[Submit Benchmark](submit-benchmark.md)** - Benchmark job details
- **[SLURM Usage Guide](../guides/slurm.md)** - Complete SLURM guide
