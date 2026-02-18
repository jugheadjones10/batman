# SLURM Usage Guide

Complete guide for using Batman on SLURM HPC clusters.

## Overview

SLURM (Simple Linux Utility for Resource Management) is a job scheduler for HPC clusters. Batman provides scripts to easily submit training, inference, and benchmarking jobs.

## Quick Start

### 1. Submit Training

```bash
./submit_train.sh --project data/projects/MyProject --gpu h100-96
```

### 2. Monitor Job

```bash
squeue -u $USER
tail -f logs/job_*.log
```

### 3. Check Results

```bash
ls runs/rfdetr_h100_*/
cat runs/rfdetr_h100_*/results.json
```

## Getting data to the GPU machine

If you prepare data locally and the cluster has the repo at `~/batman`, sync the project (and optionally the whole repo) to the cluster, then submit from there.

### 1. Prepare data locally

Run imports and class merges on your machine (e.g. run `scripts/project_one_data.sh` or use the importer/classes CLI). You should have a project directory such as `data/projects/One/`.

### 2. Sync to the cluster

From your **local** machine, sync the repo and project data to the login node. The training job script expects the repo at `~/batman` (see `submit_train.sh`).

**Option A – Sync full repo (simplest)**

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'runs' \
  /path/to/batman/ USER@cluster:~batman/
```

Replace `USER` and `cluster` with your SSH user and host. This keeps code and `data/projects/` in sync.

**Option B – Sync only one project**

If the repo already exists on the cluster and you only need to update one project:

```bash
rsync -avz /path/to/batman/data/projects/One/ USER@cluster:~/batman/data/projects/One/
```

### 3. On the cluster: install deps and submit

SSH to the cluster, go to the repo, ensure dependencies are installed, then submit:

```bash
ssh USER@cluster
cd ~/batman
uv sync   # if not already done
./submit_train.sh --project data/projects/One --gpu h100-96
```

Monitor with `squeue -u $USER` and `tail -f logs/job_*.log`. When the job finishes, checkpoints and logs are under `runs/rfdetr_*_*/`.

### 4. (Optional) Copy results back

To copy a run back to your laptop:

```bash
rsync -avz USER@cluster:~/batman/runs/rfdetr_h100_20260128_123456/ ./runs/
```

### Alternative: Mount GPU server and copy data

If you use the **mount GPU script** (`./mount_gpu.sh`), the remote batman repo appears at `gpu-server/`. Copy the project into that path so the cluster sees it without rsync:

```bash
# 1. Mount the GPU server (remote ~/batman → ./gpu-server)
./mount_gpu.sh

# 2. Ensure remote has data/projects, then copy your project
mkdir -p gpu-server/data/projects
cp -r data/projects/One gpu-server/data/projects/
# Or incremental sync (only changed files):
# rsync -av data/projects/One/ gpu-server/data/projects/One/

# 3. On the cluster: SSH in and submit (data is already there)
# ssh USER@cluster
# cd ~/batman && ./submit_train.sh --project data/projects/One --gpu h100-96

# 4. When done, unmount
./umount_gpu.sh
```

Because `gpu-server/` is the live remote filesystem, any copy into `gpu-server/data/projects/` is immediately visible on the cluster at `~/batman/data/projects/One`.

## GPU Types and Partitions

### Available GPUs

| GPU Type | VRAM | Partition | GRES | Nodes |
|----------|------|-----------|------|-------|
| H200 | 141GB | `gpu` | `gpu:h200:1` | 1-8 GPUs |
| H100-96 | 96GB | `h100` | `gpu:h100:1` | 1-4 GPUs |
| H100-47 | 47GB | `h100` | `gpu:h100_47:1` | 1-4 GPUs |
| A100-80 | 80GB | `a100` | `gpu:a100_80:1` | 1-4 GPUs |
| A100-40 | 40GB | `a100` | `gpu:a100_40:1` | 1-4 GPUs |
| NV (V100/Titan/T4) | Varies | `nv` | `gpu:nv:1` | Varies |

### GPU Selection Guidelines

**Training**:
- Large models: H200, H100-96
- Medium models: H100-47, A100-80
- Small models: A100-40

**Inference**:
- Production: A100-40 (cost-effective)
- High throughput: H100-96
- Batch processing: A100-80

**Benchmarking**:
- Compare all: `--gpus all`
- Deployment targets: Specific GPUs

### Check Availability

```bash
# View partition status
sinfo

# Check specific partitions
sinfo -p h100,a100,gpu

# See available GPUs
sinfo -p h100,a100,gpu -o "%P %a %l %D %N %G"
```

## Job Submission

### Training Jobs

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50 \
  --batch-size 16
```

**Default time limit**: 24 hours

### Inference Jobs

```bash
./submit_inference.sh \
  --run my_training_run \
  --input "videos/*.mp4" \
  --gpu a100-40 \
  --track
```

**Default time limit**: 4 hours

### Benchmark Jobs

```bash
./submit_benchmark.sh \
  --run my_training_run \
  --gpus h100-96,a100-80,a100-40 \
  --video test.mp4 \
  --runs 200
```

**Default time limit**: 30 minutes per GPU

## Job Management

### View Your Jobs

```bash
# List your jobs
squeue -u $USER

# Detailed view
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"

# Watch job status
watch -n 5 squeue -u $USER
```

### Job Status Codes

| Code | Status | Meaning |
|------|--------|---------|
| PD | Pending | Waiting for resources |
| R | Running | Job is executing |
| CG | Completing | Job is finishing |
| CD | Completed | Job finished successfully |
| F | Failed | Job failed |
| CA | Cancelled | Job was cancelled |

### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel by name
scancel --name=rfdetr_training

# Cancel by partition
scancel -u $USER -p h100
```

### Check Job Details

```bash
# Job info
scontrol show job <job_id>

# Job accounting
sacct -j <job_id> --format=JobID,JobName,Partition,State,Elapsed,ExitCode
```

## Monitoring Jobs

### Log Files

All jobs write to `logs/job_<job_id>.log`:

```bash
# Follow log
tail -f logs/job_<job_id>.log

# View last 100 lines
tail -100 logs/job_<job_id>.log

# Search logs
grep -i error logs/job_*.log
```

### Training Progress

```bash
# Follow training log
tail -f logs/job_<job_id>.log | grep "Epoch"

# Check tensorboard
ssh <compute_node>
tensorboard --logdir runs/my_run/tensorboard
```

### GPU Utilization

```bash
# SSH to compute node (get node name from squeue)
ssh <node_name>

# Monitor GPU
nvidia-smi -l 1

# GPU usage summary
nvidia-smi dmon
```

## Job Resources

### Time Limits

#### Set Time Limit

```bash
./submit_train.sh --time 48:00:00 ...    # 48 hours
./submit_inference.sh --time 08:00:00 ... # 8 hours
./submit_benchmark.sh --time 01:00:00 ... # 1 hour
```

#### Time Limit Warnings

**H200 on GPU Partition**: Limited to 3 hours
- Scripts automatically adjust
- Use H100-96 for longer jobs

#### Estimate Time

**Training**:
- ~5-10 minutes per epoch (typical)
- 50 epochs ≈ 4-8 hours
- Add 25-50% buffer

**Inference**:
- ~1-5 seconds per video minute
- 60-minute video ≈ 5-10 minutes
- Multiple videos: scale accordingly

**Benchmarking**:
- 100 runs ≈ 5-10 minutes
- 500 runs ≈ 15-30 minutes

### Memory Requirements

#### Adjust Batch Size

```bash
./submit_train.sh --batch-size 8 ...  # Reduce if OOM
```

#### GPU Memory by Type

| GPU | VRAM | Recommended Batch Size |
|-----|------|----------------------|
| H100-96 | 96GB | 16 |
| A100-80 | 80GB | 12-16 |
| A100-40 | 40GB | 8 |

#### System Memory

Training jobs allocate:
- CPU memory: 32GB (default)
- Adjust if needed: Edit submit scripts

### Multi-GPU Training

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --num-gpus 4 \
  --batch-size 16
```

**Effective batch size**: `batch_size × num_gpus`

**Node limits**:
- H100 nodes: 4 GPUs max
- A100 nodes: 4 GPUs max
- H200 nodes: 8 GPUs max

## Troubleshooting

### Job Pending

```bash
# Check why pending
squeue -j <job_id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.20R"
```

**Common reasons**:

#### Resources
Waiting for GPU availability.

**Solution**: Wait or try different partition

#### Priority
Lower priority job queued.

**Solution**: Wait or use different partition

#### QOSMaxJobsPerUser
Too many jobs running.

**Solution**: Cancel some jobs or wait

### Job Failed

```bash
# Check exit code
sacct -j <job_id> --format=JobID,State,ExitCode

# Check logs
cat logs/job_<job_id>.log | grep -i error
```

**Common errors**:

#### Out of Memory
```
CUDA out of memory
```

**Solution**: Reduce batch size or image size

```bash
./submit_train.sh --batch-size 4 --image-size 512 ...
```

#### File Not Found
```
FileNotFoundError: ...
```

**Solution**: Check file paths are correct

#### Module Errors
```
ModuleNotFoundError: ...
```

**Solution**: Ensure dependencies installed

```bash
uv sync
```

### Job Timeout

Job exceeded time limit.

**Solution**: Increase time limit

```bash
./submit_train.sh --time 48:00:00 ...
```

Or reduce workload:

```bash
./submit_train.sh --epochs 30 ...
```

### Permission Denied

```bash
# Make scripts executable
chmod +x submit_train.sh submit_inference.sh submit_benchmark.sh
```

## Best Practices

### 1. Test Locally First

Run on small data locally before cluster submission:

```bash
python -m cli.train --project data/projects/Test --epochs 10
```

### 2. Use Dry Run

Preview scripts before submitting:

```bash
./submit_train.sh --dry-run --project ... --gpu h100-96
```

### 3. Choose Appropriate GPU

- Don't use H100-96 when A100-40 suffices
- Save premium GPUs for large jobs
- Benchmark to find minimum required GPU

### 4. Set Realistic Time Limits

- Estimate job duration
- Add 25-50% buffer
- Don't set excessive limits (blocks resources)

### 5. Monitor Jobs

Check logs regularly:

```bash
watch -n 60 tail -50 logs/job_*.log
```

### 6. Clean Up

Remove old logs and results:

```bash
# Archive old logs
mkdir -p archive/logs
mv logs/job_*.log archive/logs/

# Remove old benchmark results
rm -rf benchmark_results/old_*/
```

### 7. Use Descriptive Names

Name output directories clearly:

```bash
./submit_train.sh \
  --output-dir runs/crane_hook_large_800px_v2 \
  ...
```

### 8. Document Jobs

Keep notes on experiments:

```bash
echo "Training crane hook detector with large model, 800px" > runs/my_run/notes.txt
```

## Common Workflows

### Workflow 1: Full Training Pipeline

```bash
# 1. Submit training
job_id=$(./submit_train.sh --project data/projects/MyProject --gpu h100-96)

# 2. Monitor
watch -n 10 squeue -j $job_id

# 3. When complete, submit inference
./submit_inference.sh --run my_run --input video.mp4 --gpu a100-40

# 4. Benchmark
./submit_benchmark.sh --run my_run --gpus all
```

### Workflow 2: Hyperparameter Search

```bash
for lr in 1e-5 1e-4 5e-4; do
  ./submit_train.sh \
    --project data/projects/MyProject \
    --gpu h100-96 \
    --lr $lr \
    --output-dir runs/lr_${lr}
done
```

### Workflow 3: Batch Inference

```bash
for video in videos/*.mp4; do
  ./submit_inference.sh \
    --run my_run \
    --input "$video" \
    --gpu a100-40
done
```

## Related

- **[Submit Training Script](../scripts/submit-train.md)** - Training details
- **[Submit Inference Script](../scripts/submit-inference.md)** - Inference details
- **[Submit Benchmark Script](../scripts/submit-benchmark.md)** - Benchmark details
- **[Training Workflow](training.md)** - Training guide
