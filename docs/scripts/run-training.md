# Run Training (Local)

Run training on the GPU cluster **from your local Mac** with automatic data pushing and result syncing. This script handles everything end-to-end: pushing your latest `manual_data` and `project.json` to the cluster, submitting the SLURM job via SSH, streaming the output, waiting for completion, and syncing lightweight training metadata (JSON files only, not large `.pth` checkpoints) back locally.

## Basic Usage

```bash
./run_training.sh --project data/projects/One
```

This accepts the same arguments as [`submit_train.sh`](submit-train.md), plus `--no-sync`, `--no-push`.

## How It Works

1. Pushes `manual_data/` and `project.json` from your local project to the GPU cluster via rsync
2. Generates the same SLURM script as `submit_train.sh`
3. Uploads the script to the GPU cluster via `scp`
4. Submits the job via `ssh ... sbatch`
5. Streams the SLURM log output in real-time via SSH
6. Waits for the job to leave the SLURM queue
7. Checks the job exit status via `sacct`
8. Syncs only JSON metadata (class_info.json, results.json, training_config.json) from the training run -- **not** the large `.pth` checkpoint files
9. If `--infer-after` was used, also syncs inference results

## Prerequisites

- **SSH access** to the GPU cluster (same credentials as `mount_gpu.sh`)
- **SSHFS mount** at `gpu-server/` -- run `./mount_gpu.sh` first (required for result syncing)

## Parameters

All parameters from [`submit_train.sh`](submit-train.md) are supported, plus:

### `--no-sync`

Skip syncing results locally after completion.

```bash
./run_training.sh --project data/projects/One --no-sync
```

### `--no-push`

Skip pushing `manual_data/` and `project.json` to the GPU before the job. Useful if you know the cluster already has the latest data.

```bash
./run_training.sh --project data/projects/One --no-push
```

### `--dry-run`

Show the generated SLURM script without submitting. No SSH connection is made.

## Examples

### Basic Training

```bash
./run_training.sh --project data/projects/One
```

### With Custom Options

```bash
./run_training.sh \
  --project data/projects/One \
  --gpu=h100-96 \
  --epochs=100 \
  --label=v2
```

### Manual Data Only

```bash
./run_training.sh \
  --project data/projects/One \
  --sources=manual_data \
  --label=manual-only
```

### Train + Infer

```bash
./run_training.sh \
  --project data/projects/One \
  --infer-after \
  --infer-test-only
```

## What Gets Synced

### Pre-job (local -> GPU)

| Data | Direction | Purpose |
|------|-----------|---------|
| `manual_data/` | local -> GPU | Your latest manually curated training images |
| `project.json` | local -> GPU | Class definitions and project config |

### Post-job (GPU -> local)

| Data | Direction | Synced? |
|------|-----------|---------|
| `class_info.json` | GPU -> local | Yes |
| `results.json` | GPU -> local | Yes |
| `training_config.json` | GPU -> local | Yes |
| `*.pth` checkpoints | GPU -> local | **No** (too large, ~400MB each) |
| Inference results | GPU -> local | Only if `--infer-after` was used |

Checkpoints remain on the GPU and are accessible via the SSHFS mount at `gpu-server/`.

## Comparison with submit_train.sh

| Aspect | `submit_train.sh` (cluster) | `run_training.sh` (local) |
|--------|----------------------------|--------------------------|
| Where it runs | On the GPU cluster | On your local Mac |
| Data push | Manual `sync.sh` needed | Automatic before job |
| Job submission | Direct `sbatch` call | `sbatch` via SSH |
| Blocking | Returns immediately | Blocks until job completes |
| Live output | Manual `tail -f` | Streamed automatically |
| Result syncing | Manual copy needed | Automatic (JSON metadata) |

## Troubleshooting

### SSHFS Mount Not Found

```
Error: SSHFS mount not found at gpu-server/
```

Run `./mount_gpu.sh` first, or use `--no-sync` if you don't need local results.

### Job Failed

The script displays the job state and error log path:

```
Job 12345 finished with state: FAILED
Error log: gpu-server/logs/slurm_12345_rfdetr-base-h100-96.err
```

## Related

- **[Run Inference (Local)](run-inference.md)** -- Local inference runner with auto-sync
- **[Submit Training](submit-train.md)** -- Cluster-side training submission
- **[Training CLI](../cli/train.md)** -- Training command reference
- **[Training Workflow](../guides/training.md)** -- Complete guide
