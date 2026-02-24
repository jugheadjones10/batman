# Run Inference (Local)

Run inference on the GPU cluster **from your local Mac** with automatic result syncing. This script handles everything end-to-end: submitting the SLURM job via SSH, streaming the output, waiting for completion, and copying inference results to your local project directory.

## Basic Usage

```bash
./run_inference.sh --project data/projects/One --latest
```

This accepts the same arguments as [`submit_inference.sh`](submit-inference.md), so the two scripts are interchangeable.

## How It Works

1. Pushes `manual_data/` and `project.json` from your local project to the GPU cluster via rsync
2. Generates the same SLURM script as `submit_inference.sh`
3. Uploads the script to the GPU cluster via `scp`
4. Submits the job via `ssh ... sbatch`
5. Streams the SLURM log output in real-time via SSH
6. Waits for the job to leave the SLURM queue
7. Checks the job exit status via `sacct`
8. Copies inference results from the SSHFS mount (`gpu-server/`) to your local project directory

## Prerequisites

- **SSH access** to the GPU cluster (same credentials as `mount_gpu.sh`)
- **SSHFS mount** at `gpu-server/` -- run `./mount_gpu.sh` first (required for result syncing)

## Parameters

All parameters from [`submit_inference.sh`](submit-inference.md) are supported, plus:

### `--no-sync`

Submit the job and wait for completion, but skip copying results locally. Useful if you just want to run inference without needing the results on your Mac immediately.

```bash
./run_inference.sh --project data/projects/One --latest --no-sync
```

### `--no-push`

Skip pushing `manual_data/` and `project.json` to the GPU before the job. Useful if you know the cluster already has the latest data.

```bash
./run_inference.sh --project data/projects/One --latest --no-push
```

### `--dry-run`

Show the generated SLURM script without submitting. No SSH connection is made.

```bash
./run_inference.sh --project data/projects/One --latest --dry-run
```

## Examples

### Latest Run with Auto-Sync

```bash
./run_inference.sh \
  --project data/projects/One \
  --latest
```

### Specific Run with Tracking

```bash
./run_inference.sh \
  --project data/projects/One \
  --run rfdetr_h100-96_20260223_230918 \
  --track \
  --frame-interval 5
```

### Test-Only Videos

```bash
./run_inference.sh \
  --project data/projects/One \
  --latest \
  --test-only
```

### Submit Without Syncing

```bash
./run_inference.sh \
  --project data/projects/One \
  --latest \
  --no-sync
```

## Output

Results are synced to your local project directory:

```
data/projects/One/inference/
└── rfdetr_h100-96_20260223_230918/
    └── video_1/
        └── 20260224_143922/
            ├── detected.mp4
            └── result.json
```

## Comparison with submit_inference.sh

| Aspect | `submit_inference.sh` | `run_inference.sh` |
|--------|----------------------|-------------------|
| Where it runs | On the GPU cluster | On your local Mac |
| Job submission | Direct `sbatch` call | `sbatch` via SSH |
| Blocking | Returns immediately | Blocks until job completes |
| Live output | Manual `tail -f` | Streamed automatically |
| Result syncing | Manual copy needed | Automatic via SSHFS mount |

Use `run_inference.sh` when working from your Mac for a seamless experience. Use `submit_inference.sh` when you're already SSH'd into the cluster or want fire-and-forget submission.

## Troubleshooting

### SSHFS Mount Not Found

```
Error: SSHFS mount not found at gpu-server/
```

Run `./mount_gpu.sh` first to mount the GPU server filesystem, or use `--no-sync` if you don't need local results.

### SSH Connection Failed

Ensure you can SSH to the cluster manually:

```bash
ssh youngjin@xlogin.comp.nus.edu.sg
```

The script uses SSH ControlMaster for connection reuse, so you'll only need to authenticate once per session.

### Job Failed

If the SLURM job fails, the script will display the job state and point you to the error log:

```
Job 12345 finished with state: FAILED
Error log: gpu-server/logs/slurm_12345_inference.err
```

## Related

- **[Run Training (Local)](run-training.md)** -- Local training runner with auto-sync
- **[Submit Inference](submit-inference.md)** -- Cluster-side inference submission
- **[Inference CLI](../cli/inference.md)** -- Local inference reference
- **[Inference Workflow](../guides/inference.md)** -- Complete guide
