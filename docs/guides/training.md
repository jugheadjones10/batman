# Training Workflow Guide

Complete end-to-end guide for training RF-DETR models with Batman.

## Overview

Training workflow:
1. **Prepare Data** - Import or create dataset
2. **Organize Classes** - Clean and standardize class names
3. **Configure Training** - Choose hyperparameters
4. **Train Model** - Run training locally or on cluster
5. **Evaluate Results** - Check metrics and visualizations
6. **Export Model** - Prepare for deployment

## Step 1: Prepare Data

### Option A: Import from COCO Zoo

```bash
python -m cli.importer coco \
  --project data/projects/MyProject \
  --create \
  --classes person car bicycle \
  --split validation \
  --max-samples 500
```

### Option B: Import from Roboflow

```bash
export ROBOFLOW_API_KEY=your_api_key

python -m cli.importer roboflow \
  --project data/projects/MyProject \
  --create \
  --workspace your-workspace \
  --rf-project your-project \
  --version 1
```

### Option C: Use Batman Web UI

1. Start development server: `./scripts/run_dev.sh`
2. Open http://localhost:5173
3. Create project and upload videos
4. Use SAM3 auto-labeling and manual correction

## Step 2: Organize Classes

### List Classes

```bash
python -m cli.classes list --project data/projects/MyProject
```

### Merge Similar Classes

```bash
python -m cli.classes merge \
  --project data/projects/MyProject \
  --source "crane-hook" "crane_hook" "hook" \
  --target "crane_hook"
```

### Rename Classes

```bash
python -m cli.classes rename \
  --project data/projects/MyProject \
  --old-name "crane-boom" \
  --new-name "crane_boom"
```

## Step 3: Configure Training

### Choose Model Size

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| base | ~28M | Balanced | Good | **Recommended for most cases** |
| large | ~76M | Slower | Highest | High accuracy needed |
| medium | ~48M | Medium | High | Balance accuracy/speed |
| small | ~10M | Fast | Lower | Edge deployment |

### Select Hyperparameters

#### Epochs

- **Quick test**: 10-20 epochs
- **Standard training**: 50 epochs
- **Fine-tuning**: 100-200 epochs

#### Batch Size

Choose based on GPU memory:

| GPU | Recommended Batch Size |
|-----|----------------------|
| H100 96GB | 16 |
| A100 80GB | 12-16 |
| A100 40GB | 8 |
| RTX 3090 | 4 |

Or use gradient accumulation:

```bash
--batch-size 4 --grad-accum 4  # Effective batch size: 16
```

#### Learning Rate

- **Fine-tuning pretrained**: `1e-5` to `1e-4` (default: `1e-4`)
- **Training from scratch**: `1e-4` to `1e-3`

#### Image Size

- **Standard**: 640 (default)
- **High detail**: 800-1024
- **Fast training**: 512

## Step 4: Train Model

### Local Training (e.g. Windows with GPU)

When you train on a **local machine** with a GPU (e.g. a Windows PC with CUDA), run the CLI from the project root. Output and TensorBoard logs are written to the run directory on disk.

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --model base \
  --epochs 50 \
  --batch-size 8 \
  --image-size 640 \
  --lr 1e-4 \
  --patience 10
```

By default the run is saved under `data/projects/MyProject/runs/rfdetr_<timestamp>`. To track training in TensorBoard, either:

- **From the Batman UI**: Open the Training page; local runs (no cluster) appear in the list with *gpu_type: local*. Click **Launch TensorBoard** to start TensorBoard and open the link.
- **From the command line**: `tensorboard --logdir data/projects/MyProject/runs/<run_name> --port 6006` then open http://localhost:6006.

No SSHFS or cluster connection is required for local GPU training.

### Local Training (explicit output dir)

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --model base \
  --epochs 50 \
  --batch-size 8 \
  --image-size 640 \
  --lr 1e-4 \
  --patience 10 \
  --output-dir runs/my_training_run
```

### SLURM Cluster Training (from Local Mac -- Recommended)

Use `run_training.sh` to push data, train, and sync results in one command:

```bash
./run_training.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50 \
  --model base \
  --batch-size 16
```

This automatically pushes your local `manual_data/`, frame metadata, labels, and `project.json` to the cluster before training, and syncs JSON metadata back when done. Requires the SSHFS mount (`./mount_gpu.sh`).

### Web UI (GPU Cluster)

The Batman web UI can submit training directly to the GPU cluster without needing SSH or shell scripts:

1. Start the dev server: `./scripts/run_dev.sh`
2. Navigate to **Training** in the sidebar
3. Enter your SSH password in the **GPU Connection Panel** (top-right)
4. Configure:
   - **Model Size** (nano / small / base / medium / large)
   - **GPU Type** and number of GPUs
   - **Training Params** (epochs, batch size, learning rate, patience, etc.)
   - **Data Sources** (manual data, imports, dataset filters)
5. Click **Submit Training**

The UI pushes your project data to the cluster, generates a SLURM script, submits it via `sbatch`, and streams the job logs in real time. You can cancel jobs, launch TensorBoard, and view metrics directly from the UI.

Results are automatically synced back to your local project when training completes.

### From the Cluster Directly

If you're already SSH'd into the cluster, use `submit_train.sh`:

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50 \
  --model base \
  --batch-size 16
```

### Multi-GPU Training

```bash
./run_training.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --num-gpus 4 \
  --batch-size 16 \
  --epochs 50
```

## Step 5: Monitor Training

### View Logs

```bash
# Local training
tail -f runs/my_training_run/*.log

# SLURM training
tail -f logs/slurm_*_rfdetr-*.out
```

### Check TensorBoard

**Local GPU training** (run directory on this machine):

```bash
tensorboard --logdir runs/my_training_run --port 6006
```

Or use the Batman UI: open the Training page and click **Launch TensorBoard** for the run (works for both cluster runs with SSHFS and local runs).

**Cluster training**: If you use the SSHFS mount (`./mount_gpu.sh`), launch TensorBoard from the UI so it reads from the mounted run directory. Otherwise run on the cluster: `tensorboard --logdir <run_path>/tensorboard --port 6006`.

Open http://localhost:6006

### Review Metrics

```bash
cat runs/my_training_run/results.json
```

Key metrics:
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Loss**: Training loss (lower is better)
- **Precision**: Fraction of correct detections
- **Recall**: Fraction of objects detected

### Run directory: JSON files and syncing

Each training run lives under `data/projects/<Project>/runs/<run_name>/`. These JSON files appear there and how they get there depends on whether you train **locally** or submit to the **GPU cluster** (UI or `submit_train.sh`).

| File | Purpose | Created by | Syncing behaviour |
|------|---------|------------|-------------------|
| **meta.json** | Run metadata for the UI and API: run id/name, status (queued/running/completed/failed/cancelled), SLURM job id, config snapshot, created_at, started_at, completed_at. | **Backend (Batman API)** when you submit a job. | **Local only.** Written on the machine running the Batman server. Updated in place when the background poll sees the job start or finish (`status`, `started_at`, `completed_at`). The list endpoint also updates it from SLURM when you still show as running (so the UI flips to “Completed” without waiting for the poll). |
| **training_config.json** | Exact training invocation: full command, timestamp, hostname, working directory, parsed arguments (project, output_dir, model, epochs, batch_size, etc.), environment (Python path, version). | **Training CLI** (`cli.train`) at the **start** of the run, on the machine where training runs (local or GPU server). | **GPU → local:** When you submit via the UI, this file is created on the cluster. After the job completes, the backend’s **result sync** pulls `*.json` from the cluster run dir into the local run dir, so you get `training_config.json` locally. |
| **results.json** | Final evaluation metrics: COCO-style mAP, precision, recall, per-class metrics (`class_map`), and optionally `checkpoint_path`. | **RF-DETR training/eval** (inside the training run) on the machine where training runs. | **GPU → local:** Same as above. Created on the cluster; synced down with other `*.json` when the job finishes. Used by the UI/API for run cards (e.g. mAP, best checkpoint) when present. |
| **class_info.json** | Class names, number of classes, and model id (e.g. `rf-detr-base`) for this run. Used by inference and the UI to load the right classes for a checkpoint. | **Training CLI** (`cli.train`) at the **end** of training (after the trainer returns), on the machine where training runs. | **GPU → local:** Created on the cluster; synced down with other `*.json`. Required for inference so the server knows class names for this run. |

**Summary**

- **meta.json**: Only on the Batman server host; created and updated by the backend; not synced from the cluster.
- **training_config.json**, **results.json**, **class_info.json**: Created on the machine that runs training (cluster when you submit via UI). When training is submitted through the UI, after the job completes the backend runs `sync_results(remote_run, local_run, ["*.json"])`, so those three files are **pulled from the GPU cluster** into your local run directory. Checkpoints (e.g. `best.pth`) are **not** synced by default; only `*.json` are.

## Step 6: Evaluate Results

### Check Training Configuration

```bash
cat runs/my_training_run/training_config.json
```

### View Validation Images

```bash
open runs/my_training_run/val_images/
```

### Test Inference

```bash
python -m cli.inference \
  --project data/projects/MyProject \
  --run my_training_run \
  --confidence 0.5
```

## Step 7: Iterate and Improve

### If Overfitting (High train accuracy, low val accuracy)

1. Reduce model complexity: Use `--model base` instead of `large`
2. Increase regularization: Lower learning rate
3. Add more data: Import additional samples
4. Early stopping: Use `--patience 10`

### If Underfitting (Low train and val accuracy)

1. Increase model capacity: Use `--model large`
2. Train longer: Increase `--epochs`
3. Check data quality: Review annotations
4. Adjust learning rate: Try `--lr 5e-5`

### If Slow Convergence

1. Increase learning rate: Try `--lr 5e-4`
2. Increase batch size: `--batch-size 16`
3. Use gradient accumulation: `--grad-accum 4`

### If Class Imbalance

Filter to focus on specific classes:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --filter-classes "rare_class" \
  --epochs 100
```

### Using Manual Data Subdatasets

Organize manually curated images into subdirectories inside `manual_data/` to create named datasets you can include or exclude per training run:

```
manual_data/
  crane_closeups/     # Dataset of close-up crane images
  worker_shots/       # Dataset of worker images
  negative_examples/  # Hard negatives you may want to toggle
```

Train with only specific datasets:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --sources manual_data,imports \
  --manual-datasets crane_closeups,worker_shots
```

Or exclude datasets:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --sources manual_data,imports \
  --exclude-manual-datasets negative_examples
```

See the [Training CLI docs](../cli/train.md#manual-data-subdatasets) for the full directory layout and naming conventions.

## Step 8: Export and Deploy

### Export for Inference

```bash
python -m cli.train \
  --checkpoint runs/my_training_run/best.pth \
  --export exports/my_model_v1 \
  --classes person car bicycle
```

### Test Exported Model

```bash
python -m cli.inference \
  --project data/projects/MyProject \
  --run my_training_run \
  --confidence 0.5 \
  --track
```

## Common Training Scenarios

### Scenario 1: Quick Prototype

Fast training for initial testing:

```bash
python -m cli.train \
  --project data/projects/Test \
  --model base \
  --epochs 20 \
  --batch-size 8 \
  --image-size 512
```

### Scenario 2: Production Model

High-quality training for deployment:

```bash
./submit_train.sh \
  --project data/projects/Production \
  --gpu h100-96 \
  --model large \
  --epochs 100 \
  --batch-size 16 \
  --image-size 800 \
  --patience 15
```

### Scenario 3: Limited GPU Memory

Training on smaller GPU:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --model base \
  --batch-size 2 \
  --grad-accum 8 \
  --image-size 512 \
  --epochs 50
```

### Scenario 4: Fine-tune Existing Model

Resume and continue training:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --resume runs/previous_run/checkpoint_epoch_25.pth \
  --output-dir runs/continued_training \
  --epochs 50 \
  --lr 1e-5
```

## Best Practices

### 1. Start Small

Begin with small datasets and quick training:

```bash
python -m cli.importer coco \
  --project data/projects/Test \
  --create \
  --classes person \
  --max-samples 100

python -m cli.train \
  --project data/projects/Test \
  --epochs 20
```

### 2. Use Default Settings First

Default settings work well for most cases:

```bash
python -m cli.train --project data/projects/MyProject
```

### 3. Monitor Training Closely

Watch for:
- Loss decreasing steadily
- mAP increasing
- Validation metrics improving

### 4. Save Checkpoints

Training automatically saves:
- `best.pth` - Best validation performance
- `checkpoint_last.pth` - Latest checkpoint
- `checkpoint_epoch_N.pth` - Periodic saves

### 5. Organize Experiments

Use descriptive output directories:

```bash
--output-dir runs/crane_hook_base_800px
--output-dir runs/person_car_large_v2
```

### 6. Document Training

Keep notes on:
- Dataset version
- Hyperparameters used
- Results and observations
- Issues encountered

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Reduce image size
--image-size 512

# Use gradient accumulation
--batch-size 4 --grad-accum 4
```

### Loss Not Decreasing

- Check learning rate (try `--lr 1e-4`)
- Verify data quality
- Ensure sufficient epochs
- Check for data loading errors in logs

### Training Stalls

- Monitor GPU usage: `nvidia-smi`
- Check disk space
- Review logs for errors
- Restart with lower batch size

### Poor Validation Performance

- Check for overfitting (train mAP >> val mAP)
- Add more training data
- Use smaller model
- Increase patience for early stopping

## Related

- **[Training CLI](../cli/train.md)** - Command reference
- **[Run Training (Local)](../scripts/run-training.md)** - Local runner with auto-sync
- **[Submit Training Script](../scripts/submit-train.md)** - Cluster-side SLURM training
- **[Inference Workflow](inference.md)** - Next steps
- **[Importer CLI](../cli/importer.md)** - Data import
