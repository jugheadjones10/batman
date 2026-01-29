# Submit Training Script

Submit RF-DETR training jobs to SLURM clusters with automatic GPU configuration and resource allocation.

## Basic Usage

```bash
./submit_train.sh --project data/projects/MyProject
```

## Command Builder

<div class="command-builder-widget" data-tool="submit_train" data-params='[
  {"name": "gpu", "type": "choice", "choices": ["h200", "h100-96", "h100-47", "a100-80", "a100-40", "nv"], "default": "h100-96", "description": "GPU type", "group": "GPU"},
  {"name": "num-gpus", "type": "number", "default": 1, "min": 1, "max": 8, "description": "Number of GPUs", "group": "GPU"},
  {"name": "project", "type": "path", "default": "data/projects/Test", "description": "Project directory", "group": "Training"},
  {"name": "epochs", "type": "number", "default": 50, "min": 1, "description": "Training epochs", "group": "Training"},
  {"name": "batch-size", "type": "number", "description": "Batch size (auto-set if not specified)", "group": "Training"},
  {"name": "image-size", "type": "number", "default": 640, "min": 320, "max": 1280, "step": 32, "description": "Image size", "group": "Training"},
  {"name": "lr", "type": "text", "default": "1e-4", "description": "Learning rate", "group": "Training"},
  {"name": "patience", "type": "number", "default": 10, "min": 0, "description": "Early stopping patience", "group": "Training"},
  {"name": "model", "type": "choice", "choices": ["base", "large"], "default": "base", "description": "Model size", "group": "Training"},
  {"name": "output-dir", "type": "path", "description": "Output directory (auto-generated if not set)", "group": "Training"},
  {"name": "filter-classes", "type": "text", "description": "Only train on specific classes", "group": "Training"},
  {"name": "partition", "type": "text", "description": "SLURM partition (auto-detected if not set)", "group": "SLURM"},
  {"name": "time", "type": "text", "default": "24:00:00", "description": "Time limit (HH:MM:SS)", "group": "SLURM"},
  {"name": "prepare-only", "type": "flag", "description": "Only prepare dataset", "group": "Other"},
  {"name": "dry-run", "type": "flag", "description": "Show script without submitting", "group": "Other"}
]'></div>

## Parameters

### GPU Options

#### `--gpu=TYPE`

GPU type to use.

- **Choices**: `h200`, `h100-96`, `h100-47`, `a100-80`, `a100-40`, `nv`
- **Default**: `h100-96`

| GPU     | VRAM   | Best For                        |
| ------- | ------ | ------------------------------- |
| h200    | 141GB  | Largest models, biggest batches |
| h100-96 | 96GB   | Large models, training          |
| h100-47 | 47GB   | Medium models                   |
| a100-80 | 80GB   | General purpose                 |
| a100-40 | 40GB   | Smaller models                  |
| nv      | Varies | V100/Titan/T4 (legacy)          |

#### `--num-gpus=N`

Number of GPUs for distributed training.

- **Default**: `1`
- **Range**: `1-8`

Multi-GPU training uses PyTorch `torchrun` for distributed data parallel.

### Training Options

#### `--project=PATH`

Project directory containing labeled data.

- **Default**: `data/projects/Test`

#### `--epochs=N`

Number of training epochs.

- **Default**: `50`

#### `--batch-size=N`

Batch size per GPU.

- **Default**: Auto-set based on GPU type:
  - H200/H100-96: `16`
  - H100-47/A100-80: `12`
  - A100-40: `8`
  - NV: `4`

#### `--image-size=N`

Input image size (must be multiple of 32).

- **Default**: `640`

#### `--lr=RATE`

Learning rate.

- **Default**: `1e-4`

#### `--patience=N`

Early stopping patience (0 to disable).

- **Default**: `10`

#### `--model=SIZE`

Model architecture size.

- **Choices**: `base`, `large`
- **Default**: `base`

#### `--output-dir=PATH`

Output directory for training run.

- **Default**: Auto-generated as `runs/rfdetr_{GPU_TYPE}_{TIMESTAMP}/`

#### `--filter-classes=NAMES`

Only train on specific classes (pipe-separated for multi-word).

```bash
--filter-classes="crane hook|crane-hook"
```

### SLURM Options

#### `--partition=NAME`

SLURM partition to use.

- **Default**: Auto-detected based on GPU type

| GPU              | Default Partition |
| ---------------- | ----------------- |
| h200             | `gpu`             |
| h100-96, h100-47 | `h100`            |
| a100-80, a100-40 | `a100`            |
| nv               | `nv`              |

#### `--time=LIMIT`

Time limit in format `HH:MM:SS`.

- **Default**: `24:00:00`
- **Note**: H200 on `gpu` partition limited to 3 hours

### Other Options

#### `--prepare-only`

Only prepare dataset without training.

#### `--dry-run`

Show generated SLURM script without submitting.

## Examples

### Example 1: Basic Training

Train with default settings:

```bash
./submit_train.sh --project data/projects/MyProject
```

### Example 2: Custom Configuration

Train with specific hyperparameters:

```bash
./submit_train.sh \
  --project data/projects/CraneHook \
  --gpu h100-96 \
  --epochs 100 \
  --batch-size 16 \
  --image-size 800 \
  --lr 5e-5 \
  --model large
```

### Example 3: Multi-GPU Training

Distributed training on 4 GPUs:

```bash
./submit_train.sh \
  --project data/projects/LargeDataset \
  --gpu h100-96 \
  --num-gpus 4 \
  --batch-size 16 \
  --epochs 50
```

### Example 4: Filter Classes

Train only on specific classes:

```bash
./submit_train.sh \
  --project data/projects/MultiClass \
  --filter-classes="person|pedestrian" \
  --gpu h100-96
```

### Example 5: Extended Training

Long training with large model:

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --model large \
  --epochs 200 \
  --patience 20 \
  --time 48:00:00
```

### Example 6: Budget-Friendly Training

Use smaller GPU:

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu a100-40 \
  --batch-size 8 \
  --image-size 512
```

### Example 7: Dry Run

Preview SLURM script before submitting:

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --dry-run
```

### Example 8: Dataset Preparation Only

Prepare dataset without training:

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --prepare-only
```

## Output

### Job Submission

```bash
$ ./submit_train.sh --project data/projects/Test

Submitted batch job 123456
Job ID: 123456
Output directory: runs/rfdetr_h100_20260128_105030/
Log file: logs/job_123456.log

To monitor:
  tail -f logs/job_123456.log
  squeue -j 123456
```

### Generated Files

```
runs/rfdetr_h100_20260128_105030/
├── best.pth                 # Best checkpoint
├── checkpoint_last.pth      # Latest checkpoint
├── training_config.json     # Configuration
├── results.json             # Training metrics
├── tensorboard/             # TensorBoard logs
└── val_images/              # Validation visualizations

logs/
└── job_123456.log          # SLURM job log
```

## Monitoring

### View Job Status

```bash
# List your jobs
squeue -u $USER

# Watch specific job
watch -n 5 squeue -j 123456
```

### Follow Logs

```bash
# Follow log output
tail -f logs/job_123456.log

# View full log
cat logs/job_123456.log
```

### Check TensorBoard

```bash
# On compute node
module load tensorboard
tensorboard --logdir runs/rfdetr_h100_20260128_105030/tensorboard --port 6006

# Forward port to local machine
ssh -L 6006:localhost:6006 user@cluster
```

## Multi-GPU Training

### Configuration

Multi-GPU training uses PyTorch Distributed Data Parallel:

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --num-gpus 4 \
  --gpu h100-96
```

### Effective Batch Size

With `--num-gpus=4` and `--batch-size=16`:

- Batch size per GPU: 16
- Total effective batch size: 64

### GPU Limits

Check node GPU count:

| Node Type  | Max GPUs |
| ---------- | -------- |
| H100 nodes | 4        |
| A100 nodes | 4        |
| H200 nodes | 8        |

Script validates GPU count against limits.

## Best Practices

### 1. Choose Appropriate GPU

- **Large datasets**: H100-96
- **Medium datasets**: A100-80
- **Small datasets**: A100-40
- **Experimentation**: A100-40 (cost-effective)

### 2. Start with Default Batch Size

Let the script auto-configure:

```bash
./submit_train.sh --project ... --gpu h100-96
# Uses batch size 16
```

### 3. Use Dry Run First

Preview before submitting:

```bash
./submit_train.sh --dry-run ...
```

### 4. Monitor Resource Usage

Check GPU utilization during training:

```bash
ssh <node_name>
nvidia-smi -l 1
```

### 5. Set Appropriate Time Limits

Estimate training time:

- ~1 epoch = 5-10 minutes (typical dataset)
- 50 epochs = 4-8 hours
- Add buffer: Set `--time 12:00:00`

## Troubleshooting

### Out of Memory

Reduce batch size or image size:

```bash
./submit_train.sh --batch-size 4 --image-size 512 ...
```

### H200 Time Limit

H200 limited to 3 hours on `gpu` partition:

```bash
# Use H100-96 for longer jobs
./submit_train.sh --gpu h100-96 --time 24:00:00 ...
```

### Job Pending

Check partition availability:

```bash
sinfo -p h100,a100,gpu
```

### Training Stalled

Check logs for errors:

```bash
tail -100 logs/job_*.log
```

## Related

- **[Training CLI](../cli/train.md)** - Local training
- **[Submit Inference](submit-inference.md)** - Run inference
- **[SLURM Guide](../guides/slurm.md)** - Complete SLURM guide
- **[Training Workflow](../guides/training.md)** - End-to-end workflow
