# Training CLI

Train RF-DETR models on Batman project data or existing COCO datasets.

## Overview

The training CLI handles:

- Dataset preparation from Batman projects or COCO format
- Model training with configurable hyperparameters
- Model export and inference testing
- Support for multiple GPU devices (CUDA, MPS, CPU)

## Basic Usage

```bash
python -m cli.train --project data/projects/MyProject
```

## Command Builder

<div class="command-builder-widget" data-tool="train" data-params='[
  {"name": "project", "type": "path", "required": false, "description": "Path to Batman project directory", "group": "Input"},
  {"name": "dataset", "type": "path", "required": false, "description": "Path to existing COCO format dataset", "group": "Input"},
  {"name": "checkpoint", "type": "path", "required": false, "description": "Path to trained checkpoint for inference/export", "group": "Input"},
  {"name": "output-dataset", "type": "path", "default": "datasets/rfdetr_coco", "description": "Output directory for prepared COCO dataset", "group": "Data Preparation"},
  {"name": "train-split", "type": "number", "default": 0.70, "min": 0, "max": 1, "step": 0.05, "description": "Training data fraction", "group": "Data Preparation"},
  {"name": "val-split", "type": "number", "default": 0.15, "min": 0, "max": 1, "step": 0.05, "description": "Validation data fraction", "group": "Data Preparation"},
  {"name": "test-split", "type": "number", "default": 0.15, "min": 0, "max": 1, "step": 0.05, "description": "Test data fraction", "group": "Data Preparation"},
  {"name": "video-id", "type": "number", "default": -1, "description": "Video ID to process, -1 for imports", "group": "Data Preparation"},
  {"name": "filter-classes", "type": "text", "description": "Only train on these classes (pipe-separated)", "group": "Data Preparation"},
  {"name": "prepare-only", "type": "flag", "description": "Only prepare dataset, do not train", "group": "Data Preparation"},
  {"name": "no-clean", "type": "flag", "description": "Do not remove existing dataset directory", "group": "Data Preparation"},
  {"name": "output-dir", "type": "path", "default": "runs/rfdetr_run", "description": "Output directory for training run", "group": "Training"},
  {"name": "model", "type": "choice", "choices": ["nano", "small", "base", "medium", "large"], "default": "base", "description": "Model architecture size", "group": "Training"},
  {"name": "epochs", "type": "number", "default": 50, "min": 1, "description": "Number of training epochs", "group": "Training"},
  {"name": "batch-size", "type": "number", "default": 8, "min": 1, "description": "Batch size", "group": "Training"},
  {"name": "image-size", "type": "number", "default": 640, "min": 320, "max": 1280, "step": 32, "description": "Input image size", "group": "Training"},
  {"name": "lr", "type": "text", "default": "1e-4", "description": "Learning rate", "group": "Training"},
  {"name": "device", "type": "choice", "choices": ["auto", "cuda", "mps", "cpu"], "default": "auto", "description": "Device for training", "group": "Training"},
  {"name": "num-workers", "type": "number", "default": 2, "min": 0, "description": "Data loader workers", "group": "Training"},
  {"name": "patience", "type": "number", "default": 10, "min": 0, "description": "Early stopping patience (0 to disable)", "group": "Training"},
  {"name": "resume", "type": "path", "description": "Resume training from checkpoint", "group": "Training"},
  {"name": "grad-accum", "type": "number", "default": 1, "min": 1, "description": "Gradient accumulation steps", "group": "Training"},
  {"name": "mps-fallback", "type": "flag", "description": "Enable MPS CPU fallback", "group": "Training"},
  {"name": "inference", "type": "text", "description": "Run inference on image(s) after training", "group": "Inference"},
  {"name": "confidence", "type": "number", "default": 0.5, "min": 0, "max": 1, "step": 0.05, "description": "Detection confidence threshold", "group": "Inference"},
  {"name": "inference-output", "type": "path", "description": "Output directory for annotated images", "group": "Inference"},
  {"name": "export", "type": "path", "description": "Export model to directory", "group": "Export"},
  {"name": "classes", "type": "text", "description": "Class names for inference/export", "group": "General"},
  {"name": "seed", "type": "number", "default": 42, "description": "Random seed", "group": "General"}
]'></div>

## Parameters

### Input Sources (Choose One)

#### `--project PATH`

Path to Batman project directory containing labeled data.

```bash
python -m cli.train --project data/projects/MyProject
```

#### `--dataset PATH`

Path to existing COCO format dataset (skip preparation).

```bash
python -m cli.train --dataset datasets/my_coco_dataset
```

#### `--checkpoint PATH`

Path to trained checkpoint (for inference or export only).

```bash
python -m cli.train --checkpoint runs/my_run/best.pth --inference img.jpg
```

### Data Preparation

#### `--output-dataset PATH`

Output directory for prepared COCO dataset.

- **Default**: `datasets/rfdetr_coco`

#### `--train-split FRACTION`

Fraction of data for training.

- **Default**: `0.70` (70%)

#### `--val-split FRACTION`

Fraction of data for validation.

- **Default**: `0.15` (15%)

#### `--test-split FRACTION`

Fraction of data for testing.

- **Default**: `0.15` (15%)

#### `--video-id ID`

Video ID to process. Use `-1` for imported frames.

- **Default**: `-1`

#### `--filter-classes CLASSES`

Only train on specific classes (pipe-separated for multi-word).

```bash
--filter-classes "crane hook|crane-hook"
```

#### `--prepare-only`

Only prepare dataset without training.

#### `--no-clean`

Don't remove existing dataset directory before preparing.

### Training Configuration

#### `--output-dir PATH`

Output directory for training run (checkpoints, logs, configs).

- **Default**: `runs/rfdetr_run`

#### `--model SIZE`

Model architecture size.

- **Choices**: `nano`, `small`, `base`, `medium`, `large`
- **Default**: `base`

| Model  | Parameters | Speed    | Accuracy |
| ------ | ---------- | -------- | -------- |
| nano   | ~3M        | Fastest  | Lower    |
| small  | ~10M       | Fast     | Good     |
| base   | ~28M       | Balanced | Better   |
| medium | ~48M       | Slower   | High     |
| large  | ~76M       | Slowest  | Highest  |

#### `--epochs N`

Number of training epochs.

- **Default**: `50`

#### `--batch-size N`

Batch size per device.

- **Default**: `8`
- **Recommendations**:
  - A100-80GB / H100: 16
  - A100-40GB: 12
  - V100-32GB: 8
  - RTX 3090: 4

#### `--image-size N`

Input image size (must be multiple of 32).

- **Default**: `640`
- **Common values**: `512`, `640`, `800`, `1024`

#### `--lr RATE`

Learning rate.

- **Default**: `1e-4`
- **For fine-tuning**: `1e-5` to `1e-4`
- **From scratch**: `1e-3` to `1e-4`

#### `--device TYPE`

Device for training.

- **Choices**: `auto`, `cuda`, `mps`, `cpu`
- **Default**: `auto` (CUDA > MPS > CPU)

#### `--num-workers N`

Number of data loader worker processes.

- **Default**: `2`

#### `--patience N`

Early stopping patience (epochs without improvement). Set to `0` to disable.

- **Default**: `10`

#### `--resume PATH`

Resume training from checkpoint.

```bash
--resume runs/my_run/checkpoint_epoch_20.pth
```

#### `--grad-accum N`

Gradient accumulation steps (for effective larger batch size).

- **Default**: `1`
- **Example**: `--batch-size 4 --grad-accum 4` (effective batch size: 16)

#### `--mps-fallback`

Enable MPS CPU fallback for unsupported operations (macOS).

### Inference (Post-Training)

#### `--inference IMAGES`

Run inference on image(s) after training.

```bash
--inference "img1.jpg img2.jpg"
--inference "path/to/images/*.jpg"
```

#### `--confidence THRESHOLD`

Detection confidence threshold.

- **Default**: `0.5`
- **Range**: `0.0` to `1.0`

#### `--inference-output PATH`

Output directory for annotated images.

### Export

#### `--export PATH`

Export trained model to directory.

```bash
--export exports/my_model
```

### General

#### `--classes NAMES`

Class names for inference/export (when not using project).

```bash
--classes "person,car,bicycle"
```

#### `--seed N`

Random seed for reproducibility.

- **Default**: `42`

## Examples

### Example 1: Basic Training

Train on a Batman project with default settings:

```bash
python -m cli.train \
  --project data/projects/CraneHook
```

### Example 2: Custom Configuration

Train with custom hyperparameters:

```bash
python -m cli.train \
  --project data/projects/CraneHook \
  --model large \
  --epochs 100 \
  --batch-size 16 \
  --image-size 800 \
  --lr 5e-5 \
  --patience 15 \
  --output-dir runs/crane_hook_large
```

### Example 3: Filter Classes

Train only on specific classes:

```bash
python -m cli.train \
  --project data/projects/MultiClass \
  --filter-classes "crane hook|crane-hook" \
  --epochs 50
```

### Example 4: Prepare Dataset Only

Prepare dataset without training:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --output-dataset datasets/my_dataset \
  --prepare-only
```

### Example 5: Train on Existing Dataset

Train on pre-prepared COCO dataset:

```bash
python -m cli.train \
  --dataset datasets/my_dataset \
  --model base \
  --epochs 50 \
  --batch-size 8
```

### Example 6: Resume Training

Resume interrupted training:

```bash
python -m cli.train \
  --dataset datasets/my_dataset \
  --resume runs/my_run/checkpoint_epoch_25.pth \
  --output-dir runs/my_run
```

### Example 7: Train and Inference

Train and immediately test on images:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --epochs 30 \
  --inference "test_images/*.jpg" \
  --confidence 0.6 \
  --inference-output results/
```

### Example 8: Small GPU (Gradient Accumulation)

Train on limited GPU memory:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --batch-size 2 \
  --grad-accum 4 \
  --image-size 512
```

### Example 9: Export Trained Model

Export model for deployment:

```bash
python -m cli.train \
  --checkpoint runs/my_run/best.pth \
  --export exports/my_model_v1 \
  --classes "person,car,bicycle"
```

## Output Structure

Training creates this structure:

```
runs/
  └── <output-dir>/
      ├── best.pth                 # Best model checkpoint
      ├── checkpoint_last.pth      # Latest checkpoint
      ├── checkpoint_epoch_N.pth   # Periodic checkpoints
      ├── training_config.json     # Full training configuration
      ├── results.json             # Training metrics
      ├── tensorboard/             # TensorBoard logs
      └── val_images/              # Validation visualizations
```

## Training Logs

Monitor training progress:

```bash
# View training config
cat runs/my_run/training_config.json

# View final results
cat runs/my_run/results.json

# Monitor with TensorBoard
tensorboard --logdir runs/my_run/tensorboard
```

## Tips & Best Practices

### 1. Start with Base Model

Use `--model base` for balanced speed/accuracy:

```bash
python -m cli.train --project data/projects/MyProject --model base
```

### 2. Adjust Batch Size for Your GPU

Monitor GPU memory usage and increase batch size:

```bash
# Check GPU memory
nvidia-smi

# Increase if you have headroom
--batch-size 16
```

### 3. Use Gradient Accumulation for Small GPUs

Simulate larger batch size:

```bash
--batch-size 4 --grad-accum 4  # Effective batch size: 16
```

### 4. Enable Early Stopping

Prevent overfitting with patience:

```bash
--patience 10  # Stop if no improvement for 10 epochs
```

### 5. Filter Classes for Focused Training

Train on specific classes:

```bash
--filter-classes "person|pedestrian"
```

### 6. Prepare Once, Train Multiple Times

Separate dataset preparation from training:

```bash
# Prepare dataset
python -m cli.train --project data/projects/MyProject --prepare-only

# Train multiple configurations
python -m cli.train --dataset datasets/rfdetr_coco --model base --epochs 50
python -m cli.train --dataset datasets/rfdetr_coco --model large --epochs 100
```

## Related

- **[Inference CLI](inference.md)** - Run trained models
- **[Submit Training Script](../scripts/submit-train.md)** - SLURM training
- **[Training Workflow Guide](../guides/training.md)** - Complete workflow
- **[Importer CLI](importer.md)** - Import training data
