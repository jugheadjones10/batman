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
  {"name": "output-dataset", "type": "path", "description": "Output directory for prepared COCO dataset (default: {project}/exports/coco)", "group": "Data Preparation"},
  {"name": "train-split", "type": "number", "default": 0.70, "min": 0, "max": 1, "step": 0.05, "description": "Training data fraction", "group": "Data Preparation"},
  {"name": "val-split", "type": "number", "default": 0.15, "min": 0, "max": 1, "step": 0.05, "description": "Validation data fraction", "group": "Data Preparation"},
  {"name": "test-split", "type": "number", "default": 0.15, "min": 0, "max": 1, "step": 0.05, "description": "Test data fraction", "group": "Data Preparation"},
  {"name": "video-id", "type": "text", "default": "imports", "description": "Video ID(s): all, imports (default), or specific ID", "group": "Data Preparation"},
  {"name": "filter-classes", "type": "text", "description": "Only train on these classes (pipe-separated)", "group": "Data Preparation"},
  {"name": "prepare-only", "type": "flag", "description": "Only prepare dataset, do not train", "group": "Data Preparation"},
  {"name": "max-frames-per-class", "type": "number", "min": 1, "description": "Cap frames per class to this number (random sample, deterministic with seed)", "group": "Data Preparation"},
  {"name": "no-clean", "type": "flag", "description": "Do not remove existing dataset directory", "group": "Data Preparation"},
  {"name": "sources", "type": "text", "description": "Data sources (comma-separated): manual_data,imports. Overrides --video-id.", "group": "Data Preparation"},
  {"name": "manual-split-strategy", "type": "choice", "choices": ["proportional", "val_only", "train_only", "all_splits"], "default": "train_only", "description": "How to distribute manual data across splits", "group": "Data Preparation"},
  {"name": "manual-datasets", "type": "text", "description": "Only include these manual subdatasets (comma-separated). Use (root) for root-level images.", "group": "Data Preparation"},
  {"name": "exclude-manual-datasets", "type": "text", "description": "Exclude these manual subdatasets (comma-separated). Mutually exclusive with --manual-datasets.", "group": "Data Preparation"},
  {"name": "output-dir", "type": "path", "description": "Output directory for training run (default: {project}/runs/rfdetr_run)", "group": "Training"},
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
  {"name": "infer-after", "type": "flag", "description": "Run inference on project videos after training", "group": "Post-training Inference"},
  {"name": "infer-test-only", "type": "flag", "description": "With --infer-after, only run on test-only videos", "group": "Post-training Inference"},
  {"name": "export", "type": "path", "description": "Export model to directory", "group": "Export"},
  {"name": "classes", "type": "text", "description": "Class names for export (space-separated)", "group": "General"},
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
python -m cli.train --checkpoint runs/my_run/best.pth --export exports/my_model
```

### Data Preparation

#### `--output-dataset PATH`

Output directory for prepared COCO dataset.

- **Default**: `None` (auto: `{project}/exports/coco` when using `--project`)

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

Video ID(s) to process. Accepts `'all'`, `'imports'` (default), or a specific video ID.

- **Default**: `imports`
- **Type**: `str`

#### `--filter-classes CLASSES`

Only train on specific classes (pipe-separated for multi-word).

```bash
--filter-classes "crane hook|crane-hook"
```

#### `--prepare-only`

Only prepare dataset without training.

#### `--no-clean`

Don't remove existing dataset directory before preparing.

#### `--sources TYPES`

Data sources to include (comma-separated). Valid values: `manual_data`, `imports`. When set, overrides `--video-id` and always excludes video frames.

```bash
--sources manual_data
--sources manual_data,imports
```

#### `--manual-split-strategy STRATEGY`

How to distribute manual data across train/val/test splits.

- **Choices**: `proportional`, `val_only`, `train_only`, `all_splits`
- **Default**: `train_only`

| Strategy | Behavior |
| --- | --- |
| `train_only` | All manual data goes to train split |
| `val_only` | All manual data goes to validation split |
| `proportional` | Distribute across all splits proportionally |
| `all_splits` | Include manual data in every split |

#### `--manual-datasets NAMES`

Only include specific manual data subdatasets (comma-separated). Subdatasets correspond to subdirectories inside `manual_data/`. Use `(root)` to include root-level images. Mutually exclusive with `--exclude-manual-datasets`.

```bash
--manual-datasets crane_closeups,worker_shots
--manual-datasets "(root),crane_closeups"
```

#### `--exclude-manual-datasets NAMES`

Exclude specific manual data subdatasets (comma-separated). All other manual datasets are included. Mutually exclusive with `--manual-datasets`.

```bash
--exclude-manual-datasets negative_examples
```

See [Manual Data Subdatasets](#manual-data-subdatasets) below for the directory layout.

#### `--max-frames-per-class N`

Cap the number of frames per class to roughly `N` by randomly down-sampling classes that exceed the limit. Classes with fewer than `N` frames are kept as-is. Sampling is deterministic when combined with `--seed` (default 42), so the same command always produces the same split.

```bash
--max-frames-per-class 300
```

This is useful for **class balancing** -- if one class has 1500 frames and another has 250, capping at 300 brings them to a similar scale without losing the smaller class.

### Training Configuration

#### `--output-dir PATH`

Output directory for training run (checkpoints, logs, configs).

- **Default**: `None` (auto: `{project}/runs/rfdetr_run` when using `--project`)

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

### Post-training Inference

#### `--infer-after`

Run inference on project videos after training completes. Requires `--project`.

#### `--infer-test-only`

With `--infer-after`, only run inference on videos marked with `exclude_from_training=true`.

### Export

#### `--export PATH`

Export trained model to directory.

```bash
--export exports/my_model
```

### General

#### `--classes NAMES`

Class names for export (when not using project). Space-separated.

```bash
--classes person car bicycle
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

### Example 4: Class-Balanced Training

Balance classes by capping each to ~300 frames:

```bash
python -m cli.train \
  --project data/projects/CraneHook \
  --max-frames-per-class 300 \
  --seed 42
```

The `--seed` value is saved in `training_config.json`, so the exact same dataset can be reproduced by re-running with the same arguments.

### Example 5: Prepare Dataset Only

Prepare dataset without training:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --output-dataset datasets/my_dataset \
  --prepare-only
```

### Example 6: Train on Existing Dataset

Train on pre-prepared COCO dataset:

```bash
python -m cli.train \
  --dataset datasets/my_dataset \
  --model base \
  --epochs 50 \
  --batch-size 8
```

### Example 7: Resume Training

Resume interrupted training:

```bash
python -m cli.train \
  --dataset datasets/my_dataset \
  --resume runs/my_run/checkpoint_epoch_25.pth \
  --output-dir runs/my_run
```

### Example 8: Train and Run Inference

Train and immediately run inference on project videos:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --epochs 30 \
  --infer-after
```

To only infer on test-only videos:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --epochs 30 \
  --infer-after \
  --infer-test-only
```

### Example 9: Small GPU (Gradient Accumulation)

Train on limited GPU memory:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --batch-size 2 \
  --grad-accum 4 \
  --image-size 512
```

### Example 10: Export Trained Model

Export model for deployment:

```bash
python -m cli.train \
  --checkpoint runs/my_run/best.pth \
  --export exports/my_model_v1 \
  --classes person car bicycle
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

## Manual Data Subdatasets

The `manual_data/` folder supports subdirectories to organize images into named datasets. Root-level images remain as the default dataset for backward compatibility.

### Directory Layout

```
manual_data/
  image_a.jpg                      # Root-level -> dataset "(root)"
  crane_closeups/
    img1.jpg                       # Subdataset "crane_closeups"
    img2.jpg
  worker_shots/
    img3.jpg                       # Subdataset "worker_shots"
```

After syncing, each subdirectory gets its own `frames.json`:

```
frames/
  manual_data/                     # Root-level images
    frames.json
  manual_data__crane_closeups/     # Subdataset
    frames.json
  manual_data__worker_shots/       # Subdataset
    frames.json
```

### Including Specific Datasets

```bash
# Only use crane_closeups and worker_shots manual datasets
python -m cli.train \
  --project data/projects/MyProject \
  --sources manual_data,imports \
  --manual-datasets crane_closeups,worker_shots
```

### Excluding Specific Datasets

```bash
# Use all manual datasets except negative_examples
python -m cli.train \
  --project data/projects/MyProject \
  --sources manual_data,imports \
  --exclude-manual-datasets negative_examples
```

### Including Root-Level Images

To include root-level images (those directly in `manual_data/`, not in a subdirectory), use the special name `(root)`:

```bash
--manual-datasets "(root),crane_closeups"
```

## Related

- **[Inference CLI](inference.md)** - Run trained models
- **[Submit Training Script](../scripts/submit-train.md)** - SLURM training
- **[Training Workflow Guide](../guides/training.md)** - Complete workflow
- **[Importer CLI](importer.md)** - Import training data
