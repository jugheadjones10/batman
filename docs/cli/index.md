# CLI Tools Overview

Batman provides a comprehensive set of command-line tools for training, inference, benchmarking, and data management.

## Tool Categories

### 🎓 Training & Inference
- **[train](train.md)** - Train RF-DETR models on Batman projects or COCO datasets
- **[label](label.md)** - SAM3 auto-labeling on video frames or manual data
- **[inference](inference.md)** - Run inference on images and videos with tracking support

### ⚡ Performance Testing
- **[benchmark_latency](benchmark-latency.md)** - Measure inference latency with detailed statistics
- **[compare_latency](compare-latency.md)** - Compare benchmark results across GPUs

### 🎬 Video Tools
- **[create_latency_video](create-latency-video.md)** - Create latency-delayed visualization videos
- **[create_sidebyside_video](create-sidebyside-video.md)** - Create side-by-side comparison videos

### 📹 Video Management
- **[videos](videos.md)** - Add, list, remove, and manage project videos

### 📦 Data Management
- **[importer](importer.md)** - Import datasets from Roboflow or COCO Zoo
- **[classes](classes.md)** - List, rename, and merge object classes

## Common Usage Patterns

### Model Specification

Most tools accept one of three ways to specify a model:

```bash
# Option 1: By checkpoint path
--checkpoint path/to/model.pth

# Option 2: By run name (auto-finds checkpoint)
--run rfdetr_h100_20260120_105925

# Option 3: Use latest run
--latest
```

### Project Structure

Tools expect this directory structure:

```
runs/
  └── <run_name>/
      ├── best.pth              # Best checkpoint
      ├── checkpoint_best.pth   # Alternative name
      ├── training_config.json  # Training configuration
      └── results.json          # Training results

data/projects/
  └── <project_name>/
      ├── project.json          # Project metadata
      ├── videos/               # Video files
      ├── frames/               # Extracted frames
      └── labels/               # Annotations
```

### Device Selection

Most tools support automatic device detection:

```bash
# Auto-detect best device (CUDA > MPS > CPU)
--device auto

# Force specific device
--device cuda
--device mps
--device cpu
```

### Output Directories

Default output locations:

| Tool | Default Output |
|------|----------------|
| Training | `{project}/runs/rfdetr_{timestamp}/` |
| Inference | `{project}/inference/{run_name}/{video_id}/` |
| Benchmarking | `benchmark_results/{timestamp}/` |
| Dataset Export | `{project}/exports/coco` |

## Installation

All CLI tools are available after installing Batman:

```bash
# Install with uv
uv sync

# Run tools
uv run python -m cli.train --help
uv run python -m cli.inference --help
```

## Common Arguments

### Help and Version

```bash
# Get help for any tool
python -m cli.<tool> --help

# Get help for subcommands
python -m cli.importer --help
python -m cli.importer roboflow --help
```

### Confidence Thresholds

For inference and benchmarking:

```bash
# Detection confidence (inference default: 0.0, training default: 0.5)
--confidence 0.7

# Tracking threshold (default: 0.25)
--track-thresh 0.3
```

### Model Sizes

When specifying model architecture:

```bash
# For training
--model base    # RF-DETR-Base
--model large   # RF-DETR-Large

# Also supports
--model nano    # RF-DETR-Nano
--model small   # RF-DETR-Small
--model medium  # RF-DETR-Medium
```

## Quick Examples

### Train a Model

```bash
uv run python -m cli.train \
  --project data/projects/MyProject \
  --epochs 50 \
  --batch-size 8 \
  --model base
```

### Run Inference

```bash
uv run python -m cli.inference \
  --project data/projects/MyProject \
  --run my_training_run \
  --track \
  --confidence 0.6
```

### Benchmark Performance

```bash
uv run python -m cli.benchmark_latency \
  --run my_training_run \
  --video test_video.mp4 \
  --runs 100
```

### Import Data

```bash
uv run python -m cli.importer coco \
  --project data/projects/NewProject \
  --create \
  --classes person car bicycle
```

## Tips & Best Practices

### 1. Use Run Names for Convenience

Instead of specifying full checkpoint paths, use run names:

```bash
# Easier to type and remember
--run rfdetr_h100_20260120_105925

# Instead of
--checkpoint runs/rfdetr_h100_20260120_105925/best.pth
```

### 2. Enable Optimization for Inference

Most inference tools support model optimization:

```bash
# Enable automatic optimization (default)
python -m cli.inference --project data/projects/MyProject --run my_run

# Disable if you encounter issues
python -m cli.inference --project data/projects/MyProject --run my_run --no-optimize
```

### 3. Use Tracking for Videos

Enable tracking for better temporal consistency:

```bash
python -m cli.inference \
  --project data/projects/MyProject \
  --run my_run \
  --track \
  --track-buffer 30
```

### 4. Save JSON for Analysis

Inference outputs are saved to `{project}/inference/{run_name}/{video_id}/`:

```bash
python -m cli.inference --project data/projects/MyProject --run my_run
# Saves: result.json + detected.mp4 per video
```

### 5. Specify Classes for Clarity

Load class names from projects or specify manually:

```bash
# From project
--project data/projects/MyProject

# Manual specification (space-separated)
--classes person car bicycle
```

## Next Steps

- **[Training Guide](../guides/training.md)** - Learn the complete training workflow
- **[Inference Guide](../guides/inference.md)** - Master inference techniques
- **[SLURM Scripts](../scripts/index.md)** - Run jobs on HPC clusters
- **[Benchmarking Guide](../guides/benchmarking.md)** - Performance testing strategies
