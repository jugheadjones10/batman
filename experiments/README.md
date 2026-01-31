# Experiments Directory

This directory contains a Hydra-based experiment runner for conducting class imbalance studies on object detection models using RF-DETR. The system is designed to run multiple experiments with different data sampling configurations, either locally or on a SLURM cluster.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Configuration System (Hydra)](#configuration-system-hydra)
- [Scripts Reference](#scripts-reference)
  - [train_experiment.py](#train_experimentpy)
  - [collect_results.py](#collect_resultspy)
  - [resume_experiments.py](#resume_experimentspy)
  - [run_all.sh](#run_allsh)
- [Running Experiments](#running-experiments)
- [Output Structure](#output-structure)

---

## Overview

The experiments framework enables systematic evaluation of how class imbalance affects object detection model performance. Each experiment:

1. **Prepares a dataset** with specified class sampling fractions
2. **Trains an RF-DETR model** on the prepared dataset
3. **Runs inference** on test videos
4. **Generates a summary** with metrics and results

The framework uses [Hydra](https://hydra.cc/) for configuration management, allowing easy parameter sweeps and SLURM integration via `submitit`.

---

## Directory Structure

```
experiments/
├── README.md                 # This documentation file
├── __init__.py               # Python package marker
├── train_experiment.py       # Main experiment runner
├── collect_results.py        # Results aggregation script
├── resume_experiments.py     # Resume incomplete experiments
├── run_all.sh                # Shell script to run all experiments
└── conf/                     # Hydra configuration directory
    ├── config.yaml           # Main configuration file
    └── experiment/           # Experiment-specific configs
        ├── exp_person_25.yaml
        ├── exp_person_50.yaml
        ├── exp_person_75.yaml
        └── exp_person_100.yaml
```

---

## Configuration System (Hydra)

### Main Configuration (`conf/config.yaml`)

The main config file defines default settings and references experiment-specific overrides:

```yaml
defaults:
  - experiment: exp_person_100  # Default experiment
  - _self_

# Project settings
project_dir: data/projects/Test
output_dataset: datasets/experiment_${experiment.name}

# Resume from existing experiment (set to path to skip training)
resume_from: null

# Classes to train on
classes:
  - crane hook
  - person

# Training settings
training:
  epochs: 50
  batch_size: 16
  image_size: 640
  lr: 1e-4
  patience: 10
  model: base

# Test videos for inference after training
test_videos:
  - crane_hook+human_short.mp4
  - crane_hook_1_short.mp4

# Hydra output directory config
hydra:
  run:
    dir: experiments/outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: experiments/multirun/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${experiment.name}
```

### Key Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `project_dir` | Path to the BATMAN project containing source videos |
| `output_dataset` | Where to save the prepared COCO-format dataset |
| `resume_from` | Path to resume from (skips training if set) |
| `classes` | List of object classes to train on |
| `training.epochs` | Number of training epochs |
| `training.batch_size` | Batch size for training |
| `training.image_size` | Input image resolution |
| `training.lr` | Learning rate |
| `training.patience` | Early stopping patience |
| `training.model` | RF-DETR model size (`base`, `large`) |
| `test_videos` | Videos for post-training inference |

### Experiment Configurations (`conf/experiment/`)

Each experiment config defines the `frame_sample_fractions` which controls how much data from each class is used:

**exp_person_25.yaml** - Person at 25% (1:4 ratio)
```yaml
name: exp_person_25
description: "Person at 25% (1:4 ratio)"
frame_sample_fractions:
  person: 0.25
  crane hook: 1.0
```

**exp_person_50.yaml** - Person at 50% (1:2 ratio)
```yaml
name: exp_person_50
description: "Person at 50% (1:2 ratio)"
frame_sample_fractions:
  person: 0.50
  crane hook: 1.0
```

**exp_person_75.yaml** - Person at 75% (~1:1.3 ratio)
```yaml
name: exp_person_75
description: "Person at 75% (~1:1.3 ratio)"
frame_sample_fractions:
  person: 0.75
  crane hook: 1.0
```

**exp_person_100.yaml** - Person at 100% (1:1 balanced)
```yaml
name: exp_person_100
description: "Person at 100% (1:1 balanced)"
frame_sample_fractions:
  person: 1.0
  crane hook: 1.0
```

### Variable Interpolation

Hydra supports variable interpolation using `${...}` syntax:
- `${experiment.name}` - References the current experiment's name
- `${now:%Y-%m-%d_%H-%M-%S}` - Current timestamp
- `${training.model}` - References training model size

---

## Scripts Reference

### train_experiment.py

The main experiment runner script. Uses Hydra for configuration management.

#### Usage

```bash
# Single experiment (local)
python experiments/train_experiment.py experiment=exp_person_25

# All experiments via SLURM
python experiments/train_experiment.py --multirun \
    experiment=exp_person_25,exp_person_50,exp_person_75,exp_person_100

# Preview config without running
python experiments/train_experiment.py experiment=exp_person_25 --cfg job

# Run locally (not via SLURM)
python experiments/train_experiment.py experiment=exp_person_25 hydra/launcher=basic

# Resume from existing training (run inference + summary only)
python experiments/train_experiment.py experiment=exp_person_25 resume_from=/path/to/experiment/dir
```

#### Functions

##### `run_training_subprocess(dataset_dir, output_dir, model_size, epochs, batch_size, image_size, lr, patience) -> tuple[Path, float, dict]`

Runs RF-DETR training in a subprocess to isolate `sys.exit()` calls. RF-DETR's `train()` method calls `sys.exit(0)` on completion, which would terminate the main process. Running in a subprocess prevents this.

**Parameters:**
- `dataset_dir` (Path): Directory containing the prepared COCO dataset
- `output_dir` (Path): Directory to save training outputs
- `model_size` (str): RF-DETR model size ("base" or "large")
- `epochs` (int): Number of training epochs
- `batch_size` (int): Training batch size
- `image_size` (int): Input image resolution
- `lr` (float): Learning rate
- `patience` (int): Early stopping patience

**Returns:** Tuple of (checkpoint_path, training_time_seconds, metrics)

---

##### `find_best_checkpoint(output_dir) -> Path`

Finds the best checkpoint file in the training output directory.

**Parameters:**
- `output_dir` (Path): Training output directory

**Returns:** Path to the best checkpoint file

**Search order:**
1. `checkpoint_best_total.pth`
2. `checkpoint_best_ema.pth`
3. `checkpoint_best_regular.pth`
4. `checkpoint.pth`

---

##### `load_training_metrics(output_dir) -> dict`

Loads training metrics from `results.json` if available.

**Parameters:**
- `output_dir` (Path): Training output directory

**Returns:** Dictionary of metrics (mAP@50, mAP@75, per-class AP, etc.)

---

##### `run_inference_on_videos(checkpoint_path, class_names, test_videos, output_dir, model_size) -> dict[str, dict]`

Runs inference on test videos using the trained model and saves annotated outputs.

**Parameters:**
- `checkpoint_path` (Path): Path to trained model checkpoint
- `class_names` (list[str]): List of class names
- `test_videos` (list[str]): List of video paths to process
- `output_dir` (Path): Directory to save inference outputs
- `model_size` (str): Model size for inference

**Returns:** Dictionary mapping video name to inference statistics:
- `total_frames`: Total number of frames processed
- `keyframes`: Number of keyframes
- `total_detections`: Total detections across all frames
- `avg_inference_time_ms`: Average inference time per frame
- `fps`: Processing speed in frames per second

---

##### `main(cfg: DictConfig) -> float`

Main entry point decorated with `@hydra.main`. Executes the complete experiment pipeline:

1. Prepares dataset with class sampling
2. Trains the RF-DETR model
3. Runs inference on test videos
4. Saves experiment summary

**Parameters:**
- `cfg` (DictConfig): Hydra configuration object

**Returns:** Primary metric (mAP@50) for Hydra optimization

---

### collect_results.py

Aggregates and compares results from multiple experiment runs.

#### Usage

```bash
# Collect from a specific multirun directory
python experiments/collect_results.py experiments/multirun/2026-01-30_12-00-00/

# Collect from the latest multirun
python experiments/collect_results.py --latest

# Output as markdown table
python experiments/collect_results.py --latest --format markdown

# Save to JSON file
python experiments/collect_results.py --latest -o results.json
```

#### Functions

##### `find_latest_multirun(base_dir) -> Path | None`

Finds the most recent multirun directory.

**Parameters:**
- `base_dir` (Path): Base directory to search (typically `experiments/multirun/`)

**Returns:** Path to the latest multirun directory, or None if not found

---

##### `collect_results(multirun_dir) -> list[dict]`

Aggregates results from all experiments in a multirun directory by reading `experiment_summary.json` from each experiment subdirectory.

**Parameters:**
- `multirun_dir` (Path): Path to the multirun directory

**Returns:** List of experiment result dictionaries

---

##### `print_table(results, format) -> None`

Prints a comparison table of experiment results.

**Parameters:**
- `results` (list[dict]): List of experiment result dictionaries
- `format` (str): Output format ("text" or "markdown")

**Output includes:**
- Experiment name
- Person class percentage
- Training images/annotations count
- mAP@50 and mAP@75 metrics
- Training time
- Per-class AP@50 values

---

##### `main()`

CLI entry point. Parses arguments and orchestrates result collection and display.

**Arguments:**
- `multirun_dir`: Path to multirun directory (positional)
- `--latest`: Use the latest multirun directory
- `--format`: Output format (text/markdown)
- `--output`, `-o`: Save results to JSON file

---

### resume_experiments.py

Resumes experiments where training completed but post-training steps (inference, summary) failed. This is particularly useful when RF-DETR's `sys.exit()` behavior prevented inference and summary generation from running.

#### Usage

```bash
# Resume all experiments via SLURM in one batch (recommended)
python experiments/resume_experiments.py --slurm experiments/multirun/2026-01-30_17-14-22/

# Dry run first to see what will be submitted
python experiments/resume_experiments.py --slurm --dry-run experiments/multirun/2026-01-30_17-14-22/

# Resume locally (only for testing, not recommended for heavy workloads)
python experiments/resume_experiments.py --local experiments/multirun/2026-01-30_17-14-22/
```

#### How It Works

The script:
1. Scans the multirun directory for experiments with completed training (checkpoint files exist)
2. Identifies experiments missing `experiment_summary.json` (post-training steps didn't complete)
3. Submits all experiments in a **single `--multirun` command** to SLURM

The key trick is using Hydra interpolation for `resume_from`:
```
resume_from=/path/to/multirun/${experiment.name}
```
This automatically resolves to the correct path for each experiment in the sweep.

#### Functions

##### `find_experiments_to_resume(base_dir) -> list[tuple[Path, str]]`

Identifies experiments that need to be resumed. An experiment needs resuming if:
- It has a `training/checkpoint*.pth` file (training completed)
- It doesn't have `experiment_summary.json` (post-training didn't complete)

**Parameters:**
- `base_dir` (Path): Multirun directory or single experiment directory

**Returns:** List of (experiment_dir, experiment_name) tuples

---

##### `resume_all_experiments_slurm(experiments, multirun_dir, dry_run) -> bool`

Resumes all experiments via SLURM in a single `--multirun` command. This submits all experiments as a SLURM job array, similar to how the original training was submitted.

**Parameters:**
- `experiments` (list[tuple[Path, str]]): List of (experiment_dir, experiment_name) tuples
- `multirun_dir` (Path): Path to the multirun directory
- `dry_run` (bool): If True, only show what would be done

**Returns:** True if jobs submitted successfully, False otherwise

---

##### `resume_experiment_local(exp_dir, exp_name, dry_run) -> bool`

Resumes a single experiment locally (runs sequentially).

**Parameters:**
- `exp_dir` (Path): Experiment directory
- `exp_name` (str): Experiment name
- `dry_run` (bool): If True, only show what would be done

**Returns:** True if successful, False otherwise

---

##### `resume_experiment_slurm(exp_dir, exp_name, dry_run) -> bool`

Resumes a single experiment by submitting to SLURM. Uses `--multirun` flag to activate Hydra's submitit launcher.

**Parameters:**
- `exp_dir` (Path): Experiment directory
- `exp_name` (str): Experiment name
- `dry_run` (bool): If True, only show what would be done

**Returns:** True if job submitted successfully, False otherwise

---

**SLURM Configuration:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| Partition | `gpu-long` | SLURM partition |
| GPU Type | `h100-96` | H100 96GB GPU |
| Timeout | 120 minutes | 2 hours for inference |
| CPUs | 8 | CPUs per task |
| Memory | 64GB | RAM per job |
| Array Parallelism | 4 | Max concurrent jobs |

---

##### `main()`

CLI entry point for resuming experiments.

**Arguments:**
- `directory`: Path to multirun or experiment directory (required)
- `--slurm`: Submit all jobs to SLURM in one batch (recommended)
- `--local`: Run locally one by one (not recommended)
- `--dry-run`: Preview without executing

---

### run_all.sh

Shell script to submit all experiments to SLURM or run locally.

#### Usage

```bash
# Submit all experiments to SLURM
./experiments/run_all.sh

# Preview without submitting
./experiments/run_all.sh --dry-run

# Run locally (no SLURM)
./experiments/run_all.sh --local
```

#### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PARTITION` | `gpu-long` | SLURM partition |
| `GPU_TYPE` | `h100-96` | GPU type (H100 96GB) |
| `CPUS_PER_TASK` | `8` | CPUs per job |
| `MEM_GB` | `64` | Memory per job |
| `TIMEOUT_MIN` | `1440` | Timeout (24 hours) |
| `ARRAY_PARALLELISM` | `4` | Max parallel jobs |
| `EXPERIMENTS` | `exp_person_25,...` | Experiments to run |

#### Supported GPU Types (NUS SoC)

| Type | Description | Nodes |
|------|-------------|-------|
| `h100-96` | H100 96GB | xgpi0-9 (2/node) |
| `h100-47` | H100 47GB MIG | xgpi10-19 (4/node) |
| `a100-80` | A100 80GB | xgph0-9 |
| `a100-40` | A100 40GB | xgpg0-9, xgph10-19 |
| `nv` | Older GPUs | V100, Titan V/RTX, T4 |

---

## Running Experiments

### Quick Start

1. **Single experiment locally:**
   ```bash
   python experiments/train_experiment.py experiment=exp_person_25
   ```

2. **All experiments on SLURM:**
   ```bash
   ./experiments/run_all.sh
   ```

3. **Monitor SLURM jobs:**
   ```bash
   squeue -u $USER
   ```

4. **Collect results after completion:**
   ```bash
   python experiments/collect_results.py --latest --format markdown
   ```

### Override Parameters at Runtime

Hydra allows command-line overrides:

```bash
# Change training epochs
python experiments/train_experiment.py experiment=exp_person_25 training.epochs=100

# Change batch size and learning rate
python experiments/train_experiment.py experiment=exp_person_25 training.batch_size=8 training.lr=5e-5

# Use a different model size
python experiments/train_experiment.py experiment=exp_person_25 training.model=large
```

### Creating New Experiments

1. Create a new YAML file in `conf/experiment/`:
   ```yaml
   # conf/experiment/exp_custom.yaml
   name: exp_custom
   description: "Custom experiment with 10% person data"
   frame_sample_fractions:
     person: 0.10
     crane hook: 1.0
   ```

2. Run the new experiment:
   ```bash
   python experiments/train_experiment.py experiment=exp_custom
   ```

---

## Output Structure

### Single Run (`experiments/outputs/`)

```
experiments/outputs/{experiment_name}/{timestamp}/
├── training/
│   ├── checkpoint_best_total.pth    # Best model checkpoint
│   ├── checkpoint.pth               # Latest checkpoint
│   ├── results.json                 # Training metrics
│   └── class_info.json              # Class mapping
├── inference/
│   ├── detected_{video_name}.mp4    # Annotated video
│   └── {video_stem}_detections.json # Detection results
├── experiment_summary.json          # Complete experiment summary
└── .hydra/
    ├── config.yaml                  # Resolved config
    ├── hydra.yaml                   # Hydra metadata
    └── overrides.yaml               # CLI overrides
```

### Multi-run (`experiments/multirun/`)

```
experiments/multirun/{timestamp}/
├── exp_person_25/
│   ├── training/
│   ├── inference/
│   ├── experiment_summary.json
│   └── .hydra/
├── exp_person_50/
│   └── ...
├── exp_person_75/
│   └── ...
├── exp_person_100/
│   └── ...
└── comparison_summary.json          # Generated by collect_results.py
```

### Experiment Summary JSON

Each experiment generates an `experiment_summary.json`:

```json
{
  "experiment": "exp_person_25",
  "description": "Person at 25% (1:4 ratio)",
  "frame_sample_fractions": {
    "person": 0.25,
    "crane hook": 1.0
  },
  "dataset_stats": {
    "train_images": 150,
    "train_annotations": 420,
    "val_images": 30,
    "val_annotations": 85,
    "test_images": 20,
    "test_annotations": 60,
    "classes": ["crane hook", "person"]
  },
  "training": {
    "time_seconds": 3600.5,
    "checkpoint": "/path/to/checkpoint.pth",
    "config": {
      "epochs": 50,
      "batch_size": 16,
      "image_size": 640,
      "lr": 0.0001,
      "patience": 10,
      "model": "base"
    }
  },
  "metrics": {
    "mAP@50": 0.85,
    "mAP@75": 0.72,
    "AP@50_crane hook": 0.92,
    "AP@50_person": 0.78
  },
  "inference_results": {
    "video_name.mp4": {
      "total_frames": 1000,
      "keyframes": 100,
      "total_detections": 2500,
      "avg_inference_time_ms": 15.2,
      "fps": 65.8
    }
  }
}
```

---

## Troubleshooting

### Common Issues

1. **"No checkpoint found" error**
   - Training may have failed. Check training logs in `.submitit/` directory
   - Verify GPU memory is sufficient for the batch size

2. **Training completes but no `experiment_summary.json`**
   - This happens because RF-DETR's `train()` method calls `sys.exit(0)` internally
   - The training subprocess completes, but the main script exits before running inference
   - **Solution:** Use `resume_experiments.py` to complete post-training steps:
     ```bash
     python experiments/resume_experiments.py --slurm experiments/multirun/<timestamp>/
     ```

3. **SLURM launcher not working (runs locally instead)**
   - Hydra's `submitit_slurm` launcher only activates with `--multirun` flag
   - Even for single experiments, you must use `--multirun`:
     ```bash
     # Wrong (runs locally):
     python experiments/train_experiment.py experiment=exp_person_25 hydra/launcher=submitit_slurm
     
     # Correct (submits to SLURM):
     python experiments/train_experiment.py --multirun experiment=exp_person_25 hydra/launcher=submitit_slurm
     ```

4. **SLURM jobs failing immediately**
   - Check partition availability: `sinfo -p gpu-long`
   - Verify GPU type exists: `scontrol show node xgpi0`

5. **OOM (Out of Memory) on login node**
   - Never run inference on the login node - always submit to SLURM
   - Use `--slurm` flag with `resume_experiments.py`, not `--local`

### Useful Commands

```bash
# Check SLURM queue
squeue -u $USER

# View job details
scontrol show job <job_id>

# Cancel all jobs
scancel -u $USER

# View experiment logs
tail -f experiments/multirun/<timestamp>/<exp>/.submitit/*_log.out

# Check which experiments need resuming
python experiments/resume_experiments.py --slurm --dry-run experiments/multirun/<timestamp>/

# Collect results after all jobs complete
python experiments/collect_results.py --latest --format markdown
```
