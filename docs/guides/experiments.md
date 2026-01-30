# Experiment Framework

The experiment framework uses [Hydra](https://hydra.cc/) with the [submitit-launcher](https://hydra.cc/docs/plugins/submitit_launcher/) plugin to run reproducible ML experiments with easy SLURM integration.

## Overview

The framework is designed to investigate how class imbalance affects model accuracy. It supports:

- **Hydra configuration**: Define experiments via YAML config groups
- **Frame-based sampling**: Subsample training data per class
- **SLURM integration**: Dispatch multiple experiments via `--multirun`
- **Automatic inference**: Run inference on test videos after training
- **Results collection**: Aggregate and compare metrics across experiments

## Directory Structure

```
experiments/
├── conf/
│   ├── config.yaml                 # Main config with defaults
│   ├── experiment/
│   │   ├── exp_person_25.yaml      # 25% person data
│   │   ├── exp_person_50.yaml      # 50% person data
│   │   ├── exp_person_75.yaml      # 75% person data
│   │   └── exp_person_100.yaml     # 100% person data
│   └── hydra/
│       └── launcher/
│           ├── slurm.yaml          # SLURM submitit config
│           └── local.yaml          # Local launcher for testing
├── train_experiment.py             # Main training script
├── collect_results.py              # Results aggregation
├── outputs/                        # Local runs (auto-created)
└── multirun/                       # SLURM runs (auto-created)
```

## Quick Start

### Run All Experiments on SLURM

```bash
cd ~/batman

# Submit all 4 experiments to SLURM
python experiments/train_experiment.py --multirun \
    experiment=exp_person_25,exp_person_50,exp_person_75,exp_person_100
```

This will:

1. Submit 4 SLURM jobs (one per experiment)
2. Each job prepares a dataset with the specified class sampling
3. Trains an RF-DETR model
4. Runs inference on test videos
5. Saves results to `experiments/multirun/<timestamp>/<experiment_name>/`

### Run Single Experiment Locally

```bash
# Test locally before submitting to cluster
python experiments/train_experiment.py experiment=exp_person_25

# Use local launcher explicitly
python experiments/train_experiment.py experiment=exp_person_25 hydra/launcher=local
```

### Preview Config Without Running

```bash
# See resolved config
python experiments/train_experiment.py experiment=exp_person_25 --cfg job

# See what multirun would do
python experiments/train_experiment.py --multirun experiment=glob(*) --info
```

### Collect Results

```bash
# After jobs complete, collect and compare results
python experiments/collect_results.py --latest

# Or specify a specific multirun directory
python experiments/collect_results.py experiments/multirun/2026-01-30_12-00-00/

# Output as markdown table
python experiments/collect_results.py --latest --format markdown
```

## Configuration

### Main Config (`experiments/conf/config.yaml`)

```yaml
defaults:
  - experiment: exp_person_100  # Default experiment
  - hydra/launcher: slurm       # Use SLURM for --multirun
  - _self_

# Project settings
project_dir: data/projects/Test
output_dataset: datasets/experiment_${experiment.name}

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
```

### Experiment Config (`experiments/conf/experiment/exp_person_25.yaml`)

```yaml
name: exp_person_25
description: "Person at 25% (1:4 ratio)"
frame_sample_fractions:
  person: 0.25
  crane hook: 1.0
```

### SLURM Launcher Config (`experiments/conf/hydra/launcher/slurm.yaml`)

```yaml
hydra:
  launcher:
    partition: gpu-long
    gpus_per_node: 1
    cpus_per_task: 8
    mem_gb: 64
    timeout_min: 1440  # 24 hours
    constraint: "h100-96"
    array_parallelism: 4  # Run up to 4 jobs in parallel
```

## Frame Sampling

The `frame_sample_fractions` parameter controls how much data from each class is included:

| Parameter | Description |
|-----------|-------------|
| `1.0` | Include all frames containing this class |
| `0.5` | Include 50% of frames containing this class |
| `0.25` | Include 25% of frames containing this class |

For example, with `person: 0.25` and `crane hook: 1.0`:

- 25% of frames containing "person" annotations are included
- 100% of frames containing "crane hook" annotations are included
- The final dataset is the union of these frame sets

This allows investigating how class imbalance affects model performance.

## Experiment Variants

The default experiment configs investigate Person:Crane hook ratios:

| Experiment | Person % | Person Frames | Crane Hook Frames | Ratio |
|------------|----------|---------------|-------------------|-------|
| exp_person_25 | 25% | ~500 | ~1,991 | 1:4 |
| exp_person_50 | 50% | ~1,000 | ~1,991 | 1:2 |
| exp_person_75 | 75% | ~1,500 | ~1,991 | ~1:1.3 |
| exp_person_100 | 100% | ~2,000 | ~1,991 | 1:1 |

## Output Structure

Each experiment produces:

```
experiments/multirun/<timestamp>/exp_person_25/
├── .hydra/                         # Hydra config files
│   ├── config.yaml                 # Resolved config
│   ├── hydra.yaml
│   └── overrides.yaml
├── train_experiment.log            # Experiment log
├── training/
│   ├── checkpoint_best_total.pth   # Best model checkpoint
│   ├── class_info.json             # Class names
│   └── tensorboard/                # TensorBoard logs
├── inference/
│   ├── detected_video1.mp4         # Annotated test videos
│   ├── video1_detections.json      # Detection results
│   └── ...
└── experiment_summary.json         # Summary metrics
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f experiments/multirun/*/exp_person_*/train_experiment.log

# View TensorBoard
tensorboard --logdir experiments/multirun/<timestamp>/
```

## Creating Custom Experiments

### Add a New Experiment Variant

1. Create `experiments/conf/experiment/my_experiment.yaml`:

```yaml
name: my_experiment
description: "My custom experiment"
frame_sample_fractions:
  person: 0.1
  crane hook: 1.0
```

2. Run it:

```bash
python experiments/train_experiment.py experiment=my_experiment
```

### Override Config from Command Line

```bash
# Change epochs
python experiments/train_experiment.py experiment=exp_person_25 training.epochs=100

# Change batch size
python experiments/train_experiment.py experiment=exp_person_25 training.batch_size=8

# Multiple overrides
python experiments/train_experiment.py experiment=exp_person_25 \
    training.epochs=100 \
    training.batch_size=8 \
    training.lr=5e-5
```

### Parameter Sweeps

```bash
# Sweep over epochs
python experiments/train_experiment.py --multirun \
    experiment=exp_person_100 \
    training.epochs=25,50,100

# Sweep over experiments and epochs
python experiments/train_experiment.py --multirun \
    experiment=exp_person_25,exp_person_100 \
    training.epochs=25,50
```

## Troubleshooting

### SLURM Job Fails

Check the SLURM output logs:

```bash
cat experiments/multirun/<timestamp>/<experiment>/.submitit/<job_id>/<job_id>_0_log.out
cat experiments/multirun/<timestamp>/<experiment>/.submitit/<job_id>/<job_id>_0_log.err
```

### Hydra Config Errors

Preview the resolved config:

```bash
python experiments/train_experiment.py experiment=exp_person_25 --cfg job
```

### Missing Dependencies

Install Hydra and submitit:

```bash
uv sync  # Will install hydra-core and hydra-submitit-launcher
```
