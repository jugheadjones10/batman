# Inference Workflow Guide

Complete guide for running inference with trained RF-DETR models using the project-centric architecture.

## Overview

All inference is tied to a project. Results are persisted under `{project}/inference/{run_name}/{video_id}/`, enabling you to compare different training runs against different videos and browse results later.

### Workflow

1. **Train a model** -- Complete a training run (CLI, Web UI, or SLURM)
2. **Select a project** -- Inference requires `--project`
3. **Choose a run** -- Pick a training run from the project
4. **Select videos** -- All project videos, specific ones, or test-only
5. **Run inference** -- Results are persisted automatically
6. **Browse & compare** -- View results in the Web UI or on disk

## Project Structure

After inference, your project looks like this:

```
data/projects/CraneHook/
├── videos/
│   ├── videos.json         # Video metadata (with exclude_from_training flag)
│   ├── video_1_clip.mp4
│   └── video_2_test.mp4
├── runs/                   # Training artifacts
│   ├── rfdetr_run_1/
│   │   ├── checkpoint_best_total.pth
│   │   ├── class_info.json
│   │   └── meta.json
│   └── rfdetr_run_2/
│       └── ...
├── inference/              # Inference results (auto-created)
│   ├── rfdetr_run_1/
│   │   ├── video_1/
│   │   │   ├── result.json
│   │   │   └── detected.mp4
│   │   └── video_2/
│   │       └── result.json
│   └── rfdetr_run_2/
│       └── video_1/
│           └── result.json
└── ...
```

## CLI Inference

### Run on All Project Videos

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1
```

### Run on Test-Only Videos

Videos marked with `exclude_from_training: true` can be targeted specifically:

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --test-only
```

### Run with Tracking

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --track \
  --frame-interval 5
```

### Use the Latest Training Run

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --latest
```

## SLURM Cluster Inference

### Basic Submission

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --gpu a100-40
```

### Test-Only Videos on Cluster

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --test-only \
  --track \
  --gpu a100-40
```

## Web UI Inference

The Web UI provides a visual interface for running and browsing inference results.

### Running Inference

1. Navigate to the **Inference** page for your project
2. Click **Run Inference**
3. Select a training run (model)
4. Select a video
5. Configure settings (confidence, tracking, etc.)
6. Click **Run & Save** -- results are persisted automatically

### Browsing Results

The Inference page shows a **runs × videos matrix**:

- Each cell shows whether inference has been run and key stats
- Click a cell to view detailed results (stats, config, detection timeline)
- Delete results you no longer need

## Managing Test Videos

You can designate videos as "test-only" so they're excluded from training datasets but available for inference.

### CLI

```bash
# Add a test video
python -m cli.videos add --project data/projects/CraneHook --test-only /path/to/test_video.mp4

# Toggle existing video to test-only
python -m cli.videos set-test --project data/projects/CraneHook video_2 --on

# List videos (shows [TEST-ONLY] flag)
python -m cli.videos list --project data/projects/CraneHook
```

### Web UI

On the Project page, each video card has a **Training / Test Only** toggle button. Click it to change the video's role.

## Comparing Runs

The project-centric structure makes it easy to compare different training runs:

1. Run inference with run A on your test videos
2. Run inference with run B on the same test videos
3. Open the Inference page to see the runs × videos matrix
4. Click cells to compare stats (detection counts, inference time, etc.)

## Checkpoint Resolution

When you specify `--run`, the CLI looks for checkpoints in this order:

1. `checkpoint_best_total.pth`
2. `checkpoint_best_ema.pth`
3. `checkpoint_best_regular.pth`
4. `best.pth`
5. `checkpoint.pth`
6. Fallback: newest `.pth` file

Class names are loaded from `class_info.json` in the run directory (authoritative), with fallback to the project's class list.

## Performance Tips

| Goal | Setting |
|------|---------|
| Maximum accuracy | `--frame-interval 1` (default) |
| 5x faster | `--frame-interval 5 --track` |
| 10x faster | `--frame-interval 10 --track` |
| High precision | `--confidence 0.7-0.8` |
| High recall | `--confidence 0.3-0.4` |

## Troubleshooting

### No Detections

1. Lower confidence: `--confidence 0.3`
2. Check class names in the run's `class_info.json`
3. Verify the correct training run

### Tracking Issues

- IDs switching: `--match-thresh 0.9`
- Lost tracks: `--track-buffer 60`
- Duplicate IDs: `--track-thresh 0.2`

### Slow Inference

- Skip frames: `--frame-interval 5`
- Don't write video: `--no-video`
- Use A100-40 on cluster (cost-effective)

## Related

- **[Inference CLI](../cli/inference.md)** -- Command reference
- **[Video Management CLI](../cli/videos.md)** -- Add/remove project videos
- **[Submit Inference Script](../scripts/submit-inference.md)** -- SLURM inference
- **[Training Workflow](training.md)** -- Train models
