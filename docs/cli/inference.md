# Inference CLI

Run RF-DETR inference on project videos with object tracking and visualization. All inference is **project-centric** -- results are persisted under `{project}/inference/{run_name}/{video_id}/`.

## Overview

The inference CLI provides:

- Project-centric inference: `--project` is required
- Runs resolve from the project's `runs/` directory
- Results automatically saved to `{project}/inference/`
- ByteTrack object tracking with Kalman filtering
- Configurable confidence thresholds and frame intervals
- Test-only video filtering via `--test-only`

## Basic Usage

```bash
# Run a training run's model on all project videos
python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1

# Run on a specific video
python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1 \
    --video video_2

# Run on test-only videos
python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1 --test-only

# Use the latest training run
python -m cli.inference --project data/projects/CraneHook --latest
```

## Parameters

### Required

#### `--project PATH` or `-p PATH`

Path to the Batman project. All runs and videos are resolved from this project.

```bash
-p data/projects/CraneHook
```

### Run Selection (Choose One)

#### `--run NAME` or `-r NAME`

Training run name. Resolves checkpoint from `{project}/runs/{name}/`.

```bash
-r rfdetr_run_1
```

#### `--latest`

Use the most recent training run in the project.

### Video Selection (Optional)

#### `--video ID` or `-v ID`

Specific video source_key(s) to process. Without this flag, all project videos are processed.

```bash
--video video_2
--video video_1 video_3
```

#### `--test-only`

Only run on videos marked with `exclude_from_training: true`.

### Model Configuration

#### `--model SIZE`

Model architecture size.

- **Choices**: `base`, `large`
- **Default**: `base`

#### `--device TYPE`

Device for inference.

- **Choices**: `auto`, `cuda`, `mps`, `cpu`
- **Default**: `auto`

### Detection

#### `--confidence THRESHOLD` or `-t THRESHOLD`

Minimum confidence to include (default: `0.0` = show all; each box labeled with its confidence).

- **Default**: `0.0`
- **Range**: `0.0` to `1.0`

### Optimization

#### `--no-optimize`

Disable model optimization (use if encountering errors).

#### `--optimize-compile`

Enable PyTorch JIT compilation for additional speedup.

### Video Options

#### `--frame-interval N` or `-n N`

Run inference every N frames (for faster processing).

- **Default**: `1` (every frame)

#### `--no-video`

Don't save annotated output video.

### Tracking Options

#### `--track`

Enable ByteTrack object tracking.

#### `--no-kalman`

Disable Kalman filter prediction on non-keyframes. Only applies when using `--frame-interval > 1` with `--track`.

#### `--track-thresh THRESHOLD`

Detection confidence threshold for tracking.

- **Default**: `0.25`

#### `--track-buffer N`

Number of frames to keep lost tracks before deletion.

- **Default**: `30`

#### `--match-thresh THRESHOLD`

IoU threshold for matching detections to tracks.

- **Default**: `0.8`

## Output Structure

Results are saved under the project's `inference/` directory:

```
data/projects/CraneHook/
└── inference/
    └── rfdetr_run_1/
        ├── video_1/
        │   ├── result.json       # Config + stats + per-frame detections
        │   └── detected.mp4      # Annotated video (if --no-video not set)
        └── video_2/
            └── result.json
```

### result.json Format

```json
{
  "run_name": "rfdetr_run_1",
  "video_id": "video_1",
  "created_at": "2026-02-20T...",
  "config": {
    "confidence_threshold": 0.0,
    "frame_interval": 1,
    "tracking": true,
    "tracking_mode": "bytetrack"
  },
  "stats": {
    "total_frames": 300,
    "keyframes": 60,
    "total_detections": 1500,
    "avg_inference_time_ms": 45.2
  },
  "frames": [
    {
      "frame_idx": 0,
      "timestamp": 0.0,
      "is_keyframe": true,
      "inference_time_ms": 42.1,
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95,
          "class_id": 0,
          "class_name": "crane_hook",
          "track_id": 1
        }
      ]
    }
  ]
}
```

## Examples

### Run on All Videos

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1
```

### With Tracking and Frame Skipping

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --track \
  --frame-interval 5
```

### Test-Only Videos

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --test-only
```

### High Confidence, Persistent Tracking

```bash
python -m cli.inference \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --track \
  --confidence 0.8 \
  --track-thresh 0.3 \
  --track-buffer 60
```

## Performance Tips

1. **Enable Tracking for Videos**: `--track --track-buffer 30`
2. **Skip Frames for Speed**: `--frame-interval 5 --track`
3. **Filter by Confidence**: Default 0.0 shows all; use e.g. `--confidence 0.5` to filter low-confidence detections
4. **Use Model Optimization**: Enabled by default (disable with `--no-optimize` only on errors)

## Troubleshooting

### No Detections

- Lower threshold: `--confidence 0.0` (default) or `--confidence 0.3`
- Verify class names match training (check `class_info.json` in run directory)
- Confirm correct training run

### Tracking Issues

- IDs switching: Increase `--match-thresh 0.9`
- Lost tracks: Increase `--track-buffer 60`

### Slow Inference

- Skip frames: `--frame-interval 5`
- Use `--no-video` to skip writing annotated output

## Related

- **[Training CLI](train.md)** -- Train models
- **[Submit Inference Script](../scripts/submit-inference.md)** -- SLURM inference
- **[Inference Workflow Guide](../guides/inference.md)** -- Complete workflow
- **[Video Management CLI](videos.md)** - Add/remove project videos
