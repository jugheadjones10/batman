# Inference CLI

Run RF-DETR inference on images or videos with object tracking and visualization.

## Overview

The inference CLI provides:

- Detection on single images or videos
- ByteTrack object tracking with Kalman filtering
- Configurable confidence thresholds
- Model optimization for faster inference
- JSON output for programmatic analysis
- Visual annotations with bounding boxes and IDs

## Basic Usage

```bash
# Inference on a video
python -m cli.inference --run my_training_run --input video.mp4

# Inference on images
python -m cli.inference --checkpoint model.pth --input img1.jpg img2.jpg
```

## Command Builder

<div class="command-builder-widget" data-tool="inference" data-params='[
  {"name": "checkpoint", "type": "path", "required": false, "description": "Path to checkpoint file", "group": "Model"},
  {"name": "run", "type": "text", "required": false, "description": "Run name (auto-finds checkpoint)", "group": "Model"},
  {"name": "latest", "type": "flag", "description": "Use the most recent run", "group": "Model"},
  {"name": "model", "type": "choice", "choices": ["base", "large"], "default": "base", "description": "Model architecture", "group": "Model"},
  {"name": "device", "type": "choice", "choices": ["auto", "cuda", "mps", "cpu"], "default": "auto", "description": "Device for inference", "group": "Model"},
  {"name": "input", "type": "text", "required": true, "description": "Input image(s) or video file(s)", "group": "Input"},
  {"name": "output", "type": "path", "default": "inference_results", "description": "Output directory", "group": "Output"},
  {"name": "no-visualizations", "type": "flag", "description": "Do not save annotated images/videos", "group": "Output"},
  {"name": "no-json", "type": "flag", "description": "Do not save JSON detection results", "group": "Output"},
  {"name": "confidence", "type": "number", "default": 0.5, "min": 0, "max": 1, "step": 0.05, "description": "Confidence threshold", "group": "Detection"},
  {"name": "no-optimize", "type": "flag", "description": "Do not optimize model", "group": "Optimization"},
  {"name": "optimize-compile", "type": "flag", "description": "Use JIT compilation", "group": "Optimization"},
  {"name": "frame-interval", "type": "number", "default": 1, "min": 1, "description": "Run inference every N frames", "group": "Video"},
  {"name": "track", "type": "flag", "description": "Enable ByteTrack tracking", "group": "Tracking"},
  {"name": "no-kalman", "type": "flag", "description": "Disable Kalman prediction", "group": "Tracking"},
  {"name": "track-thresh", "type": "number", "default": 0.25, "min": 0, "max": 1, "step": 0.05, "description": "ByteTrack detection threshold", "group": "Tracking"},
  {"name": "track-buffer", "type": "number", "default": 30, "min": 1, "description": "Frames to keep lost tracks", "group": "Tracking"},
  {"name": "match-thresh", "type": "number", "default": 0.8, "min": 0, "max": 1, "step": 0.05, "description": "ByteTrack IoU threshold", "group": "Tracking"},
  {"name": "project", "type": "path", "description": "Load class names from project", "group": "Classes"},
  {"name": "classes", "type": "text", "description": "Override class names", "group": "Classes"}
]'></div>

## Parameters

### Model Selection (Choose One)

#### `--checkpoint PATH` or `-c PATH`

Path to model checkpoint file.

```bash
-c runs/my_run/best.pth
```

#### `--run NAME` or `-r NAME`

Run name to auto-find checkpoint.

```bash
-r rfdetr_h100_20260120_105925
```

The tool searches for checkpoints in `runs/<name>/` with names:

- `best.pth`
- `checkpoint_best.pth`
- `model.pth`

#### `--latest`

Use the most recent training run.

```bash
--latest
```

### Model Configuration

#### `--model SIZE`

Model architecture size.

- **Choices**: `base`, `large`
- **Default**: `base`

#### `--device TYPE`

Device for inference.

- **Choices**: `auto`, `cuda`, `mps`, `cpu`
- **Default**: `auto`

### Input

#### `--input FILES` or `-i FILES` (Required)

Input image(s) or video file(s). Supports multiple files and wildcards.

```bash
# Single file
--input video.mp4

# Multiple files
--input video1.mp4 video2.mp4

# Wildcards
--input "videos/*.mp4"
--input "images/*.jpg"
```

### Output

#### `--output PATH` or `-o PATH`

Output directory for results.

- **Default**: `inference_results`

#### `--no-visualizations`

Skip saving annotated images/videos (JSON only).

#### `--no-json`

Skip saving JSON detection results (visualizations only).

### Detection

#### `--confidence THRESHOLD` or `-t THRESHOLD`

Confidence threshold for detections.

- **Default**: `0.5`
- **Range**: `0.0` to `1.0`
- **Lower values**: More detections, more false positives
- **Higher values**: Fewer detections, fewer false positives

### Optimization

#### `--no-optimize`

Disable model optimization (use if encountering errors).

#### `--optimize-compile`

Enable PyTorch JIT compilation for additional speedup.

### Video Options

#### `--frame-interval N` or `-n N`

Run inference every N frames (for faster processing).

- **Default**: `1` (every frame)
- **Example**: `-n 5` (every 5th frame)

### Tracking Options

#### `--track`

Enable ByteTrack object tracking.

!!! note
Tracking provides temporal consistency and assigns persistent IDs to objects.

#### `--no-kalman`

Disable Kalman filter prediction on non-keyframes.

Only applies when using `--frame-interval > 1` with `--track`.

#### `--track-thresh THRESHOLD`

Detection confidence threshold for tracking.

- **Default**: `0.25`
- **Lower than `--confidence`** to associate low-confidence detections

#### `--track-buffer N`

Number of frames to keep lost tracks before deletion.

- **Default**: `30`
- **Higher values**: More persistent tracks across occlusions

#### `--match-thresh THRESHOLD`

IoU threshold for matching detections to tracks.

- **Default**: `0.8`
- **Range**: `0.0` to `1.0`

### Class Names

#### `--project PATH` or `-p PATH`

Load class names from Batman project.

```bash
-p data/projects/MyProject
```

#### `--classes NAMES`

Manually specify class names (comma-separated).

```bash
--classes "person,car,bicycle"
```

## Examples

### Example 1: Basic Video Inference

Run inference on a video with default settings:

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4
```

### Example 2: Video with Tracking

Enable tracking for temporal consistency:

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --track \
  --confidence 0.6
```

### Example 3: Fast Processing (Skip Frames)

Process every 5th frame with Kalman prediction:

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --frame-interval 5 \
  --track
```

### Example 4: Multiple Videos

Process multiple videos:

```bash
python -m cli.inference \
  --run my_training_run \
  --input video1.mp4 video2.mp4 video3.mp4 \
  --track
```

### Example 5: Image Batch

Process all images in a directory:

```bash
python -m cli.inference \
  --checkpoint runs/my_run/best.pth \
  --input "images/*.jpg" \
  --confidence 0.7
```

### Example 6: High Confidence, Persistent Tracking

Strict detection with long track memory:

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --track \
  --confidence 0.8 \
  --track-thresh 0.3 \
  --track-buffer 60 \
  --match-thresh 0.7
```

### Example 7: Custom Classes

Override class names:

```bash
python -m cli.inference \
  --checkpoint model.pth \
  --input video.mp4 \
  --classes "crane_hook,crane_boom,crane_cable"
```

### Example 8: JSON Output Only

Skip visualizations, save only JSON:

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --no-visualizations
```

### Example 9: Use Latest Run

Automatically use the most recent training run:

```bash
python -m cli.inference \
  --latest \
  --input video.mp4 \
  --track
```

## Output Structure

Inference creates this structure:

```
inference_results/
  └── 20260128_143022/        # Timestamp
      ├── detected_video.mp4  # Annotated video
      ├── detections.json     # JSON detections
      └── frames/             # Individual frames (if images)
          ├── img1_detected.jpg
          └── img2_detected.jpg
```

### JSON Format

Detections are saved in this format:

```json
{
  "video": "video.mp4",
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "detections": [
        {
          "bbox": [x1, y1, x2, y2],
          "confidence": 0.95,
          "class_id": 0,
          "class_name": "person",
          "track_id": 1
        }
      ]
    }
  ]
}
```

## Tracking Explained

### ByteTrack Algorithm

ByteTrack uses a two-stage association:

1. **High-confidence detections** (≥ `--confidence`) → Match to existing tracks
2. **Low-confidence detections** (≥ `--track-thresh`) → Match to remaining tracks

This recovers objects during partial occlusions.

### Kalman Filter Prediction

When using `--frame-interval > 1`:

- **Keyframes** (every N frames): Run detection
- **Non-keyframes**: Predict box positions using Kalman filter
- Use `--no-kalman` to disable prediction (shows keyframes only)

### Track Lifecycle

1. **New detection** → Create tentative track
2. **Matched in consecutive frames** → Confirm track (assign ID)
3. **Not matched** → Mark as lost, keep for `--track-buffer` frames
4. **Matched again** → Recover track (reuse ID)
5. **Lost too long** → Delete track

## Performance Tips

### 1. Enable Tracking for Videos

Tracking improves consistency and helps with partial occlusions:

```bash
--track --track-buffer 30
```

### 2. Skip Frames for Faster Processing

Process every Nth frame with prediction:

```bash
--frame-interval 5 --track
```

### 3. Optimize Confidence Thresholds

- **Start with 0.5**: Good balance
- **High precision needed**: 0.7-0.8
- **High recall needed**: 0.3-0.4

### 4. Tune Tracking Parameters

For objects that disappear frequently:

```bash
--track-buffer 60 --match-thresh 0.7
```

### 5. Use Model Optimization

Model optimization is enabled by default. Disable only if errors occur:

```bash
--no-optimize
```

## Troubleshooting

### No Detections

- **Lower confidence**: `--confidence 0.3`
- **Check class names**: Ensure `--classes` or `--project` matches training
- **Verify model**: Check checkpoint is from correct training run

### False Positives

- **Raise confidence**: `--confidence 0.7`
- **Check training quality**: Review training metrics

### Tracking Issues

- **IDs switching**: Increase `--match-thresh 0.9`
- **Lost tracks**: Increase `--track-buffer 60`
- **Duplicate IDs**: Lower `--track-thresh 0.2`

### Slow Inference

- **Skip frames**: `--frame-interval 5`
- **Reduce resolution**: Resize video before inference
- **Use smaller model**: Train with `--model base` instead of `large`

## Related

- **[Training CLI](train.md)** - Train models
- **[Benchmark CLI](benchmark-latency.md)** - Measure performance
- **[Submit Inference Script](../scripts/submit-inference.md)** - SLURM inference
- **[Inference Workflow Guide](../guides/inference.md)** - Complete workflow
