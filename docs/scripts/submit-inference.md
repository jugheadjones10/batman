# Submit Inference Script

Submit RF-DETR inference jobs to SLURM clusters for processing images and videos.

## Basic Usage

```bash
./submit_inference.sh --run my_training_run --input video.mp4
```

## Command Builder

<div class="command-builder-widget" data-tool="submit_inference" data-params='[
  {"name": "run", "type": "text", "description": "Run name", "group": "Model"},
  {"name": "latest", "type": "flag", "description": "Use latest run", "group": "Model"},
  {"name": "checkpoint", "type": "path", "description": "Checkpoint path", "group": "Model"},
  {"name": "project", "type": "path", "description": "Load class names from project", "group": "Model"},
  {"name": "classes", "type": "text", "description": "Manual class names", "group": "Model"},
  {"name": "input", "type": "text", "required": true, "description": "Input files", "group": "Input"},
  {"name": "gpu", "type": "choice", "choices": ["h200", "h100-96", "h100-47", "a100-80", "a100-40", "nv"], "default": "a100-40", "description": "GPU type", "group": "GPU"},
  {"name": "time", "type": "text", "default": "04:00:00", "description": "Time limit", "group": "GPU"},
  {"name": "output", "type": "path", "description": "Output directory (auto-generated)", "group": "Output"},
  {"name": "model", "type": "choice", "choices": ["base", "large"], "default": "base", "description": "Model size", "group": "Inference"},
  {"name": "confidence", "type": "number", "default": 0.5, "min": 0, "max": 1, "step": 0.05, "description": "Confidence threshold", "group": "Inference"},
  {"name": "no-optimize", "type": "flag", "description": "Skip model optimization", "group": "Inference"},
  {"name": "frame-interval", "type": "number", "default": 1, "min": 1, "description": "Process every N frames", "group": "Video"},
  {"name": "track", "type": "flag", "description": "Enable ByteTrack", "group": "Tracking"},
  {"name": "no-kalman", "type": "flag", "description": "Disable Kalman prediction", "group": "Tracking"},
  {"name": "track-thresh", "type": "number", "default": 0.25, "min": 0, "max": 1, "step": 0.05, "description": "Track detection threshold", "group": "Tracking"},
  {"name": "track-buffer", "type": "number", "default": 30, "min": 1, "description": "Lost track buffer", "group": "Tracking"},
  {"name": "match-thresh", "type": "number", "default": 0.8, "min": 0, "max": 1, "step": 0.05, "description": "IoU match threshold", "group": "Tracking"},
  {"name": "dry-run", "type": "flag", "description": "Show script without submitting", "group": "Other"}
]'></div>

## Parameters

### Model Selection (Choose One)

#### `--run=NAME` or `-r=NAME`

Run name to auto-find checkpoint.

```bash
./submit_inference.sh -r=rfdetr_h100_20260120_105925 --input video.mp4
```

#### `--latest`

Use the most recent training run.

```bash
./submit_inference.sh --latest --input video.mp4
```

#### `--checkpoint=PATH` or `-c=PATH`

Explicit checkpoint path.

```bash
./submit_inference.sh -c=runs/my_run/best.pth --input video.mp4
```

### Class Names (Recommended)

#### `--project=PATH` or `-p=PATH`

Load class names from Batman project.

```bash
./submit_inference.sh --run my_run --project data/projects/MyProject --input video.mp4
```

#### `--classes=NAMES`

Manually specify class names (space-separated).

```bash
./submit_inference.sh --run my_run --classes "person car bicycle" --input video.mp4
```

### Input (Required)

#### `--input=FILES` or `-i=FILES`

Input image(s) or video file(s). Supports multiple files and wildcards.

```bash
# Single video
--input=video.mp4

# Multiple videos
--input="video1.mp4 video2.mp4"

# Wildcards
--input="videos/*.mp4"
```

### GPU Options

#### `--gpu=TYPE`

GPU type to use.

- **Default**: `a100-40`
- **Choices**: `h200`, `h100-96`, `h100-47`, `a100-80`, `a100-40`, `nv`

For inference, `a100-40` is typically sufficient and cost-effective.

#### `--time=LIMIT`

Time limit in format `HH:MM:SS`.

- **Default**: `04:00:00`

### Output

#### `--output=PATH` or `-o=PATH`

Output directory for results.

- **Default**: Auto-generated as `inference_results/{TIMESTAMP}/`

### Inference Options

#### `--model=SIZE`

Model architecture size.

- **Choices**: `base`, `large`
- **Default**: `base`

#### `--confidence=THRESHOLD`

Detection confidence threshold.

- **Default**: `0.5`
- **Range**: `0.0` to `1.0`

#### `--no-optimize`

Skip model optimization (use if encountering errors).

### Video Options

#### `--frame-interval=N` or `-n=N`

Run inference every N frames.

- **Default**: `1` (every frame)

### Tracking Options

#### `--track`

Enable ByteTrack object tracking.

#### `--no-kalman`

Disable Kalman filter prediction on non-keyframes.

#### `--track-thresh=THRESHOLD`

ByteTrack detection threshold.

- **Default**: `0.25`

#### `--track-buffer=N`

Frames to keep lost tracks.

- **Default**: `30`

#### `--match-thresh=THRESHOLD`

IoU threshold for matching.

- **Default**: `0.8`

### Other

#### `--dry-run`

Show generated SLURM script without submitting.

## Examples

### Example 1: Basic Inference

Run inference on a video:

```bash
./submit_inference.sh \
  --run my_training_run \
  --input video.mp4
```

### Example 2: With Tracking

Enable tracking for temporal consistency:

```bash
./submit_inference.sh \
  --run my_training_run \
  --input video.mp4 \
  --track \
  --confidence 0.6
```

### Example 3: Multiple Videos

Process multiple videos:

```bash
./submit_inference.sh \
  --run my_training_run \
  --input "video1.mp4 video2.mp4 video3.mp4" \
  --track \
  --gpu a100-80
```

### Example 4: Batch Images

Process all images in a directory:

```bash
./submit_inference.sh \
  --run my_training_run \
  --input "images/*.jpg" \
  --confidence 0.7
```

### Example 5: Fast Processing

Skip frames for faster processing:

```bash
./submit_inference.sh \
  --run my_training_run \
  --input video.mp4 \
  --frame-interval 5 \
  --track
```

### Example 6: Custom Classes

Override class names:

```bash
./submit_inference.sh \
  --checkpoint model.pth \
  --input video.mp4 \
  --classes "crane_hook crane_boom crane_cable"
```

### Example 7: Latest Run

Use most recent training automatically:

```bash
./submit_inference.sh \
  --latest \
  --input video.mp4 \
  --track
```

### Example 8: Dry Run

Preview SLURM script:

```bash
./submit_inference.sh \
  --run my_run \
  --input video.mp4 \
  --dry-run
```

## Output

### Job Submission

```bash
$ ./submit_inference.sh --run my_run --input video.mp4

Submitted batch job 123457
Job ID: 123457
Output directory: inference_results/20260128_110804/
Log file: logs/job_123457.log

To monitor:
  tail -f logs/job_123457.log
  squeue -j 123457
```

### Generated Files

```
inference_results/20260128_110804/
├── detected_video.mp4     # Annotated video
├── detections.json        # Detection results
└── metadata.json          # Run metadata

logs/
└── job_123457.log        # SLURM job log
```

### Timing Summary

Log includes timing breakdown:

```
=== Timing Summary ===
Setup time: 3.45s
Inference time: 45.23s
Total time: 48.68s
```

## Monitoring

### View Job Status

```bash
# List your jobs
squeue -u $USER

# Check specific job
squeue -j 123457
```

### Follow Logs

```bash
# Real-time log monitoring
tail -f logs/job_123457.log

# View full log
cat logs/job_123457.log
```

## Best Practices

### 1. Use Cost-Effective GPU

For inference, `a100-40` is usually sufficient:

```bash
./submit_inference.sh --gpu a100-40 ...
```

### 2. Enable Tracking for Videos

Tracking improves consistency:

```bash
./submit_inference.sh --track ...
```

### 3. Skip Frames for Speed

Process every 5th frame with prediction:

```bash
./submit_inference.sh --frame-interval 5 --track ...
```

### 4. Specify Classes

Always provide class names:

```bash
./submit_inference.sh --project data/projects/MyProject ...
# or
./submit_inference.sh --classes "person car" ...
```

### 5. Batch Multiple Videos

Submit one job for multiple videos:

```bash
./submit_inference.sh --input "videos/*.mp4" ...
```

## Troubleshooting

### No Detections

- Lower confidence: `--confidence 0.3`
- Check class names match training
- Verify checkpoint is correct

### Slow Processing

- Skip frames: `--frame-interval 5`
- Use smaller GPU: `--gpu a100-40`
- Disable optimization: `--no-optimize` (try as last resort)

### Out of Memory

- Reduce batch processing (not configurable in inference)
- Use smaller GPU: `--gpu a100-40`

### Job Timeout

Increase time limit:

```bash
./submit_inference.sh --time 08:00:00 ...
```

## Related

- **[Inference CLI](../cli/inference.md)** - Local inference
- **[Submit Training](submit-train.md)** - Train models
- **[Submit Benchmark](submit-benchmark.md)** - Benchmark performance
- **[Inference Workflow](../guides/inference.md)** - Complete guide
