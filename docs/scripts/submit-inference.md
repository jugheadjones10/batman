# Submit Inference Script

Submit RF-DETR inference jobs to SLURM clusters. All inference is **project-centric** -- `--project` is required and results are saved under `{project}/inference/`.

## Basic Usage

```bash
./submit_inference.sh --project data/projects/CraneHook --run rfdetr_run_1
```

## Parameters

### Required

#### `--project=PATH` or `-p=PATH`

Path to Batman project. All runs and videos are resolved from this project.

```bash
--project data/projects/CraneHook
```

### Run Selection (Choose One)

#### `--run=NAME` or `-r=NAME`

Training run name from `{project}/runs/`.

```bash
-r rfdetr_run_1
```

#### `--latest`

Use the most recent training run.

### Video Selection (Optional)

#### `--video=ID` or `-v=ID`

Specific video source_key(s). Without this, all project videos are processed.

```bash
--video video_2
```

#### `--test-only`

Only run on videos with `exclude_from_training: true`.

### GPU Options

#### `--gpu=TYPE`

GPU type to use.

- **Default**: `a100-40`
- **Choices**: `h200`, `h100-96`, `h100-47`, `a100-80`, `a100-40`, `nv`

For inference, `a100-40` is typically sufficient.

#### `--time=LIMIT`

Time limit in format `HH:MM:SS`.

- **Default**: `04:00:00`

### Inference Options

#### `--model=SIZE`

Model architecture size: `base` or `large` (default: `base`).

#### `--confidence=THRESHOLD`

Detection confidence threshold (default: `0.5`).

#### `--no-optimize`

Skip model optimization.

### Video Options

#### `--frame-interval=N` or `-n=N`

Run inference every N frames (default: `1`).

#### `--no-video`

Don't save annotated output video.

### Tracking Options

#### `--track`

Enable ByteTrack object tracking.

#### `--no-kalman`

Disable Kalman filter prediction on non-keyframes.

#### `--track-thresh=THRESHOLD`

ByteTrack detection threshold (default: `0.25`).

#### `--track-buffer=N`

Frames to keep lost tracks (default: `30`).

#### `--match-thresh=THRESHOLD`

IoU threshold for matching (default: `0.8`).

### Other

#### `--dry-run`

Show generated SLURM script without submitting.

## Examples

### Run on All Project Videos

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1
```

### With Tracking

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --track \
  --frame-interval 5
```

### Test-Only Videos

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --test-only
```

### Specific Video

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --video video_2
```

### Latest Run

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --latest \
  --track
```

### Dry Run

Preview SLURM script:

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --dry-run
```

## Output

Results are saved under the project:

```
data/projects/CraneHook/inference/
└── rfdetr_run_1/
    ├── video_1/
    │   ├── result.json
    │   └── detected.mp4
    └── video_2/
        └── result.json
```

### Job Monitoring

```bash
# List your jobs
squeue -u $USER

# Follow logs
tail -f logs/slurm_<JOB_ID>_inference.out
```

## Best Practices

1. **Use cost-effective GPU**: `--gpu a100-40` is sufficient for inference
2. **Enable tracking for videos**: `--track`
3. **Skip frames for speed**: `--frame-interval 5 --track`
4. **Use `--test-only`** to focus on evaluation videos
5. **Preview with `--dry-run`** before submitting

## Related

- **[Inference CLI](../cli/inference.md)** -- Local inference reference
- **[Submit Training](submit-train.md)** -- Train models on cluster
- **[Inference Workflow](../guides/inference.md)** -- Complete guide
