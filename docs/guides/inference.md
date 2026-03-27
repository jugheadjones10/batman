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

### From Your Local Mac (Recommended)

Use `run_inference.sh` to submit, wait, and auto-sync results in one command:

```bash
./run_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --gpu a100-40
```

This streams the job output in real-time and copies results to your local project directory when done. Requires the SSHFS mount (`./mount_gpu.sh`).

### From the Cluster Directly

If you're already SSH'd into the cluster, use `submit_inference.sh`:

```bash
./submit_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --gpu a100-40
```

### Test-Only Videos on Cluster

```bash
./run_inference.sh \
  --project data/projects/CraneHook \
  --run rfdetr_run_1 \
  --test-only \
  --track \
  --gpu a100-40
```

## Web UI Inference

The Web UI provides a visual interface for running and browsing inference results.

### Running Inference on GPU Cluster

1. Navigate to the **Inference** page for your project
2. Connect to the GPU cluster (enter SSH password in the connection panel)
3. Click **Run Inference** -- the default tab is **GPU Cluster**
4. Select a training run (or use "Latest")
5. Choose GPU type and configure settings (confidence, frame interval, tracking, etc.)
6. Click **Submit to GPU** -- the job is submitted to SLURM
7. View real-time logs in the **GPU Inference Logs** section
8. Results are synced back to your local project when the job completes

### Running Inference Locally

1. Navigate to the **Inference** page for your project
2. Click **Run Inference**
3. Switch to the **Local** tab
4. Select a training run (model)
5. Select a video
6. Configure settings (confidence, tracking, etc.)
7. Click **Run & Save** -- results are persisted automatically

### Browsing Results

The Inference page shows a **runs × videos matrix**:

- Each cell shows whether inference has been run and key stats
- Click a cell to view detailed results (stats, config, detection timeline)
- Delete results you no longer need

### Extracting Frames for Z-Axis Calibration

After running inference, you can extract specific frames as JPEG images along with their bounding box data. This is useful for z-axis height estimation calibration (see [Z-Axis Height Estimation](z-axis-height-estimation.md)).

1. Click a cell in the results matrix to open the detail panel
2. Click **Open Frame Viewer** in the Extract Frames section
3. The frame viewer opens in a full-screen layout similar to the annotation tool:
    - **Center**: the current frame with detection bounding box overlays (colored by class)
    - **Bottom filmstrip**: scrollable thumbnails of all inference frames with navigation controls
    - **Right sidebar**: detection details for the current frame (classes, confidence, bounding boxes)
4. Navigate frames with arrow keys or the filmstrip
5. Select frames for export:
    - **Space** toggles the current frame
    - **Cmd/Ctrl-click** on filmstrip thumbnails for multi-select
    - **Shift-click** for range selection
    - **Select all** via the floating bar
6. Click **Download ZIP** to download the selected frames

The ZIP file contains:

- **JPEG images** -- one per selected frame (`frame_000042.jpg`, etc.)
- **`detections.json`** -- bounding box data for all selected frames, including:
    - Video resolution (`width`, `height`) for converting normalized boxes to pixels
    - Per-frame detections with class name, confidence, and normalized bounding box (`x`, `y`, `width`, `height` as center + size in 0-1 range)

Example `detections.json` structure:

```json
{
  "project": "CraneHook",
  "run_name": "rfdetr_run_1",
  "video_id": "1",
  "inference_id": "20250327_143022",
  "video_resolution": { "width": 1920, "height": 1080 },
  "frames": [
    {
      "frame_number": 42,
      "timestamp": 1.4,
      "image_filename": "frame_000042.jpg",
      "detections": [
        {
          "class_name": "crane_hook",
          "class_id": 0,
          "confidence": 0.94,
          "box": { "x": 0.49, "y": 0.27, "width": 0.06, "height": 0.15 }
        }
      ]
    }
  ]
}
```

To convert normalized boxes to pixel coordinates:

```python
bbox_center_x_px = detection["box"]["x"] * video_resolution["width"]
bbox_center_y_px = detection["box"]["y"] * video_resolution["height"]
bbox_width_px = detection["box"]["width"] * video_resolution["width"]
bbox_height_px = detection["box"]["height"] * video_resolution["height"]
```

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
- **[Run Inference (Local)](../scripts/run-inference.md)** -- Local runner with auto-sync
- **[Submit Inference Script](../scripts/submit-inference.md)** -- Cluster-side SLURM inference
- **[Training Workflow](training.md)** -- Train models
