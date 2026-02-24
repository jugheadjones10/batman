# Video Management CLI

Add, list, remove, and manage videos in Batman projects.

## Overview

The video management CLI allows you to:
- **Add videos** to a project (with automatic metadata extraction)
- **List videos** in a project with metadata
- **Remove videos** and their associated frames
- **Toggle test-only flag** to exclude videos from training

## Basic Usage

```bash
# Add a video to a project
python -m cli.videos add --project data/projects/CraneHook /path/to/video.mp4

# Add as test-only (excluded from training)
python -m cli.videos add --project data/projects/CraneHook --test-only /path/to/video.mp4

# List videos
python -m cli.videos list --project data/projects/CraneHook

# Remove a video
python -m cli.videos remove --project data/projects/CraneHook video_2

# Toggle test-only flag
python -m cli.videos set-test --project data/projects/CraneHook video_2 --on
```

## Global Parameters

### `--project PATH` or `-p PATH` (Required)

Path to Batman project directory.

```bash
-p data/projects/CraneHook
```

## Subcommands

### `add` - Add Video(s)

Add one or more video files to the project. Videos are copied into the project's `videos/` directory and metadata (resolution, FPS, duration, frame count) is automatically extracted using OpenCV.

#### Parameters

##### `files` (Positional, Required)
One or more video file paths to add.

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`

##### `--test-only`
Mark added videos as test-only (`exclude_from_training=true`). These videos won't be included in training data but can be used for evaluation with `--test-only` in inference.

#### Example

```bash
python -m cli.videos add \
  --project data/projects/CraneHook \
  /path/to/video1.mp4 /path/to/video2.mp4

python -m cli.videos add \
  --project data/projects/CraneHook \
  --test-only /path/to/eval_video.mp4
```

#### Output

```
Copying video1.mp4 -> video_1_video1.mp4
Probing video metadata...
Added: video_1 -> video1.mp4
  1920x1080 @ 30.0fps, 120.5s, 3615 frames
```

### `list` - List Videos

List all videos in a project with their metadata.

#### Example

```bash
python -m cli.videos list --project data/projects/CraneHook
```

#### Output

```
Videos in CraneHook (3 total):

  video_1          crane_hook_1.mp4               1920x1080  30fps  120.5s  frames=3615
  video_2          crane_hook_2.mp4               1920x1080  30fps  60.2s   frames=1806
  video_3          eval_video.mp4                 1920x1080  30fps  45.0s   frames=1350 [TEST-ONLY]
```

### `remove` - Remove Video(s)

Remove one or more videos from the project. Deletes the video file, proxy (if any), and extracted frames.

#### Parameters

##### `video_ids` (Positional, Required)
One or more video source_key(s) to remove.

#### Example

```bash
python -m cli.videos remove --project data/projects/CraneHook video_2
```

#### Output

```
Deleted file: data/projects/CraneHook/videos/video_2_crane_hook_2.mp4
Deleted frames: data/projects/CraneHook/frames/video_2
Removed: video_2
```

### `set-test` - Toggle Test-Only Flag

Set or unset the `exclude_from_training` flag on one or more videos.

#### Parameters

##### `video_ids` (Positional, Required)
One or more video source_key(s).

##### `--on` (Default)
Set `exclude_from_training=true` (mark as test-only).

##### `--off`
Set `exclude_from_training=false` (include in training).

#### Example

```bash
# Mark as test-only
python -m cli.videos set-test --project data/projects/CraneHook video_2 --on

# Include in training again
python -m cli.videos set-test --project data/projects/CraneHook video_2 --off
```

## Use Cases

### 1. Add Evaluation Videos

Add videos specifically for evaluation (not used in training):

```bash
python -m cli.videos add \
  --project data/projects/CraneHook \
  --test-only eval_video.mp4

# Later, run inference only on test videos
python -m cli.inference \
  --project data/projects/CraneHook \
  --run my_run \
  --test-only
```

### 2. Manage Training Data

Check which videos are available and their status:

```bash
python -m cli.videos list --project data/projects/CraneHook
```

### 3. Clean Up

Remove videos no longer needed:

```bash
python -m cli.videos remove --project data/projects/CraneHook video_3
```

## Related

- **[Inference CLI](inference.md)** - Run inference on project videos
- **[Training CLI](train.md)** - Train models on project data
- **[Importer CLI](importer.md)** - Import datasets
