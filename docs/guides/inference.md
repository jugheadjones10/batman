# Inference Workflow Guide

Complete guide for running inference with trained RF-DETR models.

## Overview

Inference workflow:
1. **Select Model** - Choose trained checkpoint
2. **Prepare Input** - Organize images or videos
3. **Configure Inference** - Set thresholds and options
4. **Run Inference** - Execute detection
5. **Review Results** - Analyze outputs
6. **Optimize Performance** - Tune for speed or accuracy

## Step 1: Select Model

### Find Available Models

```bash
# List training runs
ls runs/

# Check run details
cat runs/my_training_run/results.json
cat runs/my_training_run/training_config.json
```

### Model Selection Criteria

- **Best mAP**: Use `best.pth` (automatically selected)
- **Latest**: Use `checkpoint_last.pth`
- **Specific epoch**: Use `checkpoint_epoch_N.pth`

## Step 2: Prepare Input

### Organize Files

```bash
# Single file
--input video.mp4

# Multiple files
--input video1.mp4 video2.mp4 video3.mp4

# Wildcards
--input "videos/*.mp4"
--input "images/*.jpg"
```

### Supported Formats

**Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`  
**Videos**: `.mp4`, `.avi`, `.mov`, `.mkv`

## Step 3: Configure Inference

### Confidence Threshold

Controls detection sensitivity:

| Threshold | Use Case | Trade-off |
|-----------|----------|-----------|
| 0.3-0.4 | High recall needed | More false positives |
| 0.5 (default) | Balanced | Good starting point |
| 0.7-0.8 | High precision needed | May miss detections |

### Tracking Options

Enable for videos with moving objects:

```bash
--track                    # Enable tracking
--track-buffer 30         # Keep tracks 30 frames after loss
--track-thresh 0.25       # Lower threshold for association
--match-thresh 0.8        # IoU threshold for matching
```

### Frame Interval

Process every Nth frame for speed:

```bash
--frame-interval 1        # Every frame (default)
--frame-interval 5        # Every 5th frame
--frame-interval 10       # Every 10th frame
```

## Step 4: Run Inference

### Local Inference

#### Basic Video Inference

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --confidence 0.5
```

#### With Tracking

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --track \
  --confidence 0.5 \
  --track-buffer 30
```

#### Fast Processing

```bash
python -m cli.inference \
  --run my_training_run \
  --input video.mp4 \
  --frame-interval 5 \
  --track
```

#### Batch Images

```bash
python -m cli.inference \
  --run my_training_run \
  --input "images/*.jpg" \
  --confidence 0.6
```

### SLURM Cluster Inference

#### Basic Submission

```bash
./submit_inference.sh \
  --run my_training_run \
  --input video.mp4 \
  --gpu a100-40
```

#### Multiple Videos

```bash
./submit_inference.sh \
  --run my_training_run \
  --input "videos/*.mp4" \
  --gpu a100-40 \
  --track
```

#### With Custom Classes

```bash
./submit_inference.sh \
  --run my_training_run \
  --project data/projects/MyProject \
  --input video.mp4 \
  --gpu a100-40
```

## Step 5: Review Results

### Output Structure

```
inference_results/20260128_110804/
├── detected_video.mp4      # Annotated video
├── detections.json         # JSON results
└── metadata.json           # Configuration
```

### JSON Format

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

### Analyze Results

```bash
# View annotated video
open inference_results/20260128_110804/detected_video.mp4

# Parse JSON
python -c "
import json
with open('inference_results/20260128_110804/detections.json') as f:
    data = json.load(f)
    total_detections = sum(len(frame['detections']) for frame in data['frames'])
    print(f'Total detections: {total_detections}')
"
```

## Step 6: Optimize Performance

### For Speed

1. **Skip frames**:
   ```bash
   --frame-interval 5 --track
   ```

2. **Lower resolution**:
   - Resize videos before inference
   - Use smaller model (`base` instead of `large`)

3. **Disable visualizations**:
   ```bash
   --no-visualizations
   ```

4. **Use smaller GPU** (cluster):
   ```bash
   --gpu a100-40
   ```

### For Accuracy

1. **Lower confidence**:
   ```bash
   --confidence 0.3
   ```

2. **Process all frames**:
   ```bash
   --frame-interval 1
   ```

3. **Tune tracking**:
   ```bash
   --track \
   --track-thresh 0.2 \
   --track-buffer 60 \
   --match-thresh 0.7
   ```

4. **Use larger model**:
   - Train with `--model large`

## Common Scenarios

### Scenario 1: Real-time Monitoring

Fast inference for live monitoring:

```bash
python -m cli.inference \
  --run my_training_run \
  --input camera_feed.mp4 \
  --frame-interval 10 \
  --track \
  --confidence 0.6
```

### Scenario 2: Offline Analysis

High-quality analysis of recorded footage:

```bash
python -m cli.inference \
  --run my_training_run \
  --input archive_video.mp4 \
  --frame-interval 1 \
  --track \
  --confidence 0.5 \
  --track-buffer 60
```

### Scenario 3: Batch Processing

Process multiple videos:

```bash
./submit_inference.sh \
  --run my_training_run \
  --input "videos/*.mp4" \
  --gpu a100-40 \
  --track \
  --frame-interval 5
```

### Scenario 4: High Precision Detection

Strict detection with high confidence:

```bash
python -m cli.inference \
  --run my_training_run \
  --input important_video.mp4 \
  --confidence 0.8 \
  --track
```

### Scenario 5: High Recall Detection

Catch all possible detections:

```bash
python -m cli.inference \
  --run my_training_run \
  --input surveillance.mp4 \
  --confidence 0.3 \
  --track \
  --track-thresh 0.15
```

## Tracking Deep Dive

### When to Use Tracking

✅ **Use tracking when:**
- Processing videos with moving objects
- Need persistent object IDs
- Objects may be temporarily occluded
- Want temporal consistency

❌ **Skip tracking when:**
- Processing individual images
- Objects don't move between frames
- Processing speed is critical

### Tracking Parameters Explained

#### `--track-buffer N`

How long to keep "lost" tracks:

- **Short (15-30)**: Fast-moving objects, no occlusions
- **Medium (30-60)**: General purpose, some occlusions
- **Long (60-90)**: Frequent occlusions, slow objects

#### `--track-thresh THRESHOLD`

Lower threshold for associating detections to tracks:

- **0.15-0.20**: Very permissive (may get false associations)
- **0.25 (default)**: Balanced
- **0.30-0.40**: Strict (may lose tracks)

#### `--match-thresh THRESHOLD`

IoU threshold for matching boxes:

- **0.6-0.7**: Permissive (accept loose matches)
- **0.8 (default)**: Balanced
- **0.9-0.95**: Strict (boxes must overlap closely)

#### `--no-kalman`

Disable Kalman filter prediction on non-keyframes:

Use when `--frame-interval > 1` but objects move unpredictably.

## Troubleshooting

### No Detections

**Solutions:**
1. Lower confidence: `--confidence 0.3`
2. Check class names: `--project` or `--classes`
3. Verify model: Correct training run
4. Test on training images

### Too Many False Positives

**Solutions:**
1. Raise confidence: `--confidence 0.7`
2. Retrain with more data
3. Check training metrics

### Tracking IDs Switching

**Solutions:**
1. Increase match threshold: `--match-thresh 0.9`
2. Reduce track buffer: `--track-buffer 15`
3. Adjust track threshold: `--track-thresh 0.3`

### Lost Tracks

**Solutions:**
1. Increase track buffer: `--track-buffer 60`
2. Lower track threshold: `--track-thresh 0.2`
3. Lower match threshold: `--match-thresh 0.7`

### Slow Inference

**Solutions:**
1. Skip frames: `--frame-interval 5`
2. Disable optimization: `--no-optimize` (if errors)
3. Use smaller model
4. Resize video

## Post-Processing

### Extract Statistics

```python
import json

with open('inference_results/20260128_110804/detections.json') as f:
    data = json.load(f)

# Count detections per class
class_counts = {}
for frame in data['frames']:
    for det in frame['detections']:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

print(class_counts)
```

### Filter by Confidence

```python
high_confidence = []
for frame in data['frames']:
    filtered_dets = [d for d in frame['detections'] if d['confidence'] > 0.8]
    if filtered_dets:
        high_confidence.append({
            'frame_id': frame['frame_id'],
            'detections': filtered_dets
        })
```

### Track Analysis

```python
# Find longest tracks
track_lengths = {}
for frame in data['frames']:
    for det in frame['detections']:
        track_id = det['track_id']
        track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

longest_tracks = sorted(track_lengths.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Longest tracks: {longest_tracks}")
```

## Best Practices

### 1. Start with Defaults

```bash
python -m cli.inference --run my_run --input video.mp4
```

### 2. Enable Tracking for Videos

```bash
--track
```

### 3. Tune Confidence Iteratively

Start at 0.5, adjust based on results.

### 4. Use Appropriate GPU

- **Local**: Use available GPU (auto)
- **Cluster**: A100-40 is cost-effective

### 5. Save JSON for Analysis

Don't skip JSON output:

```bash
# Saves both video and JSON
python -m cli.inference --run my_run --input video.mp4
```

### 6. Monitor Resource Usage

```bash
# While inference runs
nvidia-smi -l 1
```

## Related

- **[Inference CLI](../cli/inference.md)** - Command reference
- **[Submit Inference Script](../scripts/submit-inference.md)** - SLURM inference
- **[Training Workflow](training.md)** - Train models
- **[Benchmarking Guide](benchmarking.md)** - Measure performance
