# API Reference

Batman provides a REST API for project management, video processing, annotation, training, and inference.

## Base URL

```
http://localhost:8000
```

## Interactive API Docs

When the backend server is running, access interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible on localhost.

## API Endpoints

### Projects

#### List Projects
```http
GET /api/projects
```

Returns list of all projects.

#### Create Project
```http
POST /api/projects
Content-Type: application/json

{
  "name": "MyProject",
  "description": "Project description",
  "classes": ["class1", "class2"]
}
```

#### Get Project
```http
GET /api/projects/{name}
```

#### Update Classes
```http
PUT /api/projects/{name}/classes
Content-Type: application/json

{
  "classes": ["class1", "class2", "class3"]
}
```

#### Delete Project
```http
DELETE /api/projects/{name}
```

### Videos

#### List Videos
```http
GET /api/projects/{name}/videos
```

#### Upload Video
```http
POST /api/projects/{name}/videos
Content-Type: multipart/form-data

file: <video file>
```

#### Get Video Info
```http
GET /api/projects/{name}/videos/{id}
```

#### Extract Frames
```http
POST /api/projects/{name}/videos/{id}/extract-frames
Content-Type: application/json

{
  "interval": 30,
  "max_frames": 1000
}
```

#### List Frames
```http
GET /api/projects/{name}/videos/{id}/frames
```

#### Stream Video
```http
GET /api/projects/{name}/videos/{id}/stream
```

Returns video file for streaming.

#### Delete Video
```http
DELETE /api/projects/{name}/videos/{id}
```

### Annotations

#### List Annotations
```http
GET /api/projects/{name}/frames/{id}/annotations
```

#### Create Annotation
```http
POST /api/projects/{name}/annotations
Content-Type: application/json

{
  "frame_id": 1,
  "class_id": 0,
  "bbox": [x1, y1, x2, y2],
  "confidence": 1.0,
  "track_id": null
}
```

#### Update Annotation
```http
PUT /api/projects/{name}/annotations/{id}
Content-Type: application/json

{
  "class_id": 1,
  "bbox": [x1, y1, x2, y2]
}
```

#### Delete Annotation
```http
DELETE /api/projects/{name}/annotations/{id}
```

### Tracks

#### List Tracks
```http
GET /api/projects/{name}/videos/{id}/tracks
```

#### Update Track
```http
PUT /api/projects/{name}/tracks/{id}
Content-Type: application/json

{
  "class_id": 1
}
```

#### Split Track
```http
POST /api/projects/{name}/tracks/split
Content-Type: application/json

{
  "track_id": 1,
  "split_frame": 50
}
```

#### Merge Tracks
```http
POST /api/projects/{name}/tracks/merge
Content-Type: application/json

{
  "source_track_id": 2,
  "target_track_id": 1
}
```

### Labeling

#### Start Auto-Labeling
```http
POST /api/projects/{name}/labeling/auto-label
Content-Type: application/json

{
  "video_id": 1,
  "prompts": ["object1", "object2"]
}
```

Returns job ID.

#### Get Labeling Status
```http
GET /api/projects/{name}/labeling/auto-label/{job_id}/status
```

#### Refine Labels
```http
POST /api/projects/{name}/labeling/refine
Content-Type: application/json

{
  "video_id": 1,
  "frame_ids": [1, 2, 3]
}
```

#### Create Label Iteration
```http
POST /api/projects/{name}/labeling/create-iteration
Content-Type: application/json

{
  "name": "iteration_1",
  "description": "Initial labels"
}
```

### Training

#### Export Dataset
```http
POST /api/projects/{name}/training/export-dataset
Content-Type: application/json

{
  "format": "coco",
  "split": {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
  }
}
```

#### Start Training
```http
POST /api/projects/{name}/training/start
Content-Type: application/json

{
  "dataset_path": "datasets/my_dataset",
  "model": "base",
  "epochs": 50,
  "batch_size": 8
}
```

Returns training run ID.

#### List Training Runs
```http
GET /api/projects/{name}/training/runs
```

#### Get Training Run
```http
GET /api/projects/{name}/training/runs/{id}
```

### Inference

#### Load Model
```http
POST /api/projects/{name}/inference/load-model
Content-Type: application/json

{
  "checkpoint_path": "runs/my_run/best.pth"
}
```

#### Run on Image
```http
POST /api/projects/{name}/inference/run-on-image
Content-Type: multipart/form-data

image: <image file>
confidence: 0.5
```

#### Run on Video
```http
POST /api/projects/{name}/inference/run-on-video/{video_id}
Content-Type: application/json

{
  "confidence": 0.5,
  "track": true
}
```

#### Export Video
```http
POST /api/projects/{name}/inference/export-video/{video_id}
```

Returns annotated video file.

#### Stream Inference Results (WebSocket)
```
WS /api/projects/{name}/inference/stream/{video_id}
```

Streams real-time inference results for video.

## Data Formats

### Bounding Box Format

Bounding boxes are in `[x1, y1, x2, y2]` format (absolute coordinates):

```json
{
  "bbox": [100, 50, 200, 150]
}
```

Where:
- `x1, y1`: Top-left corner
- `x2, y2`: Bottom-right corner

### Detection Response

```json
{
  "frame_id": 0,
  "timestamp": 0.0,
  "detections": [
    {
      "bbox": [100, 50, 200, 150],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "person",
      "track_id": 1
    }
  ]
}
```

### Training Status

```json
{
  "run_id": "rfdetr_20260128_105030",
  "status": "training",
  "epoch": 25,
  "total_epochs": 50,
  "loss": 0.234,
  "metrics": {
    "mAP": 0.85,
    "precision": 0.88,
    "recall": 0.83
  }
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common HTTP status codes:
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource doesn't exist)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

## Rate Limiting

Currently no rate limiting is enforced on localhost.

## CORS

CORS is enabled for localhost development. The frontend at `http://localhost:5173` can access the API.

## Related

- **[Development Server](../scripts/run-dev.md)** - Start the API server
- **[CLI Tools](../cli/index.md)** - Command-line interface
- **[Training Workflow](../guides/training.md)** - Use the API for training
