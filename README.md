# Batman 🦇

**Local Video Auto-Label → Human Correct → Fine-Tune Real-Time Detector**

Batman is a localhost application that turns videos and object descriptions into a fine-tuned real-time detector. It uses **SAM3** (Segment Anything Model 3) as a label generator with text prompts, exemplar prompting, and built-in tracking. It provides a Roboflow-style review UI for corrections and fine-tunes smaller base models (YOLO11 or RF-DETR) for fast inference on new videos.

![Batman Architecture](https://via.placeholder.com/800x400?text=Batman+Architecture)

## 🎯 Key Features

- **Auto-labeling with SAM3**: Automatically generate bounding box annotations using text prompts, points, or boxes
- **Smart Tracking**: Link detections across frames with configurable occlusion handling
- **Human-in-the-loop Review**: Fix boxes, adjust classes, split/merge tracks
- **Exemplar Prompting**: Click once to guide detection across videos
- **Model Fine-tuning**: Train YOLO11 or RF-DETR on your labeled data
- **Real-time Inference**: Test models with live tracking overlay
- **Version Control**: Immutable label iterations and training runs

## 🏗️ Architecture

```
batman/
├── backend/                    # Python FastAPI backend
│   └── app/
│       ├── api/               # REST API routes
│       │   ├── projects.py    # Project CRUD
│       │   ├── videos.py      # Video upload & frames
│       │   ├── annotations.py # Annotation CRUD
│       │   ├── labeling.py    # Auto-labeling jobs
│       │   ├── training.py    # Model training
│       │   └── inference.py   # Model inference
│       ├── services/          # Business logic
│       │   ├── video_processor.py  # FFmpeg operations
│       │   ├── sam_labeler.py      # SAM2 integration
│       │   ├── tracker.py          # Object tracking
│       │   ├── dataset_exporter.py # YOLO/COCO export
│       │   ├── trainer.py          # Model training
│       │   └── inference_runner.py # Model inference
│       ├── db/                # SQLAlchemy models
│       ├── models/            # Pydantic schemas
│       ├── config.py          # Settings
│       └── main.py            # FastAPI app
│
├── frontend/                  # React + Vite frontend
│   └── src/
│       ├── pages/
│       │   ├── ProjectsPage.tsx   # Project list
│       │   ├── ProjectPage.tsx    # Project dashboard
│       │   ├── AnnotatePage.tsx   # Annotation editor
│       │   ├── TrainingPage.tsx   # Training config
│       │   └── InferencePage.tsx  # Model testing
│       ├── components/        # UI components
│       ├── api/              # API client
│       ├── store/            # Zustand state
│       └── types/            # TypeScript types
│
├── data/                     # Data directory (created at runtime)
│   └── projects/            # Project folders
│       └── <project>/
│           ├── project.json  # Project config
│           ├── project.sqlite # SQLite database
│           ├── videos/       # Original + proxy videos
│           ├── frames/       # Extracted frames
│           ├── labels/       # Label iterations
│           ├── exports/      # Dataset exports
│           └── runs/         # Training checkpoints
│
├── pyproject.toml           # Python dependencies (uv)
└── frontend/package.json    # Frontend dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- FFmpeg (for video processing)
- uv (Python package manager)

### Installation

1. **Clone and install Python dependencies**:
```bash
cd batman
uv sync
```

2. **Install frontend dependencies**:
```bash
cd frontend
npm install
```

3. **SAM3 model** (optional - auto-downloads if not present):
```bash
# Place your sam3.pt in the project root, or let Ultralytics download it automatically
# SAM3 is integrated into Ultralytics 8.3.237+
```

### Running the Application

**Terminal 1 - Backend**:
```bash
uv run python -m backend.app.main
# or
uv run batman
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### GPU Server Access (Optional)

If you're training on a remote GPU cluster, you can mount it locally for easy file access:

```bash
# Mount your GPU server
./mount_gpu.sh

# Access files in ./gpu-server/
# - View training results
# - Upload videos via drag & drop
# - Check logs without SSH

# Unmount when done
./umount_gpu.sh
```

See [SSHFS_SETUP.md](SSHFS_SETUP.md) for installation and configuration.

## 📖 User Guide

### 1. Create a Project

- Click "New Project" on the Projects page
- Enter a name and description
- Add object classes you want to detect (e.g., "person", "car", "dog")

### 2. Upload Videos

- Navigate to your project
- Click "Upload Video" to add MP4/AVI/MOV files
- Click "Extract Frames" to sample frames for annotation

### 3. Auto-Label with SAM2

- Go to the Annotate page
- Click "Auto-Label" to run SAM2 on all frames
- The system will:
  - Detect objects matching your class prompts
  - Link detections into tracks across frames
  - Generate bounding box annotations

### 4. Review and Correct

- Use the annotation editor to:
  - **Draw** new boxes (D key)
  - **Select** and move boxes (V key)
  - **Delete** incorrect annotations (Delete key)
  - **Change class** labels
  - **Navigate** frames (Arrow keys)

- For track-level operations:
  - **Split track**: Separate one track into two at a specific frame
  - **Merge tracks**: Combine two tracks that represent the same object

### 5. Train a Model

- Go to the Training page
- Select a base model:
  - **YOLO11-N/S/M**: Fast, lightweight detectors
  - **RF-DETR-B/L**: Transformer-based, higher accuracy
- Configure training parameters:
  - Epochs, batch size, image size
  - Learning rate preset
  - Augmentation level
- Click "Start Training"

### 6. Run Inference

- Go to the Inference page
- Select a video and trained model
- Adjust confidence/IoU thresholds
- Enable/disable tracking
- Run inference and view results
- Export annotated video

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Server
BATMAN_HOST=127.0.0.1
BATMAN_PORT=8000
BATMAN_DEBUG=true

# Paths
BATMAN_DATA_DIR=./data
BATMAN_PROJECTS_DIR=./data/projects

# SAM2
BATMAN_SAM_MODEL_PATH=./models/sam2_hiera_large.pt
BATMAN_SAM_CONFIG=sam2_hiera_l

# Tracking defaults
BATMAN_DEFAULT_TRACKING_MODE=visible_only
BATMAN_DEFAULT_MAX_AGE=30
BATMAN_DEFAULT_IOU_THRESHOLD=0.3

# Device
BATMAN_DEVICE=mps  # mps (Mac), cuda (NVIDIA), cpu
```

### Tracking Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `visible_only` | Conservative linking, tracks end quickly | Objects rarely occluded |
| `occlusion_tolerant` | Allows re-linking after disappearance | Objects frequently hidden |

### Tracking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | 30 | Frames before a track is considered lost |
| `iou_threshold` | 0.3 | Minimum IoU for box association |
| `min_hits` | 3 | Detections needed to confirm a track |

## 🧪 API Reference

### Projects

```
GET    /api/projects              # List all projects
POST   /api/projects              # Create project
GET    /api/projects/{name}       # Get project details
PUT    /api/projects/{name}/classes   # Update classes
DELETE /api/projects/{name}       # Delete project
```

### Videos

```
GET    /api/projects/{name}/videos           # List videos
POST   /api/projects/{name}/videos           # Upload video
GET    /api/projects/{name}/videos/{id}      # Get video info
POST   /api/projects/{name}/videos/{id}/extract-frames  # Extract frames
GET    /api/projects/{name}/videos/{id}/frames          # List frames
GET    /api/projects/{name}/videos/{id}/stream          # Stream video
DELETE /api/projects/{name}/videos/{id}      # Delete video
```

### Annotations

```
GET    /api/projects/{name}/frames/{id}/annotations  # List annotations
POST   /api/projects/{name}/annotations              # Create annotation
PUT    /api/projects/{name}/annotations/{id}         # Update annotation
DELETE /api/projects/{name}/annotations/{id}         # Delete annotation
```

### Tracks

```
GET    /api/projects/{name}/videos/{id}/tracks  # List tracks
PUT    /api/projects/{name}/tracks/{id}         # Update track
POST   /api/projects/{name}/tracks/split        # Split track
POST   /api/projects/{name}/tracks/merge        # Merge tracks
```

### Labeling

```
POST   /api/projects/{name}/labeling/auto-label          # Start auto-labeling
GET    /api/projects/{name}/labeling/auto-label/{job}/status  # Get status
POST   /api/projects/{name}/labeling/refine              # Refine labels
POST   /api/projects/{name}/labeling/create-iteration    # Create snapshot
```

### Training

```
POST   /api/projects/{name}/training/export-dataset  # Export dataset
POST   /api/projects/{name}/training/start           # Start training
GET    /api/projects/{name}/training/runs            # List runs
GET    /api/projects/{name}/training/runs/{id}       # Get run details
```

### Inference

```
POST   /api/projects/{name}/inference/load-model     # Load model
POST   /api/projects/{name}/inference/run-on-image   # Run on frame
POST   /api/projects/{name}/inference/run-on-video/{id}  # Run on video
POST   /api/projects/{name}/inference/export-video/{id}  # Export video
WS     /api/projects/{name}/inference/stream/{id}    # Stream results
```

## 📁 Data Formats

### YOLO Export Format

```
dataset/
├── data.yaml        # Dataset configuration
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images
└── labels/
    ├── train/      # Training labels (txt)
    ├── val/        # Validation labels
    └── test/       # Test labels
```

Each `.txt` file contains:
```
<class_id> <center_x> <center_y> <width> <height>
```
All coordinates are normalized (0-1).

### COCO Export Format

```
dataset/
├── train/
│   ├── images/
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

## 🔧 Development

### Backend Development

```bash
# Run with auto-reload
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest

# Format code
uv run ruff format backend/
uv run ruff check --fix backend/
```

### Frontend Development

```bash
cd frontend

# Development server
npm run dev

# Build for production
npm run build

# Lint
npm run lint
```

## 🗺️ Roadmap

### MVP ✅
- [x] Upload video + class list
- [x] Sample frames + run SAM2 → boxes + tracks
- [x] Basic UI to edit boxes and export YOLO/COCO
- [x] Fine-tune YOLO11 / RF-DETR
- [x] Run inference on a video and visualize + track

### v0.9 (In Progress)
- [ ] Track-level editing + split/merge
- [ ] "Refine only touched segments" rerun
- [ ] Problem queue for review acceleration
- [ ] Exemplar prompting in UI

### v1.0 (Planned)
- [ ] Multi-GPU / remote training support
- [ ] Video segmentation preview
- [ ] Active learning suggestions
- [ ] Model comparison dashboard
- [ ] Export to ONNX/TensorRT

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO
- [Meta AI](https://github.com/facebookresearch/segment-anything-2) for SAM2
- [Roboflow](https://roboflow.com) for RF-DETR and inspiration

