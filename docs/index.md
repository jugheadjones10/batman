# Batman Documentation

**Local Video Auto-Label → Human Correct → Fine-Tune Real-Time Detector**

Welcome to the Batman documentation! This guide covers all CLI tools, SLURM scripts, and workflows for training real-time object detectors.

## What is Batman?

Batman is a localhost application that turns videos and object descriptions into a fine-tuned real-time detector. It uses **SAM3** (Segment Anything Model 3) as a label generator with text prompts, exemplar prompting, and built-in tracking. It provides a Roboflow-style review UI for corrections and fine-tunes smaller base models (YOLO11 or RF-DETR) for fast inference on new videos.

## Quick Links

### 🛠️ CLI Tools
- **[Training](cli/train.md)** - Train RF-DETR models on your data
- **[Inference](cli/inference.md)** - Run inference on images and videos
- **[Benchmarking](cli/benchmark-latency.md)** - Measure inference latency and throughput
- **[Data Import](cli/importer.md)** - Import from Roboflow or COCO Zoo
- **[Class Management](cli/classes.md)** - Manage, rename, and merge classes

### 🚀 SLURM Scripts
- **[Submit Training](scripts/submit-train.md)** - Submit training jobs to SLURM cluster
- **[Submit Inference](scripts/submit-inference.md)** - Run inference on cluster GPUs
- **[Submit Benchmark](scripts/submit-benchmark.md)** - Benchmark across multiple GPU types
- **[Development Server](scripts/run-dev.md)** - Local development setup

### 📚 Guides
- **[Training Workflow](guides/training.md)** - End-to-end training guide
- **[Inference Workflow](guides/inference.md)** - Running inference effectively
- **[Benchmarking Guide](guides/benchmarking.md)** - Performance testing
- **[SLURM Usage](guides/slurm.md)** - Working with HPC clusters

### 🔌 API Reference
- **[REST API](api/index.md)** - Backend API endpoints

## Key Features

- **Auto-labeling with SAM3**: Automatically generate bounding box annotations
- **Smart Tracking**: Link detections across frames with occlusion handling
- **Human-in-the-loop Review**: Fix boxes, adjust classes, split/merge tracks
- **Model Fine-tuning**: Train YOLO11 or RF-DETR on your labeled data
- **Real-time Inference**: Test models with live tracking overlay
- **GPU Benchmarking**: Test performance across different GPU types
- **SLURM Integration**: Submit jobs to HPC clusters easily

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/batman.git
cd batman

# Install Python dependencies
uv sync

# Install documentation dependencies (optional)
pip install -r requirements-docs.txt
```

### Quick Start

1. **Start the development server**: See [Development Server](scripts/run-dev.md)
2. **Import data**: Use the [Importer CLI](cli/importer.md)
3. **Train a model**: Follow the [Training Workflow](guides/training.md)
4. **Run inference**: Check out the [Inference Workflow](guides/inference.md)

## Project Structure

```
batman/
├── cli/                    # Command-line tools
├── backend/               # FastAPI backend
├── frontend/              # React frontend
├── src/                   # Core libraries
├── scripts/               # Utility scripts
├── data/                  # Project data
└── docs/                  # This documentation
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/batman/issues)
- **Documentation**: You're here!
- **API Docs**: http://localhost:8000/docs (when server is running)

---

**Next**: Check out the [CLI Tools Overview](cli/index.md) or jump straight to [Training](cli/train.md)!
