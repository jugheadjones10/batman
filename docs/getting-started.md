# Getting Started

This guide will help you set up Batman and run your first training job.

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend development)
- **FFmpeg** (for video processing)
- **uv** (Python package manager)
- **CUDA-capable GPU** (recommended for training)

## Installation

### 1. Install System Dependencies

#### macOS
```bash
brew install ffmpeg
brew install uv
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/batman.git
cd batman

# Install Python dependencies
uv sync

# Install frontend dependencies (optional, for web UI)
cd frontend
npm install
cd ..
```

### 3. Verify Installation

```bash
# Check Python environment
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CLI tools
uv run python -m cli.train --help
uv run python -m cli.inference --help
```

## Quick Start Tutorial

### Step 1: Import Data

Import a dataset from Roboflow or COCO Zoo:

```bash
# Import from COCO Zoo
uv run python -m cli.importer coco \
  --project data/projects/MyProject \
  --create \
  --classes person car \
  --split validation \
  --max-samples 100
```

See [Data Import](cli/importer.md) for more options.

### Step 2: Train a Model

Train an RF-DETR model on your data:

```bash
uv run python -m cli.train \
  --project data/projects/MyProject \
  --model base \
  --epochs 50 \
  --batch-size 8 \
  --output-dir runs/my_first_run
```

See [Training CLI](cli/train.md) for detailed options.

### Step 3: Run Inference

Test your trained model:

```bash
uv run python -m cli.inference \
  --run my_first_run \
  --input path/to/video.mp4 \
  --track \
  --confidence 0.5
```

See [Inference CLI](cli/inference.md) for more options.

### Step 4: Benchmark Performance

Measure inference latency:

```bash
uv run python -m cli.benchmark_latency \
  --run my_first_run \
  --video path/to/video.mp4 \
  --runs 100
```

See [Benchmark CLI](cli/benchmark-latency.md) for details.

## Using SLURM (HPC Clusters)

If you have access to a SLURM cluster:

### Submit Training Job

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50
```

See [SLURM Training](scripts/submit-train.md) for all options.

### Submit Inference Job

```bash
./submit_inference.sh \
  --run my_first_run \
  --input video.mp4 \
  --gpu a100-40 \
  --track
```

See [SLURM Inference](scripts/submit-inference.md) for details.

## Development Mode

To run the web UI locally:

```bash
# Option 1: Use the development script
./scripts/run_dev.sh

# Option 2: Manual startup
# Terminal 1 - Backend
uv run python -m backend.app.main

# Terminal 2 - Frontend
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

See [Development Server](scripts/run-dev.md) for more information.

## Common Workflows

### Workflow 1: Local Training
1. [Import data](cli/importer.md) → 2. [Train model](cli/train.md) → 3. [Run inference](cli/inference.md)

### Workflow 2: Cluster Training
1. [Prepare data](cli/importer.md) → 2. [Submit training](scripts/submit-train.md) → 3. [Submit inference](scripts/submit-inference.md)

### Workflow 3: Benchmarking
1. [Train model](cli/train.md) → 2. [Submit benchmarks](scripts/submit-benchmark.md) → 3. [Compare results](cli/compare-latency.md)

## Next Steps

- **Explore CLI Tools**: [CLI Overview](cli/index.md)
- **Learn SLURM Scripts**: [SLURM Overview](scripts/index.md)
- **Read Workflow Guides**: [Training Guide](guides/training.md)
- **Check API Reference**: [API Docs](api/index.md)

## Troubleshooting

### CUDA Out of Memory

Reduce `--batch-size` or `--image-size`:

```bash
uv run python -m cli.train --batch-size 4 --image-size 512
```

### Model Not Found

Check that your run directory exists:

```bash
ls runs/
ls runs/my_run_name/
```

### FFmpeg Not Found

Ensure FFmpeg is in your PATH:

```bash
which ffmpeg
ffmpeg -version
```

### Import Errors

Reinstall dependencies:

```bash
uv sync --force
```

---

**Need help?** Check the [guides](guides/training.md) or open an issue on GitHub.
