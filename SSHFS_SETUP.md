# SSHFS Setup & Command Logging

This document explains how to use SSHFS to access your GPU server and the new command logging features.

## SSHFS Setup (Mounting Remote GPU Server)

### Installation

```bash
# Install FUSE-T (modern, kernel-extension-free SSHFS for macOS)
brew tap macos-fuse-t/homebrew-cask
brew install fuse-t
brew install fuse-t-sshfs
```

### Quick Start (Recommended)

```bash
# Mount the GPU server
./mount_gpu.sh

# When done
./umount_gpu.sh
```

That's it! Simple two-command workflow.

**Note:** Edit `mount_gpu.sh` line 23 to set your username if different from `youngjin`.

### Manual Mounting (Advanced)

If you prefer to mount manually:

```bash
# Create mount point inside your project
mkdir -p ~/Projects/batman/gpu-server

# Mount the remote server
sshfs username@xlogin.comp.nus.edu.sg:~/batman ~/Projects/batman/gpu-server

# Now you can access remote files locally!
# - Drag videos into gpu-server/ in Finder
# - View results in gpu-server/inference_results/
# - Open the remote directory in Cursor
```

### Unmounting

```bash
# When done
umount ~/Projects/batman/gpu-server
```

### Auto-mount on Startup (Optional)

Add to your `~/.zshrc`:

```bash
# Auto-mount GPU server if not already mounted
if ! mount | grep -q "gpu-server"; then
    sshfs username@xlogin.comp.nus.edu.sg:~/batman ~/Projects/batman/gpu-server \
        -o auto_cache,reconnect,defer_permissions,noappledouble 2>/dev/null || true
fi
```

## Command Logging Feature

**New**: All training and inference runs now automatically save their configuration!

### Training Runs

When you train a model, a `training_config.json` file is saved in the run directory:

```bash
python -m cli.train --project data/projects/CraneHook --epochs 50 --model base
```

Creates `runs/rfdetr_run/training_config.json`:

```json
{
  "command": "python -m cli.train --project data/projects/CraneHook --epochs 50 --model base",
  "timestamp": "2026-01-28T11:00:00.123456",
  "hostname": "gpu01.comp.nus.edu.sg",
  "working_directory": "/home/username/batman",
  "arguments": {
    "project": "data/projects/CraneHook",
    "dataset": "datasets/rfdetr_coco",
    "epochs": 50,
    "batch_size": 8,
    "model": "base",
    "lr": 0.0001,
    "device": "cuda",
    ...
  },
  "environment": {
    "python_executable": "/home/username/batman/.venv/bin/python3",
    "python_version": "3.11.5"
  }
}
```

### Inference Runs

When you run inference, an `inference_config.json` file is saved in the output directory:

```bash
python -m cli.inference --run rfdetr_h200_20260120_105925 \
    --project data/projects/CraneHook --input video.mp4
```

Creates `inference_results/inference_config.json` with all parameters.

### Reproducing Runs

Now you can easily reproduce any run:

```bash
# View the config from a previous training run
cat runs/rfdetr_h200_20260120_105925/training_config.json

# Copy the command and re-run it
python -m cli.train --project data/projects/CraneHook --epochs 50 --model base ...

# Or for inference
cat inference_results/20260128_110804/inference_config.json
```

## Workflow Example

### 1. Mount the Server

```bash
./mount_gpu.sh
```

### 2. Upload Video

```bash
# Drag in Finder, or:
cp my_video.mp4 ~/Projects/batman/gpu-server/
```

### 3. SSH and Run Inference

```bash
ssh username@xlogin.comp.nus.edu.sg
cd ~/batman
./submit_inference.sh --latest --project data/projects/CraneHook --input my_video.mp4
```

### 4. View Results Locally

```bash
# Results appear automatically in:
~/Projects/batman/gpu-server/inference_results/

# Open in QuickLook (spacebar in Finder) or drag to Desktop
```

### 5. Check the Config

```bash
# See exactly how the inference was run
cat ~/Projects/batman/gpu-server/inference_results/latest/inference_config.json
```

## Benefits

✅ **No more manual rsync** - Files sync automatically  
✅ **No more forgotten parameters** - Every run saves its config  
✅ **Drag & drop works** - Just like local files  
✅ **Open in Cursor** - Edit remote files directly  
✅ **Reproducibility** - Always know how you ran a model  

## Notes

- Large video files (>1GB) may be slow over network - consider copying to local first for playback
- The `gpu-server/` directory is in `.gitignore` so it won't be committed
- Config files are JSON format for easy parsing and version control
