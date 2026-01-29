# Training Workflow Guide

Complete end-to-end guide for training RF-DETR models with Batman.

## Overview

Training workflow:
1. **Prepare Data** - Import or create dataset
2. **Organize Classes** - Clean and standardize class names
3. **Configure Training** - Choose hyperparameters
4. **Train Model** - Run training locally or on cluster
5. **Evaluate Results** - Check metrics and visualizations
6. **Export Model** - Prepare for deployment

## Step 1: Prepare Data

### Option A: Import from COCO Zoo

```bash
python -m cli.importer coco \
  --project data/projects/MyProject \
  --create \
  --classes person car bicycle \
  --split validation \
  --max-samples 500
```

### Option B: Import from Roboflow

```bash
export ROBOFLOW_API_KEY=your_api_key

python -m cli.importer roboflow \
  --project data/projects/MyProject \
  --create \
  --workspace your-workspace \
  --rf-project your-project \
  --version 1
```

### Option C: Use Batman Web UI

1. Start development server: `./scripts/run_dev.sh`
2. Open http://localhost:5173
3. Create project and upload videos
4. Use SAM3 auto-labeling and manual correction

## Step 2: Organize Classes

### List Classes

```bash
python -m cli.classes list --project data/projects/MyProject
```

### Merge Similar Classes

```bash
python -m cli.classes merge \
  --project data/projects/MyProject \
  --source "crane-hook" "crane_hook" "hook" \
  --target "crane_hook"
```

### Rename Classes

```bash
python -m cli.classes rename \
  --project data/projects/MyProject \
  --old-name "crane-boom" \
  --new-name "crane_boom"
```

## Step 3: Configure Training

### Choose Model Size

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| base | ~28M | Balanced | Good | **Recommended for most cases** |
| large | ~76M | Slower | Highest | High accuracy needed |
| medium | ~48M | Medium | High | Balance accuracy/speed |
| small | ~10M | Fast | Lower | Edge deployment |

### Select Hyperparameters

#### Epochs

- **Quick test**: 10-20 epochs
- **Standard training**: 50 epochs
- **Fine-tuning**: 100-200 epochs

#### Batch Size

Choose based on GPU memory:

| GPU | Recommended Batch Size |
|-----|----------------------|
| H100 96GB | 16 |
| A100 80GB | 12-16 |
| A100 40GB | 8 |
| RTX 3090 | 4 |

Or use gradient accumulation:

```bash
--batch-size 4 --grad-accum 4  # Effective batch size: 16
```

#### Learning Rate

- **Fine-tuning pretrained**: `1e-5` to `1e-4` (default: `1e-4`)
- **Training from scratch**: `1e-4` to `1e-3`

#### Image Size

- **Standard**: 640 (default)
- **High detail**: 800-1024
- **Fast training**: 512

## Step 4: Train Model

### Local Training

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --model base \
  --epochs 50 \
  --batch-size 8 \
  --image-size 640 \
  --lr 1e-4 \
  --patience 10 \
  --output-dir runs/my_training_run
```

### SLURM Cluster Training

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --epochs 50 \
  --model base \
  --batch-size 16
```

### Multi-GPU Training

```bash
./submit_train.sh \
  --project data/projects/MyProject \
  --gpu h100-96 \
  --num-gpus 4 \
  --batch-size 16 \
  --epochs 50
```

## Step 5: Monitor Training

### View Logs

```bash
# Local training
tail -f runs/my_training_run/*.log

# SLURM training
tail -f logs/job_*.log
```

### Check TensorBoard

```bash
tensorboard --logdir runs/my_training_run/tensorboard --port 6006
```

Open http://localhost:6006

### Review Metrics

```bash
cat runs/my_training_run/results.json
```

Key metrics:
- **mAP (mean Average Precision)**: Overall detection accuracy
- **Loss**: Training loss (lower is better)
- **Precision**: Fraction of correct detections
- **Recall**: Fraction of objects detected

## Step 6: Evaluate Results

### Check Training Configuration

```bash
cat runs/my_training_run/training_config.json
```

### View Validation Images

```bash
open runs/my_training_run/val_images/
```

### Test Inference

```bash
python -m cli.inference \
  --run my_training_run \
  --input test_image.jpg \
  --confidence 0.5
```

## Step 7: Iterate and Improve

### If Overfitting (High train accuracy, low val accuracy)

1. Reduce model complexity: Use `--model base` instead of `large`
2. Increase regularization: Lower learning rate
3. Add more data: Import additional samples
4. Early stopping: Use `--patience 10`

### If Underfitting (Low train and val accuracy)

1. Increase model capacity: Use `--model large`
2. Train longer: Increase `--epochs`
3. Check data quality: Review annotations
4. Adjust learning rate: Try `--lr 5e-5`

### If Slow Convergence

1. Increase learning rate: Try `--lr 5e-4`
2. Increase batch size: `--batch-size 16`
3. Use gradient accumulation: `--grad-accum 4`

### If Class Imbalance

Filter to focus on specific classes:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --filter-classes "rare_class" \
  --epochs 100
```

## Step 8: Export and Deploy

### Export for Inference

```bash
python -m cli.train \
  --checkpoint runs/my_training_run/best.pth \
  --export exports/my_model_v1 \
  --classes "person,car,bicycle"
```

### Test Exported Model

```bash
python -m cli.inference \
  --checkpoint exports/my_model_v1/model.pth \
  --input test_video.mp4 \
  --confidence 0.5 \
  --track
```

## Common Training Scenarios

### Scenario 1: Quick Prototype

Fast training for initial testing:

```bash
python -m cli.train \
  --project data/projects/Test \
  --model base \
  --epochs 20 \
  --batch-size 8 \
  --image-size 512
```

### Scenario 2: Production Model

High-quality training for deployment:

```bash
./submit_train.sh \
  --project data/projects/Production \
  --gpu h100-96 \
  --model large \
  --epochs 100 \
  --batch-size 16 \
  --image-size 800 \
  --patience 15
```

### Scenario 3: Limited GPU Memory

Training on smaller GPU:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --model base \
  --batch-size 2 \
  --grad-accum 8 \
  --image-size 512 \
  --epochs 50
```

### Scenario 4: Fine-tune Existing Model

Resume and continue training:

```bash
python -m cli.train \
  --project data/projects/MyProject \
  --resume runs/previous_run/checkpoint_epoch_25.pth \
  --output-dir runs/continued_training \
  --epochs 50 \
  --lr 1e-5
```

## Best Practices

### 1. Start Small

Begin with small datasets and quick training:

```bash
python -m cli.importer coco \
  --project data/projects/Test \
  --create \
  --classes person \
  --max-samples 100

python -m cli.train \
  --project data/projects/Test \
  --epochs 20
```

### 2. Use Default Settings First

Default settings work well for most cases:

```bash
python -m cli.train --project data/projects/MyProject
```

### 3. Monitor Training Closely

Watch for:
- Loss decreasing steadily
- mAP increasing
- Validation metrics improving

### 4. Save Checkpoints

Training automatically saves:
- `best.pth` - Best validation performance
- `checkpoint_last.pth` - Latest checkpoint
- `checkpoint_epoch_N.pth` - Periodic saves

### 5. Organize Experiments

Use descriptive output directories:

```bash
--output-dir runs/crane_hook_base_800px
--output-dir runs/person_car_large_v2
```

### 6. Document Training

Keep notes on:
- Dataset version
- Hyperparameters used
- Results and observations
- Issues encountered

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch-size 4

# Reduce image size
--image-size 512

# Use gradient accumulation
--batch-size 4 --grad-accum 4
```

### Loss Not Decreasing

- Check learning rate (try `--lr 1e-4`)
- Verify data quality
- Ensure sufficient epochs
- Check for data loading errors in logs

### Training Stalls

- Monitor GPU usage: `nvidia-smi`
- Check disk space
- Review logs for errors
- Restart with lower batch size

### Poor Validation Performance

- Check for overfitting (train mAP >> val mAP)
- Add more training data
- Use smaller model
- Increase patience for early stopping

## Related

- **[Training CLI](../cli/train.md)** - Command reference
- **[Submit Training Script](../scripts/submit-train.md)** - SLURM training
- **[Inference Workflow](inference.md)** - Next steps
- **[Importer CLI](../cli/importer.md)** - Data import
