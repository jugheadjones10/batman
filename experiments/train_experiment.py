#!/usr/bin/env python3
"""
Hydra-based experiment runner for class imbalance study.

Usage:
    # Single experiment (local)
    python experiments/train_experiment.py experiment=exp_person_25

    # All experiments via SLURM
    python experiments/train_experiment.py --multirun \\
        experiment=exp_person_25,exp_person_50,exp_person_75,exp_person_100

    # Preview config without running
    python experiments/train_experiment.py experiment=exp_person_25 --cfg job

    # Run locally (not via SLURM)
    python experiments/train_experiment.py experiment=exp_person_25 hydra/launcher=local

    # Resume from existing training (run inference + summary only)
    python experiments/train_experiment.py experiment=exp_person_25 resume_from=/path/to/experiment/dir
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up PyTorch distributed training environment variables
# Required for RF-DETR even in single-GPU mode
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = str(12355 + random.randint(0, 1000))
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def run_training_subprocess(
    dataset_dir: Path,
    output_dir: Path,
    model_size: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    lr: float,
    patience: int,
) -> tuple[Path, float, dict]:
    """
    Run RF-DETR training in a subprocess to isolate sys.exit() calls.

    RF-DETR's train() method calls sys.exit(0) on completion, which would
    terminate our main process. Running in a subprocess prevents this.

    Returns:
        Tuple of (checkpoint_path, training_time_seconds, metrics)
    """
    import time

    # Create a training script to run in subprocess
    train_script = f'''
import sys
import json
import time
sys.path.insert(0, "{project_root}")

from pathlib import Path
from src.core.trainer import RFDETRTrainer, TrainingConfig

trainer = RFDETRTrainer(model_size="{model_size}")
config = TrainingConfig(
    epochs={epochs},
    batch_size={batch_size},
    image_size={image_size},
    lr={lr},
    patience={patience},
)

start_time = time.time()
try:
    result = trainer.train(
        dataset_dir=Path("{dataset_dir}"),
        output_dir=Path("{output_dir}"),
        config=config,
    )
    training_time = time.time() - start_time
    # This may not be reached if RF-DETR calls sys.exit()
    print(json.dumps({{"checkpoint": str(result.checkpoint_path), "time": training_time}}))
except SystemExit:
    # RF-DETR called sys.exit() - training completed
    training_time = time.time() - start_time
    print(json.dumps({{"completed_via_exit": True, "time": training_time}}))
'''

    log.info("Starting training in subprocess (to isolate RF-DETR sys.exit)...")
    start_time = time.time()

    # Run training in subprocess
    result = subprocess.run(
        [sys.executable, "-c", train_script],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )

    training_time = time.time() - start_time

    # Log any output
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line:
                log.info(f"[subprocess] {line}")
    if result.stderr:
        # Filter out common warnings, only log important stuff
        for line in result.stderr.strip().split("\n"):
            if line and "warning" not in line.lower():
                log.debug(f"[subprocess stderr] {line}")

    # Check if training succeeded (exit code 0 means RF-DETR finished)
    if result.returncode != 0:
        log.error(f"Training subprocess failed with exit code {result.returncode}")
        log.error(f"stderr: {result.stderr}")
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # Find best checkpoint
    checkpoint_path = find_best_checkpoint(output_dir)
    log.info(f"Training completed in {training_time:.1f}s")
    log.info(f"Best checkpoint: {checkpoint_path}")

    # Load metrics from results.json if available
    metrics = load_training_metrics(output_dir)

    return checkpoint_path, training_time, metrics


def find_best_checkpoint(output_dir: Path) -> Path:
    """Find the best checkpoint in the output directory."""
    candidates = [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "checkpoint.pth",
    ]

    for name in candidates:
        path = output_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


def load_training_metrics(output_dir: Path) -> dict:
    """Load training metrics from results.json if available."""
    results_file = output_dir / "results.json"
    if not results_file.exists():
        log.warning(f"No results.json found in {output_dir}")
        return {}

    try:
        with open(results_file) as f:
            results = json.load(f)

        # Extract relevant metrics from results.json
        # RF-DETR saves metrics in a specific format
        metrics = {}

        # Look for mAP metrics in various possible locations
        if isinstance(results, dict):
            # Try to find the best metrics
            for key in ["mAP@50", "mAP@75", "mAP", "AP50", "AP75"]:
                if key in results:
                    metrics[key] = results[key]

            # Also check for nested metrics
            if "metrics" in results:
                metrics.update(results["metrics"])

            # Check for per-class metrics
            for key, value in results.items():
                if key.startswith("AP@50_") or key.startswith("AP@75_"):
                    metrics[key] = value

        log.info(f"Loaded metrics: {metrics}")
        return metrics

    except Exception as e:
        log.warning(f"Failed to load results.json: {e}")
        return {}


def run_inference_on_videos(
    checkpoint_path: Path,
    class_names: list[str],
    test_videos: list[str],
    output_dir: Path,
    model_size: str = "base",
) -> dict[str, dict]:
    """
    Run inference on test videos and save annotated outputs.

    Args:
        checkpoint_path: Path to trained model checkpoint
        class_names: List of class names
        test_videos: List of video paths/names to process
        output_dir: Directory to save inference outputs
        model_size: Model size for inference

    Returns:
        Dict mapping video name to inference stats
    """
    import cv2

    from src.core.inference import (
        InferenceConfig,
        RFDETRInference,
        draw_detections,
        save_results_json,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference engine
    inferencer = RFDETRInference(
        checkpoint=checkpoint_path,
        class_names=class_names,
        model_size=model_size,
    )

    config = InferenceConfig(
        confidence_threshold=0.5,
        frame_interval=1,
        use_tracking=True,
        save_visualizations=True,
        save_json=True,
    )

    results = {}

    for video_name in test_videos:
        video_path = Path(video_name)

        # Try to find the video in common locations
        if not video_path.exists():
            video_path = project_root / video_name
        if not video_path.exists():
            log.warning(f"Video not found: {video_name}, skipping")
            continue

        log.info(f"Running inference on: {video_path.name}")

        # Setup output paths
        output_video = output_dir / f"detected_{video_path.name}"
        output_json = output_dir / f"{video_path.stem}_detections.json"

        # Open video for writing
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        # Process video
        all_results = []
        cap = cv2.VideoCapture(str(video_path))

        generator = inferencer.predict_video(video_path, config)

        for frame_result in generator:
            # Read frame for annotation
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_result.frame_idx)
            ret, frame = cap.read()

            if ret:
                # Draw detections and write frame
                annotated = draw_detections(frame, frame_result.detections)
                writer.write(annotated)

            all_results.append(frame_result)

        # Get final stats (generator returns stats at the end)
        try:
            stats = generator.send(None)
        except StopIteration as e:
            stats = e.value

        cap.release()
        writer.release()

        # Save JSON results
        save_results_json(all_results, output_json, stats, {"source": str(video_path)})

        log.info(f"Saved: {output_video}")

        results[video_path.name] = {
            "total_frames": stats.total_frames if stats else 0,
            "keyframes": stats.keyframes if stats else 0,
            "total_detections": stats.total_detections if stats else 0,
            "avg_inference_time_ms": stats.avg_inference_time_ms if stats else 0,
            "fps": stats.fps if stats else 0,
        }

    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Run a single experiment: prepare data, train, inference."""

    log.info(f"=" * 60)
    log.info(f"Running experiment: {cfg.experiment.name}")
    log.info(f"=" * 60)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Import here to avoid slow startup for --cfg commands
    from src.core.trainer import (
        prepare_coco_dataset,
    )

    # Check if we're resuming from an existing experiment
    resume_from = cfg.get("resume_from", None)

    if resume_from:
        # Resume mode: use existing experiment directory
        output_dir = Path(resume_from)
        if not output_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {output_dir}")
        log.info(f"RESUME MODE: Using existing experiment at {output_dir}")
        original_cwd = Path(hydra.utils.get_original_cwd())
    else:
        # Normal mode: use Hydra output directory
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        original_cwd = Path(hydra.utils.get_original_cwd())

    log.info(f"Output directory: {output_dir}")

    # Resolve paths relative to original working directory
    project_dir = original_cwd / cfg.project_dir
    dataset_dir = original_cwd / cfg.output_dataset

    training_output_dir = output_dir / "training"

    # Check if training already completed (for resume or re-run)
    training_already_done = False
    try:
        existing_checkpoint = find_best_checkpoint(training_output_dir)
        training_already_done = True
        log.info(f"Found existing checkpoint: {existing_checkpoint}")
    except FileNotFoundError:
        pass

    if training_already_done and resume_from:
        # Resume mode with existing training - skip to inference
        log.info("-" * 40)
        log.info("Step 1 & 2: SKIPPED (using existing training)")
        log.info("-" * 40)

        checkpoint_path = find_best_checkpoint(training_output_dir)
        training_time = 0.0  # Unknown for resumed experiments
        metrics = load_training_metrics(training_output_dir)

        # Try to load dataset stats from existing files or re-prepare
        stats = prepare_coco_dataset(
            project_dir=project_dir,
            output_dir=dataset_dir,
            filter_classes=list(cfg.classes),
            frame_sample_fractions=OmegaConf.to_container(cfg.experiment.frame_sample_fractions),
        )
        frame_sample_fractions = OmegaConf.to_container(cfg.experiment.frame_sample_fractions)

    else:
        # Normal mode: prepare dataset and train

        # 1. Prepare dataset with class sampling
        log.info("-" * 40)
        log.info("Step 1: Preparing dataset...")
        log.info("-" * 40)

        frame_sample_fractions = OmegaConf.to_container(cfg.experiment.frame_sample_fractions)
        log.info(f"Frame sample fractions: {frame_sample_fractions}")

        stats = prepare_coco_dataset(
            project_dir=project_dir,
            output_dir=dataset_dir,
            filter_classes=list(cfg.classes),
            frame_sample_fractions=frame_sample_fractions,
        )

        log.info(f"Dataset prepared:")
        log.info(f"  Train: {stats.train_images} images, {stats.train_annotations} annotations")
        log.info(f"  Val: {stats.val_images} images, {stats.val_annotations} annotations")
        log.info(f"  Test: {stats.test_images} images, {stats.test_annotations} annotations")
        log.info(f"  Classes: {stats.class_names}")

        # 2. Train model
        log.info("-" * 40)
        log.info("Step 2: Training model...")
        log.info("-" * 40)

        # Use subprocess to isolate RF-DETR's sys.exit() behavior
        checkpoint_path, training_time, metrics = run_training_subprocess(
            dataset_dir=dataset_dir,
            output_dir=training_output_dir,
            model_size=cfg.training.model,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            image_size=cfg.training.image_size,
            lr=cfg.training.lr,
            patience=cfg.training.patience,
        )

        log.info(f"Training complete!")
        log.info(f"  Checkpoint: {checkpoint_path}")
        log.info(f"  Training time: {training_time:.1f}s")
        log.info(f"  Metrics: {metrics}")

    # Save class info alongside checkpoint for inference
    class_info = {
        "classes": list(cfg.classes),
        "num_classes": len(cfg.classes),
        "model": f"rf-detr-{cfg.training.model}",
    }
    class_info_path = training_output_dir / "class_info.json"
    with open(class_info_path, "w") as f:
        json.dump(class_info, f, indent=2)

    # 3. Run inference on test videos
    log.info("-" * 40)
    log.info("Step 3: Running inference on test videos...")
    log.info("-" * 40)

    inference_dir = output_dir / "inference"
    inference_results = run_inference_on_videos(
        checkpoint_path=checkpoint_path,
        class_names=list(cfg.classes),
        test_videos=list(cfg.test_videos),
        output_dir=inference_dir,
        model_size=cfg.training.model,
    )

    # 4. Save experiment summary
    log.info("-" * 40)
    log.info("Step 4: Saving experiment summary...")
    log.info("-" * 40)

    summary = {
        "experiment": cfg.experiment.name,
        "description": cfg.experiment.description,
        "frame_sample_fractions": frame_sample_fractions,
        "dataset_stats": {
            "train_images": stats.train_images,
            "train_annotations": stats.train_annotations,
            "val_images": stats.val_images,
            "val_annotations": stats.val_annotations,
            "test_images": stats.test_images,
            "test_annotations": stats.test_annotations,
            "classes": stats.class_names,
        },
        "training": {
            "time_seconds": training_time,
            "checkpoint": str(checkpoint_path),
            "config": {
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "image_size": cfg.training.image_size,
                "lr": cfg.training.lr,
                "patience": cfg.training.patience,
                "model": cfg.training.model,
            },
        },
        "metrics": metrics,
        "inference_results": inference_results,
    }

    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Experiment summary saved to: {summary_path}")

    log.info("=" * 60)
    log.info(f"Experiment {cfg.experiment.name} COMPLETE")
    log.info("=" * 60)

    # Return primary metric for Hydra optimization (optional)
    return metrics.get("mAP@50", 0.0)


if __name__ == "__main__":
    main()
