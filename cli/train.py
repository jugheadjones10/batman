#!/usr/bin/env python3
"""
Train RF-DETR models on Batman project data.

This CLI wraps the core training logic from src.core.trainer.

Usage:
    # Full pipeline: prepare data + train
    python -m cli.train --project data/projects/Test --epochs 50

    # Just prepare the dataset (no training)
    python -m cli.train --project data/projects/Test --prepare-only

    # Train on existing COCO dataset
    python -m cli.train --dataset datasets/crane_hooks_coco --epochs 50

    # Run inference on a trained model
    python -m cli.train --checkpoint runs/my_run/checkpoint_best_total.pth --inference image.jpg

    # Export model
    python -m cli.train --checkpoint runs/my_run/checkpoint_best_total.pth --export models/my_model
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

# Import core training logic
from src.core.trainer import (
    DatasetStats,
    RFDETRTrainer,
    TrainingConfig,
    get_device,
    get_device_info,
    load_project_data,
    prepare_coco_dataset,
    set_seed,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def save_training_config(args: argparse.Namespace, output_dir: Path, dataset_dir: Path) -> None:
    """Save training configuration to a JSON file for reproducibility."""
    config = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "working_directory": str(Path.cwd()),
        "arguments": {
            "project": str(args.project) if args.project else None,
            "dataset": str(dataset_dir),
            "output_dir": str(output_dir),
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
            "device": args.device,
            "num_workers": args.num_workers,
            "patience": args.patience,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "video_id": args.video_id,
            "filter_classes": args.filter_classes,
            "resume": str(args.resume) if args.resume else None,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
        },
    }
    
    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"  Configuration saved to: {config_path}")


def print_dataset_stats(stats: DatasetStats) -> None:
    """Print dataset preparation statistics."""
    print(f"\n✓ Dataset prepared at: {stats.output_dir}")
    print(f"  Classes: {stats.class_names}")
    print("\n  Split statistics:")
    print(f"    Train: {stats.train_images} images, {stats.train_annotations} annotations")
    print(f"    Valid: {stats.val_images} images, {stats.val_annotations} annotations")
    print(f"    Test:  {stats.test_images} images, {stats.test_annotations} annotations")


def cmd_prepare(args: argparse.Namespace) -> DatasetStats:
    """Prepare dataset command."""
    print_header("PREPARING DATASET")

    # Load project info first
    _, annotations_data, class_names, project_config = load_project_data(
        args.project, args.video_id
    )

    print(f"✓ Loaded project: {project_config.get('name', 'Unknown')}")
    print(f"  All classes: {class_names}")
    print(f"  Total annotations: {len(annotations_data)}")

    # Parse filter classes
    filter_classes = None
    if args.filter_classes:
        filter_classes = [c.strip() for c in args.filter_classes.split("|") if c.strip()]
        print(f"\n  Filtering to classes: {filter_classes}")

    # Prepare dataset
    stats = prepare_coco_dataset(
        project_dir=args.project,
        output_dir=args.output_dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        video_id=args.video_id,
        clean=not args.no_clean,
        filter_classes=filter_classes,
        seed=args.seed,
    )

    print_dataset_stats(stats)
    return stats


def cmd_train(
    args: argparse.Namespace,
    dataset_dir: Path,
    class_names: list[str],
) -> Path:
    """Train model command."""
    print_header("TRAINING MODEL")

    # Save training configuration for reproducibility
    save_training_config(args, args.output_dir, dataset_dir)

    # Get device info
    device = get_device(args.device)
    device_info = get_device_info(device)
    print(f"\n✓ Device: {device_info['name']}")
    if "memory_gb" in device_info:
        print(f"  GPU Memory: {device_info['memory_gb']:.1f} GB")

    # Create training config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        device=device,
        num_workers=args.num_workers,
        patience=args.patience,
        grad_accum=args.grad_accum,
        resume=str(args.resume) if args.resume else None,
    )

    print(f"\n  Model: RF-DETR {args.model}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {args.output_dir}")
    print("\n  Training config:")
    print(f"    Epochs: {config.epochs}")
    print(f"    Batch size: {config.batch_size}")
    print(f"    Image size: {config.image_size}")
    print(f"    Learning rate: {config.lr}")
    print(f"    Early stopping patience: {config.patience}")
    if config.grad_accum > 1:
        print(f"    Gradient accumulation: {config.grad_accum}")

    # Train
    trainer = RFDETRTrainer(model_size=args.model)
    print("\n  Starting training...\n")

    result = trainer.train(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        config=config,
    )

    print("\n✓ Training complete!")
    print(f"  Best checkpoint: {result.checkpoint_path}")
    print(f"  Training time: {result.training_time_seconds / 60:.1f} minutes")

    # Save class info
    class_info = {
        "classes": class_names,
        "num_classes": len(class_names),
        "model": f"rf-detr-{args.model}",
    }
    info_path = args.output_dir / "class_info.json"
    with open(info_path, "w") as f:
        json.dump(class_info, f, indent=2)

    return result.checkpoint_path


def cmd_inference(args: argparse.Namespace, class_names: list[str]) -> None:
    """Run inference command."""
    print_header("RUNNING INFERENCE")

    print(f"\n  Loading model from: {args.checkpoint}")
    trainer = RFDETRTrainer(model_size=args.model, checkpoint=args.checkpoint)

    if args.inference_output:
        args.inference_output.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import supervision as sv

        has_supervision = True
    except ImportError:
        has_supervision = False
        print("  Note: Install 'supervision' for annotated output images")

    for img_path in args.inference:
        print(f"\n  Processing: {img_path.name}")

        from PIL import Image

        image = Image.open(img_path)
        detections = trainer.predict(image, threshold=args.confidence)

        print(f"    Found {len(detections)} detections")

        if len(detections) > 0 and hasattr(detections, "class_id"):
            for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
                class_name = (
                    class_names[int(class_id)]
                    if int(class_id) < len(class_names)
                    else f"class_{class_id}"
                )
                print(f"      [{i + 1}] {class_name}: {conf:.2f}")

        # Save annotated image
        if has_supervision and args.inference_output:
            image_np = np.array(image)

            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

            labels = [
                f"{class_names[int(cid)] if int(cid) < len(class_names) else f'class_{cid}'} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]

            annotated = box_annotator.annotate(image_np.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            out_path = args.inference_output / f"detected_{img_path.name}"
            Image.fromarray(annotated).save(out_path)
            print(f"    Saved: {out_path}")


def cmd_export(args: argparse.Namespace, class_names: list[str]) -> None:
    """Export model command."""
    print_header("EXPORTING MODEL")

    trainer = RFDETRTrainer(model_size=args.model, checkpoint=args.checkpoint)
    export_path = trainer.export(args.export, class_names)

    print(f"\n✓ Model exported to: {export_path}")
    print(f"✓ Class info saved to: {args.export / 'class_info.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RF-DETR models on Batman project data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: prepare data + train
  python -m cli.train --project data/projects/Test --epochs 50

  # Just prepare the dataset
  python -m cli.train --project data/projects/Test --prepare-only

  # Train on existing COCO dataset
  python -m cli.train --dataset datasets/my_coco --epochs 50

  # Filter to specific classes
  python -m cli.train --project data/projects/Test --filter-classes "crane hook|crane-hook"

  # Run inference
  python -m cli.train --checkpoint runs/run1/best.pth --inference image.jpg

  # Export model
  python -m cli.train --checkpoint runs/run1/best.pth --export models/my_model
        """,
    )

    # Input sources
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--project", type=Path, help="Path to Batman project directory")
    input_group.add_argument("--dataset", type=Path, help="Path to existing COCO format dataset")
    input_group.add_argument(
        "--checkpoint", type=Path, help="Path to trained model checkpoint (for inference/export)"
    )

    # Data preparation
    data_group = parser.add_argument_group("Data Preparation")
    data_group.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("datasets/rfdetr_coco"),
        help="Output directory for prepared COCO dataset (default: datasets/rfdetr_coco)",
    )
    data_group.add_argument(
        "--train-split", type=float, default=0.70, help="Fraction for training (default: 0.70)"
    )
    data_group.add_argument(
        "--val-split", type=float, default=0.15, help="Fraction for validation (default: 0.15)"
    )
    data_group.add_argument(
        "--test-split", type=float, default=0.15, help="Fraction for testing (default: 0.15)"
    )
    data_group.add_argument(
        "--video-id", type=int, default=-1, help="Video ID to process (-1 for imports, default: -1)"
    )
    data_group.add_argument(
        "--prepare-only", action="store_true", help="Only prepare dataset, don't train"
    )
    data_group.add_argument(
        "--no-clean", action="store_true", help="Don't remove existing dataset directory"
    )
    data_group.add_argument(
        "--filter-classes",
        type=str,
        help="Only train on these classes. Use pipe '|' delimiter, e.g., 'crane hook|crane-hook'",
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/rfdetr_run"),
        help="Output directory for training run (default: runs/rfdetr_run)",
    )
    train_group.add_argument(
        "--model", choices=["nano", "small", "base", "medium", "large"], default="base",
        help="Model size (default: base)"
    )
    train_group.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    train_group.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    train_group.add_argument(
        "--image-size", type=int, default=640, help="Input image size (default: 640)"
    )
    train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    train_group.add_argument(
        "--device", default="auto", help="Device: cuda, mps, cpu, or auto (default: auto)"
    )
    train_group.add_argument(
        "--num-workers", type=int, default=2, help="Number of data loader workers (default: 2)"
    )
    train_group.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience, 0 to disable (default: 10)"
    )
    train_group.add_argument("--resume", type=Path, help="Resume training from checkpoint")
    train_group.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)"
    )
    train_group.add_argument(
        "--mps-fallback", action="store_true", help="Enable MPS CPU fallback"
    )

    # Inference
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--inference", type=Path, nargs="+", help="Run inference on image(s)")
    infer_group.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)"
    )
    infer_group.add_argument(
        "--inference-output", type=Path, help="Output directory for annotated images"
    )

    # Export
    export_group = parser.add_argument_group("Export")
    export_group.add_argument("--export", type=Path, help="Export model to directory")

    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--classes", type=str, nargs="+", help="Class names (for inference/export without project)"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Enable MPS fallback if requested
    if args.mps_fallback:
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("✓ MPS CPU fallback enabled")

    # Validate arguments
    if not args.project and not args.dataset and not args.checkpoint:
        parser.error("Must specify --project, --dataset, or --checkpoint")

    if args.inference and not args.checkpoint:
        parser.error("--inference requires --checkpoint")

    if args.export and not args.checkpoint:
        parser.error("--export requires --checkpoint")

    # Track class names
    class_names = args.classes or []

    # === Data Preparation ===
    dataset_dir = args.dataset
    if args.project:
        stats = cmd_prepare(args)
        dataset_dir = stats.output_dir
        class_names = stats.class_names

        if args.prepare_only:
            print("\n✓ Dataset preparation complete (--prepare-only specified)")
            sys.exit(0)

    # === Training ===
    checkpoint_path = args.checkpoint
    if dataset_dir and not args.inference and not args.export:
        checkpoint_path = cmd_train(args, dataset_dir, class_names)

    # === Inference ===
    if args.inference and checkpoint_path:
        if not class_names:
            info_path = checkpoint_path.parent / "class_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    class_names = json.load(f).get("classes", [])
            else:
                print("Warning: No class names specified, using generic labels")
                class_names = [f"class_{i}" for i in range(100)]

        cmd_inference(args, class_names)

    # === Export ===
    if args.export and checkpoint_path:
        if not class_names:
            info_path = checkpoint_path.parent / "class_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    class_names = json.load(f).get("classes", [])

        cmd_export(args, class_names)

    print_header("DONE")


if __name__ == "__main__":
    main()
