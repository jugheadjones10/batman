#!/usr/bin/env python3
"""
Train RF-DETR models on Batman project data.

This CLI wraps the core training logic from src.core.trainer.
Training runs and dataset exports default to project-relative paths.

Usage:
    # Full pipeline: prepare data + train (saves under project)
    python -m cli.train --project data/projects/Test --epochs 50

    # Train then immediately run inference on all project videos
    python -m cli.train --project data/projects/Test --epochs 50 --infer-after

    # Train then run inference only on test videos
    python -m cli.train --project data/projects/Test --epochs 50 --infer-after --infer-test-only

    # Just prepare the dataset (no training)
    python -m cli.train --project data/projects/Test --prepare-only

    # Train on existing COCO dataset
    python -m cli.train --dataset datasets/crane_hooks_coco --epochs 50

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


def parse_video_id(video_id_str: str) -> int | str:
    """
    Parse video_id argument to int or special string.
    
    Args:
        video_id_str: String like 'all', 'imports', '-1', '1', etc.
        
    Returns:
        int for specific video IDs, or string for 'all'/'imports'
    """
    if video_id_str in ("all", "imports"):
        return video_id_str
    if video_id_str.lstrip("-").isdigit():
        return int(video_id_str)
    return video_id_str  # source_key e.g. roboflow_crane-hook_1


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
            "max_frames_per_class": args.max_frames_per_class,
            "sources": args.sources,
            "manual_split_strategy": args.manual_split_strategy,
            "manual_datasets": args.manual_datasets,
            "exclude_manual_datasets": args.exclude_manual_datasets,
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


def _parse_manual_dataset_filters(args: argparse.Namespace) -> tuple[list[str] | None, list[str] | None]:
    """Parse --manual-datasets / --exclude-manual-datasets into lists."""
    manual_ds = None
    exclude_ds = None
    if args.manual_datasets:
        manual_ds = [s.strip() for s in args.manual_datasets.split(",") if s.strip()]
    if args.exclude_manual_datasets:
        exclude_ds = [s.strip() for s in args.exclude_manual_datasets.split(",") if s.strip()]
    return manual_ds, exclude_ds


def cmd_prepare(args: argparse.Namespace) -> DatasetStats:
    """Prepare dataset command."""
    print_header("PREPARING DATASET")

    # Parse sources if provided
    sources_list = None
    if args.sources:
        sources_list = [s.strip() for s in args.sources.split(",") if s.strip()]
        print(f"  Data sources: {sources_list}")

    manual_ds, exclude_ds = _parse_manual_dataset_filters(args)
    if manual_ds:
        print(f"  Manual datasets (include): {manual_ds}")
    if exclude_ds:
        print(f"  Manual datasets (exclude): {exclude_ds}")

    # Parse video_id argument
    video_id = parse_video_id(args.video_id)

    # Load project info first
    _, annotations_data, class_names, project_config = load_project_data(
        args.project, video_id, sources=sources_list,
        manual_datasets=manual_ds, exclude_manual_datasets=exclude_ds,
    )

    print(f"✓ Loaded project: {project_config.get('name', 'Unknown')}")
    print(f"  All classes: {class_names}")
    print(f"  Total annotations: {len(annotations_data)}")

    # Parse filter classes
    filter_classes = None
    if args.filter_classes:
        filter_classes = [c.strip() for c in args.filter_classes.split("|") if c.strip()]
        print(f"\n  Filtering to classes: {filter_classes}")

    # Compute per-class sampling fractions from --max-frames-per-class
    frame_sample_fractions = None
    if args.max_frames_per_class is not None:
        from collections import defaultdict
        frames_meta, _, _, _ = load_project_data(
            args.project, video_id, sources=sources_list,
            manual_datasets=manual_ds, exclude_manual_datasets=exclude_ds,
        )
        frames_by_class: dict[str, set[str]] = defaultdict(set)
        for ann in annotations_data.values():
            fid = str(ann["frame_id"])
            if fid not in frames_meta:
                continue
            cid = ann["class_label_id"]
            if cid < len(class_names):
                frames_by_class[class_names[cid]].add(fid)

        active_classes = filter_classes if filter_classes else class_names
        frame_sample_fractions = {}
        cap = args.max_frames_per_class
        print(f"\n  Max frames per class: {cap} (manual data always included)")
        for cls in active_classes:
            all_fids = frames_by_class.get(cls, set())
            n = len(all_fids)
            n_manual = sum(1 for fid in all_fids if str(fid).startswith("manual_data_"))
            frac = min(1.0, cap / n) if n > 0 else 1.0
            frame_sample_fractions[cls] = frac
            # Manual frames are always kept; cap reduces non-manual frames
            n_non_manual = n - n_manual
            target_non_manual = max(0, cap - n_manual)
            sampled_non_manual = min(n_non_manual, target_non_manual)
            sampled = n_manual + (sampled_non_manual if frac < 1.0 else n_non_manual)
            manual_note = f" ({n_manual} manual, always kept)" if n_manual else ""
            print(f"    {cls}: {n} frames -> {sampled}{manual_note}")

    # Prepare dataset
    stats = prepare_coco_dataset(
        project_dir=args.project,
        output_dir=args.output_dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        video_id=video_id,
        clean=not args.no_clean,
        filter_classes=filter_classes,
        frame_sample_fractions=frame_sample_fractions,
        seed=args.seed,
        sources=sources_list,
        manual_data_split_strategy=args.manual_split_strategy,
        manual_datasets=manual_ds,
        exclude_manual_datasets=exclude_ds,
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


def cmd_infer_after(args: argparse.Namespace, run_dir: Path, class_names: list[str]) -> None:
    """Run inference on project videos immediately after training."""
    print_header("POST-TRAINING INFERENCE")

    from src.core.inference import InferenceConfig, RFDETRInference
    from src.core.project import Project
    from src.core.trainer import find_best_checkpoint
    from cli.inference import process_video, persist_result

    project = Project.load(args.project)

    checkpoint = find_best_checkpoint(run_dir)
    if checkpoint is None:
        print("  No checkpoint found in run, skipping inference")
        return

    run_name = run_dir.name
    print(f"  Run: {run_name}")
    print(f"  Checkpoint: {checkpoint}")

    if args.infer_test_only:
        videos = project.list_videos(test_only=True)
        if not videos:
            print("  No test-only videos found, skipping inference")
            return
        print(f"  Videos: {len(videos)} (test-only)")
    else:
        videos = project.list_videos()
        if not videos:
            print("  No videos in project, skipping inference")
            return
        print(f"  Videos: {len(videos)}")

    config = InferenceConfig(
        confidence_threshold=0.0,
        device=args.device,
        save_visualizations=True,
    )

    print("\n  Loading model...")
    engine = RFDETRInference(
        checkpoint=checkpoint,
        class_names=class_names,
        model_size=args.model,
    )
    engine.load_model(device=config.device, optimize=True)

    total_detections = 0
    for video_id, vid_meta in videos.items():
        video_path = Path(vid_meta["original_path"])
        if not video_path.exists():
            print(f"  Video not found: {video_path}, skipping")
            continue

        output_dir = project.inference_dir / run_name / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Processing: {vid_meta['filename']} ({video_id})")
        stats, all_results = process_video(engine, video_path, config, output_dir)

        if stats:
            persist_result(project, run_name, video_id, stats, all_results, config)
            print(f"    Frames: {stats.total_frames}, Detections: {stats.total_detections}, "
                  f"Avg: {stats.avg_inference_time_ms:.1f}ms")
            total_detections += stats.total_detections

    print(f"\n✓ Inference complete: {total_detections} total detections")
    print(f"  Results saved: {project.inference_dir / run_name}")


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

  # Train + run inference on all project videos
  python -m cli.train --project data/projects/Test --epochs 50 --infer-after

  # Train + run inference on test-only videos
  python -m cli.train --project data/projects/Test --epochs 50 --infer-after --infer-test-only

  # Just prepare the dataset
  python -m cli.train --project data/projects/Test --prepare-only

  # Train on existing COCO dataset
  python -m cli.train --dataset datasets/my_coco --epochs 50

  # Filter to specific classes
  python -m cli.train --project data/projects/Test --filter-classes "crane hook|crane-hook"

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
        default=None,
        help="Output directory for prepared COCO dataset (default: {project}/exports/coco)",
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
        "--video-id",
        type=str,
        default="imports",
        help="Video ID(s) to process: 'all', 'imports' (default), or specific ID like '-1'",
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
    data_group.add_argument(
        "--max-frames-per-class",
        type=int,
        default=None,
        help="Cap frames per class to roughly this number (randomly sampled, deterministic with --seed)",
    )
    data_group.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Data sources to include (comma-separated). Valid: manual_data,imports. "
             "When set, overrides --video-id and always excludes video frames.",
    )
    data_group.add_argument(
        "--manual-split-strategy",
        type=str,
        choices=["proportional", "val_only", "train_only", "all_splits"],
        default="train_only",
        help="How to distribute manual data across splits (default: train_only)",
    )
    data_group.add_argument(
        "--manual-datasets",
        type=str,
        default=None,
        help="Only include these manual data subdatasets (comma-separated). "
             "Use '(root)' for root-level images. Example: --manual-datasets crane_closeups,worker_shots",
    )
    data_group.add_argument(
        "--exclude-manual-datasets",
        type=str,
        default=None,
        help="Exclude these manual data subdatasets (comma-separated). "
             "Mutually exclusive with --manual-datasets. Example: --exclude-manual-datasets negative_examples",
    )

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for training run (default: {project}/runs/rfdetr_run)",
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

    # Export
    export_group = parser.add_argument_group("Export")
    export_group.add_argument("--export", type=Path, help="Export model to directory")

    # Post-training inference
    infer_group = parser.add_argument_group("Post-training Inference")
    infer_group.add_argument(
        "--infer-after", action="store_true",
        help="Run inference on project videos after training completes",
    )
    infer_group.add_argument(
        "--infer-test-only", action="store_true",
        help="With --infer-after, only run on test-only videos (exclude_from_training=true)",
    )

    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--classes", type=str, nargs="+", help="Class names (for export without project)"
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

    if args.export and not args.checkpoint:
        parser.error("--export requires --checkpoint")

    if args.manual_datasets and args.exclude_manual_datasets:
        parser.error("--manual-datasets and --exclude-manual-datasets are mutually exclusive")

    # Derive defaults from project when available
    if args.project:
        if args.output_dataset is None:
            args.output_dataset = args.project / "exports" / "coco"
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = args.project / "runs" / f"rfdetr_{timestamp}"
    else:
        if args.output_dataset is None:
            args.output_dataset = Path("datasets/rfdetr_coco")
        if args.output_dir is None:
            args.output_dir = Path("runs/rfdetr_run")

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
    if dataset_dir and not args.export:
        checkpoint_path = cmd_train(args, dataset_dir, class_names)

    # === Export ===
    if args.export and checkpoint_path:
        if not class_names:
            info_path = checkpoint_path.parent / "class_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    class_names = json.load(f).get("classes", [])

        cmd_export(args, class_names)

    # === Post-training Inference ===
    if args.infer_after and args.project and checkpoint_path:
        cmd_infer_after(args, args.output_dir, class_names)

    print_header("DONE")


if __name__ == "__main__":
    main()
