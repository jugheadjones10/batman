#!/usr/bin/env python3
"""
Fine-tune RF-DETR on a Batman project dataset.

This script converts Batman project annotations to COCO format and fine-tunes
an RF-DETR model for object detection.

Usage:
    # Full pipeline: prepare data + train
    python finetune_rfdetr.py --project data/projects/Test --epochs 50

    # Just prepare the dataset (no training)
    python finetune_rfdetr.py --project data/projects/Test --prepare-only

    # Train on existing COCO dataset
    python finetune_rfdetr.py --dataset datasets/crane_hooks_coco --epochs 50

    # Run inference on a trained model
    python finetune_rfdetr.py --checkpoint runs/my_run/checkpoint_best_total.pth --inference path/to/image.jpg

    # Export model
    python finetune_rfdetr.py --checkpoint runs/my_run/checkpoint_best_total.pth --export models/my_model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested_device: str | None = None) -> str:
    """Detect the best available device."""
    if requested_device and requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✓ Apple Silicon MPS available")
    else:
        device = "cpu"
        print("⚠ Using CPU (training will be slow)")

    return device


def load_project_data(project_dir: Path, video_id: int = -1) -> tuple[dict, dict, list[str]]:
    """
    Load project data from Batman project directory.

    Args:
        project_dir: Path to project directory
        video_id: Video ID to load frames from (-1 for Roboflow imports)

    Returns:
        Tuple of (frames_meta, annotations_data, class_names)
    """
    frames_dir = project_dir / "frames" / str(video_id)
    annotations_file = project_dir / "labels" / "current" / "annotations.json"
    project_config_file = project_dir / "project.json"

    # Verify paths exist
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    if not project_config_file.exists():
        raise FileNotFoundError(f"Project config not found: {project_config_file}")

    # Load project config
    with open(project_config_file) as f:
        project_config = json.load(f)
    class_names = project_config.get("classes", [])

    # Load frames metadata
    frames_meta_file = frames_dir / "frames.json"
    with open(frames_meta_file) as f:
        frames_meta = json.load(f)

    # Load annotations
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    print(f"✓ Loaded project: {project_config.get('name', 'Unknown')}")
    print(f"  Classes: {class_names}")
    print(f"  Frames: {len(frames_meta)}")
    print(f"  Annotations: {len(annotations_data)}")

    return frames_meta, annotations_data, class_names


def create_coco_dataset(
    frame_ids: set[str],
    frames_meta: dict,
    annotations_data: dict,
    class_names: list[str],
    output_dir: Path,
) -> tuple[int, int]:
    """
    Create a COCO format dataset from Batman internal format.

    COCO format:
    - images: list of {id, file_name, width, height}
    - annotations: list of {id, image_id, category_id, bbox, area}
    - categories: list of {id, name}

    Note: COCO bbox format is [x_min, y_min, width, height] in pixels
    """
    coco_data = {"images": [], "annotations": [], "categories": []}

    # Create categories (COCO uses 1-indexed category IDs)
    for i, name in enumerate(class_names):
        coco_data["categories"].append(
            {
                "id": i + 1,
                "name": name,
                "supercategory": "object",
            }
        )

    annotation_id = 1

    for frame_id in frame_ids:
        frame_info = frames_meta[frame_id]
        src_path = Path(frame_info["image_path"])

        if not src_path.exists():
            print(f"  Warning: Image not found: {src_path}")
            continue

        # Get image dimensions
        with Image.open(src_path) as img:
            img_width, img_height = img.size

        # Create new filename
        new_filename = f"{frame_id}.jpg"
        dst_path = output_dir / new_filename

        # Copy image
        shutil.copy(src_path, dst_path)

        # Add image entry
        image_id = int(frame_id)
        coco_data["images"].append(
            {"id": image_id, "file_name": new_filename, "width": img_width, "height": img_height}
        )

        # Get annotations for this frame
        for ann in annotations_data.values():
            if str(ann["frame_id"]) != frame_id:
                continue

            # Convert normalized center format to COCO format
            cx = ann["x"] * img_width
            cy = ann["y"] * img_height
            w = ann["width"] * img_width
            h = ann["height"] * img_height

            x_min = cx - w / 2
            y_min = cy - h / 2

            # COCO category_id is 1-indexed
            category_id = ann["class_label_id"] + 1

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    # Save COCO annotations
    coco_file = output_dir / "_annotations.coco.json"
    with open(coco_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return len(coco_data["images"]), len(coco_data["annotations"])


def prepare_dataset(
    project_dir: Path,
    output_dir: Path,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    video_id: int = -1,
    clean: bool = True,
) -> tuple[Path, list[str]]:
    """
    Prepare COCO format dataset from Batman project.

    Args:
        project_dir: Path to Batman project
        output_dir: Output directory for COCO dataset
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        video_id: Video ID to process (-1 for Roboflow imports)
        clean: Whether to remove existing output directory

    Returns:
        Tuple of (dataset_dir, class_names)
    """
    print("\n" + "=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    # Load project data
    frames_meta, annotations_data, class_names = load_project_data(project_dir, video_id)

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    if clean and output_dir.exists():
        print(f"  Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Created dataset directories:")
    print(f"  Train: {train_dir}")
    print(f"  Valid: {val_dir}")
    print(f"  Test:  {test_dir}")

    # Split frames
    all_frame_ids = list(frames_meta.keys())
    random.shuffle(all_frame_ids)

    n_total = len(all_frame_ids)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_frame_ids = set(all_frame_ids[:n_train])
    val_frame_ids = set(all_frame_ids[n_train : n_train + n_val])
    test_frame_ids = set(all_frame_ids[n_train + n_val :])

    print(f"\n✓ Split distribution:")
    print(f"  Train: {len(train_frame_ids)} ({len(train_frame_ids) / n_total * 100:.1f}%)")
    print(f"  Valid: {len(val_frame_ids)} ({len(val_frame_ids) / n_total * 100:.1f}%)")
    print(f"  Test:  {len(test_frame_ids)} ({len(test_frame_ids) / n_total * 100:.1f}%)")

    # Create datasets
    print("\n  Creating training dataset...")
    train_images, train_anns = create_coco_dataset(
        train_frame_ids, frames_meta, annotations_data, class_names, train_dir
    )
    print(f"  → Train: {train_images} images, {train_anns} annotations")

    print("  Creating validation dataset...")
    val_images, val_anns = create_coco_dataset(
        val_frame_ids, frames_meta, annotations_data, class_names, val_dir
    )
    print(f"  → Valid: {val_images} images, {val_anns} annotations")

    print("  Creating test dataset...")
    test_images, test_anns = create_coco_dataset(
        test_frame_ids, frames_meta, annotations_data, class_names, test_dir
    )
    print(f"  → Test: {test_images} images, {test_anns} annotations")

    print(f"\n✓ Dataset prepared at: {output_dir}")

    return output_dir, class_names


def train_model(
    dataset_dir: Path,
    output_dir: Path,
    model_size: str = "base",
    epochs: int = 50,
    batch_size: int = 8,
    image_size: int = 640,
    lr: float = 1e-4,
    device: str = "auto",
    num_workers: int = 2,
    patience: int = 10,
    resume: str | None = None,
    grad_accum: int = 1,
    use_mps_fallback: bool = False,
) -> Path:
    """
    Train RF-DETR model.

    Args:
        dataset_dir: Path to COCO format dataset
        output_dir: Output directory for training run
        model_size: Model size ('base', 'large')
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        lr: Learning rate
        device: Device to train on ('cuda', 'mps', 'cpu', or 'auto')
        num_workers: Number of data loader workers
        patience: Early stopping patience
        resume: Path to checkpoint to resume from
        grad_accum: Gradient accumulation steps
        use_mps_fallback: Enable MPS CPU fallback for unsupported ops

    Returns:
        Path to best checkpoint
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    # Set MPS fallback environment variable if requested
    if use_mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("✓ MPS CPU fallback enabled")

    # Get device
    device = get_device(device)

    # Import RF-DETR
    if model_size == "large":
        from rfdetr import RFDETRLarge as RFDETRModel

        model_name = "RF-DETR Large"
    else:
        from rfdetr import RFDETRBase as RFDETRModel

        model_name = "RF-DETR Base"

    print(f"\n✓ Model: {model_name}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")
    print(f"\n  Training config:")
    print(f"    Epochs: {epochs}")
    print(f"    Batch size: {batch_size}")
    print(f"    Image size: {image_size}")
    print(f"    Learning rate: {lr}")
    print(f"    Early stopping patience: {patience}")
    if grad_accum > 1:
        print(f"    Gradient accumulation: {grad_accum}")

    # Initialize model
    print("\n  Initializing model...")
    model = RFDETRModel()

    # Start training
    print("\n  Starting training...\n")

    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "lr": lr,
        "device": device,
        "output_dir": str(output_dir),
        "num_workers": num_workers,
        "early_stopping": patience > 0,
        "early_stopping_patience": patience,
    }

    if grad_accum > 1:
        train_kwargs["grad_accum_steps"] = grad_accum

    if resume:
        train_kwargs["resume"] = resume

    model.train(**train_kwargs)

    # Find best checkpoint
    checkpoint_path = output_dir / "checkpoint_best_total.pth"
    if not checkpoint_path.exists():
        # Try other names
        for name in ["checkpoint_best_ema.pth", "checkpoint_best_regular.pth", "checkpoint.pth"]:
            alt_path = output_dir / name
            if alt_path.exists():
                checkpoint_path = alt_path
                break

    print(f"\n✓ Training complete!")
    print(f"  Best checkpoint: {checkpoint_path}")

    return checkpoint_path


def run_inference(
    checkpoint_path: Path,
    image_paths: list[Path],
    class_names: list[str],
    confidence_threshold: float = 0.5,
    output_dir: Path | None = None,
    model_size: str = "base",
) -> None:
    """
    Run inference on images using trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        image_paths: List of image paths to run inference on
        class_names: List of class names
        confidence_threshold: Detection confidence threshold
        output_dir: Optional output directory for annotated images
        model_size: Model size ('base' or 'large')
    """
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)

    # Import RF-DETR
    if model_size == "large":
        from rfdetr import RFDETRLarge as RFDETRModel
    else:
        from rfdetr import RFDETRBase as RFDETRModel

    print(f"\n  Loading model from: {checkpoint_path}")
    model = RFDETRModel(pretrain_weights=str(checkpoint_path))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import supervision as sv

        has_supervision = True
    except ImportError:
        has_supervision = False
        print("  Note: Install 'supervision' for annotated output images")

    for img_path in image_paths:
        print(f"\n  Processing: {img_path.name}")

        image = Image.open(img_path)
        detections = model.predict(image, threshold=confidence_threshold)

        print(f"    Found {len(detections)} detections")

        if len(detections) > 0 and hasattr(detections, "class_id"):
            for i, (class_id, conf) in enumerate(zip(detections.class_id, detections.confidence)):
                class_name = (
                    class_names[int(class_id)]
                    if int(class_id) < len(class_names)
                    else f"class_{class_id}"
                )
                print(f"      [{i + 1}] {class_name}: {conf:.2f}")

        # Save annotated image if supervision is available
        if has_supervision and output_dir:
            image_np = np.array(image)

            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

            labels = [
                f"{class_names[int(cid)] if int(cid) < len(class_names) else f'class_{cid}'} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]

            annotated = box_annotator.annotate(image_np.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)

            out_path = output_dir / f"detected_{img_path.name}"
            Image.fromarray(annotated).save(out_path)
            print(f"    Saved: {out_path}")


def export_model(
    checkpoint_path: Path,
    export_dir: Path,
    class_names: list[str],
    model_size: str = "base",
) -> None:
    """
    Export trained model for deployment.

    Args:
        checkpoint_path: Path to model checkpoint
        export_dir: Output directory for exported model
        class_names: List of class names
        model_size: Model size ('base' or 'large')
    """
    print("\n" + "=" * 60)
    print("EXPORTING MODEL")
    print("=" * 60)

    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint
    export_path = export_dir / "best.pth"
    shutil.copy(checkpoint_path, export_path)
    print(f"\n✓ Model exported to: {export_path}")

    # Save class info
    class_info = {
        "classes": class_names,
        "num_classes": len(class_names),
        "model": f"rf-detr-{model_size}",
        "checkpoint": checkpoint_path.name,
        "exported_at": datetime.now().isoformat(),
    }

    info_path = export_dir / "class_info.json"
    with open(info_path, "w") as f:
        json.dump(class_info, f, indent=2)
    print(f"✓ Class info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RF-DETR on Batman project data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: prepare data + train
  python finetune_rfdetr.py --project data/projects/Test --epochs 50

  # Just prepare the dataset
  python finetune_rfdetr.py --project data/projects/Test --prepare-only

  # Train on existing COCO dataset
  python finetune_rfdetr.py --dataset datasets/my_coco --epochs 50

  # Resume training
  python finetune_rfdetr.py --dataset datasets/my_coco --resume runs/run1/checkpoint.pth

  # Run inference
  python finetune_rfdetr.py --checkpoint runs/run1/best.pth --inference image.jpg

  # Export model
  python finetune_rfdetr.py --checkpoint runs/run1/best.pth --export models/my_model
        """,
    )

    # Input sources (mutually exclusive)
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
        "--train-split",
        type=float,
        default=0.70,
        help="Fraction of data for training (default: 0.70)",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    data_group.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of data for testing (default: 0.15)",
    )
    data_group.add_argument(
        "--video-id",
        type=int,
        default=-1,
        help="Video ID to process (-1 for Roboflow imports, default: -1)",
    )
    data_group.add_argument(
        "--prepare-only", action="store_true", help="Only prepare dataset, don't train"
    )
    data_group.add_argument(
        "--no-clean", action="store_true", help="Don't remove existing dataset directory"
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
        "--model", choices=["base", "large"], default="base", help="Model size (default: base)"
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
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience, 0 to disable (default: 10)",
    )
    train_group.add_argument("--resume", type=Path, help="Resume training from checkpoint")
    train_group.add_argument(
        "--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)"
    )
    train_group.add_argument(
        "--mps-fallback",
        action="store_true",
        help="Enable MPS CPU fallback for unsupported operations",
    )

    # Inference
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--inference", type=Path, nargs="+", help="Run inference on image(s)")
    infer_group.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
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
        "--classes",
        type=str,
        nargs="+",
        help="Class names (required for inference/export without project)",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

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
        dataset_dir, class_names = prepare_dataset(
            project_dir=args.project,
            output_dir=args.output_dataset,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            video_id=args.video_id,
            clean=not args.no_clean,
        )

        if args.prepare_only:
            print("\n✓ Dataset preparation complete (--prepare-only specified)")
            sys.exit(0)

    # === Training ===
    checkpoint_path = args.checkpoint
    if dataset_dir and not args.inference and not args.export:
        checkpoint_path = train_model(
            dataset_dir=dataset_dir,
            output_dir=args.output_dir,
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            lr=args.lr,
            device=args.device,
            num_workers=args.num_workers,
            patience=args.patience,
            resume=str(args.resume) if args.resume else None,
            grad_accum=args.grad_accum,
            use_mps_fallback=args.mps_fallback,
        )

    # === Inference ===
    if args.inference and checkpoint_path:
        if not class_names:
            # Try to load from class_info.json
            info_path = checkpoint_path.parent / "class_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    class_names = json.load(f).get("classes", [])
            else:
                print("Warning: No class names specified, using generic labels")
                class_names = [f"class_{i}" for i in range(100)]

        run_inference(
            checkpoint_path=checkpoint_path,
            image_paths=args.inference,
            class_names=class_names,
            confidence_threshold=args.confidence,
            output_dir=args.inference_output,
            model_size=args.model,
        )

    # === Export ===
    if args.export and checkpoint_path:
        if not class_names:
            info_path = checkpoint_path.parent / "class_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    class_names = json.load(f).get("classes", [])

        export_model(
            checkpoint_path=checkpoint_path,
            export_dir=args.export,
            class_names=class_names,
            model_size=args.model,
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
