"""
Core training logic for RF-DETR and YOLO models.

This module contains pure business logic for:
- Dataset preparation (Batman format → COCO format)
- Model training
- Inference
- Model export

Usage:
    from src.core.trainer import RFDETRTrainer, prepare_coco_dataset

    # Prepare dataset
    result = prepare_coco_dataset(
        project_dir=Path("data/projects/Test"),
        output_dir=Path("datasets/my_coco"),
    )

    # Train model
    trainer = RFDETRTrainer(model_size="base")
    checkpoint = trainer.train(dataset_dir, output_dir, epochs=50)

    # Run inference
    detections = trainer.predict(image_path)
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass
class DatasetStats:
    """Statistics about a prepared dataset."""

    train_images: int = 0
    train_annotations: int = 0
    val_images: int = 0
    val_annotations: int = 0
    test_images: int = 0
    test_annotations: int = 0
    class_names: list[str] = field(default_factory=list)
    output_dir: Path | None = None


@dataclass
class TrainingResult:
    """Result of a training run."""

    checkpoint_path: Path
    training_time_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "completed"


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 50
    batch_size: int = 8
    image_size: int = 640
    lr: float = 1e-4
    device: str = "auto"
    num_workers: int = 2
    patience: int = 10
    grad_accum: int = 1
    resume: str | None = None


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
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info(device: str) -> dict[str, Any]:
    """Get information about the selected device."""
    info = {"device": device}

    if device == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif device == "mps":
        info["name"] = "Apple Silicon MPS"
    else:
        info["name"] = "CPU"

    return info


def load_project_data(
    project_dir: Path, video_id: int | str | None = None
) -> tuple[dict, dict, list[str], dict]:
    """
    Load project data from Batman project directory.

    Args:
        project_dir: Path to project directory
        video_id: Video ID(s) to load frames from:
            - None or "all": Load all frames from all video directories
            - "imports": Load only imported datasets (negative video IDs)
            - int: Load from specific video ID

    Returns:
        Tuple of (frames_meta, annotations_data, class_names, project_config)

    Raises:
        FileNotFoundError: If required files don't exist
    """
    frames_base_dir = project_dir / "frames"
    annotations_file = project_dir / "labels" / "current" / "annotations.json"
    project_config_file = project_dir / "project.json"

    # Verify paths exist
    if not frames_base_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_base_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    if not project_config_file.exists():
        raise FileNotFoundError(f"Project config not found: {project_config_file}")

    # Load project config
    with open(project_config_file) as f:
        project_config = json.load(f)
    class_names = project_config.get("classes", [])

    # Determine which video directories to load
    video_dirs_to_load = []
    
    if video_id is None or video_id == "all":
        # Load all video directories
        for video_dir in frames_base_dir.iterdir():
            if video_dir.is_dir() and (video_dir / "frames.json").exists():
                video_dirs_to_load.append(video_dir)
    elif video_id == "imports":
        # Load only imports (negative video IDs)
        for video_dir in frames_base_dir.iterdir():
            if video_dir.is_dir() and (video_dir / "frames.json").exists():
                try:
                    vid = int(video_dir.name)
                    if vid < 0:
                        video_dirs_to_load.append(video_dir)
                except ValueError:
                    pass
    else:
        # Load specific video ID
        video_dir = frames_base_dir / str(video_id)
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        video_dirs_to_load.append(video_dir)
    
    if not video_dirs_to_load:
        raise FileNotFoundError(f"No video directories found in {frames_base_dir}")

    # Load and merge frames metadata from all directories
    frames_meta = {}
    for video_dir in video_dirs_to_load:
        frames_meta_file = video_dir / "frames.json"
        with open(frames_meta_file) as f:
            dir_frames = json.load(f)
            frames_meta.update(dir_frames)

    # Load annotations
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    return frames_meta, annotations_data, class_names, project_config


def create_coco_split(
    frame_ids: set[str],
    frames_meta: dict,
    annotations_data: dict,
    class_names: list[str],
    output_dir: Path,
    original_class_names: list[str] | None = None,
) -> tuple[int, int]:
    """
    Create a COCO format dataset split from Batman internal format.

    COCO format:
    - images: list of {id, file_name, width, height}
    - annotations: list of {id, image_id, category_id, bbox, area}
    - categories: list of {id, name}

    Note: COCO bbox format is [x_min, y_min, width, height] in pixels

    Args:
        frame_ids: Set of frame IDs to include
        frames_meta: Frame metadata dict
        annotations_data: Annotations dict
        class_names: List of class names to include (filtered if filter_classes was used)
        output_dir: Output directory
        original_class_names: Original full list of class names (for ID remapping)

    Returns:
        Tuple of (num_images, num_annotations)
    """
    coco_data: dict[str, list] = {"images": [], "annotations": [], "categories": []}

    # Build class ID mapping (original ID -> new filtered ID)
    if original_class_names is None:
        original_class_names = class_names

    # Map original class index -> new class index (0-indexed for RF-DETR compatibility)
    # Note: Standard COCO uses 1-indexed, but RF-DETR expects 0-indexed categories
    class_id_map = {}
    for new_idx, name in enumerate(class_names):
        original_idx = original_class_names.index(name) if name in original_class_names else new_idx
        class_id_map[original_idx] = new_idx  # 0-indexed for RF-DETR

    # Create categories (0-indexed for RF-DETR)
    for i, name in enumerate(class_names):
        coco_data["categories"].append({"id": i, "name": name, "supercategory": "object"})

    annotation_id = 1

    for frame_id in frame_ids:
        if frame_id not in frames_meta:
            continue

        frame_info = frames_meta[frame_id]
        src_path = Path(frame_info["image_path"])

        if not src_path.exists():
            continue

        # Get image dimensions
        with Image.open(src_path) as img:
            img_width, img_height = img.size

        # Get annotations for this frame
        frame_annotations = []
        for ann in annotations_data.values():
            if str(ann["frame_id"]) != frame_id:
                continue

            # Check if this annotation's class is in our filtered set
            original_class_id = ann["class_label_id"]
            if original_class_id not in class_id_map:
                continue

            # Convert normalized center format to COCO format
            cx = ann["x"] * img_width
            cy = ann["y"] * img_height
            w = ann["width"] * img_width
            h = ann["height"] * img_height

            x_min = cx - w / 2
            y_min = cy - h / 2

            category_id = class_id_map[original_class_id]

            frame_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(frame_id),
                    "category_id": category_id,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        # Only include images with annotations
        if frame_annotations:
            # Copy image
            new_filename = f"{frame_id}.jpg"
            dst_path = output_dir / new_filename
            shutil.copy(src_path, dst_path)

            # Add image entry
            coco_data["images"].append(
                {
                    "id": int(frame_id),
                    "file_name": new_filename,
                    "width": img_width,
                    "height": img_height,
                }
            )

            coco_data["annotations"].extend(frame_annotations)

    # Save COCO annotations
    coco_file = output_dir / "_annotations.coco.json"
    with open(coco_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return len(coco_data["images"]), len(coco_data["annotations"])


def prepare_coco_dataset(
    project_dir: Path,
    output_dir: Path,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    video_id: int | str | None = "imports",
    clean: bool = True,
    filter_classes: list[str] | None = None,
    seed: int = 42,
) -> DatasetStats:
    """
    Prepare COCO format dataset from Batman project.

    Args:
        project_dir: Path to Batman project
        output_dir: Output directory for COCO dataset
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        video_id: Video ID(s) to process: 'all', 'imports' (default), or specific ID
        clean: Whether to remove existing output directory
        filter_classes: If specified, only include these classes (by name)
        seed: Random seed for reproducible splits

    Returns:
        DatasetStats with counts and class names

    Raises:
        FileNotFoundError: If project files don't exist
        ValueError: If filter_classes contains unknown class names
    """
    set_seed(seed)

    # Load project data
    frames_meta, annotations_data, original_class_names, _ = load_project_data(
        project_dir, video_id
    )

    # Apply class filtering if specified
    if filter_classes:
        invalid_classes = [c for c in filter_classes if c not in original_class_names]
        if invalid_classes:
            raise ValueError(
                f"Unknown classes: {invalid_classes}. Available: {original_class_names}"
            )
        class_names = filter_classes
    else:
        class_names = original_class_names

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Split frames
    all_frame_ids = list(frames_meta.keys())
    random.shuffle(all_frame_ids)

    n_total = len(all_frame_ids)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_frame_ids = set(all_frame_ids[:n_train])
    val_frame_ids = set(all_frame_ids[n_train : n_train + n_val])
    test_frame_ids = set(all_frame_ids[n_train + n_val :])

    # Create datasets
    train_images, train_anns = create_coco_split(
        train_frame_ids,
        frames_meta,
        annotations_data,
        class_names,
        train_dir,
        original_class_names,
    )

    val_images, val_anns = create_coco_split(
        val_frame_ids,
        frames_meta,
        annotations_data,
        class_names,
        val_dir,
        original_class_names,
    )

    test_images, test_anns = create_coco_split(
        test_frame_ids,
        frames_meta,
        annotations_data,
        class_names,
        test_dir,
        original_class_names,
    )

    return DatasetStats(
        train_images=train_images,
        train_annotations=train_anns,
        val_images=val_images,
        val_annotations=val_anns,
        test_images=test_images,
        test_annotations=test_anns,
        class_names=class_names,
        output_dir=output_dir,
    )


class RFDETRTrainer:
    """
    RF-DETR model trainer.

    Handles training, inference, and export of RF-DETR models.
    """

    def __init__(self, model_size: str = "base", checkpoint: Path | None = None):
        """
        Initialize trainer.

        Args:
            model_size: Model size ('nano', 'small', 'base', 'medium', 'large')
            checkpoint: Optional path to pre-trained checkpoint
        """
        self.model_size = model_size
        self.checkpoint = checkpoint
        self._model = None

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the RF-DETR model based on size."""
        if self.model_size == "large":
            from rfdetr import RFDETRLarge as Model
        elif self.model_size == "medium":
            from rfdetr import RFDETRMedium as Model
        elif self.model_size == "small":
            from rfdetr import RFDETRSmall as Model
        elif self.model_size == "nano":
            from rfdetr import RFDETRNano as Model
        else:
            from rfdetr import RFDETRBase as Model

        if self.checkpoint:
            return Model(pretrain_weights=str(self.checkpoint))
        return Model()

    def train(
        self,
        dataset_dir: Path,
        output_dir: Path,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """
        Train RF-DETR model.

        Args:
            dataset_dir: Path to COCO format dataset
            output_dir: Output directory for training run
            config: Training configuration

        Returns:
            TrainingResult with checkpoint path and metrics
        """
        import time

        if config is None:
            config = TrainingConfig()

        device = get_device(config.device)

        # Resolution must be divisible by 56 for RF-DETR
        resolution = (config.image_size // 56) * 56
        if resolution < 280:
            resolution = 280

        # Build training arguments
        train_kwargs = {
            "dataset_dir": str(dataset_dir),
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "resolution": resolution,
            "output_dir": str(output_dir),
            "num_workers": config.num_workers,
            "early_stopping": config.patience > 0,
            "early_stopping_patience": config.patience,
        }

        if config.grad_accum > 1:
            train_kwargs["grad_accum_steps"] = config.grad_accum

        if config.resume:
            train_kwargs["resume"] = config.resume

        # Only add device if not mps
        if device != "mps":
            train_kwargs["device"] = device

        # Train
        start_time = time.time()
        self.model.train(**train_kwargs)
        training_time = time.time() - start_time

        # Find best checkpoint
        checkpoint_path = self._find_best_checkpoint(output_dir)

        return TrainingResult(
            checkpoint_path=checkpoint_path,
            training_time_seconds=training_time,
            status="completed",
        )

    def _find_best_checkpoint(self, output_dir: Path) -> Path:
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

    def predict(self, image: Path | Image.Image | np.ndarray, threshold: float = 0.5) -> Any:
        """
        Run inference on an image.

        Args:
            image: Image path, PIL Image, or numpy array
            threshold: Detection confidence threshold

        Returns:
            Detections object from RF-DETR
        """
        if isinstance(image, Path):
            image = Image.open(image)

        return self.model.predict(image, threshold=threshold)

    def export(
        self,
        export_dir: Path,
        class_names: list[str],
    ) -> Path:
        """
        Export trained model for deployment.

        Args:
            export_dir: Output directory for exported model
            class_names: List of class names

        Returns:
            Path to exported model
        """
        if not self.checkpoint:
            raise ValueError("No checkpoint to export. Train or load a model first.")

        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint
        export_path = export_dir / "best.pth"
        shutil.copy(self.checkpoint, export_path)

        # Save class info
        class_info = {
            "classes": class_names,
            "num_classes": len(class_names),
            "model": f"rf-detr-{self.model_size}",
            "checkpoint": self.checkpoint.name,
            "exported_at": datetime.now().isoformat(),
        }

        info_path = export_dir / "class_info.json"
        with open(info_path, "w") as f:
            json.dump(class_info, f, indent=2)

        return export_path


def measure_latency(
    checkpoint_path: Path,
    image_size: int = 640,
    warmup_runs: int = 5,
    test_runs: int = 20,
) -> float:
    """
    Measure inference latency on a dummy image.

    Args:
        checkpoint_path: Path to model checkpoint
        image_size: Image size for inference
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs

    Returns:
        Average latency in milliseconds
    """
    import time

    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=str(checkpoint_path))
    dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup_runs):
        model.predict(dummy_img)

    # Measure
    times = []
    for _ in range(test_runs):
        start = time.perf_counter()
        model.predict(dummy_img)
        times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times))
