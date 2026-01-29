"""Data import utilities for Roboflow, COCO Zoo, and local COCO datasets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.core.project import Project


@dataclass
class ImportStats:
    """Statistics from an import operation."""

    images_imported: int = 0
    annotations_imported: int = 0
    classes_added: list[str] = field(default_factory=list)
    splits_imported: list[str] = field(default_factory=list)


# COCO class names (80 classes, 0-indexed)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class DataImporter:
    """
    Import datasets from various sources into a Batman project.

    Supports:
    - Roboflow datasets (via API)
    - COCO Zoo datasets (via FiftyOne)
    - Local COCO-format datasets
    
    Each import gets a unique negative video_id to keep imports separate.
    """

    def __init__(self, project: Project):
        self.project = project

    def import_roboflow(
        self,
        api_key: str,
        workspace: str,
        rf_project: str,
        version: int,
        format: str = "coco",
        on_progress: Callable[[str, int, str], None] | None = None,
    ) -> ImportStats:
        """
        Import a dataset from Roboflow.
        
        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            rf_project: Roboflow project name
            version: Dataset version number
            format: Download format (coco recommended)
            on_progress: Optional callback(status, progress_percent, message)
            
        Returns:
            Import statistics
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "roboflow package not installed. Install with: pip install roboflow"
            )

        def progress(status: str, pct: int, msg: str):
            if on_progress:
                on_progress(status, pct, msg)
            else:
                print(f"[{pct:3d}%] {msg}")

        progress("downloading", 0, "Connecting to Roboflow...")

        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        rf_proj = rf.workspace(workspace).project(rf_project)

        progress("downloading", 10, f"Downloading {workspace}/{rf_project}/v{version}...")

        # Download dataset
        temp_dir = self.project.imports_dir / "roboflow_temp"
        dataset = rf_proj.version(version).download(format, location=str(temp_dir))
        dataset_path = Path(dataset.location)

        progress("processing", 30, "Download complete. Processing images...")

        # Get unique video ID for this import
        video_id = self.project.get_next_import_video_id()

        # Register this import with the project
        import_metadata = {
            "type": "roboflow",
            "workspace": workspace,
            "project": rf_project,
            "version": version,
            "format": format,
            "video_id": video_id,
            "imported_at": datetime.utcnow().isoformat(),
        }
        import_id = self.project.register_import(import_metadata)

        # Process the downloaded dataset
        stats = self._process_coco_dataset(
            dataset_path=dataset_path,
            video_id=video_id,
            source="roboflow",
            import_id=import_id,
            on_progress=on_progress,
            progress_start=30,
            progress_end=90,
        )

        progress("processing", 95, "Cleaning up...")

        # Clean up temp download
        shutil.rmtree(dataset_path.parent, ignore_errors=True)

        progress("complete", 100, f"Imported {stats.images_imported} images with {stats.annotations_imported} annotations")

        return stats

    def import_coco_zoo(
        self,
        classes: list[str],
        split: str = "validation",
        max_samples: int | None = None,
        on_progress: Callable[[str, int, str], None] | None = None,
    ) -> ImportStats:
        """
        Import specific classes from COCO dataset using FiftyOne.

        Args:
            classes: List of COCO class names to import (e.g., ["person", "car"])
            split: Dataset split - "train", "validation", or "test"
            max_samples: Maximum number of samples to import (None for all)
            on_progress: Optional callback(status, progress_percent, message)

        Returns:
            Import statistics
        """
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz
        except ImportError:
            raise ImportError(
                "fiftyone package not installed. Install with: pip install fiftyone"
            )

        def progress(status: str, pct: int, msg: str):
            if on_progress:
                on_progress(status, pct, msg)
            else:
                print(f"[{pct:3d}%] {msg}")

        # Validate classes
        invalid_classes = [c for c in classes if c not in COCO_CLASSES]
        if invalid_classes:
            raise ValueError(
                f"Invalid COCO classes: {invalid_classes}. "
                f"Valid classes: {COCO_CLASSES}"
            )

        progress("downloading", 0, f"Loading COCO {split} with classes: {classes}...")

        # Load from FiftyOne zoo (downloads only needed images)
        dataset_name = f"coco-2017-{split}-{'_'.join(classes)[:30]}-{max_samples or 'all'}"

        # Delete existing dataset with same name if exists
        if dataset_name in fo.list_datasets():
            fo.delete_dataset(dataset_name)

        progress("downloading", 10, "Downloading from COCO dataset zoo...")

        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=split,
            classes=classes,
            max_samples=max_samples,
            dataset_name=dataset_name,
        )

        progress("processing", 40, f"Downloaded {len(dataset)} samples. Processing...")

        # Get unique video ID for this import
        video_id = self.project.get_next_import_video_id()

        # Register this import with the project
        import_metadata = {
            "type": "coco_zoo",
            "classes": classes,
            "split": split,
            "max_samples": max_samples,
            "video_id": video_id,
            "imported_at": datetime.utcnow().isoformat(),
        }
        import_id = self.project.register_import(import_metadata)

        # Process into our format
        stats = self._process_fiftyone_dataset(
            dataset=dataset,
            video_id=video_id,
            source="coco_zoo",
            import_id=import_id,
            on_progress=on_progress,
            progress_start=40,
            progress_end=95,
        )

        progress("processing", 98, "Cleaning up FiftyOne dataset...")

        # Clean up FiftyOne dataset (we've copied the data)
        fo.delete_dataset(dataset_name)

        progress(
            "complete",
            100,
            f"Imported {stats.images_imported} images with {stats.annotations_imported} annotations",
        )

        return stats

    def _process_fiftyone_dataset(
        self,
        dataset,  # fo.Dataset
        video_id: int,
        source: str,
        import_id: str,
        on_progress: Callable[[str, int, str], None] | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
    ) -> ImportStats:
        """Process a FiftyOne dataset into our project format."""
        stats = ImportStats()

        # Load existing project data
        annotations = self.project.load_annotations()
        frames_meta = self.project.load_frames_meta(video_id)
        existing_frame_count = len(frames_meta)

        # Ensure frames directory exists
        frames_dir = self.project.frames_dir / str(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)

        total_samples = len(dataset)
        processed = 0

        for sample in dataset:
            # Get image path and metadata
            img_path = Path(sample.filepath)
            if not img_path.exists():
                processed += 1
                continue

            img_width = sample.metadata.width if sample.metadata else None
            img_height = sample.metadata.height if sample.metadata else None

            # Load image dimensions if not in metadata
            if img_width is None or img_height is None:
                from PIL import Image

                with Image.open(img_path) as img:
                    img_width, img_height = img.size

            # Generate frame ID
            frame_idx = existing_frame_count + stats.images_imported
            frame_id = video_id * 1000000 + frame_idx

            # Copy image to frames directory
            dest_path = frames_dir / f"{frame_id:08d}.jpg"
            shutil.copy(img_path, dest_path)

            # Add frame metadata
            frames_meta[str(frame_id)] = {
                "video_id": video_id,
                "frame_number": frame_idx,
                "timestamp": 0.0,
                "image_path": str(dest_path),
                "is_approved": True,
                "needs_review": False,
                "source": source,
                "import_id": import_id,  # Reference to imports.json
                "original_filename": img_path.name,
                "split": "train",  # FiftyOne doesn't track original split after filtering
            }

            stats.images_imported += 1

            # Process detections
            detections = sample.ground_truth
            if detections is not None:
                for det in detections.detections:
                    # FiftyOne uses normalized [x, y, width, height] format
                    # where x, y is top-left corner
                    x, y, w, h = det.bounding_box

                    # Convert to normalized center format
                    cx = x + w / 2
                    cy = y + h / 2

                    # Get or add class
                    class_name = det.label
                    class_idx = self.project.add_class(class_name, source=source)
                    if class_name not in stats.classes_added:
                        stats.classes_added.append(class_name)

                    # Generate annotation ID
                    ann_id = max([int(k) for k in annotations.keys()], default=0) + 1
                    now = datetime.utcnow()

                    annotations[str(ann_id)] = {
                        "frame_id": frame_id,
                        "class_label_id": class_idx,
                        "track_id": None,
                        "x": cx,
                        "y": cy,
                        "width": w,
                        "height": h,
                        "confidence": 1.0,
                        "source": source,
                        "is_exemplar": False,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }

                    stats.annotations_imported += 1

            processed += 1

            # Report progress
            if on_progress and processed % 50 == 0:
                pct = progress_start + int(
                    (processed / total_samples) * (progress_end - progress_start)
                )
                on_progress("processing", pct, f"Processing images... {processed}/{total_samples}")

        # Save everything
        self.project.save_frames_meta(video_id, frames_meta)
        self.project.save_annotations(annotations)
        self.project.frame_count += stats.images_imported
        self.project.save()

        stats.splits_imported.append("imported")
        return stats

    def import_local_coco(
        self,
        coco_path: Path | str,
        on_progress: Callable[[str, int, str], None] | None = None,
    ) -> ImportStats:
        """
        Import from a local COCO-format dataset.
        
        The path can be:
        - A directory containing train/, valid/, test/ subdirectories
        - A single directory with images and _annotations.coco.json
        
        Args:
            coco_path: Path to COCO dataset directory
            on_progress: Optional callback(status, progress_percent, message)
            
        Returns:
            Import statistics
        """
        coco_path = Path(coco_path)

        def progress(status: str, pct: int, msg: str):
            if on_progress:
                on_progress(status, pct, msg)
            else:
                print(f"[{pct:3d}%] {msg}")

        progress("processing", 0, f"Importing from {coco_path}...")

        # Get unique video ID for this import
        video_id = self.project.get_next_import_video_id()

        # Register this import with the project
        import_metadata = {
            "type": "local_coco",
            "path": str(coco_path.absolute()),
            "video_id": video_id,
            "imported_at": datetime.utcnow().isoformat(),
        }
        import_id = self.project.register_import(import_metadata)

        stats = self._process_coco_dataset(
            dataset_path=coco_path,
            video_id=video_id,
            source="local_coco",
            import_id=import_id,
            on_progress=on_progress,
            progress_start=0,
            progress_end=95,
        )

        progress("complete", 100, f"Imported {stats.images_imported} images with {stats.annotations_imported} annotations")

        return stats

    def _process_coco_dataset(
        self,
        dataset_path: Path,
        video_id: int,
        source: str,
        import_id: str,
        on_progress: Callable[[str, int, str], None] | None = None,
        progress_start: int = 0,
        progress_end: int = 100,
    ) -> ImportStats:
        """
        Process a COCO-format dataset directory.
        
        Handles both:
        - Multi-split datasets (train/, valid/, test/ subdirectories)
        - Single-split datasets (images + _annotations.coco.json in root)
        """
        stats = ImportStats()

        # Load existing project data
        annotations = self.project.load_annotations()
        frames_meta = self.project.load_frames_meta(video_id)
        existing_frame_count = len(frames_meta)

        # Ensure frames directory exists
        frames_dir = self.project.frames_dir / str(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Determine splits to process
        splits_to_process = []

        # Check for multi-split structure
        for split in ["train", "valid", "test", "val"]:
            split_path = dataset_path / split
            if split_path.exists() and (split_path / "_annotations.coco.json").exists():
                splits_to_process.append((split, split_path))

        # Check for single-split structure
        if not splits_to_process:
            coco_file = dataset_path / "_annotations.coco.json"
            if coco_file.exists():
                splits_to_process.append(("data", dataset_path))

        if not splits_to_process:
            raise FileNotFoundError(
                f"No COCO annotations found in {dataset_path}. "
                "Expected _annotations.coco.json in root or train/valid/test subdirectories."
            )

        # Count total images for progress
        total_images = 0
        for split_name, split_path in splits_to_process:
            coco_file = split_path / "_annotations.coco.json"
            with open(coco_file) as f:
                coco_data = json.load(f)
            total_images += len(coco_data.get("images", []))

        if total_images == 0:
            raise ValueError("No images found in dataset")

        processed_images = 0

        # Process each split
        for split_name, split_path in splits_to_process:
            # Normalize split name
            normalized_split = "valid" if split_name == "val" else split_name
            stats.splits_imported.append(normalized_split)

            coco_file = split_path / "_annotations.coco.json"
            with open(coco_file) as f:
                coco_data = json.load(f)

            # Map COCO category IDs to our class indices
            coco_categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
            category_to_class = {}

            for coco_id, class_name in coco_categories.items():
                class_idx = self.project.add_class(class_name, source=source)
                category_to_class[coco_id] = class_idx
                if class_name not in [c for c in stats.classes_added]:
                    # Only count as "added" if it was new to the project
                    if self.project.class_sources.get(class_name) == source:
                        stats.classes_added.append(class_name)

            # Process images
            for coco_img in coco_data.get("images", []):
                img_filename = coco_img["file_name"]
                img_path = split_path / img_filename

                if not img_path.exists():
                    # Try without subdirectory (some COCO exports put images directly)
                    img_path = dataset_path / img_filename
                    if not img_path.exists():
                        processed_images += 1
                        continue

                # Generate frame ID
                frame_idx = existing_frame_count + stats.images_imported
                frame_id = video_id * 1000000 + frame_idx

                # Copy image to frames directory
                dest_path = frames_dir / f"{frame_id:08d}.jpg"
                shutil.copy(img_path, dest_path)

                # Add frame metadata
                frames_meta[str(frame_id)] = {
                    "video_id": video_id,
                    "frame_number": frame_idx,
                    "timestamp": 0.0,
                    "image_path": str(dest_path),
                    "is_approved": True,
                    "needs_review": False,
                    "source": source,
                    "import_id": import_id,  # Reference to imports.json
                    "original_filename": img_filename,
                    "split": normalized_split,
                }

                stats.images_imported += 1
                processed_images += 1

                # Process annotations for this image
                img_width = coco_img["width"]
                img_height = coco_img["height"]

                for ann in coco_data.get("annotations", []):
                    if ann["image_id"] != coco_img["id"]:
                        continue

                    # Convert COCO bbox [x_min, y_min, width, height] to normalized center format
                    x_min, y_min, box_w, box_h = ann["bbox"]

                    cx = (x_min + box_w / 2) / img_width
                    cy = (y_min + box_h / 2) / img_height
                    nw = box_w / img_width
                    nh = box_h / img_height

                    # Generate annotation ID
                    ann_id = max([int(k) for k in annotations.keys()], default=0) + 1
                    now = datetime.utcnow()

                    annotations[str(ann_id)] = {
                        "frame_id": frame_id,
                        "class_label_id": category_to_class[ann["category_id"]],
                        "track_id": None,
                        "x": cx,
                        "y": cy,
                        "width": nw,
                        "height": nh,
                        "confidence": 1.0,
                        "source": source,
                        "is_exemplar": False,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }

                    stats.annotations_imported += 1

                # Report progress
                if on_progress and processed_images % 50 == 0:
                    pct = progress_start + int((processed_images / total_images) * (progress_end - progress_start))
                    on_progress("processing", pct, f"Processing images... {processed_images}/{total_images}")

        # Save everything
        self.project.save_frames_meta(video_id, frames_meta)
        self.project.save_annotations(annotations)
        self.project.frame_count += stats.images_imported
        self.project.save()

        return stats
