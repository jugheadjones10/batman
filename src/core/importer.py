"""Data import utilities for Roboflow and local COCO datasets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.core.project import Project


@dataclass
class ImportStats:
    """Statistics from an import operation."""

    images_imported: int = 0
    annotations_imported: int = 0
    classes_added: list[str] = field(default_factory=list)
    splits_imported: list[str] = field(default_factory=list)


class DataImporter:
    """
    Import datasets from various sources into a Batman project.
    
    Supports:
    - Roboflow datasets (via API)
    - Local COCO-format datasets
    """

    # Video IDs for imported data (negative to distinguish from real videos)
    ROBOFLOW_VIDEO_ID = -1
    LOCAL_COCO_VIDEO_ID = -2

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

        # Process the downloaded dataset
        stats = self._process_coco_dataset(
            dataset_path=dataset_path,
            video_id=self.ROBOFLOW_VIDEO_ID,
            source="roboflow",
            on_progress=on_progress,
            progress_start=30,
            progress_end=90,
        )

        progress("processing", 95, "Cleaning up...")

        # Clean up temp download
        shutil.rmtree(dataset_path.parent, ignore_errors=True)

        progress("complete", 100, f"Imported {stats.images_imported} images with {stats.annotations_imported} annotations")

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

        stats = self._process_coco_dataset(
            dataset_path=coco_path,
            video_id=self.LOCAL_COCO_VIDEO_ID,
            source="local_coco",
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
