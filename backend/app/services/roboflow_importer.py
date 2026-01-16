"""Service for importing datasets from Roboflow."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator

from loguru import logger


class RoboflowImporter:
    """Import datasets from Roboflow into the project."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.frames_dir = project_path / "frames"
        self.labels_dir = project_path / "labels" / "current"
        
    async def import_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        format: str = "coco",
    ) -> dict:
        """
        Import a Roboflow dataset into the project.
        
        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Roboflow project name  
            version: Dataset version number
            format: Download format (coco recommended)
            
        Returns:
            Import statistics
        """
        from roboflow import Roboflow
        
        logger.info(f"Importing Roboflow dataset: {workspace}/{project}/v{version}")
        
        # Initialize Roboflow
        rf = Roboflow(api_key=api_key)
        rf_project = rf.workspace(workspace).project(project)
        dataset = rf_project.version(version).download(format, location=str(self.project_path / "imports" / "roboflow_temp"))
        
        dataset_path = Path(dataset.location)
        
        # Track stats
        stats = {
            "images_imported": 0,
            "annotations_imported": 0,
            "classes_added": [],
            "splits_imported": [],
        }
        
        # Load existing project config
        config_path = self.project_path / "project.json"
        with open(config_path) as f:
            config = json.load(f)
        
        existing_classes = config.get("classes", [])
        # Track class sources: {"ClassName": "manual" | "roboflow" | "local_coco"}
        class_sources = config.get("class_sources", {})
        # Mark existing classes as manual if not tracked
        for cls in existing_classes:
            if cls not in class_sources:
                class_sources[cls] = "manual"
        
        # Load existing annotations
        annotations_path = self.labels_dir / "annotations.json"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        if annotations_path.exists():
            with open(annotations_path) as f:
                annotations_meta = json.load(f)
        else:
            annotations_meta = {}
        
        # Use a special video_id for imported data (negative to distinguish from real videos)
        import_video_id = -1  # Roboflow imports get video_id = -1
        
        # Check for existing import frames
        import_frames_dir = self.frames_dir / str(import_video_id)
        import_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing frames meta or create new
        frames_meta_path = import_frames_dir / "frames.json"
        if frames_meta_path.exists():
            with open(frames_meta_path) as f:
                frames_meta = json.load(f)
            existing_frame_count = len(frames_meta)
        else:
            frames_meta = {}
            existing_frame_count = 0
        
        # Process each split (train, valid, test)
        for split in ["train", "valid", "test"]:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
                
            stats["splits_imported"].append(split)
            
            # Find annotation file
            coco_file = split_path / "_annotations.coco.json"
            if not coco_file.exists():
                logger.warning(f"No COCO annotations found in {split}")
                continue
            
            with open(coco_file) as f:
                coco_data = json.load(f)
            
            # Map COCO category IDs to our class indices
            coco_categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
            category_to_class = {}
            
            for coco_id, class_name in coco_categories.items():
                if class_name not in existing_classes:
                    existing_classes.append(class_name)
                    class_sources[class_name] = "roboflow"
                    stats["classes_added"].append(class_name)
                category_to_class[coco_id] = existing_classes.index(class_name)
            
            # Map COCO image IDs to file names
            coco_images = {img["id"]: img for img in coco_data.get("images", [])}
            
            # Process images
            for coco_img in coco_data.get("images", []):
                img_filename = coco_img["file_name"]
                img_path = split_path / img_filename
                
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    continue
                
                # Generate frame ID
                frame_idx = existing_frame_count + stats["images_imported"]
                frame_id = import_video_id * 1000000 + frame_idx
                
                # Copy image to frames directory
                dest_path = import_frames_dir / f"{frame_id:08d}.jpg"
                shutil.copy(img_path, dest_path)
                
                # Add frame metadata
                frames_meta[str(frame_id)] = {
                    "video_id": import_video_id,
                    "frame_number": frame_idx,
                    "timestamp": 0.0,  # No timestamp for imported images
                    "image_path": str(dest_path),
                    "is_approved": True,  # Roboflow data is pre-approved
                    "needs_review": False,
                    "source": "roboflow",
                    "original_filename": img_filename,
                    "split": split,
                }
                
                stats["images_imported"] += 1
                
                # Find annotations for this image
                img_width = coco_img["width"]
                img_height = coco_img["height"]
                
                for ann in coco_data.get("annotations", []):
                    if ann["image_id"] != coco_img["id"]:
                        continue
                    
                    # Convert COCO bbox [x_min, y_min, width, height] to normalized center format
                    x_min, y_min, box_w, box_h = ann["bbox"]
                    
                    # Normalize and convert to center format
                    cx = (x_min + box_w / 2) / img_width
                    cy = (y_min + box_h / 2) / img_height
                    nw = box_w / img_width
                    nh = box_h / img_height
                    
                    # Generate annotation ID
                    ann_id = max([int(k) for k in annotations_meta.keys()], default=0) + 1
                    now = datetime.utcnow()
                    
                    annotations_meta[str(ann_id)] = {
                        "frame_id": frame_id,
                        "class_label_id": category_to_class[ann["category_id"]],
                        "track_id": None,
                        "x": cx,
                        "y": cy,
                        "width": nw,
                        "height": nh,
                        "confidence": 1.0,
                        "source": "roboflow",
                        "is_exemplar": False,
                        "exemplar_type": None,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                    
                    stats["annotations_imported"] += 1
        
        # Save frames metadata
        with open(frames_meta_path, "w") as f:
            json.dump(frames_meta, f, indent=2)
        
        # Save annotations
        with open(annotations_path, "w") as f:
            json.dump(annotations_meta, f, indent=2)
        
        # Update project config
        config["classes"] = existing_classes
        config["class_sources"] = class_sources
        config["frame_count"] = config.get("frame_count", 0) + stats["images_imported"]
        config["annotation_count"] = len(annotations_meta)
        config["updated_at"] = datetime.utcnow().isoformat()
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Clean up temp download
        shutil.rmtree(dataset_path.parent, ignore_errors=True)
        
        logger.info(f"Import complete: {stats['images_imported']} images, {stats['annotations_imported']} annotations")
        
        return stats
    
    async def import_dataset_with_progress(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        format: str = "coco",
    ) -> AsyncGenerator[dict, None]:
        """
        Import a Roboflow dataset with progress updates.
        
        Yields progress updates as dict with:
        - status: "downloading" | "processing" | "complete" | "error"
        - progress: 0-100 percentage
        - message: Human-readable message
        - For complete: includes full stats
        """
        from roboflow import Roboflow
        
        logger.info(f"Importing Roboflow dataset: {workspace}/{project}/v{version}")
        
        yield {
            "status": "downloading",
            "progress": 0,
            "message": "Connecting to Roboflow..."
        }
        
        # Initialize Roboflow and download
        rf = Roboflow(api_key=api_key)
        rf_project = rf.workspace(workspace).project(project)
        
        yield {
            "status": "downloading",
            "progress": 10,
            "message": f"Downloading {workspace}/{project}/v{version}..."
        }
        
        dataset = rf_project.version(version).download(
            format, 
            location=str(self.project_path / "imports" / "roboflow_temp")
        )
        
        dataset_path = Path(dataset.location)
        
        yield {
            "status": "processing",
            "progress": 30,
            "message": "Download complete. Processing images..."
        }
        
        # Track stats
        stats = {
            "images_imported": 0,
            "annotations_imported": 0,
            "classes_added": [],
            "splits_imported": [],
        }
        
        # Load existing project config
        config_path = self.project_path / "project.json"
        with open(config_path) as f:
            config = json.load(f)
        
        existing_classes = config.get("classes", [])
        class_sources = config.get("class_sources", {})
        for cls in existing_classes:
            if cls not in class_sources:
                class_sources[cls] = "manual"
        
        # Load existing annotations
        annotations_path = self.labels_dir / "annotations.json"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        if annotations_path.exists():
            with open(annotations_path) as f:
                annotations_meta = json.load(f)
        else:
            annotations_meta = {}
        
        import_video_id = -1
        import_frames_dir = self.frames_dir / str(import_video_id)
        import_frames_dir.mkdir(parents=True, exist_ok=True)
        
        frames_meta_path = import_frames_dir / "frames.json"
        if frames_meta_path.exists():
            with open(frames_meta_path) as f:
                frames_meta = json.load(f)
            existing_frame_count = len(frames_meta)
        else:
            frames_meta = {}
            existing_frame_count = 0
        
        # Count total images first for progress calculation
        total_images = 0
        for split in ["train", "valid", "test"]:
            split_path = dataset_path / split
            if split_path.exists():
                coco_file = split_path / "_annotations.coco.json"
                if coco_file.exists():
                    with open(coco_file) as f:
                        coco_data = json.load(f)
                    total_images += len(coco_data.get("images", []))
        
        if total_images == 0:
            yield {
                "status": "error",
                "progress": 100,
                "message": "No images found in dataset"
            }
            return
        
        processed_images = 0
        
        # Process each split (train, valid, test)
        for split in ["train", "valid", "test"]:
            split_path = dataset_path / split
            if not split_path.exists():
                continue
                
            stats["splits_imported"].append(split)
            
            coco_file = split_path / "_annotations.coco.json"
            if not coco_file.exists():
                logger.warning(f"No COCO annotations found in {split}")
                continue
            
            with open(coco_file) as f:
                coco_data = json.load(f)
            
            # Map COCO category IDs to our class indices
            coco_categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
            category_to_class = {}
            
            for coco_id, class_name in coco_categories.items():
                if class_name not in existing_classes:
                    existing_classes.append(class_name)
                    class_sources[class_name] = "roboflow"
                    stats["classes_added"].append(class_name)
                category_to_class[coco_id] = existing_classes.index(class_name)
            
            coco_images = {img["id"]: img for img in coco_data.get("images", [])}
            
            # Process images
            for coco_img in coco_data.get("images", []):
                img_filename = coco_img["file_name"]
                img_path = split_path / img_filename
                
                if not img_path.exists():
                    logger.warning(f"Image not found: {img_path}")
                    processed_images += 1
                    continue
                
                frame_idx = existing_frame_count + stats["images_imported"]
                frame_id = import_video_id * 1000000 + frame_idx
                
                dest_path = import_frames_dir / f"{frame_id:08d}.jpg"
                shutil.copy(img_path, dest_path)
                
                frames_meta[str(frame_id)] = {
                    "video_id": import_video_id,
                    "frame_number": frame_idx,
                    "timestamp": 0.0,
                    "image_path": str(dest_path),
                    "is_approved": True,
                    "needs_review": False,
                    "source": "roboflow",
                    "original_filename": img_filename,
                    "split": split,
                }
                
                stats["images_imported"] += 1
                processed_images += 1
                
                # Find annotations for this image
                img_width = coco_img["width"]
                img_height = coco_img["height"]
                
                for ann in coco_data.get("annotations", []):
                    if ann["image_id"] != coco_img["id"]:
                        continue
                    
                    x_min, y_min, box_w, box_h = ann["bbox"]
                    cx = (x_min + box_w / 2) / img_width
                    cy = (y_min + box_h / 2) / img_height
                    nw = box_w / img_width
                    nh = box_h / img_height
                    
                    ann_id = max([int(k) for k in annotations_meta.keys()], default=0) + 1
                    now = datetime.utcnow()
                    
                    annotations_meta[str(ann_id)] = {
                        "frame_id": frame_id,
                        "class_label_id": category_to_class[ann["category_id"]],
                        "track_id": None,
                        "x": cx,
                        "y": cy,
                        "width": nw,
                        "height": nh,
                        "confidence": 1.0,
                        "source": "roboflow",
                        "is_exemplar": False,
                        "exemplar_type": None,
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    }
                    
                    stats["annotations_imported"] += 1
                
                # Yield progress every 10 images or at the end
                if processed_images % 10 == 0 or processed_images == total_images:
                    # Progress: 30% for download, 60% for processing (30-90), 10% for saving
                    processing_progress = 30 + int((processed_images / total_images) * 60)
                    yield {
                        "status": "processing",
                        "progress": processing_progress,
                        "message": f"Processing images... {processed_images}/{total_images}",
                        "images_processed": processed_images,
                        "total_images": total_images,
                    }
        
        yield {
            "status": "processing",
            "progress": 92,
            "message": "Saving metadata..."
        }
        
        # Save frames metadata
        with open(frames_meta_path, "w") as f:
            json.dump(frames_meta, f, indent=2)
        
        # Save annotations
        with open(annotations_path, "w") as f:
            json.dump(annotations_meta, f, indent=2)
        
        # Update project config
        config["classes"] = existing_classes
        config["class_sources"] = class_sources
        config["frame_count"] = config.get("frame_count", 0) + stats["images_imported"]
        config["annotation_count"] = len(annotations_meta)
        config["updated_at"] = datetime.utcnow().isoformat()
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        yield {
            "status": "processing",
            "progress": 97,
            "message": "Cleaning up..."
        }
        
        # Clean up temp download
        shutil.rmtree(dataset_path.parent, ignore_errors=True)
        
        logger.info(f"Import complete: {stats['images_imported']} images, {stats['annotations_imported']} annotations")
        
        yield {
            "status": "complete",
            "progress": 100,
            "message": f"Imported {stats['images_imported']} images with {stats['annotations_imported']} annotations",
            "images_imported": stats["images_imported"],
            "annotations_imported": stats["annotations_imported"],
            "classes_added": stats["classes_added"],
            "splits_imported": stats["splits_imported"],
        }
    
    async def import_from_local_coco(
        self,
        coco_dir: Path,
        split: str = "train",
    ) -> dict:
        """
        Import from a local COCO-format dataset directory.
        
        Args:
            coco_dir: Path to directory containing images and _annotations.coco.json
            split: Name of this split (for metadata)
            
        Returns:
            Import statistics
        """
        logger.info(f"Importing local COCO dataset from {coco_dir}")
        
        coco_file = coco_dir / "_annotations.coco.json"
        if not coco_file.exists():
            raise FileNotFoundError(f"COCO annotations not found: {coco_file}")
        
        stats = {
            "images_imported": 0,
            "annotations_imported": 0,
            "classes_added": [],
        }
        
        # Load project config
        config_path = self.project_path / "project.json"
        with open(config_path) as f:
            config = json.load(f)
        
        existing_classes = config.get("classes", [])
        class_sources = config.get("class_sources", {})
        for cls in existing_classes:
            if cls not in class_sources:
                class_sources[cls] = "manual"
        
        # Load annotations
        annotations_path = self.labels_dir / "annotations.json"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        if annotations_path.exists():
            with open(annotations_path) as f:
                annotations_meta = json.load(f)
        else:
            annotations_meta = {}
        
        # Import frames directory
        import_video_id = -2  # Local COCO imports get video_id = -2
        import_frames_dir = self.frames_dir / str(import_video_id)
        import_frames_dir.mkdir(parents=True, exist_ok=True)
        
        frames_meta_path = import_frames_dir / "frames.json"
        if frames_meta_path.exists():
            with open(frames_meta_path) as f:
                frames_meta = json.load(f)
            existing_frame_count = len(frames_meta)
        else:
            frames_meta = {}
            existing_frame_count = 0
        
        # Load COCO data
        with open(coco_file) as f:
            coco_data = json.load(f)
        
        # Map categories
        coco_categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
        category_to_class = {}
        
        for coco_id, class_name in coco_categories.items():
            if class_name not in existing_classes:
                existing_classes.append(class_name)
                class_sources[class_name] = "local_coco"
                stats["classes_added"].append(class_name)
            category_to_class[coco_id] = existing_classes.index(class_name)
        
        # Process images
        for coco_img in coco_data.get("images", []):
            img_filename = coco_img["file_name"]
            img_path = coco_dir / img_filename
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            frame_idx = existing_frame_count + stats["images_imported"]
            frame_id = import_video_id * 1000000 + frame_idx
            
            dest_path = import_frames_dir / f"{frame_id:08d}.jpg"
            shutil.copy(img_path, dest_path)
            
            frames_meta[str(frame_id)] = {
                "video_id": import_video_id,
                "frame_number": frame_idx,
                "timestamp": 0.0,
                "image_path": str(dest_path),
                "is_approved": True,
                "needs_review": False,
                "source": "local_coco",
                "original_filename": img_filename,
                "split": split,
            }
            
            stats["images_imported"] += 1
            
            img_width = coco_img["width"]
            img_height = coco_img["height"]
            
            for ann in coco_data.get("annotations", []):
                if ann["image_id"] != coco_img["id"]:
                    continue
                
                x_min, y_min, box_w, box_h = ann["bbox"]
                cx = (x_min + box_w / 2) / img_width
                cy = (y_min + box_h / 2) / img_height
                nw = box_w / img_width
                nh = box_h / img_height
                
                ann_id = max([int(k) for k in annotations_meta.keys()], default=0) + 1
                now = datetime.utcnow()
                
                annotations_meta[str(ann_id)] = {
                    "frame_id": frame_id,
                    "class_label_id": category_to_class[ann["category_id"]],
                    "track_id": None,
                    "x": cx,
                    "y": cy,
                    "width": nw,
                    "height": nh,
                    "confidence": 1.0,
                    "source": "local_coco",
                    "is_exemplar": False,
                    "exemplar_type": None,
                    "created_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                }
                
                stats["annotations_imported"] += 1
        
        # Save
        with open(frames_meta_path, "w") as f:
            json.dump(frames_meta, f, indent=2)
        
        with open(annotations_path, "w") as f:
            json.dump(annotations_meta, f, indent=2)
        
        config["classes"] = existing_classes
        config["class_sources"] = class_sources
        config["frame_count"] = config.get("frame_count", 0) + stats["images_imported"]
        config["annotation_count"] = len(annotations_meta)
        config["updated_at"] = datetime.utcnow().isoformat()
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Import complete: {stats['images_imported']} images, {stats['annotations_imported']} annotations")
        
        return stats

