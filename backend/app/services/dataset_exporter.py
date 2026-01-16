"""Dataset export service for YOLO and COCO formats."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Literal

from loguru import logger


class DatasetExporter:
    """Exports labeled data to various formats."""

    def __init__(self, project_path: Path):
        self.project_path = project_path

    async def export(
        self,
        frames: list[dict],
        annotations: list[dict],
        classes: list[str],
        format: Literal["yolo", "coco", "both"] = "both",
        output_dir: Path = None,
        split_by_video: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
    ) -> dict:
        """
        Export dataset to specified format.

        Args:
            frames: List of frame dictionaries
            annotations: List of annotation dictionaries
            classes: List of class names
            format: Export format ('yolo', 'coco', or 'both')
            output_dir: Output directory (default: project_path/exports)
            split_by_video: Split train/val/test by video, not frame
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation

        Returns:
            Export result with paths and statistics
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.project_path / "exports" / f"export_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Group frames by video for splitting
        video_frames = {}
        for frame in frames:
            vid_id = frame.get("video_id", 0)
            if vid_id not in video_frames:
                video_frames[vid_id] = []
            video_frames[vid_id].append(frame)

        # Create annotations lookup
        frame_annotations = {}
        for ann in annotations:
            fid = ann["frame_id"]
            if fid not in frame_annotations:
                frame_annotations[fid] = []
            frame_annotations[fid].append(ann)

        # Split videos into train/val/test
        video_ids = list(video_frames.keys())
        n_videos = len(video_ids)
        n_train = max(1, int(n_videos * train_ratio))
        n_val = max(1, int(n_videos * val_ratio))

        train_videos = set(video_ids[:n_train])
        val_videos = set(video_ids[n_train : n_train + n_val])
        test_videos = set(video_ids[n_train + n_val :])

        # If only one video, split frames instead
        if n_videos == 1:
            all_frames = video_frames[video_ids[0]]
            n_frames = len(all_frames)
            n_train_frames = int(n_frames * train_ratio)
            n_val_frames = int(n_frames * val_ratio)

            splits = {
                "train": all_frames[:n_train_frames],
                "val": all_frames[n_train_frames : n_train_frames + n_val_frames],
                "test": all_frames[n_train_frames + n_val_frames :],
            }
        else:
            splits = {
                "train": [f for vid in train_videos for f in video_frames.get(vid, [])],
                "val": [f for vid in val_videos for f in video_frames.get(vid, [])],
                "test": [f for vid in test_videos for f in video_frames.get(vid, [])],
            }

        results = {
            "format": format,
            "output_path": str(output_dir),
            "train_images": len(splits["train"]),
            "val_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "total_annotations": len(annotations),
            "classes": classes,
        }

        if format in ("yolo", "both"):
            await self._export_yolo(
                output_dir / "yolo",
                splits,
                frame_annotations,
                classes,
            )
            results["yolo_path"] = str(output_dir / "yolo")

        if format in ("coco", "both"):
            await self._export_coco(
                output_dir / "coco",
                splits,
                frame_annotations,
                classes,
            )
            results["coco_path"] = str(output_dir / "coco")

        logger.info(f"Dataset exported to {output_dir}")
        return results

    async def _export_yolo(
        self,
        output_dir: Path,
        splits: dict,
        frame_annotations: dict,
        classes: list[str],
    ):
        """Export in YOLO format (txt files with normalized coords)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create data.yaml
        data_yaml = {
            "path": str(output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(classes),
            "names": classes,
        }

        with open(output_dir / "data.yaml", "w") as f:
            import yaml
            yaml.dump(data_yaml, f, default_flow_style=False)

        # Export each split
        for split_name, split_frames in splits.items():
            images_dir = output_dir / "images" / split_name
            labels_dir = output_dir / "labels" / split_name
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for frame in split_frames:
                frame_id = frame["id"]
                image_path = Path(frame["image_path"])

                if not image_path.exists():
                    continue

                # Copy image
                new_image_name = f"{frame_id:08d}.jpg"
                shutil.copy(image_path, images_dir / new_image_name)

                # Create label file
                label_path = labels_dir / f"{frame_id:08d}.txt"
                anns = frame_annotations.get(frame_id, [])

                with open(label_path, "w") as f:
                    for ann in anns:
                        class_id = ann.get("class_label_id", 0)
                        # Handle both formats: nested {"box": {...}} or flat {x, y, width, height}
                        if "box" in ann and isinstance(ann["box"], dict):
                            box = ann["box"]
                        else:
                            box = {
                                "x": ann.get("x", 0),
                                "y": ann.get("y", 0),
                                "width": ann.get("width", 0),
                                "height": ann.get("height", 0),
                            }
                        # YOLO format: class_id center_x center_y width height
                        f.write(
                            f"{class_id} {box['x']:.6f} {box['y']:.6f} "
                            f"{box['width']:.6f} {box['height']:.6f}\n"
                        )

    async def _export_coco(
        self,
        output_dir: Path,
        splits: dict,
        frame_annotations: dict,
        classes: list[str],
    ):
        """Export in COCO JSON format (RF-DETR compatible structure)."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # RF-DETR expects: train/, valid/, test/ with _annotations.coco.json in each
        split_name_map = {"train": "train", "val": "valid", "test": "test"}

        for split_name, split_frames in splits.items():
            coco_split_name = split_name_map.get(split_name, split_name)
            split_dir = output_dir / coco_split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            coco_data = {
                "info": {
                    "description": "Batman Auto-Label Export",
                    "date_created": datetime.now().isoformat(),
                },
                "licenses": [],
                "categories": [
                    {"id": i, "name": name, "supercategory": "object"}
                    for i, name in enumerate(classes)
                ],
                "images": [],
                "annotations": [],
            }

            annotation_id = 1

            for frame in split_frames:
                frame_id = frame["id"]
                image_path = Path(frame["image_path"])

                if not image_path.exists():
                    continue

                # Get image dimensions
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size

                # Copy image to split folder (images alongside annotations)
                new_image_name = f"{frame_id:08d}.jpg"
                shutil.copy(image_path, split_dir / new_image_name)

                # Add image entry
                coco_data["images"].append({
                    "id": frame_id,
                    "file_name": new_image_name,
                    "width": width,
                    "height": height,
                })

                # Add annotations
                anns = frame_annotations.get(frame_id, [])
                for ann in anns:
                    # Handle both formats: nested {"box": {...}} or flat {x, y, width, height}
                    if "box" in ann and isinstance(ann["box"], dict):
                        box = ann["box"]
                    else:
                        box = {
                            "x": ann.get("x", 0),
                            "y": ann.get("y", 0),
                            "width": ann.get("width", 0),
                            "height": ann.get("height", 0),
                        }

                    # Convert normalized center format to COCO format (x, y, w, h in pixels)
                    x = (box["x"] - box["width"] / 2) * width
                    y = (box["y"] - box["height"] / 2) * height
                    w = box["width"] * width
                    h = box["height"] * height

                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": frame_id,
                        "category_id": ann.get("class_label_id", 0),
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    })
                    annotation_id += 1

            # Save COCO JSON - RF-DETR expects _annotations.coco.json
            with open(split_dir / "_annotations.coco.json", "w") as f:
                json.dump(coco_data, f, indent=2)

