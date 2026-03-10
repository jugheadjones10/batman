"""SAM3-based auto-labeling for CLI parity with the backend API."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.core.project import Project


def _get_frame_image_path(project: Project, frame_id: str | int) -> Path | None:
    """Resolve frame_id to image path. Returns None if not found."""
    frames_dir = project.frames_dir
    if not frames_dir.exists():
        return None
    fid_str = str(frame_id)
    for video_dir in frames_dir.iterdir():
        if not video_dir.is_dir():
            continue
        meta = project.load_frames_meta(video_dir.name)
        if fid_str in meta:
            path = Path(meta[fid_str].get("image_path", ""))
            return path if path.exists() else None
    return None


async def _run_sam_on_frame(image_path: Path, class_prompts: list[str]) -> list[dict[str, Any]]:
    """Run SAM3 on a single frame. Uses backend labeler if available."""
    try:
        from backend.app.services.sam_labeler import sam_labeler

        return await sam_labeler.label_frame(image_path, class_prompts, exemplars=None)
    except Exception as e:
        logger.warning(f"SAM3 label_frame failed for {image_path}: {e}")
        return []


def auto_label_frames(
    project: Project,
    frame_ids: list[str | int],
    class_descriptions: dict[str, str] | None = None,
    confidence: float = 0.25,
    skip_labeled: bool = True,
) -> list[dict[str, Any]]:
    """
    Run SAM3 auto-labeling on the given frames.

    Args:
        project: Loaded Project instance.
        frame_ids: List of frame IDs to process.
        class_descriptions: Optional map class_name -> prompt; defaults to class name.
        confidence: Confidence threshold (passed to backend; may not be applied if backend ignores it).
        skip_labeled: If True, skip frames that already have at least one annotation.

    Returns:
        List of created annotation records (with id, frame_id, class_label_id, box, etc.).
    """
    annotations = project.load_annotations()
    labeled_ids = {str(ann.get("frame_id")) for ann in annotations.values()} if skip_labeled else set()
    classes = project.classes
    if not classes:
        logger.warning("No classes defined")
        return []
    class_prompts = [
        (class_descriptions or {}).get(c, c) for c in classes
    ]
    next_id = max((int(k) for k in annotations.keys() if k.isdigit()), default=0) + 1
    created: list[dict[str, Any]] = []

    for frame_id in frame_ids:
        if skip_labeled and str(frame_id) in labeled_ids:
            continue
        image_path = _get_frame_image_path(project, frame_id)
        if not image_path:
            logger.warning(f"No image path for frame_id={frame_id}")
            continue
        detections = asyncio.run(_run_sam_on_frame(image_path, class_prompts))
        for det in detections:
            box = det["box"]
            ann = {
                "frame_id": frame_id,
                "class_label_id": det.get("class_id", 0),
                "track_id": None,
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "confidence": det.get("confidence", 1.0),
                "source": "auto",
                "is_exemplar": False,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            aid = str(next_id)
            annotations[aid] = ann
            created.append({**ann, "id": next_id})
            next_id += 1

    if created:
        project.save_annotations(annotations)
        project.save()  # Persist updated annotation_count to project.json
        logger.info(f"Created {len(created)} annotations on {len(set(a['frame_id'] for a in created))} frames")
    return created
