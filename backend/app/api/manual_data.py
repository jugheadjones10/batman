"""Manual data folder API - scan and serve images from project_root/manual_data/."""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from PIL import Image

from backend.app.api.projects import get_project_path, load_project_config, save_project_config

router = APIRouter(prefix="/projects/{project_name}/manual-data", tags=["manual-data"])

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE_KEY = "manual_data"


def _get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get width and height of an image using Pillow."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.warning(f"Could not read image dimensions for {image_path}: {e}")
        return 0, 0


@router.post("/sync")
async def sync_manual_data(project_name: str):
    """
    Scan manual_data/ folder for images and update frames.json.

    Creates or updates frames/manual_data/frames.json with entries for each
    image file. Images remain in manual_data/; frames.json stores metadata
    and paths for annotation/training.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    manual_data_dir = project_path / "manual_data"
    frames_dir = project_path / "frames" / SOURCE_KEY
    frames_meta_path = frames_dir / "frames.json"

    # Create manual_data if it doesn't exist (empty folder)
    manual_data_dir.mkdir(parents=True, exist_ok=True)

    # Discover image files (sorted for stable ordering)
    image_files = sorted(
        f for f in manual_data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    # Load existing frames.json to preserve metadata for unchanged files
    existing_meta: dict = {}
    if frames_meta_path.exists():
        with open(frames_meta_path) as f:
            existing_meta = json.load(f)

    # Build filename -> existing entry for quick lookup
    filename_to_entry = {}
    for frame_id, data in existing_meta.items():
        path = Path(data.get("image_path", ""))
        if path.name:
            filename_to_entry[path.name] = (frame_id, data)

    frames_meta = {}
    added = 0
    removed = 0
    migrations: list[tuple[str, str]] = []  # (old_frame_id, new_frame_id)

    for i, image_path in enumerate(image_files):
        filename = image_path.name
        frame_id = f"{SOURCE_KEY}_{i:06d}"

        if filename in filename_to_entry:
            old_frame_id, old_data = filename_to_entry[filename]
            # Preserve existing entry, update image_path if needed
            image_path_full = str(manual_data_dir / filename)
            entry = {
                **old_data,
                "video_id": SOURCE_KEY,
                "frame_number": i,
                "timestamp": 0.0,
                "image_path": image_path_full,
                "source": SOURCE_KEY,
                "original_filename": filename,
            }
            if not entry.get("width") or not entry.get("height"):
                w, h = _get_image_dimensions(image_path)
                entry["width"] = w
                entry["height"] = h
            frames_meta[frame_id] = entry
            if old_frame_id != frame_id:
                migrations.append((old_frame_id, frame_id))
        else:
            width, height = _get_image_dimensions(image_path)
            image_path_full = str(manual_data_dir / filename)
            frames_meta[frame_id] = {
                "video_id": SOURCE_KEY,
                "frame_number": i,
                "timestamp": 0.0,
                "image_path": image_path_full,
                "is_approved": False,
                "needs_review": True,
                "source": SOURCE_KEY,
                "original_filename": filename,
                "width": width,
                "height": height,
            }
            added += 1

    # Count removed (files that were in frames.json but no longer on disk)
    for frame_id, data in existing_meta.items():
        path = Path(data.get("image_path", ""))
        if path.name and path.name not in [f.name for f in image_files]:
            removed += 1

    frames_dir.mkdir(parents=True, exist_ok=True)
    with open(frames_meta_path, "w") as f:
        json.dump(frames_meta, f, indent=2)

    # Apply annotation migrations in a single pass (order matters when IDs swap)
    if migrations:
        _apply_annotation_migrations(project_path, migrations)

    # Update project config frame count (sum all frames across all sources)
    config = load_project_config(project_path)
    total_frames = 0
    frames_root = project_path / "frames"
    if frames_root.exists():
        for meta_file in frames_root.rglob("frames.json"):
            with open(meta_file) as f:
                total_frames += len(json.load(f))
    config["frame_count"] = total_frames
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)

    return {
        "images_found": len(image_files),
        "images_added": added,
        "images_removed": removed,
        "total": len(frames_meta),
    }


def _apply_annotation_migrations(project_path: Path, migrations: list[tuple[str, str]]):
    """
    Update annotations: old_frame_id -> new_frame_id.
    Handles reordering correctly by building a full mapping first.
    """
    if not migrations:
        return
    ann_path = project_path / "labels" / "current" / "annotations.json"
    if not ann_path.exists():
        return
    # Build mapping: may need multiple hops (e.g. A->B, B->A)
    mapping: dict[str, str] = dict(migrations)
    with open(ann_path) as f:
        data = json.load(f)
    changed = False
    for ann in data.values():
        fid = str(ann.get("frame_id"))
        if fid in mapping:
            ann["frame_id"] = mapping[fid]
            changed = True
    if changed:
        with open(ann_path, "w") as f:
            json.dump(data, f, indent=2)


@router.get("/images")
async def list_manual_data_images(
    project_name: str,
    offset: int = 0,
    limit: int = 500,
):
    """
    List manual_data images with annotation counts.

    Returns paginated list with filename, frame_id, dimensions, annotation_count, url.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    frames_meta_path = project_path / "frames" / SOURCE_KEY / "frames.json"
    if not frames_meta_path.exists():
        return {"total": 0, "offset": offset, "limit": limit, "images": []}

    with open(frames_meta_path) as f:
        frames_meta = json.load(f)

    # Sort by frame_number for stable ordering
    sorted_items = sorted(
        frames_meta.items(),
        key=lambda x: x[1].get("frame_number", 0),
    )
    total = len(sorted_items)
    paginated = sorted_items[offset : offset + limit]

    # Load annotation counts
    annotations_path = project_path / "labels" / "current" / "annotations.json"
    annotation_counts: dict[str, int] = {}
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        for ann in annotations.values():
            fid = str(ann.get("frame_id"))
            annotation_counts[fid] = annotation_counts.get(fid, 0) + 1

    images = []
    for frame_id, data in paginated:
        filename = data.get("original_filename", Path(data.get("image_path", "")).name)
        width = data.get("width", 0)
        height = data.get("height", 0)
        if not width or not height:
            img_path = Path(data.get("image_path", ""))
            if not img_path.is_absolute():
                img_path = project_path / img_path
            if img_path.exists():
                w, h = _get_image_dimensions(img_path)
                width, height = w, h
        url = f"/api/projects/{project_name}/manual-data/image/{filename}"
        images.append({
            "filename": filename,
            "frame_id": frame_id,
            "width": width,
            "height": height,
            "annotation_count": annotation_counts.get(str(frame_id), 0),
            "url": url,
        })

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": images,
    }


@router.get("/image/{filename}")
async def get_manual_data_image(project_name: str, filename: str):
    """Serve an image file from manual_data/ folder."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Security: ensure filename doesn't escape the directory
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    image_path = project_path / "manual_data" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine media type from extension
    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    media_type = media_types.get(ext, "image/jpeg")

    return FileResponse(image_path, media_type=media_type)
