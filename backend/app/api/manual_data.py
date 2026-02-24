"""Manual data folder API - scan and serve images from project_root/manual_data/."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from PIL import Image

from backend.app.api.projects import get_project_path, load_project_config, save_project_config

router = APIRouter(prefix="/projects/{project_name}/manual-data", tags=["manual-data"])

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


def _source_key_for_dataset(dataset: str | None) -> str:
    """Return the frames/ directory name (source key) for a manual dataset.

    None or empty string means root-level images -> "manual_data".
    A named dataset "foo" -> "manual_data__foo".
    """
    if not dataset:
        return SOURCE_KEY
    return f"{SOURCE_KEY}__{dataset}"


def _sync_one_dataset(
    project_path: Path,
    image_dir: Path,
    source_key: str,
) -> dict:
    """Sync a single image directory into its frames.json.

    Returns a summary dict with images_found, images_added, images_removed, total.
    Also returns the list of annotation migrations to apply.
    """
    frames_dir = project_path / "frames" / source_key
    frames_meta_path = frames_dir / "frames.json"

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ) if image_dir.exists() else []

    existing_meta: dict = {}
    if frames_meta_path.exists():
        with open(frames_meta_path) as f:
            existing_meta = json.load(f)

    filename_to_entry = {}
    for frame_id, data in existing_meta.items():
        path = Path(data.get("image_path", ""))
        if path.name:
            filename_to_entry[path.name] = (frame_id, data)

    frames_meta = {}
    added = 0
    removed = 0
    migrations: list[tuple[str, str]] = []

    for i, img_path in enumerate(image_files):
        filename = img_path.name
        frame_id = f"{source_key}_{i:06d}"

        if filename in filename_to_entry:
            old_frame_id, old_data = filename_to_entry[filename]
            image_path_full = str(image_dir / filename)
            entry = {
                **old_data,
                "video_id": source_key,
                "frame_number": i,
                "timestamp": 0.0,
                "image_path": image_path_full,
                "source": source_key,
                "original_filename": filename,
            }
            if not entry.get("width") or not entry.get("height"):
                w, h = _get_image_dimensions(img_path)
                entry["width"] = w
                entry["height"] = h
            frames_meta[frame_id] = entry
            if old_frame_id != frame_id:
                migrations.append((old_frame_id, frame_id))
        else:
            width, height = _get_image_dimensions(img_path)
            image_path_full = str(image_dir / filename)
            frames_meta[frame_id] = {
                "video_id": source_key,
                "frame_number": i,
                "timestamp": 0.0,
                "image_path": image_path_full,
                "is_approved": False,
                "needs_review": True,
                "source": source_key,
                "original_filename": filename,
                "width": width,
                "height": height,
            }
            added += 1

    current_filenames = {f.name for f in image_files}
    for frame_id, data in existing_meta.items():
        path = Path(data.get("image_path", ""))
        if path.name and path.name not in current_filenames:
            removed += 1

    frames_dir.mkdir(parents=True, exist_ok=True)
    with open(frames_meta_path, "w") as f:
        json.dump(frames_meta, f, indent=2)

    return {
        "images_found": len(image_files),
        "images_added": added,
        "images_removed": removed,
        "total": len(frames_meta),
        "migrations": migrations,
    }


@router.post("/sync")
async def sync_manual_data(project_name: str):
    """Scan manual_data/ folder (including subdirectories) for images and update frames.

    Root-level images go to frames/manual_data/frames.json (backward compatible).
    Each subdirectory ``manual_data/{name}/`` gets its own
    frames/manual_data__{name}/frames.json with frame IDs prefixed accordingly.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    manual_data_dir = project_path / "manual_data"
    manual_data_dir.mkdir(parents=True, exist_ok=True)

    all_migrations: list[tuple[str, str]] = []
    dataset_results: dict[str, dict] = {}

    # 1) Sync root-level images
    root_result = _sync_one_dataset(project_path, manual_data_dir, SOURCE_KEY)
    all_migrations.extend(root_result.pop("migrations"))
    dataset_results["(root)"] = root_result

    # 2) Sync each subdirectory as a named dataset
    subdirs = sorted(
        d for d in manual_data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    for sub_dir in subdirs:
        source_key = _source_key_for_dataset(sub_dir.name)
        sub_result = _sync_one_dataset(project_path, sub_dir, source_key)
        all_migrations.extend(sub_result.pop("migrations"))
        dataset_results[sub_dir.name] = sub_result

    # 3) Clean up frames dirs for subdatasets that no longer exist on disk
    current_subdir_names = {d.name for d in subdirs}
    frames_root = project_path / "frames"
    if frames_root.exists():
        for frames_sub in frames_root.iterdir():
            if not frames_sub.is_dir():
                continue
            if not frames_sub.name.startswith(f"{SOURCE_KEY}__"):
                continue
            dataset_name = frames_sub.name[len(f"{SOURCE_KEY}__"):]
            if dataset_name not in current_subdir_names:
                logger.info(f"Removing stale frames dir: {frames_sub}")
                shutil.rmtree(frames_sub)

    # Apply annotation migrations
    if all_migrations:
        _apply_annotation_migrations(project_path, all_migrations)

    # Update project config frame count
    config = load_project_config(project_path)
    total_frames = 0
    if frames_root.exists():
        for meta_file in frames_root.rglob("frames.json"):
            with open(meta_file) as f:
                total_frames += len(json.load(f))
    config["frame_count"] = total_frames
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)

    total_found = sum(r["images_found"] for r in dataset_results.values())
    total_added = sum(r["images_added"] for r in dataset_results.values())
    total_removed = sum(r["images_removed"] for r in dataset_results.values())
    total_images = sum(r["total"] for r in dataset_results.values())

    return {
        "images_found": total_found,
        "images_added": total_added,
        "images_removed": total_removed,
        "total": total_images,
        "datasets": dataset_results,
    }


def _apply_annotation_migrations(project_path: Path, migrations: list[tuple[str, str]]):
    """Update annotations: old_frame_id -> new_frame_id."""
    if not migrations:
        return
    ann_path = project_path / "labels" / "current" / "annotations.json"
    if not ann_path.exists():
        return
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


@router.get("/datasets")
async def list_datasets(project_name: str):
    """List manual data datasets (subdirectories of manual_data/).

    Returns dataset names and image counts. The root-level images are
    represented as dataset ``(root)``.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    manual_data_dir = project_path / "manual_data"
    frames_root = project_path / "frames"
    datasets: list[dict] = []

    # Root-level dataset
    root_meta_path = frames_root / SOURCE_KEY / "frames.json"
    root_count = 0
    if root_meta_path.exists():
        with open(root_meta_path) as f:
            root_count = len(json.load(f))
    if root_count > 0:
        datasets.append({"name": "(root)", "source_key": SOURCE_KEY, "image_count": root_count})

    # Subdirectory datasets
    if manual_data_dir.exists():
        for sub_dir in sorted(manual_data_dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            source_key = _source_key_for_dataset(sub_dir.name)
            meta_path = frames_root / source_key / "frames.json"
            count = 0
            if meta_path.exists():
                with open(meta_path) as f:
                    count = len(json.load(f))
            datasets.append({
                "name": sub_dir.name,
                "source_key": source_key,
                "image_count": count,
            })

    return {"datasets": datasets}


@router.get("/images")
async def list_manual_data_images(
    project_name: str,
    dataset: str | None = None,
    offset: int = 0,
    limit: int = 500,
):
    """List manual_data images with annotation counts.

    Optional ``dataset`` query param filters to a specific dataset name.
    Use ``(root)`` for root-level images, or a subdirectory name.
    If omitted, images from all manual datasets are returned.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    frames_root = project_path / "frames"

    # Determine which frames.json files to load
    if dataset is not None:
        source_key = SOURCE_KEY if dataset == "(root)" else _source_key_for_dataset(dataset)
        meta_path = frames_root / source_key / "frames.json"
        if not meta_path.exists():
            return {"total": 0, "offset": offset, "limit": limit, "images": [], "dataset": dataset}
        with open(meta_path) as f:
            frames_meta = json.load(f)
    else:
        # Load all manual datasets
        frames_meta = {}
        if frames_root.exists():
            for sub in sorted(frames_root.iterdir()):
                if not sub.is_dir():
                    continue
                if sub.name != SOURCE_KEY and not sub.name.startswith(f"{SOURCE_KEY}__"):
                    continue
                meta_path = sub / "frames.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        frames_meta.update(json.load(f))

    sorted_items = sorted(
        frames_meta.items(),
        key=lambda x: (x[1].get("source", ""), x[1].get("frame_number", 0)),
    )
    total = len(sorted_items)
    paginated = sorted_items[offset : offset + limit]

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

        source = data.get("source", SOURCE_KEY)
        if source == SOURCE_KEY:
            url = f"/api/projects/{project_name}/manual-data/image/{filename}"
        else:
            ds_name = source[len(f"{SOURCE_KEY}__"):]
            url = f"/api/projects/{project_name}/manual-data/image/{ds_name}/{filename}"

        images.append({
            "filename": filename,
            "frame_id": frame_id,
            "dataset": data.get("source", SOURCE_KEY),
            "width": width,
            "height": height,
            "annotation_count": annotation_counts.get(str(frame_id), 0),
            "url": url,
        })

    result = {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": images,
    }
    if dataset is not None:
        result["dataset"] = dataset
    return result


@router.get("/image/{filename:path}")
async def get_manual_data_image(project_name: str, filename: str):
    """Serve an image file from manual_data/ folder.

    Supports both root-level images (``image.jpg``) and subdataset images
    (``dataset_name/image.jpg``).
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    image_path = project_path / "manual_data" / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Ensure the resolved path is still inside manual_data/
    try:
        image_path.resolve().relative_to((project_path / "manual_data").resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

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
