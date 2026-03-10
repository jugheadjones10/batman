"""Manual data folder API - scan and serve images from project_root/manual_data/."""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from PIL import Image

from backend.app.api.projects import get_project_path, load_project_config, save_project_config

router = APIRouter(prefix="/projects/{project_name}/manual-data", tags=["manual-data"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE_KEY = "manual_data"


def _sanitize_dataset_name(name: str) -> str:
    """Sanitize subdirectory name for filesystem (alnum, ._- space, +)."""
    if not name or not name.strip():
        return ""
    s = "".join(c for c in name.strip() if c.isalnum() or c in "._- +").strip()
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _sanitize_filename(name: str) -> str:
    """Sanitize filename: keep alnum, ._- and one extension."""
    base = Path(name).stem
    ext = (Path(name).suffix or "").lower()
    safe_base = "".join(c for c in base if c.isalnum() or c in "._- ").strip() or "image"
    safe_base = re.sub(r"\s+", "_", safe_base)
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"
    return safe_base + ext


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


@router.patch("/datasets/{dataset_name}")
async def rename_dataset(project_name: str, dataset_name: str, body: dict):
    """Rename a manual-data subdirectory dataset.

    Body: ``{"new_name": "..."}``.
    Renames the folder on disk, migrates frame IDs, and updates annotations.
    Cannot rename the root dataset ``(root)``.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if dataset_name == "(root)":
        raise HTTPException(status_code=400, detail="Cannot rename the root dataset")

    new_name_raw: str = body.get("new_name", "").strip()
    new_name = _sanitize_dataset_name(new_name_raw)
    if not new_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    if new_name == dataset_name:
        return {"name": dataset_name, "message": "No change"}

    manual_data_dir = project_path / "manual_data"
    old_dir = manual_data_dir / dataset_name
    new_dir = manual_data_dir / new_name

    if not old_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    if new_dir.exists():
        raise HTTPException(status_code=409, detail=f"A dataset named '{new_name}' already exists")

    # Rename the images directory
    old_dir.rename(new_dir)

    old_source_key = _source_key_for_dataset(dataset_name)
    new_source_key = _source_key_for_dataset(new_name)
    frames_root = project_path / "frames"
    old_frames_dir = frames_root / old_source_key
    new_frames_dir = frames_root / new_source_key

    migrations: list[tuple[str, str]] = []

    if old_frames_dir.exists():
        # Load old frames.json and build new one with updated frame_ids/paths
        old_meta_path = old_frames_dir / "frames.json"
        with open(old_meta_path) as f:
            old_meta: dict = json.load(f)

        new_meta: dict = {}
        for old_id, data in old_meta.items():
            # Frame IDs look like: manual_data__oldname_000042
            if old_id.startswith(f"{old_source_key}_"):
                suffix = old_id[len(old_source_key):]  # e.g. "_000042"
                new_id = f"{new_source_key}{suffix}"
            else:
                new_id = old_id  # shouldn't happen, keep as-is

            # Update paths pointing into the old directory
            old_image_path = Path(data.get("image_path", ""))
            try:
                rel = old_image_path.relative_to(old_dir)
                new_image_path = str(new_dir / rel)
            except ValueError:
                new_image_path = data.get("image_path", "")

            new_meta[new_id] = {
                **data,
                "video_id": new_source_key,
                "source": new_source_key,
                "image_path": new_image_path,
            }
            if new_id != old_id:
                migrations.append((old_id, new_id))

        # Write new frames dir
        new_frames_dir.mkdir(parents=True, exist_ok=True)
        with open(new_frames_dir / "frames.json", "w") as f:
            json.dump(new_meta, f, indent=2)

        # Remove old frames dir
        shutil.rmtree(old_frames_dir)

    # Migrate annotations
    if migrations:
        _apply_annotation_migrations(project_path, migrations)

    # Update project config
    config = load_project_config(project_path)
    total_frames = 0
    if frames_root.exists():
        for meta_file in frames_root.rglob("frames.json"):
            with open(meta_file) as f:
                total_frames += len(json.load(f))
    config["frame_count"] = total_frames
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)

    return {
        "old_name": dataset_name,
        "new_name": new_name,
        "frames_migrated": len(migrations),
    }


@router.post("/upload")
async def upload_manual_data_images(
    project_name: str,
    dataset: str | None = None,
    files: list[UploadFile] = File(...),
):
    """Upload image files into manual_data/ or manual_data/{dataset}/, then sync.

    Optional query param ``dataset`` is the subdirectory name (e.g. "crane_closeups").
    If omitted, images are stored in the root manual_data/ folder.
    Accepts multiple files via multipart/form-data (field name: files).
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    manual_data_dir = project_path / "manual_data"
    manual_data_dir.mkdir(parents=True, exist_ok=True)

    if dataset is not None and dataset.strip():
        safe_dataset = _sanitize_dataset_name(dataset)
        if not safe_dataset:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        target_dir = manual_data_dir / safe_dataset
    else:
        safe_dataset = ""
        target_dir = manual_data_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    seen_stem_index: dict[str, int] = {}  # stem -> last used index (0 = original name)
    saved: list[str] = []

    for upload in files:
        if not upload.filename:
            continue
        ext = Path(upload.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            logger.warning(f"Skipping non-image file: {upload.filename}")
            continue
        safe_name = _sanitize_filename(upload.filename)
        stem, suf = Path(safe_name).stem, Path(safe_name).suffix
        idx = seen_stem_index.get(stem, -1)
        idx += 1
        seen_stem_index[stem] = idx
        final_name = f"{stem}_{idx}{suf}" if idx else safe_name
        dest_path = target_dir / final_name
        while dest_path.exists():
            idx += 1
            seen_stem_index[stem] = idx
            final_name = f"{stem}_{idx}{suf}"
            dest_path = target_dir / final_name
        try:
            content = await upload.read()
            dest_path.write_bytes(content)
            saved.append(dest_path.name)
        except Exception as e:
            logger.exception(f"Failed to save {upload.filename}")
            raise HTTPException(status_code=500, detail=f"Failed to save {upload.filename}: {e}")

    if not saved:
        return {
            "uploaded": 0,
            "dataset": dataset or "(root)",
            "filenames": [],
            "sync": None,
        }

    # Run sync so new images appear in frames.json
    sync_result = await sync_manual_data(project_name)
    return {
        "uploaded": len(saved),
        "dataset": safe_dataset or "(root)",
        "filenames": saved,
        "sync": sync_result,
    }


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


@router.delete("/images/{frame_id}")
async def delete_manual_data_image(project_name: str, frame_id: str):
    """Delete a single manual-data image by its frame_id.

    Removes the file from disk, drops it from the relevant frames.json, and
    removes any annotations that reference this frame.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    frames_root = project_path / "frames"

    # Locate which frames.json contains this frame_id
    target_meta_path: Path | None = None
    frame_data: dict | None = None

    if frames_root.exists():
        for sub in frames_root.iterdir():
            if not sub.is_dir():
                continue
            if sub.name != SOURCE_KEY and not sub.name.startswith(f"{SOURCE_KEY}__"):
                continue
            meta_path = sub / "frames.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if frame_id in meta:
                target_meta_path = meta_path
                frame_data = meta[frame_id]
                break

    if target_meta_path is None or frame_data is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete the file from disk
    image_path = Path(frame_data.get("image_path", ""))
    if image_path.exists():
        image_path.unlink()

    # Remove from frames.json
    with open(target_meta_path) as f:
        meta = json.load(f)
    del meta[frame_id]
    with open(target_meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Remove annotations referencing this frame
    annotations_deleted = 0
    ann_path = project_path / "labels" / "current" / "annotations.json"
    if ann_path.exists():
        with open(ann_path) as f:
            annotations = json.load(f)
        keys_to_delete = [k for k, v in annotations.items() if str(v.get("frame_id")) == frame_id]
        for k in keys_to_delete:
            del annotations[k]
            annotations_deleted += 1
        if keys_to_delete:
            with open(ann_path, "w") as f:
                json.dump(annotations, f, indent=2)

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

    return {
        "deleted": True,
        "frame_id": frame_id,
        "annotations_deleted": annotations_deleted,
    }


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
