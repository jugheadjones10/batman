"""Import API routes for external datasets."""

import asyncio
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config, save_project_config
from src.core.project import Project
from src.core.importer import DataImporter

router = APIRouter(prefix="/projects/{project_name}/import", tags=["import"])

# Thread pool for running sync code in async context
_executor = ThreadPoolExecutor(max_workers=2)


class RoboflowImportRequest(BaseModel):
    """Request to import a Roboflow dataset."""

    api_key: str
    workspace: str
    project: str
    version: int
    format: str = "coco"


class LocalCocoImportRequest(BaseModel):
    """Request to import a local COCO dataset."""

    path: str  # Path to COCO directory


class CocoZooImportRequest(BaseModel):
    """Request to import from COCO dataset zoo via FiftyOne."""

    classes: list[str]  # e.g., ["person", "car"]
    split: str = "validation"  # "train", "validation", or "test"
    max_samples: int | None = None  # None for all matching images


class ImportResult(BaseModel):
    """Result of a dataset import."""

    images_imported: int
    annotations_imported: int
    classes_added: list[str]
    splits_imported: list[str] = []
    message: str


@router.post("/roboflow", response_model=ImportResult)
async def import_from_roboflow(
    project_name: str,
    request: RoboflowImportRequest,
):
    """
    Import a dataset from Roboflow.

    Requires a Roboflow API key. Get one at https://app.roboflow.com/settings/api

    Example:
        POST /api/projects/MyProject/import/roboflow
        {
            "api_key": "your_api_key",
            "workspace": "your-workspace",
            "project": "your-project",
            "version": 1
        }
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        # Load project using shared core
        project = Project.load(project_path)
        importer = DataImporter(project)

        # Run sync import in thread pool
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            _executor,
            lambda: importer.import_roboflow(
                api_key=request.api_key,
                workspace=request.workspace,
                rf_project=request.project,
                version=request.version,
                format=request.format,
            ),
        )

        return ImportResult(
            images_imported=stats.images_imported,
            annotations_imported=stats.annotations_imported,
            classes_added=stats.classes_added,
            splits_imported=stats.splits_imported,
            message=f"Successfully imported {stats.images_imported} images with {stats.annotations_imported} annotations",
        )
    except ImportError as e:
        logger.error(f"Roboflow import failed - missing dependency: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Roboflow import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/roboflow/stream")
async def import_from_roboflow_stream(
    project_name: str,
    request: RoboflowImportRequest,
):
    """
    Import a dataset from Roboflow with streaming progress updates.

    Returns Server-Sent Events with progress information.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    async def generate_progress() -> AsyncGenerator[str, None]:
        progress_updates: list[dict] = []
        import_complete = False
        import_error: str | None = None
        final_stats = None

        def on_progress(status: str, pct: int, msg: str):
            progress_updates.append(
                {"status": status, "progress": pct, "message": msg}
            )

        def run_import():
            nonlocal import_complete, import_error, final_stats
            try:
                project = Project.load(project_path)
                importer = DataImporter(project)
                stats = importer.import_roboflow(
                    api_key=request.api_key,
                    workspace=request.workspace,
                    rf_project=request.project,
                    version=request.version,
                    format=request.format,
                    on_progress=on_progress,
                )
                final_stats = stats
                import_complete = True
            except Exception as e:
                import_error = str(e)
                logger.error(f"Roboflow import failed: {e}")

        # Start import in background thread
        loop = asyncio.get_event_loop()
        import_task = loop.run_in_executor(_executor, run_import)

        # Stream progress updates
        last_sent = 0
        while not import_complete and not import_error:
            await asyncio.sleep(0.1)

            # Send any new progress updates
            while last_sent < len(progress_updates):
                yield f"data: {json.dumps(progress_updates[last_sent])}\n\n"
                last_sent += 1

        # Wait for task to complete
        await import_task

        # Send final status
        if import_error:
            yield f"data: {json.dumps({'status': 'error', 'progress': 100, 'message': import_error})}\n\n"
        elif final_stats:
            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'message': f'Imported {final_stats.images_imported} images', 'images_imported': final_stats.images_imported, 'annotations_imported': final_stats.annotations_imported, 'classes_added': final_stats.classes_added, 'splits_imported': final_stats.splits_imported})}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/coco-zoo", response_model=ImportResult)
async def import_from_coco_zoo(
    project_name: str,
    request: CocoZooImportRequest,
):
    """
    Import specific classes from COCO dataset using FiftyOne.

    This downloads only the images that contain the specified classes,
    not the entire COCO dataset.

    Example:
        POST /api/projects/MyProject/import/coco-zoo
        {
            "classes": ["person", "car"],
            "split": "validation",
            "max_samples": 500
        }
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        # Load project using shared core
        project = Project.load(project_path)
        importer = DataImporter(project)

        # Run sync import in thread pool
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            _executor,
            lambda: importer.import_coco_zoo(
                classes=request.classes,
                split=request.split,
                max_samples=request.max_samples,
            ),
        )

        return ImportResult(
            images_imported=stats.images_imported,
            annotations_imported=stats.annotations_imported,
            classes_added=stats.classes_added,
            splits_imported=stats.splits_imported,
            message=f"Successfully imported {stats.images_imported} images with {stats.annotations_imported} annotations from COCO",
        )
    except ImportError as e:
        logger.error(f"COCO Zoo import failed - missing dependency: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"COCO Zoo import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/coco-zoo/stream")
async def import_from_coco_zoo_stream(
    project_name: str,
    request: CocoZooImportRequest,
):
    """
    Import from COCO dataset zoo with streaming progress updates.

    Returns Server-Sent Events with progress information.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    async def generate_progress() -> AsyncGenerator[str, None]:
        progress_updates: list[dict] = []
        import_complete = False
        import_error: str | None = None
        final_stats = None

        def on_progress(status: str, pct: int, msg: str):
            progress_updates.append(
                {"status": status, "progress": pct, "message": msg}
            )

        def run_import():
            nonlocal import_complete, import_error, final_stats
            try:
                project = Project.load(project_path)
                importer = DataImporter(project)
                stats = importer.import_coco_zoo(
                    classes=request.classes,
                    split=request.split,
                    max_samples=request.max_samples,
                    on_progress=on_progress,
                )
                final_stats = stats
                import_complete = True
            except Exception as e:
                import_error = str(e)
                logger.error(f"COCO Zoo import failed: {e}")

        # Start import in background thread
        loop = asyncio.get_event_loop()
        import_task = loop.run_in_executor(_executor, run_import)

        # Stream progress updates
        last_sent = 0
        while not import_complete and not import_error:
            await asyncio.sleep(0.1)

            # Send any new progress updates
            while last_sent < len(progress_updates):
                yield f"data: {json.dumps(progress_updates[last_sent])}\n\n"
                last_sent += 1

        # Wait for task to complete
        await import_task

        # Send final status
        if import_error:
            yield f"data: {json.dumps({'status': 'error', 'progress': 100, 'message': import_error})}\n\n"
        elif final_stats:
            yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'message': f'Imported {final_stats.images_imported} images', 'images_imported': final_stats.images_imported, 'annotations_imported': final_stats.annotations_imported, 'classes_added': final_stats.classes_added, 'splits_imported': final_stats.splits_imported})}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/local-coco", response_model=ImportResult)
async def import_from_local_coco(
    project_name: str,
    request: LocalCocoImportRequest,
):
    """
    Import a dataset from a local COCO-format directory.

    The directory can contain:
    - train/, valid/, test/ subdirectories with _annotations.coco.json each
    - Or a single directory with images and _annotations.coco.json

    Example:
        POST /api/projects/MyProject/import/local-coco
        {
            "path": "/path/to/coco/dataset"
        }
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    coco_dir = Path(request.path)
    if not coco_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.path}")

    try:
        # Load project using shared core
        project = Project.load(project_path)
        importer = DataImporter(project)

        # Run sync import in thread pool
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            _executor,
            lambda: importer.import_local_coco(coco_path=coco_dir),
        )

        return ImportResult(
            images_imported=stats.images_imported,
            annotations_imported=stats.annotations_imported,
            classes_added=stats.classes_added,
            splits_imported=stats.splits_imported,
            message=f"Successfully imported {stats.images_imported} images with {stats.annotations_imported} annotations",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Local COCO import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ImportedDatasetInfo(BaseModel):
    """Info about an imported dataset."""

    video_id: str | int  # source_key (e.g. roboflow_crane-hook_1) or legacy -1, -2
    source: str
    image_count: int
    annotation_count: int
    sample_images: list[str]


@router.get("/datasets")
async def list_imported_datasets(project_name: str):
    """List all imported datasets (frame dirs that are not in videos.json)."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    frames_dir = project_path / "frames"
    annotations_path = project_path / "labels" / "current" / "annotations.json"

    # Load videos.json to exclude uploaded videos (video_1, video_2 or 1, 2)
    videos_meta = {}
    videos_json = project_path / "videos" / "videos.json"
    if videos_json.exists():
        with open(videos_json) as f:
            videos_meta = json.load(f)

    config = load_project_config(project_path)
    classes = config.get("classes", [])

    # Build frame_id -> dir_name for annotation counts (frame_id may be int or str)
    frame_to_video: dict[str, str] = {}
    annotations_by_video: dict[str, int] = {}
    classes_by_video: dict[str, set[int]] = {}
    if frames_dir.exists() and annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        for video_dir in frames_dir.iterdir():
            if not video_dir.is_dir() or video_dir.name in videos_meta:
                continue
            meta_path = video_dir / "frames.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                frames_meta = json.load(f)
            for frame_id in frames_meta.keys():
                frame_to_video[str(frame_id)] = video_dir.name
        for ann in annotations.values():
            fid = str(ann.get("frame_id"))
            vid = frame_to_video.get(fid)
            if vid is not None:
                annotations_by_video[vid] = annotations_by_video.get(vid, 0) + 1
                class_id = ann.get("class_label_id", 0)
                classes_by_video.setdefault(vid, set()).add(class_id)

    # Load imports metadata for source_key -> source type
    imports_metadata = {}
    imports_path = project_path / "imports" / "imports.json"
    if imports_path.exists():
        with open(imports_path) as f:
            imports_metadata = json.load(f)
    video_id_to_source = {}
    for import_id, import_meta in imports_metadata.items():
        vid = import_meta.get("source_key") or import_meta.get("video_id")
        if vid is not None:
            video_id_to_source[str(vid)] = import_meta.get("type", "unknown")

    datasets = []
    if not frames_dir.exists():
        return datasets

    for video_dir in frames_dir.iterdir():
        if not video_dir.is_dir() or video_dir.name in videos_meta:
            continue
        if video_dir.name == "manual_data" or video_dir.name.startswith("manual_data__"):
            continue
        meta_path = video_dir / "frames.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            frames_meta = json.load(f)

        source = video_id_to_source.get(video_dir.name)
        if not source and frames_meta:
            first_frame = next(iter(frames_meta.values()))
            source = first_frame.get("source", "unknown")

        image_files = [
            f for f in video_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
        sample_files = random.sample(image_files, min(6, len(image_files)))
        sample_urls = [
            f"/api/projects/{project_name}/import/image/{video_dir.name}/{f.name}"
            for f in sample_files
        ]

        vid_out = int(video_dir.name) if video_dir.name.lstrip("-").isdigit() else video_dir.name
        class_ids = sorted(classes_by_video.get(video_dir.name, set()))
        class_names = [classes[cid] if cid < len(classes) else f"class_{cid}" for cid in class_ids]
        datasets.append({
            "video_id": vid_out,
            "source": source or "unknown",
            "image_count": len(frames_meta),
            "annotation_count": annotations_by_video.get(video_dir.name, 0),
            "sample_images": sample_urls,
            "classes": class_names,
        })

    return datasets


@router.get("/image/{video_id}/{filename}")
async def get_imported_image(project_name: str, video_id: str, filename: str):
    """Get an imported image file (video_id is source_key or legacy -1, -2)."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    image_path = project_path / "frames" / video_id / filename
    if not image_path.exists():
        if video_id == "manual_data":
            image_path = project_path / "manual_data" / filename
        elif video_id.startswith("manual_data__"):
            dataset_name = video_id[len("manual_data__"):]
            image_path = project_path / "manual_data" / dataset_name / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

    suffix = image_path.suffix.lower()
    media_type = {
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(suffix, "image/jpeg")

    return FileResponse(image_path, media_type=media_type)


@router.get("/images/{video_id}")
async def list_imported_images(
    project_name: str,
    video_id: str,
    offset: int = 0,
    limit: int = 50,
):
    """List images in an imported dataset with pagination."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    frames_dir = project_path / "frames" / video_id
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    meta_path = frames_dir / "frames.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Dataset metadata not found")

    with open(meta_path) as f:
        frames_meta = json.load(f)

    # Sort by frame_id (string or numeric)
    def sort_key(item):
        k = item[0]
        return (0, int(k)) if isinstance(k, str) and k.isdigit() else (1, k)

    sorted_frames = sorted(frames_meta.items(), key=sort_key)
    total = len(sorted_frames)
    paginated = sorted_frames[offset : offset + limit]

    images = []
    for frame_id, frame_data in paginated:
        fname = Path(frame_data["image_path"]).name
        images.append({
            "frame_id": frame_id,
            "url": f"/api/projects/{project_name}/import/image/{video_id}/{fname}",
            "original_filename": frame_data.get("original_filename", ""),
            "split": frame_data.get("split", ""),
        })

    return {"total": total, "offset": offset, "limit": limit, "images": images}


@router.delete("/datasets/{video_id}")
async def delete_imported_dataset(project_name: str, video_id: str):
    """Delete an imported dataset (video_id is source_key or legacy -1, -2)."""
    import shutil

    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Reject uploaded video dirs (video_1, video_2 or numeric >= 0)
    if video_id.startswith("video_") or (video_id.lstrip("-").isdigit() and int(video_id) >= 0):
        raise HTTPException(
            status_code=400, detail="Can only delete imported datasets, not uploaded videos"
        )

    frames_dir = project_path / "frames" / video_id
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    meta_path = frames_dir / "frames.json"
    frame_ids = set()
    if meta_path.exists():
        with open(meta_path) as f:
            frames_meta = json.load(f)
        frame_ids = set(frames_meta.keys())

    annotations_path = project_path / "labels" / "current" / "annotations.json"
    deleted_annotations = 0
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)

        new_annotations = {}
        for ann_id, ann_data in annotations.items():
            if str(ann_data.get("frame_id")) not in frame_ids:
                new_annotations[ann_id] = ann_data
            else:
                deleted_annotations += 1

        with open(annotations_path, "w") as f:
            json.dump(new_annotations, f, indent=2)

    # Delete frames directory
    shutil.rmtree(frames_dir)

    # Update project config
    config = load_project_config(project_path)
    config["frame_count"] = max(0, config.get("frame_count", 0) - len(frame_ids))
    config["annotation_count"] = max(0, config.get("annotation_count", 0) - deleted_annotations)
    save_project_config(project_path, config)

    return {
        "message": f"Deleted {len(frame_ids)} images and {deleted_annotations} annotations",
        "images_deleted": len(frame_ids),
        "annotations_deleted": deleted_annotations,
    }
