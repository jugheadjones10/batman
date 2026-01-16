"""Import API routes for external datasets."""

import json
import random
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config, save_project_config
from backend.app.services.roboflow_importer import RoboflowImporter

router = APIRouter(prefix="/projects/{project_name}/import", tags=["import"])


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
    split: str = "train"


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
        importer = RoboflowImporter(project_path)
        stats = await importer.import_dataset(
            api_key=request.api_key,
            workspace=request.workspace,
            project=request.project,
            version=request.version,
            format=request.format,
        )
        
        return ImportResult(
            images_imported=stats["images_imported"],
            annotations_imported=stats["annotations_imported"],
            classes_added=stats["classes_added"],
            splits_imported=stats["splits_imported"],
            message=f"Successfully imported {stats['images_imported']} images with {stats['annotations_imported']} annotations",
        )
    except Exception as e:
        logger.error(f"Roboflow import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/local-coco", response_model=ImportResult)
async def import_from_local_coco(
    project_name: str,
    request: LocalCocoImportRequest,
):
    """
    Import a dataset from a local COCO-format directory.
    
    The directory should contain:
    - _annotations.coco.json
    - Image files referenced in the annotations
    
    Example:
        POST /api/projects/MyProject/import/local-coco
        {
            "path": "/path/to/coco/dataset/train",
            "split": "train"
        }
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    coco_dir = Path(request.path)
    if not coco_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.path}")
    
    try:
        importer = RoboflowImporter(project_path)
        stats = await importer.import_from_local_coco(
            coco_dir=coco_dir,
            split=request.split,
        )
        
        return ImportResult(
            images_imported=stats["images_imported"],
            annotations_imported=stats["annotations_imported"],
            classes_added=stats["classes_added"],
            message=f"Successfully imported {stats['images_imported']} images with {stats['annotations_imported']} annotations",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Local COCO import failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ImportedDatasetInfo(BaseModel):
    """Info about an imported dataset."""
    video_id: int  # -1 for roboflow, -2 for local_coco
    source: str  # "roboflow" | "local_coco"
    image_count: int
    annotation_count: int
    sample_images: list[str]  # URLs to sample images


@router.get("/datasets")
async def list_imported_datasets(project_name: str):
    """List all imported datasets in this project."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    frames_dir = project_path / "frames"
    annotations_path = project_path / "labels" / "current" / "annotations.json"
    
    # Load annotations to count per-dataset
    annotations_by_video = {}
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        # Count annotations by frame's video_id
        frame_to_video = {}
        for video_dir in frames_dir.iterdir():
            if not video_dir.is_dir():
                continue
            try:
                video_id = int(video_dir.name)
            except ValueError:
                continue
            
            meta_path = video_dir / "frames.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    frames_meta = json.load(f)
                for frame_id in frames_meta.keys():
                    frame_to_video[int(frame_id)] = video_id
        
        for ann in annotations.values():
            frame_id = ann.get("frame_id")
            video_id = frame_to_video.get(frame_id, 0)
            if video_id < 0:  # Only count imported datasets
                annotations_by_video[video_id] = annotations_by_video.get(video_id, 0) + 1
    
    datasets = []
    
    # Check for imported datasets (negative video IDs)
    source_map = {-1: "roboflow", -2: "local_coco"}
    
    for video_dir in frames_dir.iterdir():
        if not video_dir.is_dir():
            continue
        try:
            video_id = int(video_dir.name)
        except ValueError:
            continue
        
        if video_id >= 0:
            continue  # Skip regular videos
        
        meta_path = video_dir / "frames.json"
        if not meta_path.exists():
            continue
        
        with open(meta_path) as f:
            frames_meta = json.load(f)
        
        # Get sample images (up to 6)
        image_files = [
            f for f in video_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ]
        sample_files = random.sample(image_files, min(6, len(image_files)))
        sample_urls = [
            f"/api/projects/{project_name}/import/image/{video_id}/{f.name}"
            for f in sample_files
        ]
        
        datasets.append({
            "video_id": video_id,
            "source": source_map.get(video_id, "unknown"),
            "image_count": len(frames_meta),
            "annotation_count": annotations_by_video.get(video_id, 0),
            "sample_images": sample_urls,
        })
    
    return datasets


@router.get("/image/{video_id}/{filename}")
async def get_imported_image(project_name: str, video_id: int, filename: str):
    """Get an imported image file."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    image_path = project_path / "frames" / str(video_id) / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/jpeg")


@router.get("/images/{video_id}")
async def list_imported_images(
    project_name: str, 
    video_id: int,
    offset: int = 0,
    limit: int = 50,
):
    """List images in an imported dataset with pagination."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    frames_dir = project_path / "frames" / str(video_id)
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    meta_path = frames_dir / "frames.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Dataset metadata not found")
    
    with open(meta_path) as f:
        frames_meta = json.load(f)
    
    # Sort by frame_id
    sorted_frames = sorted(frames_meta.items(), key=lambda x: int(x[0]))
    total = len(sorted_frames)
    
    # Paginate
    paginated = sorted_frames[offset:offset + limit]
    
    images = []
    for frame_id, frame_data in paginated:
        images.append({
            "frame_id": int(frame_id),
            "url": f"/api/projects/{project_name}/import/image/{video_id}/{Path(frame_data['image_path']).name}",
            "original_filename": frame_data.get("original_filename", ""),
            "split": frame_data.get("split", ""),
        })
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "images": images,
    }


@router.delete("/datasets/{video_id}")
async def delete_imported_dataset(project_name: str, video_id: int):
    """Delete an imported dataset."""
    import shutil
    
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    
    if video_id >= 0:
        raise HTTPException(status_code=400, detail="Can only delete imported datasets (negative video_id)")
    
    frames_dir = project_path / "frames" / str(video_id)
    if not frames_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Load frames to delete
    meta_path = frames_dir / "frames.json"
    frame_ids = set()
    if meta_path.exists():
        with open(meta_path) as f:
            frames_meta = json.load(f)
        frame_ids = {int(fid) for fid in frames_meta.keys()}
    
    # Delete annotations for these frames
    annotations_path = project_path / "labels" / "current" / "annotations.json"
    deleted_annotations = 0
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        new_annotations = {}
        for ann_id, ann_data in annotations.items():
            if ann_data.get("frame_id") not in frame_ids:
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

