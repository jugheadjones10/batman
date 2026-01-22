"""Project management API routes."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.app.config import settings
from backend.app.models.project import (
    ClassLabelCreate,
    ClassLabelInfo,
    LabelIterationInfo,
    ProjectConfig,
    ProjectCreate,
    ProjectInfo,
)

router = APIRouter(prefix="/projects", tags=["projects"])


def get_project_path(project_name: str) -> Path:
    """Get path to project directory."""
    return settings.projects_dir / project_name


def load_project_config(project_path: Path) -> dict:
    """Load project configuration from disk."""
    config_path = project_path / "project.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def save_project_config(project_path: Path, config: dict):
    """Save project configuration to disk."""
    config_path = project_path / "project.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


@router.get("", response_model=list[ProjectInfo])
async def list_projects():
    """List all projects."""
    projects = []

    if not settings.projects_dir.exists():
        return projects

    for project_dir in settings.projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        config = load_project_config(project_dir)
        if not config:
            continue

        projects.append(
            ProjectInfo(
                name=config.get("name", project_dir.name),
                path=str(project_dir),
                description=config.get("description"),
                classes=config.get("classes", []),
                config=ProjectConfig(**config.get("config", {})),
                video_count=config.get("video_count", 0),
                frame_count=config.get("frame_count", 0),
                annotation_count=config.get("annotation_count", 0),
                current_iteration=config.get("current_iteration", 0),
                created_at=datetime.fromisoformat(config.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(config.get("updated_at", datetime.utcnow().isoformat())),
            )
        )

    return projects


@router.post("", response_model=ProjectInfo)
async def create_project(data: ProjectCreate):
    """Create a new project."""
    # Sanitize project name for filesystem
    safe_name = "".join(c for c in data.name if c.isalnum() or c in "._- ").strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid project name")

    project_path = get_project_path(safe_name)

    if project_path.exists():
        raise HTTPException(status_code=400, detail="Project already exists")

    # Create project structure
    project_path.mkdir(parents=True)
    (project_path / "videos").mkdir()
    (project_path / "frames").mkdir()
    (project_path / "labels").mkdir()
    (project_path / "exports").mkdir()
    (project_path / "runs").mkdir()

    now = datetime.utcnow()

    config = {
        "name": data.name,
        "description": data.description,
        "classes": data.classes,
        "config": ProjectConfig().model_dump(),
        "video_count": 0,
        "frame_count": 0,
        "annotation_count": 0,
        "current_iteration": 0,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    save_project_config(project_path, config)

    logger.info(f"Created project: {data.name}")

    return ProjectInfo(
        name=data.name,
        path=str(project_path),
        description=data.description,
        classes=data.classes,
        config=ProjectConfig(),
        created_at=now,
        updated_at=now,
    )


@router.get("/{project_name}", response_model=ProjectInfo)
async def get_project(project_name: str):
    """Get project details."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    if not config:
        raise HTTPException(status_code=404, detail="Project configuration not found")

    return ProjectInfo(
        name=config.get("name", project_name),
        path=str(project_path),
        description=config.get("description"),
        classes=config.get("classes", []),
        config=ProjectConfig(**config.get("config", {})),
        video_count=config.get("video_count", 0),
        frame_count=config.get("frame_count", 0),
        annotation_count=config.get("annotation_count", 0),
        current_iteration=config.get("current_iteration", 0),
        created_at=datetime.fromisoformat(config.get("created_at", datetime.utcnow().isoformat())),
        updated_at=datetime.fromisoformat(config.get("updated_at", datetime.utcnow().isoformat())),
    )


@router.put("/{project_name}/config", response_model=ProjectInfo)
async def update_project_config(project_name: str, new_config: ProjectConfig):
    """Update project configuration."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    config["config"] = new_config.model_dump()
    config["updated_at"] = datetime.utcnow().isoformat()

    save_project_config(project_path, config)

    return await get_project(project_name)


@router.put("/{project_name}/classes", response_model=ProjectInfo)
async def update_project_classes(project_name: str, classes: list[str]):
    """Update project class list."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    config["classes"] = classes
    config["updated_at"] = datetime.utcnow().isoformat()

    save_project_config(project_path, config)

    return await get_project(project_name)


@router.delete("/{project_name}")
async def delete_project(project_name: str):
    """Delete a project."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    shutil.rmtree(project_path)
    logger.info(f"Deleted project: {project_name}")

    return {"message": "Project deleted"}


@router.get("/{project_name}/iterations", response_model=list[LabelIterationInfo])
async def list_iterations(project_name: str):
    """List all label iterations for a project."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    iterations = []
    labels_dir = project_path / "labels"

    for iter_dir in sorted(labels_dir.iterdir()):
        if not iter_dir.is_dir():
            continue

        meta_path = iter_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                iterations.append(
                    LabelIterationInfo(
                        id=meta.get("id", 0),
                        version=meta.get("version", 0),
                        description=meta.get("description"),
                        total_annotations=meta.get("total_annotations", 0),
                        total_tracks=meta.get("total_tracks", 0),
                        approved_frames=meta.get("approved_frames", 0),
                        is_active=meta.get("is_active", False),
                        created_at=datetime.fromisoformat(meta.get("created_at", datetime.utcnow().isoformat())),
                    )
                )

    return iterations


@router.post("/{project_name}/iterations/{iteration_id}/activate")
async def activate_iteration(project_name: str, iteration_id: int):
    """Set an iteration as the active one."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    labels_dir = project_path / "labels"

    # Deactivate all iterations
    for iter_dir in labels_dir.iterdir():
        if not iter_dir.is_dir():
            continue
        meta_path = iter_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["is_active"] = (meta.get("id", 0) == iteration_id)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    config = load_project_config(project_path)
    config["current_iteration"] = iteration_id
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)

    return {"message": f"Iteration {iteration_id} activated"}


@router.get("/{project_name}/classes/details")
async def get_classes_with_details(project_name: str):
    """Get classes with source information and annotation counts."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    classes = config.get("classes", [])
    class_sources = config.get("class_sources", {})
    
    # Load frames metadata to identify which frames are from which source
    frames_by_source = {}  # frame_id -> source
    
    # Check for imported frames (video_id = -1 for roboflow, -2 for local_coco)
    frames_dir = project_path / "frames"
    if frames_dir.exists():
        for video_dir in frames_dir.iterdir():
            if video_dir.is_dir():
                frames_meta_file = video_dir / "frames.json"
                if frames_meta_file.exists():
                    with open(frames_meta_file) as f:
                        frames_meta = json.load(f)
                    for frame_id, meta in frames_meta.items():
                        source = meta.get("source", "video")
                        frames_by_source[int(frame_id)] = source
    
    # Count annotations per class, broken down by source
    annotations_path = project_path / "labels" / "current" / "annotations.json"
    # Structure: {class_id: {"total": N, "sources": {"roboflow": N, "video": N, ...}}}
    class_stats = {i: {"total": 0, "sources": {}} for i in range(len(classes))}
    
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        for ann in annotations.values():
            class_id = ann.get("class_label_id", 0)
            if class_id not in class_stats:
                continue
            
            frame_id = ann.get("frame_id", 0)
            # Determine source: check annotation source first, then frame source
            ann_source = ann.get("source", "manual")
            if ann_source in ("roboflow", "local_coco"):
                source = ann_source
            elif frame_id in frames_by_source:
                source = frames_by_source[frame_id]
            elif frame_id < 0:
                # Negative frame IDs are from imports
                source = "roboflow" if frame_id >= -1000000 else "local_coco"
            else:
                source = "video"
            
            class_stats[class_id]["total"] += 1
            class_stats[class_id]["sources"][source] = class_stats[class_id]["sources"].get(source, 0) + 1
    
    result = []
    for i, cls in enumerate(classes):
        stats = class_stats.get(i, {"total": 0, "sources": {}})
        result.append({
            "id": i,
            "name": cls,
            "source": class_sources.get(cls, "manual"),
            "annotation_count": stats["total"],
            "annotation_sources": stats["sources"],
        })
    
    return result


@router.delete("/{project_name}/classes/{class_name}")
async def delete_class(project_name: str, class_name: str, delete_annotations: bool = True):
    """Delete a class and optionally its annotations."""
    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    classes = config.get("classes", [])
    class_sources = config.get("class_sources", {})
    
    if class_name not in classes:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")
    
    class_id = classes.index(class_name)
    annotations_deleted = 0
    
    # Handle annotations
    annotations_path = project_path / "labels" / "current" / "annotations.json"
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        if delete_annotations:
            # Remove annotations for this class
            new_annotations = {}
            for ann_id, ann in annotations.items():
                if ann.get("class_label_id") == class_id:
                    annotations_deleted += 1
                else:
                    # Shift class IDs for classes after the deleted one
                    if ann.get("class_label_id", 0) > class_id:
                        ann["class_label_id"] -= 1
                    new_annotations[ann_id] = ann
            annotations = new_annotations
        else:
            # Just shift class IDs, don't delete annotations
            for ann in annotations.values():
                if ann.get("class_label_id", 0) > class_id:
                    ann["class_label_id"] -= 1
        
        with open(annotations_path, "w") as f:
            json.dump(annotations, f, indent=2)
    
    # Remove class from list
    classes.remove(class_name)
    
    # Remove from class_sources
    if class_name in class_sources:
        del class_sources[class_name]
    
    config["classes"] = classes
    config["class_sources"] = class_sources
    config["annotation_count"] = config.get("annotation_count", 0) - annotations_deleted
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)
    
    return {
        "message": f"Deleted class '{class_name}'",
        "annotations_deleted": annotations_deleted,
    }


from pydantic import BaseModel as PydanticBaseModel

class ClassRenameRequest(PydanticBaseModel):
    old_name: str
    new_name: str


class ClassMergeRequest(PydanticBaseModel):
    source_classes: list[str]  # Classes to merge FROM
    target_class: str  # Class to merge INTO


@router.post("/{project_name}/classes/rename")
async def rename_class(project_name: str, request: ClassRenameRequest):
    """Rename a class."""
    from src.core import ClassManager, Project

    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        project = Project.load(project_path)
        manager = ClassManager(project)
        result = manager.rename_class(request.old_name, request.new_name)
        return {"message": result.message}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{project_name}/classes/merge")
async def merge_classes(project_name: str, request: ClassMergeRequest):
    """Merge multiple classes into one."""
    from src.core import ClassManager, Project

    project_path = get_project_path(project_name)

    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        project = Project.load(project_path)
        manager = ClassManager(project)
        result = manager.merge_classes(request.source_classes, request.target_class)
        return {
            "message": result.message,
            "annotations_updated": result.annotations_updated,
            "classes_removed": result.classes_removed,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

