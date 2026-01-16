"""Annotation management API routes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.app.api.projects import get_project_path, load_project_config, save_project_config
from backend.app.models.annotation import (
    AnnotationCreate,
    AnnotationInfo,
    AnnotationUpdate,
    BoundingBox,
    ExemplarCreate,
    ExemplarInfo,
    ProblemQueueItem,
    TrackInfo,
    TrackMergeRequest,
    TrackSplitRequest,
    TrackUpdate,
)
from backend.app.services.tracker import detect_problems

router = APIRouter(prefix="/projects/{project_name}", tags=["annotations"])


def _load_annotations_meta(project_path: Path) -> dict:
    """Load annotations metadata."""
    meta_path = project_path / "labels" / "current" / "annotations.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _save_annotations_meta(project_path: Path, data: dict):
    """Save annotations metadata."""
    labels_dir = project_path / "labels" / "current"
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta_path = labels_dir / "annotations.json"
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)


def _load_tracks_meta(project_path: Path) -> dict:
    """Load tracks metadata."""
    meta_path = project_path / "labels" / "current" / "tracks.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _save_tracks_meta(project_path: Path, data: dict):
    """Save tracks metadata."""
    labels_dir = project_path / "labels" / "current"
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta_path = labels_dir / "tracks.json"
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)


# ============== Annotations ==============


@router.get("/frames/{frame_id}/annotations", response_model=list[AnnotationInfo])
async def list_frame_annotations(project_name: str, frame_id: int):
    """List all annotations for a frame."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    annotations_meta = _load_annotations_meta(project_path)
    config = load_project_config(project_path)
    classes = config.get("classes", [])

    # Color palette
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    annotations = []
    for ann_id, ann_data in annotations_meta.items():
        if ann_data.get("frame_id") != frame_id:
            continue

        class_id = ann_data.get("class_label_id", 0)
        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

        annotations.append(
            AnnotationInfo(
                id=int(ann_id),
                frame_id=ann_data["frame_id"],
                class_label_id=class_id,
                class_name=class_name,
                class_color=colors[class_id % len(colors)],
                box=BoundingBox(
                    x=ann_data["x"],
                    y=ann_data["y"],
                    width=ann_data["width"],
                    height=ann_data["height"],
                ),
                track_id=ann_data.get("track_id"),
                confidence=ann_data.get("confidence", 1.0),
                source=ann_data.get("source", "manual"),
                is_exemplar=ann_data.get("is_exemplar", False),
                exemplar_type=ann_data.get("exemplar_type"),
                created_at=datetime.fromisoformat(ann_data.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(ann_data.get("updated_at", datetime.utcnow().isoformat())),
            )
        )

    return annotations


@router.post("/annotations", response_model=AnnotationInfo)
async def create_annotation(project_name: str, data: AnnotationCreate):
    """Create a new annotation."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    annotations_meta = _load_annotations_meta(project_path)
    config = load_project_config(project_path)
    classes = config.get("classes", [])

    # Generate new annotation ID
    ann_id = max([int(k) for k in annotations_meta.keys()], default=0) + 1
    now = datetime.utcnow()

    # Color palette
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    annotations_meta[str(ann_id)] = {
        "frame_id": data.frame_id,
        "class_label_id": data.class_label_id,
        "track_id": data.track_id,
        "x": data.box.x,
        "y": data.box.y,
        "width": data.box.width,
        "height": data.box.height,
        "confidence": 1.0,
        "source": data.source,
        "is_exemplar": data.is_exemplar,
        "exemplar_type": data.exemplar_type,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    _save_annotations_meta(project_path, annotations_meta)

    # Update project stats
    config["annotation_count"] = len(annotations_meta)
    config["updated_at"] = now.isoformat()
    save_project_config(project_path, config)

    class_id = data.class_label_id
    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

    return AnnotationInfo(
        id=ann_id,
        frame_id=data.frame_id,
        class_label_id=data.class_label_id,
        class_name=class_name,
        class_color=colors[class_id % len(colors)],
        box=data.box,
        track_id=data.track_id,
        confidence=1.0,
        source=data.source,
        is_exemplar=data.is_exemplar,
        exemplar_type=data.exemplar_type,
        created_at=now,
        updated_at=now,
    )


@router.put("/annotations/{annotation_id}", response_model=AnnotationInfo)
async def update_annotation(project_name: str, annotation_id: int, data: AnnotationUpdate):
    """Update an annotation."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    annotations_meta = _load_annotations_meta(project_path)
    config = load_project_config(project_path)
    classes = config.get("classes", [])
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    if str(annotation_id) not in annotations_meta:
        raise HTTPException(status_code=404, detail="Annotation not found")

    ann_data = annotations_meta[str(annotation_id)]
    now = datetime.utcnow()

    # Update fields
    if data.class_label_id is not None:
        ann_data["class_label_id"] = data.class_label_id
    if data.box is not None:
        ann_data["x"] = data.box.x
        ann_data["y"] = data.box.y
        ann_data["width"] = data.box.width
        ann_data["height"] = data.box.height
    if data.track_id is not None:
        ann_data["track_id"] = data.track_id
    if data.source is not None:
        ann_data["source"] = data.source
    if data.is_exemplar is not None:
        ann_data["is_exemplar"] = data.is_exemplar
    if data.exemplar_type is not None:
        ann_data["exemplar_type"] = data.exemplar_type

    ann_data["updated_at"] = now.isoformat()
    _save_annotations_meta(project_path, annotations_meta)

    class_id = ann_data["class_label_id"]
    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

    return AnnotationInfo(
        id=annotation_id,
        frame_id=ann_data["frame_id"],
        class_label_id=class_id,
        class_name=class_name,
        class_color=colors[class_id % len(colors)],
        box=BoundingBox(
            x=ann_data["x"],
            y=ann_data["y"],
            width=ann_data["width"],
            height=ann_data["height"],
        ),
        track_id=ann_data.get("track_id"),
        confidence=ann_data.get("confidence", 1.0),
        source=ann_data.get("source", "manual"),
        is_exemplar=ann_data.get("is_exemplar", False),
        exemplar_type=ann_data.get("exemplar_type"),
        created_at=datetime.fromisoformat(ann_data.get("created_at", now.isoformat())),
        updated_at=now,
    )


@router.delete("/annotations/{annotation_id}")
async def delete_annotation(project_name: str, annotation_id: int):
    """Delete an annotation."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    annotations_meta = _load_annotations_meta(project_path)

    if str(annotation_id) not in annotations_meta:
        raise HTTPException(status_code=404, detail="Annotation not found")

    del annotations_meta[str(annotation_id)]
    _save_annotations_meta(project_path, annotations_meta)

    # Update project stats
    config = load_project_config(project_path)
    config["annotation_count"] = len(annotations_meta)
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)

    return {"message": "Annotation deleted"}


# ============== Tracks ==============


@router.get("/videos/{video_id}/tracks", response_model=list[TrackInfo])
async def list_tracks(project_name: str, video_id: int):
    """List all tracks for a video."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    tracks_meta = _load_tracks_meta(project_path)
    annotations_meta = _load_annotations_meta(project_path)
    config = load_project_config(project_path)
    classes = config.get("classes", [])
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    tracks = []
    for track_id, track_data in tracks_meta.items():
        if track_data.get("video_id") != video_id:
            continue

        class_id = track_data.get("class_label_id", 0)
        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

        # Count annotations in this track
        ann_count = sum(
            1 for ann in annotations_meta.values()
            if ann.get("track_id") == int(track_id)
        )

        tracks.append(
            TrackInfo(
                id=int(track_id),
                track_id=track_data["track_id"],
                class_label_id=class_id,
                class_name=class_name,
                class_color=colors[class_id % len(colors)],
                video_id=video_id,
                start_frame=track_data["start_frame"],
                end_frame=track_data["end_frame"],
                annotation_count=ann_count,
                is_approved=track_data.get("is_approved", False),
                needs_review=track_data.get("needs_review", True),
            )
        )

    return tracks


@router.put("/tracks/{track_id}", response_model=TrackInfo)
async def update_track(project_name: str, track_id: int, data: TrackUpdate):
    """Update a track."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    tracks_meta = _load_tracks_meta(project_path)
    config = load_project_config(project_path)
    classes = config.get("classes", [])
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    if str(track_id) not in tracks_meta:
        raise HTTPException(status_code=404, detail="Track not found")

    track_data = tracks_meta[str(track_id)]

    if data.class_label_id is not None:
        track_data["class_label_id"] = data.class_label_id
    if data.is_approved is not None:
        track_data["is_approved"] = data.is_approved
    if data.needs_review is not None:
        track_data["needs_review"] = data.needs_review

    track_data["updated_at"] = datetime.utcnow().isoformat()
    _save_tracks_meta(project_path, tracks_meta)

    class_id = track_data["class_label_id"]
    class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"

    return TrackInfo(
        id=track_id,
        track_id=track_data["track_id"],
        class_label_id=class_id,
        class_name=class_name,
        class_color=colors[class_id % len(colors)],
        video_id=track_data["video_id"],
        start_frame=track_data["start_frame"],
        end_frame=track_data["end_frame"],
        annotation_count=0,
        is_approved=track_data.get("is_approved", False),
        needs_review=track_data.get("needs_review", True),
    )


@router.post("/tracks/split")
async def split_track(project_name: str, data: TrackSplitRequest):
    """Split a track at a specific frame."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    tracks_meta = _load_tracks_meta(project_path)
    annotations_meta = _load_annotations_meta(project_path)

    if str(data.track_id) not in tracks_meta:
        raise HTTPException(status_code=404, detail="Track not found")

    old_track = tracks_meta[str(data.track_id)]

    if data.split_frame <= old_track["start_frame"] or data.split_frame >= old_track["end_frame"]:
        raise HTTPException(status_code=400, detail="Split frame must be within track range")

    # Create new track for second half
    new_track_id = max([int(k) for k in tracks_meta.keys()], default=0) + 1
    new_external_track_id = max([t["track_id"] for t in tracks_meta.values()], default=0) + 1

    tracks_meta[str(new_track_id)] = {
        "track_id": new_external_track_id,
        "video_id": old_track["video_id"],
        "class_label_id": old_track["class_label_id"],
        "start_frame": data.split_frame,
        "end_frame": old_track["end_frame"],
        "is_approved": False,
        "needs_review": True,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    # Update old track end frame
    old_track["end_frame"] = data.split_frame - 1
    old_track["updated_at"] = datetime.utcnow().isoformat()

    # Update annotations - move those >= split_frame to new track
    for ann_id, ann_data in annotations_meta.items():
        if ann_data.get("track_id") == data.track_id:
            # Need to check frame number - would need frame metadata
            pass

    _save_tracks_meta(project_path, tracks_meta)

    logger.info(f"Split track {data.track_id} at frame {data.split_frame}")

    return {
        "message": "Track split successfully",
        "original_track_id": data.track_id,
        "new_track_id": new_track_id,
    }


@router.post("/tracks/merge")
async def merge_tracks(project_name: str, data: TrackMergeRequest):
    """Merge two tracks into one."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    tracks_meta = _load_tracks_meta(project_path)
    annotations_meta = _load_annotations_meta(project_path)

    if str(data.source_track_id) not in tracks_meta:
        raise HTTPException(status_code=404, detail="Source track not found")
    if str(data.target_track_id) not in tracks_meta:
        raise HTTPException(status_code=404, detail="Target track not found")

    source_track = tracks_meta[str(data.source_track_id)]
    target_track = tracks_meta[str(data.target_track_id)]

    # Expand target track range
    target_track["start_frame"] = min(source_track["start_frame"], target_track["start_frame"])
    target_track["end_frame"] = max(source_track["end_frame"], target_track["end_frame"])
    target_track["updated_at"] = datetime.utcnow().isoformat()

    # Move annotations from source to target
    for ann_id, ann_data in annotations_meta.items():
        if ann_data.get("track_id") == data.source_track_id:
            ann_data["track_id"] = data.target_track_id

    # Delete source track
    del tracks_meta[str(data.source_track_id)]

    _save_tracks_meta(project_path, tracks_meta)
    _save_annotations_meta(project_path, annotations_meta)

    logger.info(f"Merged track {data.source_track_id} into {data.target_track_id}")

    return {
        "message": "Tracks merged successfully",
        "merged_track_id": data.target_track_id,
    }


# ============== Problem Queue ==============


@router.get("/problem-queue", response_model=list[ProblemQueueItem])
async def get_problem_queue(project_name: str, video_id: Optional[int] = None):
    """Get the problem queue for review acceleration."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    tracks_meta = _load_tracks_meta(project_path)
    annotations_meta = _load_annotations_meta(project_path)

    # Build track data for problem detection
    tracks_data = []
    for track_id, track_data in tracks_meta.items():
        if video_id is not None and track_data.get("video_id") != video_id:
            continue

        # Collect boxes for this track
        boxes = {}
        for ann_id, ann_data in annotations_meta.items():
            if ann_data.get("track_id") == int(track_id):
                frame_num = ann_data.get("frame_number", 0)
                boxes[frame_num] = {
                    "x": ann_data["x"],
                    "y": ann_data["y"],
                    "width": ann_data["width"],
                    "height": ann_data["height"],
                }

        tracks_data.append({
            "track_id": int(track_id),
            "boxes": boxes,
        })

    # Detect problems
    problems = detect_problems(tracks_data, {})

    # Convert to response format
    return [
        ProblemQueueItem(
            frame_id=0,  # Would need proper mapping
            frame_number=p["frame_number"],
            timestamp=0.0,  # Would need proper mapping
            video_id=video_id or 0,
            problem_type=p["problem_type"],
            severity=p["severity"],
            description=p["description"],
            affected_track_ids=p["affected_track_ids"],
        )
        for p in problems
    ]


# ============== Exemplars ==============


@router.get("/exemplars", response_model=list[ExemplarInfo])
async def list_exemplars(project_name: str):
    """List all exemplars."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    classes = config.get("classes", [])

    meta_path = project_path / "labels" / "current" / "exemplars.json"
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        exemplars_meta = json.load(f)

    return [
        ExemplarInfo(
            id=int(ex_id),
            class_label_id=ex_data["class_label_id"],
            class_name=classes[ex_data["class_label_id"]] if ex_data["class_label_id"] < len(classes) else f"class_{ex_data['class_label_id']}",
            frame_id=ex_data["frame_id"],
            box=BoundingBox(
                x=ex_data["x"],
                y=ex_data["y"],
                width=ex_data["width"],
                height=ex_data["height"],
            ),
            exemplar_type=ex_data.get("exemplar_type", "anchor"),
            created_at=datetime.fromisoformat(ex_data.get("created_at", datetime.utcnow().isoformat())),
        )
        for ex_id, ex_data in exemplars_meta.items()
    ]


@router.post("/exemplars", response_model=ExemplarInfo)
async def create_exemplar(project_name: str, data: ExemplarCreate):
    """Create a new exemplar."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    classes = config.get("classes", [])

    labels_dir = project_path / "labels" / "current"
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta_path = labels_dir / "exemplars.json"

    if meta_path.exists():
        with open(meta_path) as f:
            exemplars_meta = json.load(f)
    else:
        exemplars_meta = {}

    ex_id = max([int(k) for k in exemplars_meta.keys()], default=0) + 1
    now = datetime.utcnow()

    exemplars_meta[str(ex_id)] = {
        "class_label_id": data.class_label_id,
        "frame_id": data.frame_id,
        "x": data.box.x,
        "y": data.box.y,
        "width": data.box.width,
        "height": data.box.height,
        "exemplar_type": data.exemplar_type,
        "created_at": now.isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(exemplars_meta, f, indent=2)

    return ExemplarInfo(
        id=ex_id,
        class_label_id=data.class_label_id,
        class_name=classes[data.class_label_id] if data.class_label_id < len(classes) else f"class_{data.class_label_id}",
        frame_id=data.frame_id,
        box=data.box,
        exemplar_type=data.exemplar_type,
        created_at=now,
    )


@router.delete("/exemplars/{exemplar_id}")
async def delete_exemplar(project_name: str, exemplar_id: int):
    """Delete an exemplar."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    meta_path = project_path / "labels" / "current" / "exemplars.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Exemplar not found")

    with open(meta_path) as f:
        exemplars_meta = json.load(f)

    if str(exemplar_id) not in exemplars_meta:
        raise HTTPException(status_code=404, detail="Exemplar not found")

    del exemplars_meta[str(exemplar_id)]

    with open(meta_path, "w") as f:
        json.dump(exemplars_meta, f, indent=2)

    return {"message": "Exemplar deleted"}

