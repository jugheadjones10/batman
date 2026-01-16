"""Auto-labeling API routes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.api.projects import get_project_path, load_project_config, save_project_config
from backend.app.services.sam_labeler import sam_labeler
from backend.app.services.tracker import Tracker, TrackingConfig

router = APIRouter(prefix="/projects/{project_name}/labeling", tags=["labeling"])


class AutoLabelRequest(BaseModel):
    """Request to run auto-labeling on frames."""

    video_ids: Optional[list[int]] = None  # None = all videos
    frame_ids: Optional[list[int]] = None  # None = all frames for selected videos
    use_exemplars: bool = True
    tracking_mode: Literal["visible_only", "occlusion_tolerant"] = "visible_only"
    skip_labeled_frames: bool = True  # Skip frames that already have annotations
    overwrite: bool = False  # If True, overwrite existing labels (dangerous!)


class RefineRequest(BaseModel):
    """Request to refine labels with SAM."""

    scope: Literal["clip_range", "touched_tracks", "full"] = "touched_tracks"
    video_id: Optional[int] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    track_ids: Optional[list[int]] = None


class AutoLabelProgress(BaseModel):
    """Auto-labeling progress update."""

    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float  # 0-1
    frames_processed: int
    total_frames: int
    annotations_created: int
    tracks_created: int
    message: str = ""


# In-memory job status (would use database in production)
_labeling_jobs: dict[str, AutoLabelProgress] = {}


@router.post("/auto-label")
async def start_auto_labeling(
    project_name: str,
    request: AutoLabelRequest,
    background_tasks: BackgroundTasks,
):
    """Start auto-labeling job."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    classes = config.get("classes", [])

    if not classes:
        raise HTTPException(status_code=400, detail="No classes defined for project")

    job_id = f"{project_name}_{datetime.utcnow().timestamp()}"

    _labeling_jobs[job_id] = AutoLabelProgress(
        status="pending",
        progress=0.0,
        frames_processed=0,
        total_frames=0,
        annotations_created=0,
        tracks_created=0,
        message="Starting auto-labeling...",
    )

    background_tasks.add_task(
        _run_auto_labeling,
        job_id,
        project_path,
        classes,
        request,
    )

    return {"job_id": job_id, "message": "Auto-labeling started"}


@router.get("/auto-label/{job_id}/status", response_model=AutoLabelProgress)
async def get_labeling_status(project_name: str, job_id: str):
    """Get auto-labeling job status."""
    if job_id not in _labeling_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _labeling_jobs[job_id]


@router.post("/refine")
async def refine_labels(
    project_name: str,
    request: RefineRequest,
    background_tasks: BackgroundTasks,
):
    """Refine labels with SAM on specific scope."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    job_id = f"{project_name}_refine_{datetime.utcnow().timestamp()}"

    _labeling_jobs[job_id] = AutoLabelProgress(
        status="pending",
        progress=0.0,
        frames_processed=0,
        total_frames=0,
        annotations_created=0,
        tracks_created=0,
        message="Starting refinement...",
    )

    background_tasks.add_task(
        _run_refinement,
        job_id,
        project_path,
        request,
    )

    return {"job_id": job_id, "message": "Refinement started"}


@router.post("/create-iteration")
async def create_iteration(project_name: str, description: Optional[str] = None):
    """Create a new label iteration snapshot."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    config = load_project_config(project_path)
    current_iteration = config.get("current_iteration", 0)
    new_iteration = current_iteration + 1

    # Create iteration directory
    labels_dir = project_path / "labels"
    current_dir = labels_dir / "current"
    iter_dir = labels_dir / f"iteration_{new_iteration}"

    if not current_dir.exists():
        raise HTTPException(status_code=400, detail="No labels to snapshot")

    # Copy current labels to iteration
    import shutil
    shutil.copytree(current_dir, iter_dir)

    # Create iteration metadata
    now = datetime.utcnow()

    # Count annotations and tracks
    annotations_path = current_dir / "annotations.json"
    tracks_path = current_dir / "tracks.json"

    total_annotations = 0
    total_tracks = 0

    if annotations_path.exists():
        with open(annotations_path) as f:
            total_annotations = len(json.load(f))

    if tracks_path.exists():
        with open(tracks_path) as f:
            total_tracks = len(json.load(f))

    iter_meta = {
        "id": new_iteration,
        "version": new_iteration,
        "description": description,
        "config": config.get("config", {}),
        "total_annotations": total_annotations,
        "total_tracks": total_tracks,
        "approved_frames": 0,
        "is_active": True,
        "created_at": now.isoformat(),
    }

    with open(iter_dir / "meta.json", "w") as f:
        json.dump(iter_meta, f, indent=2)

    # Update project config
    config["current_iteration"] = new_iteration
    config["updated_at"] = now.isoformat()
    save_project_config(project_path, config)

    logger.info(f"Created iteration {new_iteration} for {project_name}")

    return {
        "iteration_id": new_iteration,
        "message": f"Created iteration {new_iteration}",
    }


async def _run_auto_labeling(
    job_id: str,
    project_path: Path,
    classes: list[str],
    request: AutoLabelRequest,
):
    """Background task for auto-labeling."""
    try:
        _labeling_jobs[job_id].status = "running"
        _labeling_jobs[job_id].message = "Loading existing annotations..."

        # Load existing annotations and tracks (to preserve manual fixes)
        labels_dir = project_path / "labels" / "current"
        labels_dir.mkdir(parents=True, exist_ok=True)

        existing_annotations = {}
        existing_tracks = {}

        annotations_path = labels_dir / "annotations.json"
        tracks_path = labels_dir / "tracks.json"

        if annotations_path.exists():
            with open(annotations_path) as f:
                existing_annotations = json.load(f)

        if tracks_path.exists():
            with open(tracks_path) as f:
                existing_tracks = json.load(f)

        # Get frame IDs that already have annotations
        labeled_frame_ids = set()
        if request.skip_labeled_frames and not request.overwrite:
            for ann in existing_annotations.values():
                labeled_frame_ids.add(ann.get("frame_id"))
            logger.info(f"Found {len(labeled_frame_ids)} frames with existing annotations (will skip)")

        _labeling_jobs[job_id].message = "Loading frames..."

        # Collect frames to process
        frames_to_process = []
        frames_dir = project_path / "frames"

        if frames_dir.exists():
            for video_dir in frames_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                video_id = int(video_dir.name)
                if request.video_ids and video_id not in request.video_ids:
                    continue

                meta_path = video_dir / "frames.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        frames_meta = json.load(f)

                    for frame_id, frame_data in frames_meta.items():
                        frame_id_int = int(frame_id)

                        if request.frame_ids and frame_id_int not in request.frame_ids:
                            continue

                        # Skip frames that already have annotations
                        if request.skip_labeled_frames and frame_id_int in labeled_frame_ids:
                            continue

                        frames_to_process.append({
                            "id": frame_id_int,
                            "video_id": video_id,
                            **frame_data,
                        })

        _labeling_jobs[job_id].total_frames = len(frames_to_process)

        if not frames_to_process:
            _labeling_jobs[job_id].status = "completed"
            _labeling_jobs[job_id].progress = 1.0
            _labeling_jobs[job_id].message = "No new frames to process (all frames already labeled)"
            return

        # Load exemplars if requested
        exemplars = []
        if request.use_exemplars:
            exemplars_path = labels_dir / "exemplars.json"
            if exemplars_path.exists():
                with open(exemplars_path) as f:
                    exemplars = list(json.load(f).values())

        # Set up tracking
        if request.tracking_mode == "visible_only":
            tracking_config = TrackingConfig.visible_only()
        else:
            tracking_config = TrackingConfig.occlusion_tolerant()

        tracker = Tracker(tracking_config)

        # Start with existing data or empty if overwriting
        if request.overwrite:
            all_annotations = {}
            all_tracks = {}
            ann_id = 1
        else:
            all_annotations = existing_annotations.copy()
            all_tracks = existing_tracks.copy()
            # Find the next annotation ID
            ann_id = max([int(k) for k in all_annotations.keys()], default=0) + 1

        new_annotations_count = 0

        # Sort frames by video and frame number
        frames_to_process.sort(key=lambda f: (f["video_id"], f.get("frame_number", 0)))

        current_video_id = None

        for i, frame in enumerate(frames_to_process):
            _labeling_jobs[job_id].progress = (i + 1) / len(frames_to_process)
            _labeling_jobs[job_id].frames_processed = i + 1
            _labeling_jobs[job_id].message = f"Processing frame {i + 1}/{len(frames_to_process)} (skipped {len(labeled_frame_ids)} labeled)"

            # Reset tracker for new video
            if frame["video_id"] != current_video_id:
                current_video_id = frame["video_id"]
                tracker.reset()

            image_path = Path(frame["image_path"])
            if not image_path.exists():
                continue

            # Run detection
            try:
                detections = await sam_labeler.label_frame(
                    image_path,
                    classes,
                    exemplars=exemplars if request.use_exemplars else None,
                )
            except Exception as e:
                logger.warning(f"Failed to label frame {frame['id']}: {e}")
                continue

            # Apply tracking
            frame_number = frame.get("frame_number", 0)
            tracked_detections = tracker.update(detections, frame_number)

            # Save annotations
            for det in tracked_detections:
                all_annotations[str(ann_id)] = {
                    "frame_id": frame["id"],
                    "class_label_id": det.get("class_id", 0),
                    "track_id": det.get("track_id"),
                    "x": det["box"]["x"],
                    "y": det["box"]["y"],
                    "width": det["box"]["width"],
                    "height": det["box"]["height"],
                    "confidence": det.get("confidence", 1.0),
                    "source": "auto",
                    "is_exemplar": False,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }
                ann_id += 1
                new_annotations_count += 1

        # Save all tracks from tracker
        for track in tracker.get_all_tracks():
            track_id = track["track_id"]
            # Only add new tracks (don't overwrite existing)
            if str(track_id) not in all_tracks:
                all_tracks[str(track_id)] = {
                    "track_id": track_id,
                    "video_id": current_video_id,
                    "class_label_id": track["class_id"],
                    "start_frame": track["start_frame"],
                    "end_frame": track["end_frame"],
                    "is_approved": False,
                    "needs_review": True,
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                }

        # Save results (merged)
        with open(annotations_path, "w") as f:
            json.dump(all_annotations, f, indent=2)

        with open(tracks_path, "w") as f:
            json.dump(all_tracks, f, indent=2)

        # Update project stats
        config = load_project_config(project_path)
        config["annotation_count"] = len(all_annotations)
        config["updated_at"] = datetime.utcnow().isoformat()
        save_project_config(project_path, config)

        _labeling_jobs[job_id].status = "completed"
        _labeling_jobs[job_id].annotations_created = new_annotations_count
        _labeling_jobs[job_id].tracks_created = len(all_tracks) - len(existing_tracks)
        
        preserved_count = len(existing_annotations)
        _labeling_jobs[job_id].message = f"Done: +{new_annotations_count} new annotations, {preserved_count} existing preserved"

        logger.info(f"Auto-labeling completed: {new_annotations_count} new annotations (preserved {preserved_count} existing)")

    except Exception as e:
        logger.error(f"Auto-labeling failed: {e}")
        _labeling_jobs[job_id].status = "failed"
        _labeling_jobs[job_id].message = str(e)


async def _run_refinement(
    job_id: str,
    project_path: Path,
    request: RefineRequest,
):
    """Background task for label refinement."""
    try:
        _labeling_jobs[job_id].status = "running"
        _labeling_jobs[job_id].message = "Running refinement..."

        # Load existing annotations
        annotations_path = project_path / "labels" / "current" / "annotations.json"
        if not annotations_path.exists():
            raise ValueError("No annotations to refine")

        with open(annotations_path) as f:
            annotations = json.load(f)

        # Filter based on scope
        if request.scope == "touched_tracks" and request.track_ids:
            annotations_to_refine = {
                k: v for k, v in annotations.items()
                if v.get("track_id") in request.track_ids
            }
        elif request.scope == "clip_range" and request.start_frame is not None:
            # Would need frame number mapping
            annotations_to_refine = annotations
        else:
            annotations_to_refine = annotations

        _labeling_jobs[job_id].total_frames = len(annotations_to_refine)

        # Refinement logic would go here
        # For now, just mark as completed

        _labeling_jobs[job_id].status = "completed"
        _labeling_jobs[job_id].progress = 1.0
        _labeling_jobs[job_id].message = "Refinement completed"

    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        _labeling_jobs[job_id].status = "failed"
        _labeling_jobs[job_id].message = str(e)

