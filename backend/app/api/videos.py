"""Video management API routes."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config, save_project_config
from backend.app.models.video import (
    FrameExtractionRequest,
    FrameInfo,
    SamplingConfig,
    VideoInfo,
    VideoUploadResponse,
)
from backend.app.services.video_processor import VideoProcessor
from src.core.project import Project

router = APIRouter(prefix="/projects/{project_name}/videos", tags=["videos"])


@router.get("", response_model=list[VideoInfo])
async def list_videos(project_name: str):
    """List all videos in a project."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    videos = []
    videos_dir = project_path / "videos"

    if not videos_dir.exists():
        return videos

    # Load video metadata (keys: legacy "1","2" or new "video_1","video_2")
    meta_path = videos_dir / "videos.json"
    if meta_path.exists():
        with open(meta_path) as f:
            videos_meta = json.load(f)

        for vid_id, vid_data in videos_meta.items():
            annotation_count = _count_video_annotations(project_path, vid_id)
            vid_id_out = vid_id if isinstance(vid_id, str) and not vid_id.isdigit() else int(vid_id)

            videos.append(
                VideoInfo(
                    id=vid_id_out,
                    filename=vid_data["filename"],
                    width=vid_data["width"],
                    height=vid_data["height"],
                    fps=vid_data["fps"],
                    duration=vid_data["duration"],
                    total_frames=vid_data["total_frames"],
                    has_proxy=vid_data.get("has_proxy", False),
                    frame_count=vid_data.get("frame_count", 0),
                    annotation_count=annotation_count,
                    exclude_from_training=vid_data.get("exclude_from_training", False),
                    created_at=datetime.fromisoformat(vid_data.get("created_at", datetime.utcnow().isoformat())),
                )
            )

    return videos


@router.post("", response_model=VideoUploadResponse)
async def upload_video(
    project_name: str,
    file: UploadFile = File(...),
    create_proxy: bool = True,
    exclude_from_training: bool = False,
):
    """Upload a video to the project."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    videos_dir = project_path / "videos"
    videos_dir.mkdir(exist_ok=True)

    # Load or create videos metadata
    meta_path = videos_dir / "videos.json"
    if meta_path.exists():
        with open(meta_path) as f:
            videos_meta = json.load(f)
    else:
        videos_meta = {}

    # New uploads use source_key (video_1, video_2)
    project = Project.load(project_path)
    video_id = project.get_next_video_source_key()

    # Save uploaded file
    video_path = videos_dir / f"{video_id}_{file.filename}"
    async with aiofiles.open(video_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    logger.info(f"Uploaded video: {video_path}")

    # Get video info
    try:
        info = await VideoProcessor.get_video_info(video_path)
    except Exception as e:
        # Clean up on failure
        video_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to process video: {e}")

    # Create proxy video if requested (video_id is e.g. video_1)
    proxy_path = None
    if create_proxy:
        proxy_path = videos_dir / f"{video_id}_proxy.mp4"
        try:
            await VideoProcessor.create_proxy_video(video_path, proxy_path)
        except Exception as e:
            logger.warning(f"Failed to create proxy video: {e}")

    now = datetime.utcnow()

    # Save video metadata (key is source_key e.g. video_1)
    videos_meta[video_id] = {
        "filename": file.filename,
        "original_path": str(video_path),
        "proxy_path": str(proxy_path) if proxy_path and proxy_path.exists() else None,
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "duration": info["duration"],
        "total_frames": info["total_frames"],
        "has_proxy": proxy_path is not None and proxy_path.exists(),
        "frame_count": 0,
        "exclude_from_training": exclude_from_training,
        "created_at": now.isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(videos_meta, f, indent=2)

    # Update project config
    config = load_project_config(project_path)
    config["video_count"] = len(videos_meta)
    config["updated_at"] = now.isoformat()
    save_project_config(project_path, config)

    return VideoUploadResponse(
        video=VideoInfo(
            id=video_id,
            filename=file.filename,
            width=info["width"],
            height=info["height"],
            fps=info["fps"],
            duration=info["duration"],
            total_frames=info["total_frames"],
            has_proxy=proxy_path is not None and proxy_path.exists(),
            created_at=now,
        ),
        message="Video uploaded successfully",
    )


@router.get("/{video_id}", response_model=VideoInfo)
async def get_video(project_name: str, video_id: str):
    """Get video details (video_id: video_1 or legacy 1)."""
    project_path = get_project_path(project_name)
    videos_meta = await _load_videos_meta(project_path)

    if video_id not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    vid_data = videos_meta[video_id]
    annotation_count = _count_video_annotations(project_path, video_id)
    vid_id_out = video_id if not video_id.isdigit() else int(video_id)

    return VideoInfo(
        id=vid_id_out,
        filename=vid_data["filename"],
        width=vid_data["width"],
        height=vid_data["height"],
        fps=vid_data["fps"],
        duration=vid_data["duration"],
        total_frames=vid_data["total_frames"],
        has_proxy=vid_data.get("has_proxy", False),
        frame_count=vid_data.get("frame_count", 0),
        annotation_count=annotation_count,
        exclude_from_training=vid_data.get("exclude_from_training", False),
        created_at=datetime.fromisoformat(vid_data.get("created_at", datetime.utcnow().isoformat())),
    )


class VideoUpdateRequest(BaseModel):
    exclude_from_training: Optional[bool] = None


@router.patch("/{video_id}", response_model=VideoInfo)
async def update_video(project_name: str, video_id: str, update: VideoUpdateRequest):
    """Update video metadata (e.g. toggle exclude_from_training)."""
    project_path = get_project_path(project_name)
    videos_meta = await _load_videos_meta(project_path)

    if video_id not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    if update.exclude_from_training is not None:
        videos_meta[video_id]["exclude_from_training"] = update.exclude_from_training

    meta_path = project_path / "videos" / "videos.json"
    with open(meta_path, "w") as f:
        json.dump(videos_meta, f, indent=2)

    vid_data = videos_meta[video_id]
    annotation_count = _count_video_annotations(project_path, video_id)
    vid_id_out = video_id if not video_id.isdigit() else int(video_id)

    return VideoInfo(
        id=vid_id_out,
        filename=vid_data["filename"],
        width=vid_data["width"],
        height=vid_data["height"],
        fps=vid_data["fps"],
        duration=vid_data["duration"],
        total_frames=vid_data["total_frames"],
        has_proxy=vid_data.get("has_proxy", False),
        frame_count=vid_data.get("frame_count", 0),
        annotation_count=annotation_count,
        exclude_from_training=vid_data.get("exclude_from_training", False),
        created_at=datetime.fromisoformat(vid_data.get("created_at", datetime.utcnow().isoformat())),
    )


@router.get("/{video_id}/stream")
async def stream_video(project_name: str, video_id: str, proxy: bool = True):
    """Stream video file (original or proxy)."""
    project_path = get_project_path(project_name)
    videos_meta = await _load_videos_meta(project_path)

    if video_id not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    vid_data = videos_meta[video_id]

    if proxy and vid_data.get("proxy_path"):
        video_path = Path(vid_data["proxy_path"])
    else:
        video_path = Path(vid_data["original_path"])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=vid_data["filename"],
    )


@router.get("/{video_id}/frame/{frame_number}")
async def get_frame_image(project_name: str, video_id: str, frame_number: int):
    """Get a specific frame as JPEG."""
    project_path = get_project_path(project_name)
    videos_meta = await _load_videos_meta(project_path)

    if video_id not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    vid_data = videos_meta[video_id]
    video_path = Path(vid_data["original_path"])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    frame_bytes = VideoProcessor.get_frame(video_path, frame_number)
    if frame_bytes is None:
        raise HTTPException(status_code=404, detail="Frame not found")

    return StreamingResponse(
        iter([frame_bytes]),
        media_type="image/jpeg",
    )


@router.post("/{video_id}/extract-frames", response_model=list[FrameInfo])
async def extract_frames(
    project_name: str,
    video_id: str,
    sampling: SamplingConfig = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """Extract frames from video at specified intervals."""
    if sampling is None:
        sampling = SamplingConfig()

    project_path = get_project_path(project_name)
    videos_meta = await _load_videos_meta(project_path)

    if video_id not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    vid_data = videos_meta[video_id]
    video_path = Path(vid_data["original_path"])

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    frames_dir = project_path / "frames" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    extracted = await VideoProcessor.extract_frames(
        video_path,
        frames_dir,
        mode=sampling.mode,
        interval=sampling.interval,
        start_time=start_time,
        end_time=end_time,
    )

    # New source_key (video_1) uses string frame_id; legacy (1) uses int
    use_string_frame_id = video_id.startswith("video_")
    frames_meta = {}
    frame_infos = []

    for i, frame_data in enumerate(extracted):
        if use_string_frame_id:
            frame_id = f"{video_id}_{i:06d}"
            new_path = frames_dir / f"{frame_id}.jpg"
            old_path = Path(frame_data["image_path"])
            if old_path.exists():
                old_path.rename(new_path)
            image_path = str(new_path)
        else:
            frame_id = int(video_id) * 1000000 + i
            image_path = frame_data["image_path"]
        frames_meta[str(frame_id)] = {
            "video_id": video_id,
            "frame_number": frame_data["frame_number"],
            "timestamp": frame_data["timestamp"],
            "image_path": image_path,
            "is_approved": False,
            "needs_review": True,
        }
        frame_infos.append(
            FrameInfo(
                id=frame_id,
                video_id=video_id,
                frame_number=frame_data["frame_number"],
                timestamp=frame_data["timestamp"],
                image_path=image_path,
                is_approved=False,
                needs_review=True,
            )
        )

    meta_path = frames_dir / "frames.json"
    with open(meta_path, "w") as f:
        json.dump(frames_meta, f, indent=2)

    videos_meta[video_id]["frame_count"] = len(extracted)
    videos_meta_path = project_path / "videos" / "videos.json"
    with open(videos_meta_path, "w") as f:
        json.dump(videos_meta, f, indent=2)

    config = load_project_config(project_path)
    config["frame_count"] = config.get("frame_count", 0) + len(extracted)
    config["updated_at"] = datetime.utcnow().isoformat()
    save_project_config(project_path, config)

    logger.info(f"Extracted {len(extracted)} frames from video {video_id}")

    return frame_infos


@router.get("/{video_id}/frames", response_model=list[FrameInfo])
async def list_frames(project_name: str, video_id: str):
    """List all extracted frames for a video."""
    project_path = get_project_path(project_name)
    frames_dir = project_path / "frames" / video_id

    if not frames_dir.exists():
        return []

    meta_path = frames_dir / "frames.json"
    if not meta_path.exists():
        return []

    with open(meta_path) as f:
        frames_meta = json.load(f)

    return [
        FrameInfo(
            id=frame_id if isinstance(frame_id, str) and not frame_id.isdigit() else int(frame_id),
            video_id=frame_data["video_id"],
            frame_number=frame_data["frame_number"],
            timestamp=frame_data["timestamp"],
            image_path=frame_data["image_path"],
            is_approved=frame_data.get("is_approved", False),
            needs_review=frame_data.get("needs_review", True),
        )
        for frame_id, frame_data in frames_meta.items()
    ]


@router.delete("/{video_id}")
async def delete_video(project_name: str, video_id: str):
    """Delete a video and its frames."""
    project_path = get_project_path(project_name)
    videos_meta = await _load_videos_meta(project_path)

    if video_id not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    vid_data = videos_meta[video_id]

    video_path = Path(vid_data["original_path"])
    if video_path.exists():
        video_path.unlink()

    if vid_data.get("proxy_path"):
        proxy_path = Path(vid_data["proxy_path"])
        if proxy_path.exists():
            proxy_path.unlink()

    frames_dir = project_path / "frames" / video_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    del videos_meta[video_id]
    videos_meta_path = project_path / "videos" / "videos.json"
    with open(videos_meta_path, "w") as f:
        json.dump(videos_meta, f, indent=2)

    logger.info(f"Deleted video {video_id}")
    return {"message": "Video deleted"}


async def _load_videos_meta(project_path: Path) -> dict:
    """Load videos metadata."""
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    meta_path = project_path / "videos" / "videos.json"
    if not meta_path.exists():
        return {}

    with open(meta_path) as f:
        return json.load(f)


def _count_video_annotations(project_path: Path, video_id: str | int) -> int:
    """Count total annotations for a video across all its frames."""
    frames_dir = project_path / "frames" / str(video_id)
    if not frames_dir.exists():
        return 0
    
    frames_meta_path = frames_dir / "frames.json"
    if not frames_meta_path.exists():
        return 0
    
    with open(frames_meta_path) as f:
        frames_meta = json.load(f)

    # frame_id in annotations may be int (legacy) or str (source_key_000042)
    frame_ids = set(frames_meta.keys())

    annotations_path = project_path / "labels" / "current" / "annotations.json"
    if not annotations_path.exists():
        return 0

    with open(annotations_path) as f:
        annotations = json.load(f)

    total = 0
    for ann in annotations.values():
        if str(ann.get("frame_id")) in frame_ids:
            total += 1
    return total

