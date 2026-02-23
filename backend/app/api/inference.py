"""Inference API routes."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.models.training import InferenceConfig
from backend.app.services.inference_runner import inference_runner
from backend.app.services.tracker import TrackingConfig
from src.core.project import Project

router = APIRouter(prefix="/projects/{project_name}/inference", tags=["inference"])


class LoadModelRequest(BaseModel):
    run_id: int


@router.post("/load-model")
async def load_model(project_name: str, request: LoadModelRequest):
    """Load a trained model for inference."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_config = load_project_config(project_path)
    classes = project_config.get("classes", [])

    runs_dir = project_path / "runs"
    checkpoint_path = None
    model_type = "yolo"
    run_name = None

    for run_dir in runs_dir.iterdir():
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if meta["id"] == request.run_id:
            checkpoint_path = meta.get("checkpoint_path")
            run_name = run_dir.name
            if meta["base_model"].startswith("rfdetr"):
                model_type = "rfdetr"

            class_info_path = run_dir / "class_info.json"
            if class_info_path.exists():
                with open(class_info_path) as f:
                    class_info = json.load(f)
                classes = class_info.get("classes", classes)
            break

    if not checkpoint_path:
        raise HTTPException(status_code=404, detail="Model checkpoint not found")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found")

    await inference_runner.load_model(checkpoint_path, classes, model_type)
    inference_runner.current_run_name = run_name

    return {"message": "Model loaded successfully", "run_name": run_name}


@router.post("/run-on-image")
async def run_on_image(
    project_name: str,
    frame_id: int,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
):
    """Run inference on a single frame."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if inference_runner.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    frames_dir = project_path / "frames"
    image_path = None

    for video_dir in frames_dir.iterdir():
        if not video_dir.is_dir():
            continue
        meta_path = video_dir / "frames.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            frames_meta = json.load(f)

        if str(frame_id) in frames_meta:
            image_path = Path(frames_meta[str(frame_id)]["image_path"])
            break

    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    result = await inference_runner.run_on_image(
        image_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )

    return result


@router.post("/run-on-video/{video_id}")
async def run_on_video(
    project_name: str,
    video_id: str,
    config: InferenceConfig,
):
    """Run inference on a video, persist results, and return them."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if inference_runner.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    run_name = getattr(inference_runner, "current_run_name", None)
    if not run_name:
        raise HTTPException(status_code=400, detail="No run associated with loaded model")

    videos_meta_path = project_path / "videos" / "videos.json"
    if not videos_meta_path.exists():
        raise HTTPException(status_code=404, detail="No videos found")

    with open(videos_meta_path) as f:
        videos_meta = json.load(f)

    if str(video_id) not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(videos_meta[str(video_id)]["original_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    if config.tracking_mode == "visible_only":
        tracking_config = TrackingConfig.visible_only()
    else:
        tracking_config = TrackingConfig.occlusion_tolerant()

    result = await inference_runner.run_on_video_full(
        video_path,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        enable_tracking=config.enable_tracking,
        tracking_config=tracking_config,
        detection_interval=config.detection_interval,
    )

    project = Project.load(project_path)
    persist_data = {
        "run_name": run_name,
        "video_id": video_id,
        "created_at": datetime.utcnow().isoformat(),
        "config": {
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
            "frame_interval": config.detection_interval,
            "tracking": config.enable_tracking,
            "tracking_mode": config.tracking_mode,
        },
        "stats": {
            "total_frames": result["total_frames"],
            "keyframes": sum(1 for r in result.get("results", []) if r.get("is_keyframe", True)),
            "total_detections": sum(len(r.get("detections", [])) for r in result.get("results", [])),
            "avg_inference_time_ms": result.get("avg_inference_time_ms", 0),
        },
        "frames": result.get("results", []),
    }
    project.save_inference_result(run_name, video_id, persist_data)

    result["persisted"] = True
    result["run_name"] = run_name
    return result


@router.get("/results")
async def list_inference_results(project_name: str):
    """List all persisted inference results as a matrix of runs x videos."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    raw_results = project.list_inference_results()

    runs_set: set[str] = set()
    videos_set: set[str] = set()
    results_map: dict[str, dict[str, dict]] = {}

    for (run_name, vid_id), summary in raw_results.items():
        runs_set.add(run_name)
        videos_set.add(vid_id)
        results_map.setdefault(run_name, {})[vid_id] = {
            **summary,
            "has_video": (project.inference_dir / run_name / vid_id / "detected.mp4").exists(),
        }

    runs = sorted(runs_set)
    videos = sorted(videos_set)

    padded: dict[str, dict[str, dict | None]] = {}
    for r in runs:
        padded[r] = {}
        for v in videos:
            padded[r][v] = results_map.get(r, {}).get(v)

    return {
        "runs": runs,
        "videos": videos,
        "results": padded,
    }


@router.get("/results/{run_name}/{video_id}")
async def get_inference_result(project_name: str, run_name: str, video_id: str):
    """Load a specific persisted inference result."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    result = project.get_inference_result(run_name, video_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference result not found")

    result["has_video"] = (project.inference_dir / run_name / video_id / "detected.mp4").exists()
    return result


@router.delete("/results/{run_name}/{video_id}")
async def delete_inference_result(project_name: str, run_name: str, video_id: str):
    """Delete a persisted inference result."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    deleted = project.delete_inference_result(run_name, video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Inference result not found")

    return {"message": "Inference result deleted"}


@router.post("/export-video/{video_id}")
async def export_annotated_video(
    project_name: str,
    video_id: str,
    config: InferenceConfig,
):
    """Export video with detection overlay, saved under inference results."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if inference_runner.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    run_name = getattr(inference_runner, "current_run_name", None)
    if not run_name:
        raise HTTPException(status_code=400, detail="No run associated with loaded model")

    videos_meta_path = project_path / "videos" / "videos.json"
    with open(videos_meta_path) as f:
        videos_meta = json.load(f)

    if str(video_id) not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(videos_meta[str(video_id)]["original_path"])
    output_dir = project_path / "inference" / run_name / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "detected.mp4"

    if config.tracking_mode == "visible_only":
        tracking_config = TrackingConfig.visible_only()
    else:
        tracking_config = TrackingConfig.occlusion_tolerant()

    result = await inference_runner.run_on_video_full(
        video_path,
        output_path=output_path,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        enable_tracking=config.enable_tracking,
        tracking_config=tracking_config,
        detection_interval=config.detection_interval,
    )

    return {
        "output_path": str(output_path),
        "total_frames": result["total_frames"],
        "avg_fps": result["avg_fps"],
        "avg_inference_time_ms": result["avg_inference_time_ms"],
    }


@router.websocket("/stream/{video_id}")
async def stream_inference(
    websocket: WebSocket,
    project_name: str,
    video_id: str,
):
    """Stream real-time inference results via WebSocket."""
    await websocket.accept()

    project_path = get_project_path(project_name)
    if not project_path.exists():
        await websocket.close(code=1008, reason="Project not found")
        return

    if inference_runner.model is None:
        await websocket.close(code=1008, reason="No model loaded")
        return

    videos_meta_path = project_path / "videos" / "videos.json"
    if not videos_meta_path.exists():
        await websocket.close(code=1008, reason="No videos found")
        return

    with open(videos_meta_path) as f:
        videos_meta = json.load(f)

    if str(video_id) not in videos_meta:
        await websocket.close(code=1008, reason="Video not found")
        return

    video_path = Path(videos_meta[str(video_id)]["original_path"])

    try:
        config_data = await websocket.receive_json()
        config = InferenceConfig(**config_data)

        if config.tracking_mode == "visible_only":
            tracking_config = TrackingConfig.visible_only()
        else:
            tracking_config = TrackingConfig.occlusion_tolerant()

        async for result in inference_runner.run_on_video(
            video_path,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            enable_tracking=config.enable_tracking,
            tracking_config=tracking_config,
        ):
            await websocket.send_json(result)

        await websocket.close()

    except Exception as e:
        logger.error(f"Streaming inference error: {e}")
        await websocket.close(code=1011, reason=str(e))
