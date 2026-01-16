"""Inference API routes."""

import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.models.training import InferenceConfig, InferenceResult
from backend.app.services.inference_runner import inference_runner
from backend.app.services.tracker import TrackingConfig

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

    # Find the training run
    runs_dir = project_path / "runs"
    checkpoint_path = None
    model_type = "yolo"

    for run_dir in runs_dir.iterdir():
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if meta["id"] == request.run_id:
            checkpoint_path = meta.get("checkpoint_path")
            if meta["base_model"].startswith("rfdetr"):
                model_type = "rfdetr"
            break

    if not checkpoint_path:
        raise HTTPException(status_code=404, detail="Model checkpoint not found")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found")

    await inference_runner.load_model(checkpoint_path, classes, model_type)

    return {"message": "Model loaded successfully"}


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

    # Find the frame
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
    video_id: int,
    config: InferenceConfig,
):
    """Run inference on a video and return results."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if inference_runner.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Get video path
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

    # Set up tracking config
    if config.tracking_mode == "visible_only":
        tracking_config = TrackingConfig.visible_only()
    else:
        tracking_config = TrackingConfig.occlusion_tolerant()

    # Run inference
    result = await inference_runner.run_on_video_full(
        video_path,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        enable_tracking=config.enable_tracking,
        tracking_config=tracking_config,
        detection_interval=config.detection_interval,
    )

    return result


@router.websocket("/stream/{video_id}")
async def stream_inference(
    websocket: WebSocket,
    project_name: str,
    video_id: int,
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

    # Get video path
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
        # Receive config from client
        config_data = await websocket.receive_json()
        config = InferenceConfig(**config_data)

        if config.tracking_mode == "visible_only":
            tracking_config = TrackingConfig.visible_only()
        else:
            tracking_config = TrackingConfig.occlusion_tolerant()

        # Stream inference results
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


@router.post("/export-video/{video_id}")
async def export_annotated_video(
    project_name: str,
    video_id: int,
    config: InferenceConfig,
):
    """Export video with detection overlay."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if inference_runner.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    # Get video path
    videos_meta_path = project_path / "videos" / "videos.json"
    with open(videos_meta_path) as f:
        videos_meta = json.load(f)

    if str(video_id) not in videos_meta:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(videos_meta[str(video_id)]["original_path"])
    output_path = project_path / "exports" / f"inference_{video_id}.mp4"
    output_path.parent.mkdir(exist_ok=True)

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

