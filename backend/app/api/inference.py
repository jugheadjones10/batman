"""Inference API routes."""

import asyncio
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.models.training import InferenceConfig, InferenceGPUSubmitRequest
from backend.app.services.gpu_service import GPUJobState, gpu_service
from backend.app.services.inference_runner import inference_runner
from backend.app.services.tracker import TrackingConfig
from src.core.project import Project

SGT = timezone(timedelta(hours=8))

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
    model_type = "rfdetr"
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
            model_field = meta.get("model", meta.get("base_model", ""))
            if "rfdetr" in model_field or "rf-detr" in model_field:
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
    confidence_threshold: float = 0.0,
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
    result_dir = project.save_inference_result(run_name, video_id, persist_data)

    result["persisted"] = True
    result["run_name"] = run_name
    result["inference_id"] = persist_data.get("inference_id")
    return result


@router.get("/results")
async def list_inference_results(project_name: str):
    """List all persisted inference results as a matrix of runs x videos.

    Each cell now contains a list of inference results (multiple runs per
    video are supported), sorted newest-first by inference_id (SGT timestamp).
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    raw_results = project.list_inference_results()

    runs_set: set[str] = set()
    videos_set: set[str] = set()
    results_map: dict[str, dict[str, list[dict]]] = {}

    for summary in raw_results:
        run_name = summary.get("run_name", "")
        vid_id = summary.get("video_id", "")
        inf_id = summary.get("inference_id", "legacy")
        runs_set.add(run_name)
        videos_set.add(vid_id)

        # Check for detected video in the timestamped dir
        if inf_id != "legacy":
            has_video = (project.inference_dir / run_name / vid_id / inf_id / "detected.mp4").exists()
        else:
            has_video = (project.inference_dir / run_name / vid_id / "detected.mp4").exists()

        entry = {**summary, "has_video": has_video}
        results_map.setdefault(run_name, {}).setdefault(vid_id, []).append(entry)

    # Sort each cell newest-first
    for r in results_map:
        for v in results_map[r]:
            results_map[r][v].sort(key=lambda x: x.get("inference_id", ""), reverse=True)

    runs = sorted(runs_set)
    videos = sorted(videos_set)

    padded: dict[str, dict[str, list[dict] | None]] = {}
    for r in runs:
        padded[r] = {}
        for v in videos:
            cell = results_map.get(r, {}).get(v)
            padded[r][v] = cell if cell else None

    return {
        "runs": runs,
        "videos": videos,
        "results": padded,
    }


@router.get("/results/{run_name}/{video_id}/{inference_id}")
async def get_inference_result(
    project_name: str, run_name: str, video_id: str, inference_id: str
):
    """Load a specific persisted inference result by its inference_id (SGT timestamp)."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    result = project.get_inference_result(run_name, video_id, inference_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference result not found")

    if inference_id != "legacy":
        result["has_video"] = (
            project.inference_dir / run_name / video_id / inference_id / "detected.mp4"
        ).exists()
    else:
        result["has_video"] = (
            project.inference_dir / run_name / video_id / "detected.mp4"
        ).exists()
    return result


@router.delete("/results/{run_name}/{video_id}/{inference_id}")
async def delete_inference_result(
    project_name: str, run_name: str, video_id: str, inference_id: str
):
    """Delete a persisted inference result."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    deleted = project.delete_inference_result(run_name, video_id, inference_id)
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
    inference_id = datetime.now(SGT).strftime("%Y%m%d_%H%M%S")
    output_dir = project_path / "inference" / run_name / video_id / inference_id
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


# ── GPU cluster inference submission ──────────────────────────────────────


@router.post("/submit-gpu")
async def submit_inference_gpu(project_name: str, request: InferenceGPUSubmitRequest):
    """Submit an inference job to the GPU cluster."""
    if not gpu_service.is_connected:
        raise HTTPException(status_code=400, detail="Not connected to GPU cluster")

    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = f"data/projects/{project_name}"

    # Push project data
    try:
        gpu_service.push_project_data(project_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to push data: {e}")

    script, job_name = gpu_service.generate_inference_script(
        project_dir=project_dir,
        run_name=request.run_name,
        use_latest=request.run_name is None,
        video_ids=request.video_ids,
        test_only=request.test_only,
        model=request.model,
        confidence=request.confidence,
        frame_interval=request.frame_interval,
        track=request.track,
        track_thresh=request.track_thresh,
        track_buffer=request.track_buffer,
        match_thresh=request.match_thresh,
        no_video=request.no_video,
        gpu_type=request.gpu.gpu_type,
        time_limit=request.gpu.time_limit,
    )

    try:
        job_id = gpu_service.submit_slurm_job(script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SLURM submission failed: {e}")

    now = datetime.utcnow()
    infer_run_name = f"inference_{request.gpu.gpu_type}_{now.strftime('%Y%m%d_%H%M%S')}"

    job_state = GPUJobState(
        job_id=job_id,
        run_name=infer_run_name,
        job_type="inference",
        gpu_type=request.gpu.gpu_type,
        project_name=project_name,
        project_dir=project_dir,
        output_dir=f"{project_dir}/inference",
        submitted_at=now.isoformat(),
        log_file=f"logs/slurm_{job_id}_{job_name}.out",
        err_file=f"logs/slurm_{job_id}_{job_name}.err",
    )
    gpu_service.track_job(job_state)
    asyncio.create_task(gpu_service.poll_job_until_done(job_state, project_path))

    return {
        "job_id": job_id,
        "run_name": infer_run_name,
        "message": "Inference job submitted to GPU cluster",
    }


@router.get("/gpu-jobs/{job_name}/logs")
async def stream_inference_logs(project_name: str, job_name: str):
    """Stream GPU inference logs via SSE."""
    tracked = gpu_service.get_tracked_job(job_name)
    if not tracked:
        raise HTTPException(status_code=404, detail="Job not found")

    if not gpu_service.is_connected:
        raise HTTPException(status_code=400, detail="Not connected to GPU cluster")

    async def event_generator():
        try:
            async for line in gpu_service.stream_logs(tracked.job_id, "rfdetr-inference"):
                data = json.dumps({"type": "log", "line": line.rstrip("\n")})
                yield f"data: {data}\n\n"
        except Exception as e:
            data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {data}\n\n"
        finally:
            data = json.dumps({"type": "done"})
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/gpu-jobs/{job_name}/cancel")
async def cancel_inference_gpu(project_name: str, job_name: str):
    """Cancel a GPU inference job."""
    tracked = gpu_service.get_tracked_job(job_name)
    if not tracked:
        raise HTTPException(status_code=404, detail="Job not found")

    if gpu_service.is_connected:
        try:
            gpu_service.cancel_job(tracked.job_id)
        except Exception as e:
            logger.warning(f"scancel failed: {e}")

    tracked.status = "cancelled"
    tracked.completed_at = datetime.utcnow().isoformat()
    return {"status": "cancelled", "message": f"Inference job '{job_name}' cancelled"}


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
