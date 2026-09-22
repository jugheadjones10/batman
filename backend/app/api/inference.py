"""Inference API routes."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.config import settings
from backend.app.models.training import InferenceConfig, InferenceGPUSubmitRequest
from backend.app.services import z_estimator
from backend.app.services.gpu_service import GPUJobState, gpu_service
from backend.app.services.inference_runner import inference_runner
from backend.app.services.tracker import TrackingConfig
from src.core.project import Project
from src.core.trainer import find_best_checkpoint

SGT = timezone(timedelta(hours=8))

router = APIRouter(prefix="/projects/{project_name}/inference", tags=["inference"])


class LoadModelRequest(BaseModel):
    # `run_name` is the unique identifier (directory name on disk). `run_id` is
    # kept for backwards compatibility but is NOT reliably unique — historical
    # meta.json files can share an id because the old generator used
    # `len(iterdir())`, so a by-id lookup can resolve to the wrong run. Prefer
    # `run_name` whenever the frontend has it.
    run_name: str | None = None
    run_id: int | None = None
    device: str | None = None  # auto, cuda, mps, cpu; default from settings


@router.post("/load-model")
async def load_model(project_name: str, request: LoadModelRequest):
    """Load a trained model for inference."""
    if request.run_name is None and request.run_id is None:
        raise HTTPException(status_code=400, detail="run_name or run_id is required")

    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_config = load_project_config(project_path)
    classes = project_config.get("classes", [])

    runs_dir = project_path / "runs"
    checkpoint_path = None
    model_type = "rfdetr"
    model_size = "base"
    run_name = None

    for run_dir in runs_dir.iterdir():
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        matches = (
            run_dir.name == request.run_name
            if request.run_name is not None
            else meta.get("id") == request.run_id
        )
        if matches:
            checkpoint_path = meta.get("checkpoint_path")
            run_name = run_dir.name
            model_field = meta.get("model", meta.get("base_model", ""))
            if "rfdetr" in model_field or "rf-detr" in model_field:
                model_type = "rfdetr"

            # Extract variant: "rf-detr-small" → "small", or from config.
            if "seg" in model_field.lower() or "xlarge" in model_field.lower():
                raise HTTPException(status_code=400, detail="Only detection checkpoints are supported")
            for variant in ("nano", "small", "medium", "base", "large"):
                if variant in model_field:
                    model_size = variant
                    break
            cfg_training = (meta.get("config") or {}).get("training", {})
            cfg_model = cfg_training.get("model")
            if cfg_model:
                model_size = cfg_model
            cfg_task = cfg_training.get("task") or meta.get("task")
            if cfg_task and cfg_task != "detection":
                raise HTTPException(status_code=400, detail="Only detection checkpoints are supported")

            # Fallback: meta may not have checkpoint_path (e.g. local run before backend wrote it)
            if not checkpoint_path:
                results_path = run_dir / "results.json"
                if results_path.exists():
                    try:
                        with open(results_path) as rf:
                            results = json.load(rf)
                        checkpoint_path = results.get("checkpoint_path")
                    except (json.JSONDecodeError, OSError):
                        pass
                if not checkpoint_path:
                    best = find_best_checkpoint(run_dir)
                    if best is not None:
                        checkpoint_path = str(best)

            class_info_path = run_dir / "class_info.json"
            if class_info_path.exists():
                with open(class_info_path) as f:
                    class_info = json.load(f)
                classes = class_info.get("classes", classes)
                # class_info.json may also carry task (written by cli/train)
                ci_task = class_info.get("task")
                if ci_task and ci_task != "detection":
                    raise HTTPException(status_code=400, detail="Only detection checkpoints are supported")
            break

    if not checkpoint_path:
        raise HTTPException(status_code=404, detail="Model checkpoint not found")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found")

    device = request.device if request.device else settings.device
    await inference_runner.load_model(
        checkpoint_path,
        classes,
        model_type,
        device=device,
        model_size=model_size,
    )
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


def _prepare_run_on_video(
    project_name: str, video_id: str, config: InferenceConfig
) -> tuple[Path, str, Path, str, Path, TrackingConfig]:
    """Validate state and resolve paths for a `run-on-video` request.

    Returns (project_path, run_name, video_path, inference_id, result_dir,
    tracking_config). Raises HTTPException on any precondition failure so the
    error surfaces before a streaming response is opened (SSE can't set status).
    """
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

    inference_id = datetime.now(SGT).strftime("%Y%m%d_%H%M%S")
    result_dir = project_path / "inference" / run_name / video_id / inference_id
    result_dir.mkdir(parents=True, exist_ok=True)

    return project_path, run_name, video_path, inference_id, result_dir, tracking_config


def _persist_inference_result(
    project_path: Path,
    run_name: str,
    video_id: str,
    inference_id: str,
    result_dir: Path,
    config: InferenceConfig,
    result: dict,
) -> dict:
    """Write result.json and return the persisted result."""
    video_resolution = _get_video_resolution(project_path, video_id)

    persist_data = {
        "run_name": run_name,
        "video_id": video_id,
        "inference_id": inference_id,
        "created_at": datetime.now(SGT).isoformat(),
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
            "total_detections": sum(
                len(r.get("detections", [])) for r in result.get("results", [])
            ),
            "avg_inference_time_ms": result.get("avg_inference_time_ms", 0),
        },
        "video_resolution": video_resolution,
        "frames": result.get("results", []),
    }
    with open(result_dir / "result.json", "w") as f:
        json.dump(persist_data, f, indent=2)

    result["persisted"] = True
    result["run_name"] = run_name
    result["inference_id"] = inference_id
    return result


@router.post("/run-on-video/{video_id}")
async def run_on_video(
    project_name: str,
    video_id: str,
    config: InferenceConfig,
):
    """Run inference on a video, persist results, and return them."""
    project_path, run_name, video_path, inference_id, result_dir, tracking_config = (
        _prepare_run_on_video(project_name, video_id, config)
    )

    result = await inference_runner.run_on_video_full(
        video_path,
        output_path=result_dir / "detected.mp4",
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        enable_tracking=config.enable_tracking,
        tracking_config=tracking_config,
        detection_interval=config.detection_interval,
    )

    return _persist_inference_result(
        project_path, run_name, video_id, inference_id, result_dir, config, result
    )


@router.post("/run-on-video/{video_id}/stream")
async def run_on_video_stream(
    project_name: str,
    video_id: str,
    config: InferenceConfig,
):
    """Run inference on a video and stream progress events via SSE.

    The stream emits JSON-encoded events in order:
      1. `{"type": "stage", "stage": "running_inference", "total_frames": N}`
      2. Repeated `{"type": "progress", "current": i, "total": N, "avg_fps": f,
         "eta_s": t}` (~5 Hz).
      3. `{"type": "stage", "stage": "encoding_video"}` while ffmpeg re-encodes.
      4. `{"type": "stage", "stage": "post_processing"}` while saving results.
      5. Terminal `{"type": "complete", "inference_id": ..., "total_frames": N,
         "avg_fps": f, ...}` OR `{"type": "error", "message": ...}`.

    Precondition errors (no model, missing video, etc.) raise HTTP 4xx *before*
    the stream opens so the client can surface them naturally.
    """
    project_path, run_name, video_path, inference_id, result_dir, tracking_config = (
        _prepare_run_on_video(project_name, video_id, config)
    )

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    SENTINEL = object()

    def on_progress(event: dict) -> None:
        # Called from the inference worker thread; bounce onto the event loop.
        # Drop updates silently if the queue is full — progress is idempotent
        # and the next tick will replace the value anyway.
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            pass

    async def runner() -> None:
        try:
            result = await inference_runner.run_on_video_full(
                video_path,
                output_path=result_dir / "detected.mp4",
                confidence_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                enable_tracking=config.enable_tracking,
                tracking_config=tracking_config,
                detection_interval=config.detection_interval,
                progress_callback=on_progress,
            )
            await queue.put({"type": "stage", "stage": "post_processing"})
            persisted = _persist_inference_result(
                project_path, run_name, video_id, inference_id, result_dir, config, result
            )
            await queue.put(
                {
                    "type": "complete",
                    "inference_id": inference_id,
                    "run_name": run_name,
                    "total_frames": persisted["total_frames"],
                    "avg_fps": persisted.get("avg_fps", 0),
                    "avg_inference_time_ms": persisted.get("avg_inference_time_ms", 0),
                }
            )
        except Exception as e:
            logger.exception(f"inference stream failed: {e}")
            await queue.put({"type": "error", "message": str(e) or type(e).__name__})
        finally:
            await queue.put(SENTINEL)

    task = asyncio.create_task(runner())

    async def event_generator():
        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            # Ensure the worker can't outlive the stream (client disconnect).
            if not task.done():
                task.cancel()
            # Best-effort await; swallow CancelledError so FastAPI cleanup runs.
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
            result_base = project.inference_dir / run_name / vid_id / inf_id
        else:
            result_base = project.inference_dir / run_name / vid_id
        has_video = (result_base / "detected.mp4").exists()
        has_z_video = (result_base / "detected_z.mp4").exists()

        entry = {**summary, "has_video": has_video, "has_z_video": has_z_video}
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
async def get_inference_result(project_name: str, run_name: str, video_id: str, inference_id: str):
    """Load a specific persisted inference result by its inference_id (SGT timestamp)."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project.load(project_path)
    result = project.get_inference_result(run_name, video_id, inference_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference result not found")

    if inference_id != "legacy":
        result_base = project.inference_dir / run_name / video_id / inference_id
    else:
        result_base = project.inference_dir / run_name / video_id
    result["has_video"] = (result_base / "detected.mp4").exists()
    result["has_z_video"] = (result_base / "detected_z.mp4").exists()
    result["has_raw_video"] = (result_base / "detected_raw.mp4").exists()
    result["has_bytetrack_video"] = (result_base / "detected_bytetrack.mp4").exists()
    return result


@router.get("/results/{run_name}/{video_id}/{inference_id}/video")
async def get_inference_result_video(
    request: Request,
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
    variant: str | None = None,
):
    """Stream the detected video (with overlay) for an inference result. Supports Range for seeking.
    
    Use ?variant=z to stream the Z-overlay video (detected_z.mp4) instead of the default.
    Use ?variant=raw to stream the no-tracker baseline (detected_raw.mp4).
    Use ?variant=bytetrack to stream the ByteTrack comparison video (detected_bytetrack.mp4).
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    variant_files = {
        "z": "detected_z.mp4",
        "raw": "detected_raw.mp4",
        "bytetrack": "detected_bytetrack.mp4",
    }
    filename = variant_files.get(variant or "", "detected.mp4")
    if inference_id == "legacy":
        video_path = project_path / "inference" / run_name / video_id / filename
    else:
        video_path = (
            project_path / "inference" / run_name / video_id / inference_id / filename
        )

    video_path = video_path.resolve()
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")

    # #region agent log
    import time as _time

    open("/home/batman/batman/.cursor/debug-b2be69.log", "a").write(
        json.dumps(
            {
                "sessionId": "b2be69",
                "hypothesisId": "H3",
                "location": "inference.py:video_endpoint",
                "message": "video endpoint hit",
                "data": {
                    "video_path": str(video_path),
                    "file_size": file_size,
                    "range": range_header,
                    "exists": video_path.exists(),
                },
                "timestamp": int(_time.time() * 1000),
            }
        )
        + "\n"
    )
    # #endregion

    if range_header:
        # Parse "bytes=start-end" (end may be missing)
        try:
            range_str = range_header.strip().lower().replace("bytes=", "")
            parts = range_str.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            end = min(end, file_size - 1)
            if start > end or start < 0:
                raise ValueError("Invalid range")
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")

        content_length = end - start + 1

        async def stream_range():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            stream_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(content_length),
            },
        )

    # No Range header: return full file (some players need this for initial load)
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


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


# ── Z-axis height estimation ─────────────────────────────────────────────


class ZCalibrationLabel(BaseModel):
    frame_number: int
    z_mm: float
    detection_index: int = 0


class ZCalibrationRequest(BaseModel):
    labels: list[ZCalibrationLabel]
    reference_class: str
    length_mm: float | None = None
    target_classes: list[str] = Field(default_factory=list)
    measurement_source: Literal["bbox_longer_side", "round_feature_equivalent_length"] = (
        "bbox_longer_side"
    )
    round_feature_diameter_mm: float | None = None
    feature_to_spreader_z_offset_mm: float = 0


def _resolve_result_dir(
    project_path: Path, run_name: str, video_id: str, inference_id: str
) -> Path:
    if inference_id == "legacy":
        return project_path / "inference" / run_name / video_id
    return project_path / "inference" / run_name / video_id / inference_id


def _get_video_resolution(project_path: Path, video_id: str) -> dict:
    videos_meta_path = project_path / "videos" / "videos.json"
    if not videos_meta_path.exists():
        raise HTTPException(status_code=404, detail="No videos found")
    with open(videos_meta_path) as f:
        videos_meta = json.load(f)
    vid_meta = videos_meta.get(str(video_id))
    if vid_meta is None:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(vid_meta["original_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"width": w, "height": h}


@router.get("/results/{run_name}/{video_id}/{inference_id}/z-calibration")
async def get_z_calibration(project_name: str, run_name: str, video_id: str, inference_id: str):
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    result_dir = _resolve_result_dir(project_path, run_name, video_id, inference_id)
    cal = z_estimator.load_z_calibration(result_dir)
    return {"z_calibration": cal}


@router.post("/results/{run_name}/{video_id}/{inference_id}/z-calibration")
async def save_z_calibration(
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
    request: ZCalibrationRequest,
):
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    result_dir = _resolve_result_dir(project_path, run_name, video_id, inference_id)
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Inference result not found")

    video_resolution = _get_video_resolution(project_path, video_id)

    equivalent_size_ratio: float | None = None
    if request.measurement_source == z_estimator.ROUND_FEATURE_EQUIVALENT_LENGTH:
        if request.length_mm is None or request.length_mm <= 0:
            raise HTTPException(
                status_code=400,
                detail="Round feature calibration requires a positive container/spreader length",
            )
        if request.round_feature_diameter_mm is None or request.round_feature_diameter_mm <= 0:
            raise HTTPException(
                status_code=400,
                detail="Round feature calibration requires a positive round feature diameter",
            )
        equivalent_size_ratio = request.length_mm / request.round_feature_diameter_mm

    calibration_data: dict[str, Any] = {
        "labels": [label.model_dump() for label in request.labels],
        "reference_class": request.reference_class,
        "targets": list(request.target_classes),
        "video_resolution": video_resolution,
        "measurement_source": request.measurement_source,
        "feature_to_spreader_z_offset_mm": request.feature_to_spreader_z_offset_mm,
    }
    if request.length_mm is not None:
        calibration_data["length_mm"] = request.length_mm
    if request.measurement_source == z_estimator.ROUND_FEATURE_EQUIVALENT_LENGTH:
        calibration_data["round_feature_diameter_mm"] = request.round_feature_diameter_mm
        calibration_data["equivalent_size_ratio"] = equivalent_size_ratio

    z_estimator.save_z_calibration(result_dir, calibration_data)
    return {"message": "Z calibration saved", "labels_count": len(request.labels)}


@router.post("/results/{run_name}/{video_id}/{inference_id}/z-estimate")
async def apply_z_estimation(
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
):
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    result_dir = _resolve_result_dir(project_path, run_name, video_id, inference_id)

    try:
        model = z_estimator.apply_z_to_result(result_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"message": "Z estimation applied", "model": model}


@router.post("/results/{run_name}/{video_id}/{inference_id}/z-export-video")
async def export_z_video(
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
):
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    result_dir = _resolve_result_dir(project_path, run_name, video_id, inference_id)
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Inference result not found")

    with open(result_path) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    has_z = any(d.get("z_mm") is not None for frame in frames for d in frame.get("detections", []))
    if not has_z:
        raise HTTPException(status_code=400, detail="No Z values found — run z-estimate first")

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

    import subprocess as _sp

    from src.core.inference import Detection as DetObj
    from src.core.inference import draw_detections

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = result_dir / "detected_z.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (vid_w, vid_h))

    frame_map = {f["frame_number"]: f for f in frames}

    try:
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = frame_map.get(frame_num)
            if frame_data:
                det_objects = [
                    DetObj(
                        bbox=(
                            (d["box"]["x"] - d["box"]["width"] / 2) * vid_w,
                            (d["box"]["y"] - d["box"]["height"] / 2) * vid_h,
                            (d["box"]["x"] + d["box"]["width"] / 2) * vid_w,
                            (d["box"]["y"] + d["box"]["height"] / 2) * vid_h,
                        ),
                        class_id=d.get("class_id", 0),
                        class_name=d.get("class_name", ""),
                        confidence=d.get("confidence", 1.0),
                        track_id=d.get("track_id"),
                        z_mm=d.get("z_mm"),
                    )
                    for d in frame_data.get("detections", [])
                ]
                frame = draw_detections(frame, det_objects)

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    if output_path.exists():
        tmp_path = output_path.with_suffix(".tmp.mp4")
        try:
            result_ffmpeg = _sp.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(output_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(tmp_path),
                ],
                capture_output=True,
                timeout=300,
            )
            if result_ffmpeg.returncode == 0 and tmp_path.exists():
                tmp_path.replace(output_path)
            else:
                logger.warning(f"ffmpeg re-encode failed (rc={result_ffmpeg.returncode})")
        except FileNotFoundError:
            logger.warning("ffmpeg not found; video remains in mp4v codec")
        except Exception as e:
            logger.warning(f"ffmpeg re-encode error: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return {"message": "Video re-exported with Z overlays", "output_path": str(output_path)}


@router.get("/results/{run_name}/{video_id}/{inference_id}/bytetrack-frames")
async def get_bytetrack_frames(
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
    track_activation_threshold: float = Query(0.25, ge=0, le=1),
    lost_track_buffer: int = Query(30, ge=1, le=600),
    minimum_matching_threshold: float = Query(0.8, ge=0, le=1),
):
    """Return a `frames[]` list with per-frame detections re-associated by
    sv.ByteTrack (including Kalman gap-fills for tracks lost within
    `lost_track_buffer`).

    This is the same transformation used to render the ByteTrack comparison
    video, but returned as JSON so the frontend can drive a second schematic
    and compare raw-vs-tracked stability. Computed on the fly each call; no
    video decoding is performed.
    """
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    result_dir = _resolve_result_dir(project_path, run_name, video_id, inference_id)
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Inference result not found")

    with open(result_path) as f:
        result_data = json.load(f)

    # Need fps for ByteTrack's frame-rate-derived buffer math, and video
    # dimensions to convert normalized boxes to the pixel space the tracker
    # expects (its default kalman motion model is scale-aware).
    videos_meta_path = project_path / "videos" / "videos.json"
    fps = 30.0
    vid_w = 1920
    vid_h = 1080
    if videos_meta_path.exists():
        try:
            with open(videos_meta_path) as f:
                vm = json.load(f)
            v = vm.get(str(video_id)) or {}
            fps = float(v.get("fps") or fps)
            vid_w = int(v.get("width") or vid_w)
            vid_h = int(v.get("height") or vid_h)
        except Exception as e:
            logger.warning(f"Failed to read videos.json for bytetrack-frames: {e}")

    # Fall back to result.json's video_resolution if available.
    vr = result_data.get("video_resolution") or {}
    vid_w = int(vr.get("width") or vid_w)
    vid_h = int(vr.get("height") or vid_h)

    from src.core.inference import compute_bytetrack_frames

    try:
        frames = await asyncio.to_thread(
            compute_bytetrack_frames,
            result_data,
            fps,
            vid_w,
            vid_h,
            track_activation_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
        )
    except Exception as e:
        logger.exception("bytetrack-frames computation failed")
        raise HTTPException(
            status_code=500, detail=f"bytetrack-frames failed: {e}"
        ) from e

    return {
        "frames": frames,
        "bytetrack_config": {
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
        },
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
