"""Inference API routes."""

import asyncio
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.config import settings
from backend.app.models.training import InferenceConfig, InferenceGPUSubmitRequest
from backend.app.services.gpu_service import GPUJobState, gpu_service
from backend.app.services.inference_runner import inference_runner
from backend.app.services.tracker import TrackingConfig
from backend.app.services import z_estimator
from src.core.project import Project
from src.core.trainer import find_best_checkpoint

SGT = timezone(timedelta(hours=8))

router = APIRouter(prefix="/projects/{project_name}/inference", tags=["inference"])


class LoadModelRequest(BaseModel):
    run_id: int
    device: Optional[str] = None  # auto, cuda, mps, cpu; default from settings


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
    model_size = "base"
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

            # Extract variant: "rf-detr-small" → "small", or from config
            for variant in ("nano", "small", "medium", "base", "large"):
                if variant in model_field:
                    model_size = variant
                    break
            cfg_model = (meta.get("config") or {}).get("training", {}).get("model")
            if cfg_model:
                model_size = cfg_model

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
            break

    if not checkpoint_path:
        raise HTTPException(status_code=404, detail="Model checkpoint not found")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint file not found")

    device = request.device if request.device else settings.device
    await inference_runner.load_model(
        checkpoint_path, classes, model_type, device=device, model_size=model_size
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

    project = Project.load(project_path)
    inference_id = datetime.now(SGT).strftime("%Y%m%d_%H%M%S")
    result_dir = project_path / "inference" / run_name / video_id / inference_id
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "detected.mp4"

    result = await inference_runner.run_on_video_full(
        video_path,
        output_path=output_path,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        enable_tracking=config.enable_tracking,
        tracking_config=tracking_config,
        detection_interval=config.detection_interval,
    )

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
        "frames": result.get("results", []),
    }
    with open(result_dir / "result.json", "w") as f:
        json.dump(persist_data, f, indent=2)

    result["persisted"] = True
    result["run_name"] = run_name
    result["inference_id"] = inference_id
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
            has_video = (
                project.inference_dir / run_name / vid_id / inf_id / "detected.mp4"
            ).exists()
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
        result["has_video"] = (
            project.inference_dir / run_name / video_id / inference_id / "detected.mp4"
        ).exists()
    else:
        result["has_video"] = (
            project.inference_dir / run_name / video_id / "detected.mp4"
        ).exists()
    return result


@router.get("/results/{run_name}/{video_id}/{inference_id}/video")
async def get_inference_result_video(
    request: Request,
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
):
    """Stream the detected video (with overlay) for an inference result. Supports Range for seeking."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    if inference_id == "legacy":
        video_path = project_path / "inference" / run_name / video_id / "detected.mp4"
    else:
        video_path = (
            project_path / "inference" / run_name / video_id / inference_id / "detected.mp4"
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


class ExtractFramesRequest(BaseModel):
    frame_numbers: list[int]


@router.post("/results/{run_name}/{video_id}/{inference_id}/extract-frames")
async def extract_inference_frames(
    project_name: str,
    run_name: str,
    video_id: str,
    inference_id: str,
    request: ExtractFramesRequest,
):
    """Extract selected frames as JPEG images with their detection data, returned as a ZIP."""
    if not request.frame_numbers:
        raise HTTPException(status_code=400, detail="No frame numbers specified")

    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    # Load inference result
    project = Project.load(project_path)
    result = project.get_inference_result(run_name, video_id, inference_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Inference result not found")

    # Build lookup from frame_number -> frame data
    frames_by_number: dict[int, dict] = {}
    for frame in result.get("frames", []):
        frames_by_number[frame["frame_number"]] = frame

    # Resolve source video path
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

    # Open video and get resolution
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video file")

    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sort frame numbers for sequential seeking efficiency
    sorted_frames = sorted(set(request.frame_numbers))

    # Build ZIP in memory
    zip_buffer = io.BytesIO()
    export_frames = []

    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for frame_num in sorted_frames:
                if frame_num < 0 or frame_num >= total_frames:
                    logger.warning(f"Skipping out-of-range frame {frame_num}")
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_num}")
                    continue

                # Encode as JPEG
                success, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    logger.warning(f"Could not encode frame {frame_num}")
                    continue

                filename = f"frame_{frame_num:06d}.jpg"
                zf.writestr(filename, jpeg_buf.tobytes())

                # Look up detection data
                frame_data = frames_by_number.get(frame_num)
                export_frames.append(
                    {
                        "frame_number": frame_num,
                        "timestamp": frame_data["timestamp"]
                        if frame_data
                        else frame_num / cap.get(cv2.CAP_PROP_FPS),
                        "image_filename": filename,
                        "detections": frame_data.get("detections", []) if frame_data else [],
                    }
                )

            # Write detections JSON
            detections_json = {
                "project": project_name,
                "run_name": run_name,
                "video_id": video_id,
                "inference_id": inference_id,
                "video_resolution": {"width": vid_width, "height": vid_height},
                "frames": export_frames,
            }
            zf.writestr("detections.json", json.dumps(detections_json, indent=2))
    finally:
        cap.release()

    zip_buffer.seek(0)
    zip_filename = f"{project_name}_{run_name}_{video_id}_{inference_id}_frames.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


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
    class_name: str = "crane hook"


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

    calibration_data = {
        "labels": [l.model_dump() for l in request.labels],
        "class_name": request.class_name,
        "size_metric": "h_px",
        "video_resolution": video_resolution,
    }

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

    from src.core.inference import Detection as DetObj, draw_detections
    import subprocess as _sp
    import time as _time

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = result_dir / "detected.mp4"
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
