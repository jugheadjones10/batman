"""Training API routes — GPU cluster submission via Fabric/SLURM."""

import asyncio
import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.models.training import (
    DatasetExportConfig,
    DatasetExportResult,
    TrainingRunInfo,
    TrainingSubmitRequest,
)
from backend.app.services.dataset_exporter import DatasetExporter
from backend.app.services.gpu_service import GPU_CONFIGS, GPUJobState, gpu_service

router = APIRouter(prefix="/projects/{project_name}/training", tags=["training"])

# TensorBoard process tracking (kept from previous implementation)
_tensorboard_processes: dict[str, dict] = {}


def _find_free_port(start_port: int = 6006, max_attempts: int = 100) -> int:
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port in range {start_port}-{start_port + max_attempts}")


# ── Dataset export (kept) ────────────────────────────────────────────────


@router.post("/export-dataset", response_model=DatasetExportResult)
async def export_dataset(
    project_name: str,
    config: DatasetExportConfig = DatasetExportConfig(),
):
    """Export labeled data as a dataset."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_config = load_project_config(project_path)
    classes = project_config.get("classes", [])
    if not classes:
        raise HTTPException(status_code=400, detail="No classes defined")

    videos_meta_path = project_path / "videos" / "videos.json"
    video_dir_names: set[str] = set()
    if videos_meta_path.exists():
        with open(videos_meta_path) as f:
            videos_meta = json.load(f)
        video_dir_names = set(videos_meta.keys())

    allowed_sources = set(config.data_sources or ["manual_data", "imports"])
    manual_ds_include = set(config.manual_datasets) if config.manual_datasets else None
    manual_ds_exclude = (
        set(config.exclude_manual_datasets) if config.exclude_manual_datasets else None
    )

    def _is_manual_dir(name: str) -> bool:
        return name == "manual_data" or name.startswith("manual_data__")

    def _manual_dataset_name(name: str) -> str:
        return "(root)" if name == "manual_data" else name[len("manual_data__"):]

    def _should_include_manual(dir_name: str) -> bool:
        if manual_ds_include is None and manual_ds_exclude is None:
            return True
        canonical = _manual_dataset_name(dir_name)
        if manual_ds_include is not None:
            return canonical in manual_ds_include
        if manual_ds_exclude is not None:
            return canonical not in manual_ds_exclude
        return True

    frames = []
    frames_dir = project_path / "frames"
    if frames_dir.exists():
        for sub_dir in frames_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            meta_path = sub_dir / "frames.json"
            if not meta_path.exists():
                continue

            if _is_manual_dir(sub_dir.name):
                source_type = "manual_data"
            elif sub_dir.name in video_dir_names:
                source_type = "video"
            else:
                source_type = "imports"

            if source_type == "video":
                continue
            if source_type not in allowed_sources:
                continue
            if source_type == "manual_data" and not _should_include_manual(sub_dir.name):
                continue

            with open(meta_path) as f:
                frames_meta = json.load(f)
            for frame_id, frame_data in frames_meta.items():
                fid = (
                    int(frame_id)
                    if isinstance(frame_id, str) and frame_id.isdigit()
                    else frame_id
                )
                vid = (
                    int(sub_dir.name)
                    if sub_dir.name.lstrip("-").isdigit()
                    else sub_dir.name
                )
                frames.append({"id": fid, "video_id": vid, **frame_data})

    annotations_path = project_path / "labels" / "current" / "annotations.json"
    annotations = []
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations_meta = json.load(f)
        for ann_id, ann_data in annotations_meta.items():
            annotations.append({"id": int(ann_id), **ann_data})

    exporter = DatasetExporter(project_path)
    result = await exporter.export(
        frames=frames,
        annotations=annotations,
        classes=classes,
        format=config.format,
        split_by_video=config.split_by_video,
        manual_data_split_strategy=config.manual_data_split_strategy,
    )
    return DatasetExportResult(**result)


# ── Submit training to GPU cluster ───────────────────────────────────────


@router.post("/submit")
async def submit_training(project_name: str, request: TrainingSubmitRequest):
    """Submit a training job to the GPU cluster."""
    if not gpu_service.is_connected:
        raise HTTPException(status_code=400, detail="Not connected to GPU cluster")

    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_config = load_project_config(project_path)
    classes = project_config.get("classes", [])
    if not classes:
        raise HTTPException(status_code=400, detail="No classes defined")

    project_dir = f"data/projects/{project_name}"

    # Auto batch size from GPU config if not specified
    gpu_cfg = GPU_CONFIGS.get(request.gpu.gpu_type, {})
    batch_size = request.training.batch_size or gpu_cfg.get("default_batch", 8)

    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rfdetr_{request.gpu.gpu_type}_{timestamp}"
    if request.label:
        run_name = f"{run_name}_{request.label}"

    output_dir = f"{project_dir}/runs/{run_name}"
    output_dataset = f"{project_dir}/exports/coco"

    # Push project data to cluster (can take a while over SSH for many files)
    logger.info("Submitting training: pushing project data to cluster (may take a minute for large projects)...")
    try:
        await asyncio.to_thread(gpu_service.push_project_data, project_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to push data: {e}")

    # Generate SLURM script
    script, job_name = gpu_service.generate_training_script(
        project_dir=project_dir,
        output_dir=output_dir,
        output_dataset=output_dataset,
        model=request.training.model,
        epochs=request.training.epochs,
        batch_size=batch_size,
        image_size=request.training.image_size,
        lr=request.training.lr,
        patience=request.training.patience,
        grad_accum=request.training.grad_accum,
        gpu_type=request.gpu.gpu_type,
        num_gpus=request.gpu.num_gpus,
        time_limit=request.gpu.time_limit,
        filter_classes=request.data.filter_classes,
        max_frames_per_class=request.data.max_frames_per_class,
        sources=request.data.sources,
        manual_split_strategy=request.data.manual_split_strategy,
        manual_datasets=request.data.manual_datasets,
        exclude_manual_datasets=request.data.exclude_manual_datasets,
        infer_after=request.infer_after,
        infer_test_only=request.infer_test_only,
    )

    # Submit to SLURM
    try:
        job_id = await asyncio.to_thread(gpu_service.submit_slurm_job, script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SLURM submission failed: {e}")

    # Save meta.json locally
    runs_dir = project_path / "runs"
    runs_dir.mkdir(exist_ok=True)
    run_dir = runs_dir / run_name
    run_dir.mkdir(exist_ok=True)

    run_id = len(list(runs_dir.iterdir()))
    now = datetime.utcnow()

    meta = {
        "id": run_id,
        "name": run_name,
        "model": f"rf-detr-{request.training.model}",
        "status": "queued",
        "progress": 0.0,
        "slurm_job_id": job_id,
        "gpu_type": request.gpu.gpu_type,
        "config": {
            "training": request.training.model_dump(),
            "gpu": request.gpu.model_dump(),
            "data": request.data.model_dump(),
        },
        "created_at": now.isoformat(),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Track and start background polling
    job_state = GPUJobState(
        job_id=job_id,
        run_name=run_name,
        job_type="training",
        gpu_type=request.gpu.gpu_type,
        project_name=project_name,
        project_dir=project_dir,
        output_dir=output_dir,
        submitted_at=now.isoformat(),
        log_file=f"logs/slurm_{job_id}_{job_name}.out",
        err_file=f"logs/slurm_{job_id}_{job_name}.err",
    )
    gpu_service.track_job(job_state)

    asyncio.create_task(gpu_service.poll_job_until_done(job_state, project_path))

    return {
        "job_id": job_id,
        "run_name": run_name,
        "message": "Training job submitted to GPU cluster",
    }


# ── List / get training runs ────────────────────────────────────────────


def _is_local_training_run(run_dir: Path) -> bool:
    """True if run_dir looks like a local training run (no meta.json from cluster submit)."""
    if (run_dir / "meta.json").exists():
        return False
    return (
        (run_dir / "training_config.json").exists()
        or (run_dir / "results.json").exists()
        or (run_dir / "tensorboard").is_dir()
        or bool(list(run_dir.glob("*.pth")))
    )


def _local_run_info(project_name: str, run_dir: Path, run_index: int) -> TrainingRunInfo | None:
    """Build TrainingRunInfo for a local-only run (no meta.json)."""
    name = run_dir.name
    tb_key = f"{project_name}_{name}"
    tensorboard_url = None
    if tb_key in _tensorboard_processes:
        tb_info = _tensorboard_processes[tb_key]
        if tb_info["process"].poll() is None:
            tensorboard_url = f"http://localhost:{tb_info['port']}"

    config_path = run_dir / "training_config.json"
    created_at = datetime.fromtimestamp(run_dir.stat().st_mtime)
    model = "unknown"
    config = None
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            args = config.get("arguments", {})
            model = f"rf-detr-{args.get('model', 'base')}"
            ts = config.get("timestamp")
            if ts:
                created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    metrics = None
    checkpoint_path = None
    results_path = run_dir / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                metrics = json.load(f)
            checkpoint_path = metrics.get("checkpoint_path")
        except (json.JSONDecodeError, OSError):
            pass

    return TrainingRunInfo(
        id=run_index,
        name=name,
        status="completed",
        model=model,
        gpu_type="local",
        slurm_job_id=None,
        progress=1.0,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        latency_ms=None,
        tensorboard_url=tensorboard_url,
        config=config,
        started_at=None,
        completed_at=None,
        created_at=created_at,
    )


@router.get("/runs", response_model=list[TrainingRunInfo])
async def list_training_runs(project_name: str):
    """List all training runs for a project."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    runs = []
    runs_dir = project_path / "runs"
    if not runs_dir.exists():
        return runs

    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Check live SLURM status for runs that still appear active (so UI updates when job finishes)
        current_status = meta.get("status", "unknown")
        job_id = meta.get("slurm_job_id")
        if job_id and current_status in ("queued", "running", "pending") and gpu_service.is_connected:
            try:
                status_info = gpu_service.get_job_status(job_id)
                slurm_status = status_info.get("status", "unknown")
                if slurm_status in ("completed", "failed", "cancelled", "timeout"):
                    current_status = slurm_status
                    meta["status"] = slurm_status
                    meta["completed_at"] = meta.get("completed_at") or datetime.utcnow().isoformat()
                    with open(meta_path, "w") as mf:
                        json.dump(meta, mf, indent=2)
            except Exception:
                pass
        else:
            tracked = gpu_service.get_tracked_job(meta["name"])
            if tracked and tracked.status != meta.get("status"):
                meta["status"] = tracked.status

        # TensorBoard URL
        tb_key = f"{project_name}_{meta['name']}"
        tensorboard_url = None
        if tb_key in _tensorboard_processes:
            tb_info = _tensorboard_processes[tb_key]
            if tb_info["process"].poll() is None:
                tensorboard_url = f"http://localhost:{tb_info['port']}"

        runs.append(
            TrainingRunInfo(
                id=meta["id"],
                name=meta["name"],
                status=meta.get("status", "unknown"),
                model=meta.get("model", meta.get("base_model", "unknown")),
                gpu_type=meta.get("gpu_type"),
                slurm_job_id=meta.get("slurm_job_id"),
                progress=meta.get("progress", 0.0),
                metrics=meta.get("metrics"),
                checkpoint_path=meta.get("checkpoint_path"),
                latency_ms=meta.get("latency_ms"),
                tensorboard_url=tensorboard_url,
                config=meta.get("config"),
                started_at=(
                    datetime.fromisoformat(meta["started_at"])
                    if meta.get("started_at")
                    else None
                ),
                completed_at=(
                    datetime.fromisoformat(meta["completed_at"])
                    if meta.get("completed_at")
                    else None
                ),
                created_at=datetime.fromisoformat(meta["created_at"]),
            )
        )

    # Include local training runs (no meta.json): e.g. CLI runs on a local GPU machine
    seen_names = {r.name for r in runs}
    local_index = 0
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name in seen_names:
            continue
        if not _is_local_training_run(run_dir):
            continue
        info = _local_run_info(project_name, run_dir, 900000 + local_index)
        if info:
            runs.append(info)
            seen_names.add(run_dir.name)
            local_index += 1

    # Keep newest first (by created_at)
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs


# ── Log streaming (SSE) ─────────────────────────────────────────────────


@router.get("/runs/{run_name}/logs")
async def stream_training_logs(project_name: str, run_name: str):
    """Stream SLURM training logs via Server-Sent Events."""
    project_path = get_project_path(project_name)
    meta_path = project_path / "runs" / run_name / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Training run not found")

    with open(meta_path) as f:
        meta = json.load(f)

    job_id = meta.get("slurm_job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="No SLURM job ID for this run")

    if not gpu_service.is_connected:
        raise HTTPException(status_code=400, detail="Not connected to GPU cluster")

    # Derive the job name from the SLURM script naming convention
    model = meta.get("config", {}).get("training", {}).get("model", "base")
    gpu_type = meta.get("gpu_type", "a100-80")
    job_name = f"rfdetr-{model}-{gpu_type}"

    async def event_generator():
        try:
            async for stream, line in gpu_service.stream_logs(job_id, job_name):
                data = json.dumps({"type": "log", "stream": stream, "line": line.rstrip("\n")})
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


# ── Cancel ───────────────────────────────────────────────────────────────


@router.post("/runs/{run_name}/cancel")
async def cancel_training_run(project_name: str, run_name: str):
    """Cancel a running/queued training job on the GPU cluster."""
    project_path = get_project_path(project_name)
    meta_path = project_path / "runs" / run_name / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Training run not found")

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("status") not in ("running", "queued", "pending"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run with status '{meta.get('status')}'",
        )

    job_id = meta.get("slurm_job_id")
    if job_id and gpu_service.is_connected:
        try:
            gpu_service.cancel_job(job_id)
        except Exception as e:
            logger.warning(f"scancel failed: {e}")

    meta["status"] = "cancelled"
    meta["completed_at"] = datetime.utcnow().isoformat()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    tracked = gpu_service.get_tracked_job(run_name)
    if tracked:
        tracked.status = "cancelled"
        tracked.completed_at = meta["completed_at"]

    return {"status": "cancelled", "message": f"Training run '{run_name}' cancelled"}


# ── TensorBoard (kept) ──────────────────────────────────────────────────


@router.post("/runs/{run_name}/tensorboard/start")
async def start_tensorboard(project_name: str, run_name: str):
    """Start TensorBoard for a training run."""
    project_path = get_project_path(project_name)
    run_dir = project_path / "runs" / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Training run not found")

    # Also check SSHFS mount
    sshfs_run = Path("gpu-server") / f"data/projects/{project_name}/runs/{run_name}"
    logdir = str(sshfs_run) if sshfs_run.exists() else str(run_dir)

    tb_key = f"{project_name}_{run_name}"
    if tb_key in _tensorboard_processes:
        existing = _tensorboard_processes[tb_key]
        if existing["process"].poll() is None:
            return {
                "status": "already_running",
                "port": existing["port"],
                "url": f"http://localhost:{existing['port']}",
            }
        del _tensorboard_processes[tb_key]

    port = _find_free_port(6006)
    try:
        process = subprocess.Popen(
            ["tensorboard", "--logdir", logdir, "--port", str(port), "--bind_all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.sleep(1)
        if process.poll() is not None:
            stderr = process.stderr.read().decode() if process.stderr else ""
            raise HTTPException(
                status_code=500, detail=f"TensorBoard failed: {stderr}"
            )

        _tensorboard_processes[tb_key] = {"process": process, "port": port}
        return {"status": "started", "port": port, "url": f"http://localhost:{port}"}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="TensorBoard not installed")


@router.post("/runs/{run_name}/tensorboard/stop")
async def stop_tensorboard(project_name: str, run_name: str):
    tb_key = f"{project_name}_{run_name}"
    if tb_key not in _tensorboard_processes:
        raise HTTPException(status_code=404, detail="TensorBoard not running")

    process = _tensorboard_processes[tb_key]["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    del _tensorboard_processes[tb_key]
    return {"status": "stopped"}


@router.get("/runs/{run_name}/tensorboard/status")
async def get_tensorboard_status(project_name: str, run_name: str):
    tb_key = f"{project_name}_{run_name}"
    if tb_key not in _tensorboard_processes:
        return {"running": False}

    info = _tensorboard_processes[tb_key]
    if info["process"].poll() is None:
        return {"running": True, "port": info["port"], "url": f"http://localhost:{info['port']}"}
    del _tensorboard_processes[tb_key]
    return {"running": False}
