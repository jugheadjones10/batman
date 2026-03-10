"""Training API routes — GPU cluster submission via Fabric/SLURM and local GPU training."""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.models.training import (
    DatasetExportConfig,
    DatasetExportResult,
    LocalTrainingSubmitRequest,
    TrainingRunInfo,
    TrainingSubmitRequest,
)
from backend.app.services.dataset_exporter import DatasetExporter
from backend.app.services.gpu_service import GPU_CONFIGS, GPUJobState, gpu_service
from src.core.trainer import find_best_checkpoint

router = APIRouter(prefix="/projects/{project_name}/training", tags=["training"])

# TensorBoard process tracking (kept from previous implementation)
_tensorboard_processes: dict[str, dict] = {}

# Local training process tracking: run_name -> subprocess.Popen
_local_training_processes: dict[str, subprocess.Popen] = {}


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

    allowed_sources = set(config.data_sources or ["manual_data", "imports", "videos"])
    manual_ds_include = set(config.manual_datasets) if config.manual_datasets else None
    manual_ds_exclude = (
        set(config.exclude_manual_datasets) if config.exclude_manual_datasets else None
    )

    excluded_videos: set[str] = set()
    if videos_meta_path.exists():
        with open(videos_meta_path) as f:
            vm = json.load(f)
        for vid_key, vid_meta in vm.items():
            if vid_meta.get("exclude_from_training", False):
                excluded_videos.add(vid_key)

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
                source_type = "videos"
            else:
                source_type = "imports"

            if source_type not in allowed_sources:
                continue
            if source_type == "videos" and sub_dir.name in excluded_videos:
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
        exclude_videos=request.data.exclude_videos,
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


# ── Submit training locally (e.g. Windows GPU) ───────────────────────────


def _build_local_train_argv(
    project_path: Path,
    run_dir: Path,
    request: LocalTrainingSubmitRequest,
) -> list[str]:
    """Build argv for python -m cli.train (local run)."""
    argv = [
        sys.executable,
        "-m",
        "cli.train",
        "--project",
        str(project_path),
        "--output-dir",
        str(run_dir),
        "--device",
        "auto",
        "--model",
        request.training.model,
        "--epochs",
        str(request.training.epochs),
        "--image-size",
        str(request.training.image_size),
        "--lr",
        str(request.training.lr),
        "--patience",
        str(request.training.patience),
        "--grad-accum",
        str(request.training.grad_accum),
        "--train-split",
        str(request.data.train_split),
        "--val-split",
        str(request.data.val_split),
        "--test-split",
        str(request.data.test_split),
    ]
    if request.training.batch_size is not None:
        argv.extend(["--batch-size", str(request.training.batch_size)])
    if request.data.sources:
        argv.extend(["--sources", ",".join(request.data.sources)])
    if request.data.manual_split_strategy:
        argv.extend(["--manual-split-strategy", request.data.manual_split_strategy])
    if request.data.manual_datasets:
        argv.extend(["--manual-datasets", ",".join(request.data.manual_datasets)])
    if request.data.exclude_manual_datasets:
        argv.extend(["--exclude-manual-datasets", ",".join(request.data.exclude_manual_datasets)])
    if request.data.exclude_videos:
        argv.extend(["--exclude-videos", ",".join(request.data.exclude_videos)])
    if request.data.filter_classes:
        argv.extend(["--filter-classes", "|".join(request.data.filter_classes)])
    if request.data.max_frames_per_class is not None:
        argv.extend(["--max-frames-per-class", str(request.data.max_frames_per_class)])
    if request.infer_after:
        argv.append("--infer-after")
    if request.infer_test_only:
        argv.append("--infer-test-only")
    return argv


@router.post("/submit-local")
async def submit_training_local(project_name: str, request: LocalTrainingSubmitRequest):
    """Run training locally (e.g. on Windows GPU). Exports dataset then launches cli.train in background."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_config = load_project_config(project_path)
    classes = project_config.get("classes", [])
    if not classes:
        raise HTTPException(status_code=400, detail="No classes defined")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rfdetr_local_{timestamp}"
    if request.label:
        run_name = f"{run_name}_{request.label}"

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
        "status": "running",
        "progress": 0.0,
        "gpu_type": "local",
        "local_pid": None,
        "config": {
            "training": request.training.model_dump(),
            "data": request.data.model_dump(),
        },
        "created_at": now.isoformat(),
    }
    meta_path = run_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log_file = run_dir / "training.log"
    argv = _build_local_train_argv(project_path, run_dir, request)

    try:
        with open(log_file, "w") as log_f:
            process = subprocess.Popen(
                argv,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env=os.environ.copy(),
            )
    except Exception as e:
        meta["status"] = "failed"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")

    meta["local_pid"] = process.pid
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    _local_training_processes[run_name] = process

    async def poll_local_job():
        try:
            process.wait()
        except Exception:
            pass
        if run_name in _local_training_processes:
            del _local_training_processes[run_name]
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            return
        meta["status"] = "completed" if process.returncode == 0 else "failed"
        meta["completed_at"] = datetime.utcnow().isoformat()
        if process.returncode != 0:
            meta["progress"] = 0.0
        # Set checkpoint_path and metrics when run completed successfully (for inference load-model)
        if process.returncode == 0:
            run_dir = meta_path.parent
            results_path = run_dir / "results.json"
            if results_path.exists():
                try:
                    with open(results_path) as rf:
                        results = json.load(rf)
                    meta["checkpoint_path"] = results.get("checkpoint_path")
                    if results.get("metrics"):
                        meta["metrics"] = results["metrics"]
                except (json.JSONDecodeError, OSError):
                    pass
            if not meta.get("checkpoint_path"):
                best = find_best_checkpoint(run_dir)
                if best is not None:
                    meta["checkpoint_path"] = str(best)
        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    asyncio.create_task(asyncio.to_thread(poll_local_job))

    return {
        "run_name": run_name,
        "pid": process.pid,
        "message": "Local training started",
    }


# ── List / get training runs ────────────────────────────────────────────

_EPOCH_HEADER_RE = re.compile(r"Epoch:\s*\[(\d+)\]")
_BATCH_PROGRESS_RE = re.compile(r"Epoch:\s*\[\d+\]\s*\[\s*(\d+)/(\d+)\]")


def _parse_progress_from_log(log_path: Path, total_epochs: int) -> tuple[float, int | None]:
    """Read the tail of a training log and extract epoch-level progress.

    Returns (progress_fraction, current_epoch).  progress is 0.0-1.0.
    """
    if total_epochs <= 0 or not log_path.exists():
        return 0.0, None
    try:
        size = log_path.stat().st_size
        read_bytes = min(size, 8192)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - read_bytes))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return 0.0, None

    current_epoch = None
    batch_step = 0
    batch_total = 1
    for line in tail.splitlines():
        m = _EPOCH_HEADER_RE.search(line)
        if m:
            current_epoch = int(m.group(1))
            bm = _BATCH_PROGRESS_RE.search(line)
            if bm:
                batch_step = int(bm.group(1))
                batch_total = max(int(bm.group(2)), 1)

    if current_epoch is None:
        return 0.0, None

    progress = (current_epoch + batch_step / batch_total) / total_epochs
    return min(max(progress, 0.0), 1.0), current_epoch


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
        local_pid = meta.get("local_pid")
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
        elif local_pid and current_status == "running":
            proc = _local_training_processes.get(meta["name"])
            if proc is not None:
                ret = proc.poll()
                if ret is not None:
                    current_status = "completed" if ret == 0 else "failed"
                    meta["status"] = current_status
                    meta["completed_at"] = meta.get("completed_at") or datetime.utcnow().isoformat()
                    del _local_training_processes[meta["name"]]
                    with open(meta_path, "w") as mf:
                        json.dump(meta, mf, indent=2)
            else:
                # Process finished and was already cleaned up; re-read meta from disk
                # (poll_local_job may have written completed/failed)
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                    current_status = meta.get("status", current_status)
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

        # Parse live progress from training log for running jobs
        run_progress = meta.get("progress", 0.0)
        current_epoch = None
        total_epochs = None
        if current_status == "running":
            total_epochs = (meta.get("config") or {}).get("training", {}).get("epochs")
            if total_epochs:
                log_path = run_dir / "training.log"
                if not log_path.exists() and meta.get("gpu_type") != "local":
                    sshfs_log = Path("gpu-server") / f"data/projects/{project_name}/runs/{meta['name']}/training.log"
                    if sshfs_log.exists():
                        log_path = sshfs_log
                run_progress, current_epoch = _parse_progress_from_log(log_path, total_epochs)

        runs.append(
            TrainingRunInfo(
                id=meta["id"],
                name=meta["name"],
                status=current_status,
                model=meta.get("model", meta.get("base_model", "unknown")),
                gpu_type=meta.get("gpu_type"),
                slurm_job_id=meta.get("slurm_job_id"),
                progress=run_progress,
                current_epoch=current_epoch,
                total_epochs=total_epochs,
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


@router.get("/runs/{run_name}/local-logs")
async def stream_local_training_logs(project_name: str, run_name: str):
    """Stream local training log file via Server-Sent Events."""
    project_path = get_project_path(project_name)
    run_dir = project_path / "runs" / run_name
    log_path = run_dir / "training.log"
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Training run not found")
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("gpu_type") != "local":
        raise HTTPException(status_code=400, detail="Not a local training run")

    async def event_generator():
        try:
            if log_path.exists():
                with open(log_path) as f:
                    for line in f:
                        data = json.dumps({"type": "log", "stream": "stdout", "line": line.rstrip("\n")})
                        yield f"data: {data}\n\n"
            last_size = log_path.stat().st_size if log_path.exists() else 0
            for _ in range(3600):
                await asyncio.sleep(1)
                if not log_path.exists():
                    break
                try:
                    size = log_path.stat().st_size
                    if size > last_size:
                        with open(log_path) as f:
                            f.seek(last_size)
                            for line in f:
                                data = json.dumps({"type": "log", "stream": "stdout", "line": line.rstrip("\n")})
                                yield f"data: {data}\n\n"
                        last_size = log_path.stat().st_size
                except (OSError, IOError):
                    break
        except asyncio.CancelledError:
            pass
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
    """Cancel a running/queued training job (GPU cluster or local)."""
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
    local_pid = meta.get("local_pid")
    if job_id and gpu_service.is_connected:
        try:
            gpu_service.cancel_job(job_id)
        except Exception as e:
            logger.warning(f"scancel failed: {e}")
    elif local_pid and run_name in _local_training_processes:
        proc = _local_training_processes[run_name]
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception as e:
            logger.warning(f"Failed to kill local process: {e}")
        if run_name in _local_training_processes:
            del _local_training_processes[run_name]

    meta["status"] = "cancelled"
    meta["completed_at"] = datetime.utcnow().isoformat()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    tracked = gpu_service.get_tracked_job(run_name)
    if tracked:
        tracked.status = "cancelled"
        tracked.completed_at = meta["completed_at"]

    return {"status": "cancelled", "message": f"Training run '{run_name}' cancelled"}


# ── Delete ────────────────────────────────────────────────────────────────


@router.delete("/runs/{run_name}")
async def delete_training_run(project_name: str, run_name: str):
    """Delete a training run and all associated inference results."""
    import shutil

    project_path = get_project_path(project_name)
    run_dir = project_path / "runs" / run_name

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Training run not found")

    # Cancel if still running
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        local_pid = meta.get("local_pid")
        if run_name in _local_training_processes:
            proc = _local_training_processes[run_name]
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, Exception):
                try:
                    proc.kill()
                except Exception:
                    pass
            del _local_training_processes[run_name]
        job_id = meta.get("slurm_job_id")
        if job_id and gpu_service.is_connected:
            try:
                gpu_service.cancel_job(job_id)
            except Exception:
                pass

    # Stop TensorBoard if running
    tb_key = f"{project_name}_{run_name}"
    if tb_key in _tensorboard_processes:
        proc = _tensorboard_processes[tb_key]["process"]
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        del _tensorboard_processes[tb_key]

    # Delete the run directory
    shutil.rmtree(run_dir, ignore_errors=True)

    # Delete associated inference results
    inference_dir = project_path / "inference" / run_name
    if inference_dir.exists():
        shutil.rmtree(inference_dir, ignore_errors=True)

    return {"message": f"Training run '{run_name}' and associated inference results deleted"}


# ── Rename ────────────────────────────────────────────────────────────────

_VALID_NAME_RE = re.compile(r"^[\w\-. ]+$")


@router.patch("/runs/{run_name}/rename")
async def rename_training_run(project_name: str, run_name: str, body: dict):
    """Rename a training run directory, meta.json, and associated inference results."""
    new_name = (body.get("new_name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="new_name is required")
    if not _VALID_NAME_RE.match(new_name):
        raise HTTPException(status_code=400, detail="Invalid name. Use only letters, numbers, dashes, underscores, dots, and spaces.")
    if len(new_name) > 200:
        raise HTTPException(status_code=400, detail="Name too long (max 200 chars)")

    project_path = get_project_path(project_name)
    runs_dir = project_path / "runs"
    old_dir = runs_dir / run_name
    new_dir = runs_dir / new_name

    if not old_dir.exists():
        raise HTTPException(status_code=404, detail="Training run not found")
    if new_name == run_name:
        return {"name": new_name}
    if new_dir.exists():
        raise HTTPException(status_code=409, detail=f"A run named '{new_name}' already exists")

    old_dir.rename(new_dir)

    meta_path = new_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["name"] = new_name
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Rename inference results directory
    inference_dir = project_path / "inference"
    old_inf = inference_dir / run_name
    new_inf = inference_dir / new_name
    if old_inf.exists() and not new_inf.exists():
        old_inf.rename(new_inf)
        # Update run_name in result.json files
        for result_json in new_inf.rglob("result.json"):
            try:
                with open(result_json) as f:
                    data = json.load(f)
                if data.get("run_name") == run_name:
                    data["run_name"] = new_name
                    with open(result_json, "w") as f:
                        json.dump(data, f, indent=2)
            except (json.JSONDecodeError, OSError):
                pass

    # Update TensorBoard tracking if running
    old_tb_key = f"{project_name}_{run_name}"
    new_tb_key = f"{project_name}_{new_name}"
    if old_tb_key in _tensorboard_processes:
        _tensorboard_processes[new_tb_key] = _tensorboard_processes.pop(old_tb_key)

    # Update local training process tracking
    if run_name in _local_training_processes:
        _local_training_processes[new_name] = _local_training_processes.pop(run_name)

    return {"name": new_name}


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
