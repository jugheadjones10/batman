"""Training API routes."""

import asyncio
import json
import re
import subprocess
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.app.api.projects import get_project_path, load_project_config
from backend.app.models.training import (
    DatasetExportConfig,
    DatasetExportResult,
    InferenceConfig,
    TrainingConfig,
    TrainingProgress,
    TrainingRequest,
    TrainingRunInfo,
)
from backend.app.services.dataset_exporter import DatasetExporter
from backend.app.services.trainer import ModelTrainer

router = APIRouter(prefix="/projects/{project_name}/training", tags=["training"])


# In-memory training status (would use database in production)
_training_jobs: dict[str, TrainingProgress] = {}

# Track running training tasks: {run_name: asyncio.Task}
_training_tasks: dict[str, asyncio.Task] = {}

# Track TensorBoard processes: {run_name: {"process": subprocess.Popen, "port": int}}
_tensorboard_processes: dict[str, dict] = {}


def _find_free_port(start_port: int = 6006, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


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

    # Load frames
    frames = []
    frames_dir = project_path / "frames"
    if frames_dir.exists():
        for video_dir in frames_dir.iterdir():
            if not video_dir.is_dir():
                continue
            meta_path = video_dir / "frames.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    frames_meta = json.load(f)
                for frame_id, frame_data in frames_meta.items():
                    frames.append({
                        "id": int(frame_id),
                        "video_id": int(video_dir.name),
                        **frame_data,
                    })

    # Load annotations
    annotations_path = project_path / "labels" / "current" / "annotations.json"
    annotations = []
    if annotations_path.exists():
        with open(annotations_path) as f:
            annotations_meta = json.load(f)
        for ann_id, ann_data in annotations_meta.items():
            # Filter out unapproved if requested
            if not config.include_unapproved:
                # Would check frame approval status
                pass
            annotations.append({
                "id": int(ann_id),
                **ann_data,
            })

    # Export dataset
    exporter = DatasetExporter(project_path)
    result = await exporter.export(
        frames=frames,
        annotations=annotations,
        classes=classes,
        format=config.format,
        split_by_video=config.split_by_video,
    )

    return DatasetExportResult(**result)


@router.post("/start")
async def start_training(
    project_name: str,
    request: TrainingRequest,
):
    """Start a training run."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    project_config = load_project_config(project_path)
    classes = project_config.get("classes", [])

    if not classes:
        raise HTTPException(status_code=400, detail="No classes defined")

    # Check for existing dataset
    exports_dir = project_path / "exports"
    if not exports_dir.exists() or not any(exports_dir.iterdir()):
        raise HTTPException(
            status_code=400,
            detail="No exported dataset found. Export dataset first.",
        )

    # Use most recent export
    latest_export = max(exports_dir.iterdir(), key=lambda p: p.stat().st_mtime)
    
    # RF-DETR uses COCO format, YOLO uses YOLO format
    if request.config.base_model.startswith("rfdetr"):
        dataset_path = latest_export / "coco"
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail="COCO format dataset not found for RF-DETR")
    else:
        dataset_path = latest_export / "yolo"
        if not dataset_path.exists():
            raise HTTPException(status_code=400, detail="YOLO format dataset not found")

    # Create training run record
    runs_dir = project_path / "runs"
    runs_dir.mkdir(exist_ok=True)

    run_id = len(list(runs_dir.iterdir())) + 1
    run_dir = runs_dir / request.name

    now = datetime.utcnow()

    run_meta = {
        "id": run_id,
        "name": request.name,
        "label_iteration_id": request.label_iteration_id,
        "base_model": request.config.base_model,
        "config": request.config.model_dump(),
        "status": "pending",
        "progress": 0.0,
        "created_at": now.isoformat(),
    }

    run_dir.mkdir(exist_ok=True)
    with open(run_dir / "meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    _training_jobs[str(run_id)] = TrainingProgress(
        run_id=run_id,
        status="pending",
        progress=0.0,
        current_epoch=0,
        total_epochs=request.config.epochs,
    )

    # Create and track the training task
    task = asyncio.create_task(
        _run_training(
            str(run_id),
            project_path,
            run_dir,
            dataset_path,
            request.config,
        )
    )
    _training_tasks[request.name] = task

    return {
        "run_id": run_id,
        "message": "Training started",
    }


@router.get("/runs", response_model=list[TrainingRunInfo])
async def list_training_runs(project_name: str):
    """List all training runs."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    runs = []
    runs_dir = project_path / "runs"

    if not runs_dir.exists():
        return runs

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Check if TensorBoard is running for this run
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
                label_iteration_id=meta.get("label_iteration_id", 0),
                base_model=meta["base_model"],
                status=meta.get("status", "unknown"),
                progress=meta.get("progress", 0.0),
                metrics=meta.get("metrics"),
                checkpoint_path=meta.get("checkpoint_path"),
                latency_ms=meta.get("latency_ms"),
                tensorboard_url=tensorboard_url,
                started_at=datetime.fromisoformat(meta["started_at"]) if meta.get("started_at") else None,
                completed_at=datetime.fromisoformat(meta["completed_at"]) if meta.get("completed_at") else None,
                created_at=datetime.fromisoformat(meta["created_at"]),
            )
        )

    return runs


@router.get("/runs/{run_id}", response_model=TrainingRunInfo)
async def get_training_run(project_name: str, run_id: int):
    """Get training run details."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    runs_dir = project_path / "runs"

    for run_dir in runs_dir.iterdir():
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if meta["id"] == run_id:
            return TrainingRunInfo(
                id=meta["id"],
                name=meta["name"],
                label_iteration_id=meta.get("label_iteration_id", 0),
                base_model=meta["base_model"],
                status=meta.get("status", "unknown"),
                progress=meta.get("progress", 0.0),
                metrics=meta.get("metrics"),
                checkpoint_path=meta.get("checkpoint_path"),
                latency_ms=meta.get("latency_ms"),
                started_at=datetime.fromisoformat(meta["started_at"]) if meta.get("started_at") else None,
                completed_at=datetime.fromisoformat(meta["completed_at"]) if meta.get("completed_at") else None,
                created_at=datetime.fromisoformat(meta["created_at"]),
            )

    raise HTTPException(status_code=404, detail="Training run not found")


@router.get("/runs/{run_id}/progress", response_model=TrainingProgress)
async def get_training_progress(project_name: str, run_id: int):
    """Get training progress."""
    if str(run_id) not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    return _training_jobs[str(run_id)]


@router.post("/runs/{run_name}/tensorboard/start")
async def start_tensorboard(project_name: str, run_name: str):
    """Start TensorBoard for a training run."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    run_dir = project_path / "runs" / run_name
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Training run not found")

    # Check if TensorBoard is already running for this run
    tb_key = f"{project_name}_{run_name}"
    if tb_key in _tensorboard_processes:
        existing = _tensorboard_processes[tb_key]
        # Check if process is still running
        if existing["process"].poll() is None:
            return {
                "status": "already_running",
                "port": existing["port"],
                "url": f"http://localhost:{existing['port']}",
            }
        else:
            # Process died, clean up
            del _tensorboard_processes[tb_key]

    # Find a free port
    try:
        port = _find_free_port(6006)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Start TensorBoard process
    try:
        process = subprocess.Popen(
            [
                "tensorboard",
                "--logdir", str(run_dir),
                "--port", str(port),
                "--bind_all",  # Allow access from any interface
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give it a moment to start
        await asyncio.sleep(1)

        # Check if it started successfully
        if process.poll() is not None:
            stderr = process.stderr.read().decode() if process.stderr else ""
            raise HTTPException(status_code=500, detail=f"TensorBoard failed to start: {stderr}")

        _tensorboard_processes[tb_key] = {
            "process": process,
            "port": port,
        }

        logger.info(f"Started TensorBoard for {run_name} on port {port}")

        return {
            "status": "started",
            "port": port,
            "url": f"http://localhost:{port}",
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="TensorBoard not found. Install with: pip install tensorboard"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start TensorBoard: {e}")


@router.post("/runs/{run_name}/tensorboard/stop")
async def stop_tensorboard(project_name: str, run_name: str):
    """Stop TensorBoard for a training run."""
    tb_key = f"{project_name}_{run_name}"
    
    if tb_key not in _tensorboard_processes:
        raise HTTPException(status_code=404, detail="TensorBoard not running for this run")

    existing = _tensorboard_processes[tb_key]
    process = existing["process"]
    
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    del _tensorboard_processes[tb_key]
    logger.info(f"Stopped TensorBoard for {run_name}")

    return {"status": "stopped"}


@router.get("/runs/{run_name}/tensorboard/status")
async def get_tensorboard_status(project_name: str, run_name: str):
    """Get TensorBoard status for a training run."""
    tb_key = f"{project_name}_{run_name}"
    
    if tb_key not in _tensorboard_processes:
        return {"running": False}

    existing = _tensorboard_processes[tb_key]
    process = existing["process"]
    
    if process.poll() is None:
        return {
            "running": True,
            "port": existing["port"],
            "url": f"http://localhost:{existing['port']}",
        }
    else:
        # Process died, clean up
        del _tensorboard_processes[tb_key]
        return {"running": False}


@router.post("/runs/{run_name}/cancel")
async def cancel_training_run(project_name: str, run_name: str):
    """Cancel a running training job."""
    project_path = get_project_path(project_name)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    run_dir = project_path / "runs" / run_name
    meta_path = run_dir / "meta.json"
    
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Training run not found")

    # Load current meta
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("status") not in ["running", "pending"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel run with status '{meta.get('status')}'. Only 'running' or 'pending' runs can be cancelled."
        )

    # Cancel the asyncio task if it exists
    if run_name in _training_tasks:
        task = _training_tasks[run_name]
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled training task for {run_name}")
        del _training_tasks[run_name]

    # Try to kill any related training processes (RF-DETR spawns subprocesses)
    try:
        import os
        import signal
        
        # Find processes by checking for the run directory in command line
        result = subprocess.run(
            ["pgrep", "-f", str(run_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid_str in pids:
                if pid_str:
                    try:
                        pid = int(pid_str)
                        os.kill(pid, signal.SIGTERM)
                        logger.info(f"Sent SIGTERM to process {pid}")
                    except (ValueError, ProcessLookupError):
                        pass
    except Exception as e:
        logger.warning(f"Could not kill training processes: {e}")

    # Update meta.json to cancelled
    meta["status"] = "cancelled"
    meta["error"] = "Cancelled by user"
    meta["completed_at"] = datetime.utcnow().isoformat()
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Update in-memory status
    run_id = str(meta.get("id", ""))
    if run_id in _training_jobs:
        _training_jobs[run_id].status = "cancelled"

    logger.info(f"Training run {run_name} cancelled")

    return {
        "status": "cancelled",
        "message": f"Training run '{run_name}' has been cancelled",
    }


async def _monitor_training_progress(
    run_id: str,
    run_dir: Path,
    total_epochs: int,
    stop_event: asyncio.Event,
):
    """Monitor RF-DETR training progress by parsing log file."""
    log_path = run_dir / "log.txt"
    meta_path = run_dir / "meta.json"
    
    while not stop_event.is_set():
        try:
            if log_path.exists():
                with open(log_path, 'r') as f:
                    content = f.read()
                
                # Find the latest epoch from log
                # RF-DETR logs like: {"epoch": 0, ...}
                epoch_matches = re.findall(r'"epoch":\s*(\d+)', content)
                if epoch_matches:
                    current_epoch = int(epoch_matches[-1]) + 1  # +1 because epoch is 0-indexed
                    progress = min(current_epoch / total_epochs, 0.99)  # Cap at 99% until done
                    
                    # Update in-memory status
                    if run_id in _training_jobs:
                        _training_jobs[run_id].progress = progress
                        _training_jobs[run_id].current_epoch = current_epoch
                    
                    # Update meta.json
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        meta["progress"] = progress
                        meta["current_epoch"] = current_epoch
                        with open(meta_path, 'w') as f:
                            json.dump(meta, f, indent=2)
            
            await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logger.warning(f"Progress monitor error: {e}")
            await asyncio.sleep(5)


async def _run_training(
    run_id: str,
    project_path: Path,
    run_dir: Path,
    dataset_path: Path,
    config: TrainingConfig,
):
    """Background task for training."""
    stop_event = asyncio.Event()
    monitor_task = None
    run_name = run_dir.name
    
    try:
        _training_jobs[run_id].status = "running"

        # Update meta
        meta_path = run_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "running"
        meta["started_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Start progress monitor for RF-DETR
        if config.base_model.startswith("rfdetr"):
            monitor_task = asyncio.create_task(
                _monitor_training_progress(run_id, run_dir, config.epochs, stop_event)
            )

        # Run training
        trainer = ModelTrainer(project_path)
        result = await trainer.train(
            run_name=run_dir.name,
            dataset_path=dataset_path,
            base_model=config.base_model,
            config=config.model_dump(),
        )

        # Stop monitor
        stop_event.set()
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # Update meta with results
        now = datetime.utcnow()
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = result["status"]
        meta["progress"] = 1.0
        meta["checkpoint_path"] = result.get("checkpoint_path")
        meta["metrics"] = result.get("metrics")
        meta["latency_ms"] = result.get("latency_ms")
        meta["completed_at"] = now.isoformat()

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        _training_jobs[run_id].status = "completed"
        _training_jobs[run_id].progress = 1.0
        _training_jobs[run_id].metrics = result.get("metrics")

        logger.info(f"Training completed: {run_dir.name}")

    except asyncio.CancelledError:
        logger.info(f"Training cancelled: {run_dir.name}")
        
        # Stop monitor
        stop_event.set()
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Update meta to cancelled (may already be done by cancel endpoint)
        meta_path = run_dir / "meta.json"
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("status") != "cancelled":
                meta["status"] = "cancelled"
                meta["error"] = "Cancelled"
                meta["completed_at"] = datetime.utcnow().isoformat()
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
        except Exception:
            pass
        
        if run_id in _training_jobs:
            _training_jobs[run_id].status = "cancelled"
        
        # Re-raise to properly propagate cancellation
        raise

    except Exception as e:
        logger.error(f"Training failed: {e}")

        # Stop monitor
        stop_event.set()
        if monitor_task:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        meta_path = run_dir / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "failed"
        meta["error"] = str(e)
        meta["completed_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        _training_jobs[run_id].status = "failed"
    
    finally:
        # Clean up the task from tracking dict
        if run_name in _training_tasks:
            del _training_tasks[run_name]

