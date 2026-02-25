"""GPU cluster service using Fabric for SSH management and SLURM job lifecycle."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import tempfile
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from backend.app.config import settings

GPU_CONFIGS: dict[str, dict[str, Any]] = {
    "h200": {
        "gres_name": "h200-141",
        "default_batch": 16,
        "mem": "256G",
        "default_partition": "gpu",
        "max_gpus": 4,
    },
    "h100-96": {
        "gres_name": "h100-96",
        "default_batch": 16,
        "mem": "256G",
        "default_partition": "gpu-long",
        "max_gpus": 2,
    },
    "h100-47": {
        "gres_name": "h100-47",
        "default_batch": 12,
        "mem": "256G",
        "default_partition": "gpu-long",
        "max_gpus": 4,
    },
    "a100-80": {
        "gres_name": "a100-80",
        "default_batch": 12,
        "mem": "128G",
        "default_partition": "gpu-long",
        "max_gpus": 1,
    },
    "a100-40": {
        "gres_name": "a100-40",
        "default_batch": 8,
        "mem": "64G",
        "default_partition": "gpu-long",
        "max_gpus": 2,
    },
    "nv": {
        "gres_name": "nv",
        "default_batch": 4,
        "mem": "32G",
        "default_partition": "gpu-long",
        "max_gpus": 2,
    },
}


@dataclass
class GPUJobState:
    """Tracked state of a submitted SLURM job."""

    job_id: str
    run_name: str
    job_type: str  # "training" or "inference"
    gpu_type: str
    project_name: str
    project_dir: str
    output_dir: str
    submitted_at: str
    log_file: str  # remote path to SLURM stdout log
    err_file: str  # remote path to SLURM stderr log
    status: str = "queued"
    started_at: str | None = None
    completed_at: str | None = None


class GPUService:
    """Manages SSH connection to GPU cluster and SLURM job lifecycle."""

    def __init__(self) -> None:
        self._conn: Any | None = None
        self._lock = threading.Lock()
        self._jobs: dict[str, GPUJobState] = {}  # keyed by run_name
        self._poll_tasks: dict[str, asyncio.Task] = {}

    @property
    def is_connected(self) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.run("true", hide=True, timeout=5)
            return True
        except Exception:
            self._conn = None
            return False

    def connect(self, password: str) -> dict[str, str]:
        """Establish SSH connection to the GPU cluster."""
        from fabric import Connection

        conn = Connection(
            host=settings.ssh_host,
            user=settings.ssh_user,
            connect_kwargs={"password": password},
            connect_timeout=15,
        )
        result = conn.run("hostname", hide=True, timeout=10)
        hostname = result.stdout.strip()
        logger.info(f"Connected to GPU cluster: {hostname}")

        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = conn

        return {"status": "connected", "hostname": hostname}

    def disconnect(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        logger.info("Disconnected from GPU cluster")

    def _require_conn(self) -> Any:
        if self._conn is None:
            raise RuntimeError("Not connected to GPU cluster. Call connect() first.")
        return self._conn

    # ── File sync ────────────────────────────────────────────────────────

    def push_project_data(self, project_dir: str) -> dict[str, int]:
        """Push local project data to the GPU cluster (manual_data, frames metadata, labels, project.json).

        Mirrors the pre-sync logic from run_training.sh.
        """
        conn = self._require_conn()
        local_project = Path(project_dir)
        remote_project = f"{settings.remote_dir}/{project_dir}"
        files_synced = 0

        conn.run(f"mkdir -p {remote_project}", hide=True)

        # project.json
        pj = local_project / "project.json"
        if pj.exists():
            conn.put(str(pj), f"{remote_project}/project.json")
            files_synced += 1

        # manual_data/ images
        md = local_project / "manual_data"
        if md.is_dir():
            files_synced += self._push_dir(conn, md, f"{remote_project}/manual_data")

        # frames/manual_data* metadata
        frames_dir = local_project / "frames"
        if frames_dir.is_dir():
            for sub in frames_dir.iterdir():
                if sub.is_dir() and (sub.name == "manual_data" or sub.name.startswith("manual_data__")):
                    files_synced += self._push_dir(
                        conn, sub, f"{remote_project}/frames/{sub.name}"
                    )

        # frames for imported datasets (source_key dirs)
        if frames_dir.is_dir():
            for sub in frames_dir.iterdir():
                if sub.is_dir() and not sub.name.startswith("manual_data") and not sub.name.startswith("video_"):
                    fj = sub / "frames.json"
                    if fj.exists():
                        conn.run(f"mkdir -p {remote_project}/frames/{sub.name}", hide=True)
                        conn.put(str(fj), f"{remote_project}/frames/{sub.name}/frames.json")
                        files_synced += 1

        # labels/current/
        labels_dir = local_project / "labels" / "current"
        if labels_dir.is_dir():
            files_synced += self._push_dir(conn, labels_dir, f"{remote_project}/labels/current")

        logger.info(f"Pushed {files_synced} files to {remote_project}")
        return {"files_synced": files_synced}

    def _push_dir(self, conn: Any, local_dir: Path, remote_dir: str) -> int:
        """Recursively push a directory via SFTP."""
        conn.run(f"mkdir -p {remote_dir}", hide=True)
        count = 0
        for item in local_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(local_dir)
                remote_path = f"{remote_dir}/{rel}"
                parent = str(Path(remote_path).parent)
                conn.run(f"mkdir -p {parent}", hide=True)
                conn.put(str(item), remote_path)
                count += 1
        return count

    def sync_results(
        self,
        remote_path: str,
        local_path: Path,
        patterns: list[str] | None = None,
    ) -> int:
        """Pull result files from GPU cluster to local path.

        If patterns is given (e.g. ["*.json"]), only matching files are synced.
        Otherwise all files are synced.
        """
        conn = self._require_conn()
        local_path.mkdir(parents=True, exist_ok=True)

        check = conn.run(f"test -d {remote_path} && echo yes || echo no", hide=True)
        if check.stdout.strip() != "yes":
            logger.warning(f"Remote path not found: {remote_path}")
            return 0

        listing = conn.run(
            f"find {remote_path} -type f", hide=True
        ).stdout.strip().splitlines()

        count = 0
        for remote_file in listing:
            rel = os.path.relpath(remote_file, remote_path)
            if patterns:
                if not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(os.path.basename(rel), p) for p in patterns):
                    continue

            local_file = local_path / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            conn.get(remote_file, str(local_file))
            count += 1

        logger.info(f"Synced {count} files from {remote_path} to {local_path}")
        return count

    # ── SLURM job management ─────────────────────────────────────────────

    def submit_slurm_job(self, script_content: str) -> str:
        """Upload a SLURM script and submit it via sbatch. Returns the job ID."""
        conn = self._require_conn()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            local_script = f.name

        remote_script = f"/tmp/batman_slurm_{os.getpid()}_{id(script_content)}.sh"
        try:
            conn.put(local_script, remote_script)
            result = conn.run(
                f"cd {settings.remote_dir} && mkdir -p logs && sbatch {remote_script}",
                hide=True,
            )
            output = result.stdout.strip()
            match = re.search(r"Submitted batch job (\d+)", output)
            if not match:
                raise RuntimeError(f"sbatch did not return job ID: {output}")
            job_id = match.group(1)
            logger.info(f"Submitted SLURM job {job_id}")
            return job_id
        finally:
            os.unlink(local_script)
            try:
                conn.run(f"rm -f {remote_script}", hide=True)
            except Exception:
                pass

    def get_job_status(self, job_id: str) -> dict[str, str]:
        """Check SLURM job status via squeue / sacct."""
        conn = self._require_conn()

        # Try squeue first (for running/pending jobs)
        try:
            result = conn.run(
                f"squeue -j {job_id} -h -o '%T %r' 2>/dev/null || true",
                hide=True,
                timeout=10,
            )
            line = result.stdout.strip()
            if line:
                parts = line.split(None, 1)
                state = parts[0]
                reason = parts[1] if len(parts) > 1 else ""
                state_map = {
                    "PENDING": "queued",
                    "RUNNING": "running",
                    "COMPLETING": "running",
                }
                return {
                    "status": state_map.get(state, state.lower()),
                    "raw_state": state,
                    "reason": reason,
                }
        except Exception:
            pass

        # Fall back to sacct (for finished jobs)
        try:
            result = conn.run(
                f"sacct -j {job_id} --format=State --noheader -P | head -1 | tr -d ' '",
                hide=True,
                timeout=10,
            )
            state = result.stdout.strip()
            state_map = {
                "COMPLETED": "completed",
                "FAILED": "failed",
                "CANCELLED": "cancelled",
                "CANCELLED+": "cancelled",
                "TIMEOUT": "timeout",
                "OUT_OF_ME+": "failed",
                "OUT_OF_MEMORY": "failed",
            }
            return {
                "status": state_map.get(state, state.lower()),
                "raw_state": state,
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    def cancel_job(self, job_id: str) -> None:
        conn = self._require_conn()
        conn.run(f"scancel {job_id}", hide=True, timeout=10)
        logger.info(f"Cancelled SLURM job {job_id}")

    # ── Log streaming ────────────────────────────────────────────────────

    async def stream_logs(
        self,
        job_id: str,
        job_name: str,
    ) -> AsyncGenerator[str, None]:
        """Stream SLURM log output via tail -f as an async generator.

        Yields individual lines. Blocks until the job finishes or the generator
        is closed.
        """
        conn = self._require_conn()
        remote_dir = settings.remote_dir
        log_path = f"{remote_dir}/logs/slurm_{job_id}_{job_name}.out"
        err_path = f"{remote_dir}/logs/slurm_{job_id}_{job_name}.err"

        # Wait for log file to appear (job may be queued)
        for _ in range(120):  # up to 10 minutes
            try:
                check = conn.run(f"test -f {log_path} && echo yes || echo no", hide=True, timeout=5)
                if check.stdout.strip() == "yes":
                    break
            except Exception:
                pass

            # Yield queue status while waiting
            status = self.get_job_status(job_id)
            if status.get("status") in ("completed", "failed", "cancelled", "timeout"):
                yield f"[system] Job {job_id} finished with status: {status['status']}\n"
                return
            reason = status.get("reason", "")
            yield f"[system] Waiting for job to start... {status.get('raw_state', 'PENDING')} {reason}\n"
            await asyncio.sleep(5)
        else:
            yield f"[system] Timed out waiting for log file to appear\n"
            return

        # Stream log via subprocess ssh tail -f (Fabric doesn't support async streaming well)
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _tail_thread() -> None:
            """Run tail -f in a background thread and push lines to the queue."""
            try:
                import subprocess

                proc = subprocess.Popen(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=10",
                        f"{settings.ssh_user}@{settings.ssh_host}",
                        f"tail -f {log_path} 2>/dev/null",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    for line in iter(proc.stdout.readline, ""):
                        loop.call_soon_threadsafe(queue.put_nowait, line)
                except Exception:
                    pass
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, f"[system] Log stream error: {e}\n")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_tail_thread, daemon=True)
        thread.start()

        try:
            while True:
                # Check job status periodically
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    status = self.get_job_status(job_id)
                    if status.get("status") in ("completed", "failed", "cancelled", "timeout"):
                        yield f"[system] Job finished: {status['status']}\n"
                        return
                    continue

                if line is None:
                    break
                yield line
        finally:
            pass

    # ── SLURM script generation ──────────────────────────────────────────

    def generate_training_script(
        self,
        *,
        project_dir: str,
        output_dir: str,
        output_dataset: str,
        model: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        lr: float,
        patience: int,
        grad_accum: int,
        gpu_type: str,
        num_gpus: int,
        time_limit: str,
        filter_classes: list[str] | None = None,
        max_frames_per_class: int | None = None,
        sources: list[str] | None = None,
        manual_split_strategy: str = "train_only",
        manual_datasets: list[str] | None = None,
        exclude_manual_datasets: list[str] | None = None,
        infer_after: bool = False,
        infer_test_only: bool = False,
    ) -> tuple[str, str]:
        """Generate a SLURM training script. Returns (script_content, job_name)."""
        gpu_cfg = GPU_CONFIGS.get(gpu_type)
        if not gpu_cfg:
            raise ValueError(f"Unknown GPU type: {gpu_type}")

        if num_gpus > gpu_cfg["max_gpus"]:
            raise ValueError(
                f"{gpu_type} supports max {gpu_cfg['max_gpus']} GPUs, requested {num_gpus}"
            )

        partition = gpu_cfg["default_partition"]
        if partition == "gpu" and time_limit != "3:00:00":
            time_limit = "3:00:00"

        gres = f"gpu:{gpu_cfg['gres_name']}:{num_gpus}"
        mem = gpu_cfg["mem"]
        job_name = f"rfdetr-{model}-{gpu_type}"

        # Build CLI arguments
        cli_args = [
            f"--project {project_dir}",
            f"--output-dataset {output_dataset}",
            f"--output-dir {output_dir}",
            f"--model {model}",
            f"--epochs {epochs}",
            f"--batch-size {batch_size}",
            f"--image-size {image_size}",
            f"--lr {lr}",
            f"--patience {patience}",
            f"--grad-accum {grad_accum}",
            "--device cuda",
            "--num-workers 8",
        ]

        if filter_classes:
            cli_args.append(f'--filter-classes "{"|".join(filter_classes)}"')
        if max_frames_per_class is not None:
            cli_args.append(f"--max-frames-per-class {max_frames_per_class}")
        if sources:
            cli_args.append(f"--sources {','.join(sources)}")
        if manual_split_strategy:
            cli_args.append(f"--manual-split-strategy {manual_split_strategy}")
        if manual_datasets:
            cli_args.append(f"--manual-datasets {','.join(manual_datasets)}")
        if exclude_manual_datasets:
            cli_args.append(f"--exclude-manual-datasets {','.join(exclude_manual_datasets)}")

        cli_cmd = " \\\n    ".join(cli_args)

        if num_gpus > 1:
            train_cmd = (
                f"torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \\\n"
                f"    -m cli.train \\\n    {cli_cmd}"
            )
        else:
            train_cmd = f"python3 -m cli.train \\\n    {cli_cmd}"

        # Post-training inference
        infer_section = ""
        if infer_after:
            infer_flags = "--latest --device cuda"
            if infer_test_only:
                infer_flags += " --test-only"
            infer_section = f"""
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed (exit $TRAIN_EXIT), skipping inference"
    exit $TRAIN_EXIT
fi

echo ""
echo "============================================================"
echo "Post-Training Inference"
echo "============================================================"

python3 -m cli.inference \\
    --project {project_dir} \\
    {infer_flags}
"""

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/slurm_%j_{job_name}.out
#SBATCH --error=logs/slurm_%j_{job_name}.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --mem={mem}

NUM_GPUS={num_gpus}

echo "============================================================"
echo "RF-DETR Training Job"
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $NUM_GPUS"
echo "Started:       $(date)"
echo "============================================================"

cd ~/batman || {{ echo "Error: ~/batman not found"; exit 1; }}

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

export MASTER_ADDR=localhost
export MASTER_PORT=$((12355 + RANDOM % 1000))
export WORLD_SIZE=$NUM_GPUS
export RANK=0
export LOCAL_RANK=0

echo "Training Configuration:"
echo "  Project:     {project_dir}"
echo "  Output:      {output_dir}"
echo "  Model:       RF-DETR {model}"
echo "  Epochs:      {epochs}"
echo "  Batch Size:  {batch_size}"
echo "  LR:          {lr}"
echo ""

echo "Starting training..."

{train_cmd}
{infer_section}
EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
"""
        return script, job_name

    def generate_inference_script(
        self,
        *,
        project_dir: str,
        run_name: str | None = None,
        use_latest: bool = False,
        video_ids: list[str] | None = None,
        test_only: bool = False,
        model: str = "base",
        confidence: float = 0.5,
        frame_interval: int = 1,
        track: bool = False,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        no_video: bool = False,
        gpu_type: str = "a100-80",
        time_limit: str = "04:00:00",
    ) -> tuple[str, str]:
        """Generate a SLURM inference script. Returns (script_content, job_name)."""
        gpu_cfg = GPU_CONFIGS.get(gpu_type)
        if not gpu_cfg:
            raise ValueError(f"Unknown GPU type: {gpu_type}")

        partition = gpu_cfg["default_partition"]
        gres = f"gpu:{gpu_cfg['gres_name']}:1"
        job_name = "rfdetr-inference"

        run_arg = f"--run {run_name}" if run_name else "--latest"
        if use_latest:
            run_arg = "--latest"

        video_arg = ""
        if video_ids:
            video_arg = " ".join(f"--video {v}" for v in video_ids)

        track_args = ""
        if track:
            track_args = f"--track --track-thresh {track_thresh} --track-buffer {track_buffer} --match-thresh {match_thresh}"

        opt_flags = []
        if no_video:
            opt_flags.append("--no-video")
        if test_only:
            opt_flags.append("--test-only")
        opt_str = " ".join(opt_flags)

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time_limit}
#SBATCH --output=logs/slurm_%j_{job_name}.out
#SBATCH --error=logs/slurm_%j_{job_name}.err

echo "============================================================"
echo "RF-DETR Inference Job"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "GPU:          {gpu_type}"
echo "Project:      {project_dir}"
echo "Started:      $(date)"
echo "============================================================"

cd ~/batman
source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Starting inference..."
echo ""

python3 -m cli.inference \\
    --project {project_dir} \\
    {run_arg} \\
    {video_arg} \\
    --model {model} \\
    --confidence {confidence} \\
    --frame-interval {frame_interval} \\
    {track_args} \\
    {opt_str} \\
    --device cuda

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
"""
        return script, job_name

    # ── Job tracking ─────────────────────────────────────────────────────

    def track_job(self, state: GPUJobState) -> None:
        self._jobs[state.run_name] = state

    def get_tracked_job(self, run_name: str) -> GPUJobState | None:
        return self._jobs.get(run_name)

    def list_tracked_jobs(self) -> list[GPUJobState]:
        return list(self._jobs.values())

    async def poll_job_until_done(
        self,
        job_state: GPUJobState,
        project_path: Path,
        on_complete: Any | None = None,
    ) -> None:
        """Background task that polls SLURM status and syncs results on completion."""
        while True:
            await asyncio.sleep(15)
            try:
                status_info = self.get_job_status(job_state.job_id)
                new_status = status_info.get("status", "unknown")

                if new_status == "running" and job_state.status == "queued":
                    job_state.status = "running"
                    job_state.started_at = datetime.utcnow().isoformat()
                    self._update_meta(project_path, job_state)

                if new_status in ("completed", "failed", "cancelled", "timeout"):
                    job_state.status = new_status
                    job_state.completed_at = datetime.utcnow().isoformat()
                    self._update_meta(project_path, job_state)

                    # Sync results
                    try:
                        if job_state.job_type == "training":
                            remote_run = f"{settings.remote_dir}/{job_state.output_dir}"
                            local_run = project_path / "runs" / job_state.run_name
                            self.sync_results(remote_run, local_run, ["*.json"])
                        elif job_state.job_type == "inference":
                            remote_infer = f"{settings.remote_dir}/{job_state.project_dir}/inference/"
                            local_infer = project_path / "inference"
                            self.sync_results(remote_infer, local_infer)
                    except Exception as e:
                        logger.warning(f"Result sync failed for {job_state.run_name}: {e}")

                    if on_complete:
                        on_complete(job_state)
                    return
            except Exception as e:
                logger.warning(f"Poll error for job {job_state.job_id}: {e}")

    def _update_meta(self, project_path: Path, job_state: GPUJobState) -> None:
        """Update meta.json for a training run."""
        import json

        if job_state.job_type != "training":
            return

        meta_path = project_path / "runs" / job_state.run_name / "meta.json"
        if not meta_path.exists():
            return

        with open(meta_path) as f:
            meta = json.load(f)

        meta["status"] = job_state.status
        meta["slurm_job_id"] = job_state.job_id
        if job_state.started_at:
            meta["started_at"] = job_state.started_at
        if job_state.completed_at:
            meta["completed_at"] = job_state.completed_at

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# Singleton instance
gpu_service = GPUService()
