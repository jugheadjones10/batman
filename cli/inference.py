#!/usr/bin/env python3
"""
Run RF-DETR inference on project videos.

All inference is project-centric: results are persisted under
{project}/inference/{run_name}/{video_id}/.

Usage:
    # Run a training run's model on all project videos
    python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1

    # Run on a specific project video
    python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1 \\
        --video video_2

    # Run on test-only videos
    python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1 --test-only

    # Use the latest training run
    python -m cli.inference --project data/projects/CraneHook --latest

    # With tracking and frame skipping
    python -m cli.inference --project data/projects/CraneHook --run my_run \\
        --track --frame-interval 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.core.inference import (
    FrameResult,
    InferenceConfig,
    InferenceStats,
    RFDETRInference,
    draw_detections,
    save_results_json,
)
from src.core.project import Project
from src.core.trainer import find_best_checkpoint


def resolve_run(project: Project, args) -> tuple[Path, str]:
    """
    Resolve checkpoint path and run name from project runs.

    Returns:
        (checkpoint_path, run_name)
    """
    runs_dir = project.runs_dir
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        logger.error(f"No training runs found in {runs_dir}")
        sys.exit(1)

    if args.latest:
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        run_dir = run_dirs[0]
        logger.info(f"Using latest run: {run_dir.name}")
    else:
        run_dir = runs_dir / args.run
        if not run_dir.exists():
            logger.error(f"Run not found: {run_dir}")
            logger.info("Available runs:")
            for d in sorted(runs_dir.iterdir()):
                if d.is_dir():
                    logger.info(f"  - {d.name}")
            sys.exit(1)

    checkpoint = find_best_checkpoint(run_dir)
    if checkpoint is None:
        logger.error(f"No checkpoint found in {run_dir}")
        sys.exit(1)

    return checkpoint, run_dir.name


def resolve_class_names(project: Project, run_name: str) -> list[str]:
    """Load class names from class_info.json (authoritative), fallback to project."""
    class_info_path = project.runs_dir / run_name / "class_info.json"
    if class_info_path.exists():
        with open(class_info_path) as f:
            info = json.load(f)
        names = info.get("classes", [])
        if names:
            return names
    return project.classes


def get_target_videos(project: Project, args) -> dict[str, dict]:
    """Get the videos to run inference on based on args."""
    if args.video:
        videos_meta = project.load_videos_meta()
        vids = {}
        for vid_id in args.video:
            if vid_id in videos_meta:
                vids[vid_id] = videos_meta[vid_id]
            else:
                logger.warning(f"Video not found: {vid_id}")
        if not vids:
            logger.error("No matching videos found in project")
            sys.exit(1)
        return vids

    if args.test_only:
        vids = project.list_videos(test_only=True)
        if not vids:
            logger.error("No test-only videos found (exclude_from_training=true)")
            sys.exit(1)
        return vids

    vids = project.list_videos()
    if not vids:
        logger.error("No videos found in project")
        sys.exit(1)
    return vids


def process_video(
    engine: RFDETRInference,
    video_path: Path,
    config: InferenceConfig,
    output_dir: Path,
) -> tuple[InferenceStats | None, list[FrameResult]]:
    """Process a video file. Returns (stats, all_results)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    out_video = None
    if config.save_visualizations:
        output_video_path = output_dir / "detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    all_results: list[FrameResult] = []
    pbar = tqdm(total=total_frames, desc=f"  {video_path.name}")

    def progress_callback(current: int, total: int):
        pbar.update(1)

    cap = cv2.VideoCapture(str(video_path))

    for result in engine.predict_video(video_path, config, progress_callback):
        all_results.append(result)

        if out_video is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, result.frame_idx)
            ret, frame = cap.read()
            if ret:
                annotated = draw_detections(frame, result.detections, config.visualization_thickness)
                out_video.write(annotated)

    pbar.close()
    cap.release()

    if out_video is not None:
        out_video.release()

    inference_times = [r.inference_time_ms for r in all_results if r.is_keyframe]
    stats = InferenceStats(
        total_frames=len(all_results),
        keyframes=sum(1 for r in all_results if r.is_keyframe),
        total_detections=sum(len(r.detections) for r in all_results),
        avg_inference_time_ms=float(np.mean(inference_times)) if inference_times else 0,
        total_time_seconds=0,
        fps=0,
    )

    return stats, all_results


def persist_result(
    project: Project,
    run_name: str,
    video_id: str,
    stats: InferenceStats,
    all_results: list[FrameResult],
    config: InferenceConfig,
) -> None:
    """Save inference result to project/inference/{run_name}/{video_id}/."""
    frames_data = []
    for r in all_results:
        frames_data.append({
            "frame_idx": r.frame_idx,
            "timestamp": r.timestamp,
            "is_keyframe": r.is_keyframe,
            "inference_time_ms": r.inference_time_ms,
            "detections": [
                {
                    "bbox": list(d.bbox),
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "track_id": d.track_id,
                }
                for d in r.detections
            ],
        })

    data = {
        "run_name": run_name,
        "video_id": video_id,
        "config": {
            "confidence_threshold": config.confidence_threshold,
            "frame_interval": config.frame_interval,
            "tracking": config.use_tracking,
            "tracking_mode": "bytetrack" if config.use_tracking else "none",
        },
        "stats": {
            "total_frames": stats.total_frames,
            "keyframes": stats.keyframes,
            "total_detections": stats.total_detections,
            "avg_inference_time_ms": stats.avg_inference_time_ms,
        },
        "frames": frames_data,
    }

    result_dir = project.save_inference_result(run_name, video_id, data)

    # Move detected.mp4 from the flat video dir into the timestamped result dir
    flat_video_dir = project.inference_dir / run_name / video_id
    flat_video = flat_video_dir / "detected.mp4"
    if flat_video.exists() and result_dir != flat_video_dir:
        import shutil
        shutil.move(str(flat_video), str(result_dir / "detected.mp4"))

    logger.info(f"  Result saved: {result_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on project videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.inference --project data/projects/CraneHook --run rfdetr_run_1
  python -m cli.inference --project data/projects/CraneHook --latest
  python -m cli.inference --project data/projects/CraneHook --run my_run --video video_2
  python -m cli.inference --project data/projects/CraneHook --run my_run --test-only
  python -m cli.inference --project data/projects/CraneHook --run my_run --track --frame-interval 5
        """,
    )

    parser.add_argument(
        "--project", "-p",
        type=Path,
        required=True,
        help="Path to Batman project (required)",
    )

    run_group = parser.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run", "-r", type=str, help="Training run name")
    run_group.add_argument("--latest", action="store_true", help="Use the most recent training run")

    parser.add_argument("--video", "-v", type=str, nargs="+", help="Specific video source_key(s) to process")
    parser.add_argument("--test-only", action="store_true", help="Only run on videos with exclude_from_training=true")

    parser.add_argument("--model", choices=["base", "large"], default="base", help="Model size (default: base)")
    parser.add_argument("--device", default="auto", help="Device: cuda, mps, cpu, or auto")
    parser.add_argument("--confidence", "-t", type=float, default=0.0, help="Min confidence to include (0=all; each box shows its confidence)")

    parser.add_argument("--no-optimize", action="store_true", help="Skip model optimization")
    parser.add_argument("--optimize-compile", action="store_true", help="Use JIT compilation")

    parser.add_argument("--frame-interval", "-n", type=int, default=1, help="Inference every N frames (default: 1)")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack tracking")
    parser.add_argument("--no-kalman", action="store_true", help="Disable Kalman prediction on non-keyframes")
    parser.add_argument("--track-thresh", type=float, default=0.25, help="ByteTrack detection threshold")
    parser.add_argument("--track-buffer", type=int, default=30, help="Frames to keep lost tracks")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack IoU threshold")

    parser.add_argument("--no-video", action="store_true", help="Don't save annotated output video")

    args = parser.parse_args()

    if not args.project.exists():
        logger.error(f"Project not found: {args.project}")
        sys.exit(1)

    project = Project.load(args.project)
    logger.info(f"Project: {project.name} ({len(project.classes)} classes)")

    checkpoint, run_name = resolve_run(project, args)
    class_names = resolve_class_names(project, run_name)
    logger.info(f"Run: {run_name}")
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Classes: {class_names}")

    videos = get_target_videos(project, args)
    logger.info(f"Videos to process: {len(videos)}")

    config = InferenceConfig(
        confidence_threshold=args.confidence,
        device=args.device,
        optimize=not args.no_optimize,
        optimize_compile=args.optimize_compile,
        frame_interval=args.frame_interval,
        use_tracking=args.track,
        use_kalman_prediction=not args.no_kalman,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        save_visualizations=not args.no_video,
        save_json=False,
    )

    logger.info(f"Loading model from {checkpoint}")
    engine = RFDETRInference(
        checkpoint=checkpoint,
        class_names=class_names,
        model_size=args.model,
    )
    engine.load_model(
        device=config.device,
        optimize=config.optimize,
        optimize_compile=config.optimize_compile,
    )

    total_detections = 0
    for video_id, vid_meta in videos.items():
        video_path = Path(vid_meta["original_path"])
        if not video_path.exists():
            logger.warning(f"Video file not found: {video_path}")
            continue

        output_dir = project.inference_dir / run_name / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nProcessing: {vid_meta['filename']} ({video_id})")
        stats, all_results = process_video(engine, video_path, config, output_dir)

        if stats:
            persist_result(project, run_name, video_id, stats, all_results, config)
            logger.info(f"  Frames: {stats.total_frames}, Detections: {stats.total_detections}, Avg: {stats.avg_inference_time_ms:.1f}ms")
            total_detections += stats.total_detections

    logger.info(f"\n{'='*50}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Results saved under: {project.inference_dir / run_name}")


if __name__ == "__main__":
    main()
