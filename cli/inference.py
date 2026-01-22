#!/usr/bin/env python3
"""
Run RF-DETR inference on images or videos.

Usage:
    # Single image
    python -m cli.inference --checkpoint runs/my_run/best.pth --input image.jpg

    # Multiple images
    python -m cli.inference --checkpoint runs/my_run/best.pth --input images/*.jpg

    # Video (all frames, recommended for best quality)
    python -m cli.inference --checkpoint runs/my_run/best.pth --input video.mp4

    # Video with frame skipping + tracking + Kalman prediction (smooth interpolation)
    python -m cli.inference --checkpoint runs/my_run/best.pth --input video.mp4 \\
        --frame-interval 3 --track

    # Video with tracking but no Kalman prediction (static boxes between keyframes)
    python -m cli.inference --checkpoint runs/my_run/best.pth --input video.mp4 \\
        --frame-interval 3 --track --no-kalman

    # Skip optimization (faster startup, slower inference)
    python -m cli.inference --checkpoint runs/my_run/best.pth --input video.mp4 --no-optimize

    # Custom output directory
    python -m cli.inference --checkpoint runs/my_run/best.pth --input video.mp4 --output results/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from src.core.inference import (
    Detection,
    FrameResult,
    InferenceConfig,
    InferenceStats,
    RFDETRInference,
    draw_detections,
    save_results_json,
)


def is_video_file(path: Path) -> bool:
    """Check if path is a video file."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    return path.suffix.lower() in video_extensions


def is_image_file(path: Path) -> bool:
    """Check if path is an image file."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    return path.suffix.lower() in image_extensions


def process_single_image(
    engine: RFDETRInference,
    image_path: Path,
    output_dir: Path,
    config: InferenceConfig,
) -> tuple[int, float]:
    """
    Process a single image.
    
    Returns:
        Tuple of (detection_count, inference_time_ms)
    """
    # Load image
    image = Image.open(image_path)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Run inference
    t0 = time.time()
    detections = engine.predict_image(image, config)
    inference_time = (time.time() - t0) * 1000
    
    logger.info(f"  {image_path.name}: {len(detections)} detections in {inference_time:.1f}ms")
    
    # Save visualization
    if config.save_visualizations:
        annotated = draw_detections(image_bgr, detections, config.visualization_thickness)
        output_path = output_dir / f"detected_{image_path.name}"
        cv2.imwrite(str(output_path), annotated)
    
    # Save JSON
    if config.save_json:
        json_path = output_dir / f"{image_path.stem}_detections.json"
        save_results_json(
            [FrameResult(
                frame_idx=0,
                timestamp=0,
                detections=detections,
                inference_time_ms=inference_time,
            )],
            json_path,
            metadata={"source": str(image_path)},
        )
    
    return len(detections), inference_time


def process_video(
    engine: RFDETRInference,
    video_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    save_video: bool = True,
) -> InferenceStats:
    """Process a video file."""
    
    # Open video for writing output
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Output video writer
    out_video = None
    if save_video and config.save_visualizations:
        output_video_path = output_dir / f"detected_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process video
    all_results: list[FrameResult] = []
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}")
    
    def progress_callback(current: int, total: int):
        pbar.update(1)
    
    # Run inference generator
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    stats = None
    
    for result in engine.predict_video(video_path, config, progress_callback):
        all_results.append(result)
        
        # Write annotated frame to video
        if out_video is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, result.frame_idx)
            ret, frame = cap.read()
            if ret:
                annotated = draw_detections(frame, result.detections, config.visualization_thickness)
                out_video.write(annotated)
    
    # Get stats from generator return
    try:
        # The generator returns stats after completion
        pass
    except StopIteration as e:
        stats = e.value
    
    pbar.close()
    cap.release()
    
    if out_video is not None:
        out_video.release()
        logger.info(f"  Output video: {output_video_path}")
    
    # Calculate stats if not returned
    if stats is None:
        inference_times = [r.inference_time_ms for r in all_results if r.is_keyframe]
        stats = InferenceStats(
            total_frames=len(all_results),
            keyframes=sum(1 for r in all_results if r.is_keyframe),
            total_detections=sum(len(r.detections) for r in all_results),
            avg_inference_time_ms=np.mean(inference_times) if inference_times else 0,
            total_time_seconds=0,
            fps=0,
        )
    
    # Save JSON results
    if config.save_json:
        json_path = output_dir / f"{video_path.stem}_detections.json"
        save_results_json(
            all_results,
            json_path,
            stats=stats,
            metadata={
                "source": str(video_path),
                "fps": fps,
                "resolution": f"{width}x{height}",
                "frame_interval": config.frame_interval,
                "tracking": config.use_tracking,
                "kalman_prediction": config.use_kalman_prediction,
            },
        )
        logger.info(f"  Results JSON: {json_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on images or videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python -m cli.inference --checkpoint runs/run1/best.pth --input image.jpg

  # Video with tracking + Kalman interpolation (best for frame skipping)
  python -m cli.inference --checkpoint runs/run1/best.pth --input video.mp4 \\
      --frame-interval 3 --track

  # Video without optimization (faster startup)
  python -m cli.inference --checkpoint runs/run1/best.pth --input video.mp4 --no-optimize

  # Batch process images
  python -m cli.inference --checkpoint runs/run1/best.pth --input images/*.jpg

Notes:
  - Model optimization (enabled by default) may take a few seconds at startup
    but improves inference speed
  - When using --frame-interval > 1 with --track, Kalman prediction smoothly
    interpolates bounding boxes between keyframes (disable with --no-kalman)
        """,
    )
    
    # Required
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Input image(s) or video file(s)",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("inference_results"),
        help="Output directory (default: inference_results)",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Don't save annotated images/videos",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON detection results",
    )
    
    # Model
    parser.add_argument(
        "--model",
        choices=["base", "large"],
        default="base",
        help="Model size (default: base)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: cuda, mps, cpu, or auto (default: auto)",
    )
    
    # Detection
    parser.add_argument(
        "--confidence", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    
    # Optimization
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Don't optimize model for inference (skip optimize_for_inference())",
    )
    parser.add_argument(
        "--optimize-compile",
        action="store_true",
        help="Use JIT compilation for optimization (may fail on some systems)",
    )

    # Video options
    parser.add_argument(
        "--frame-interval", "-n",
        type=int,
        default=1,
        help="Run inference every N frames (default: 1 = all frames)",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable ByteTrack tracking between frames",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable Kalman prediction on non-keyframes (only with --track)",
    )
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.25,
        help="ByteTrack: detection threshold for tracking (default: 0.25)",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="ByteTrack: frames to keep lost tracks (default: 30)",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=0.8,
        help="ByteTrack: IoU threshold for matching (default: 0.8)",
    )
    
    # Classes
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        help="Override class names",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Build config
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
        save_visualizations=not args.no_visualizations,
        save_json=not args.no_json,
    )
    
    # Initialize engine
    logger.info(f"Loading model from {args.checkpoint}")
    engine = RFDETRInference(
        checkpoint=args.checkpoint,
        class_names=args.classes,
        model_size=args.model,
    )
    engine.load_model(
        device=config.device,
        optimize=config.optimize,
        optimize_compile=config.optimize_compile,
    )
    
    logger.info(f"Classes: {engine.class_names}")
    
    # Process inputs
    total_detections = 0
    total_time = 0
    
    for input_path in args.input:
        if not input_path.exists():
            logger.warning(f"Input not found: {input_path}")
            continue
        
        if is_video_file(input_path):
            logger.info(f"\nProcessing video: {input_path}")
            stats = process_video(engine, input_path, args.output, config)
            
            logger.info(f"  Total frames: {stats.total_frames}")
            logger.info(f"  Keyframes (inference): {stats.keyframes}")
            logger.info(f"  Total detections: {stats.total_detections}")
            logger.info(f"  Avg inference time: {stats.avg_inference_time_ms:.1f}ms")
            
            total_detections += stats.total_detections
            
        elif is_image_file(input_path):
            logger.info(f"\nProcessing image: {input_path}")
            count, time_ms = process_single_image(engine, input_path, args.output, config)
            total_detections += count
            total_time += time_ms
            
        else:
            logger.warning(f"Unsupported file type: {input_path}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
