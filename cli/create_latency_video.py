#!/usr/bin/env python3
"""
Create a latency-delayed inference video.

This script reads annotated frames and creates a video where each frame appears
at the time when its inference actually completed, simulating real-time latency.

Usage:
    python -m cli.create_latency_video \\
        --video original.mp4 \\
        --results benchmark_results.json \\
        --frames frames/ \\
        --output detected_latency.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def create_latency_video(
    video_path: Path,
    results_json_path: Path,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    """
    Create a video with realistic latency delays.

    Each frame appears at the timestamp when its inference would have completed
    in a real-time scenario. This simulates what the output would look like
    if the input was a live video feed.

    Args:
        video_path: Path to original video file
        results_json_path: Path to benchmark_results.json with per-frame timing
        frames_dir: Directory containing annotated frames (frame_00000.jpg, etc.)
        output_path: Path for output video

    Returns:
        Path to created video file
    """
    logger.info(f"Creating latency-delayed video: {output_path}")

    # Load benchmark results
    with open(results_json_path) as f:
        results = json.load(f)

    # Extract video info and per-frame timings
    if "video_info" not in results:
        raise ValueError("Results JSON must contain video_info (requires video benchmark)")

    video_info = results["video_info"]
    per_frame_results = results.get("per_frame_results", [])
    
    if not per_frame_results:
        raise ValueError("Results JSON must contain per_frame_results")

    fps = video_info["fps"]
    width = video_info["width"]
    height = video_info["height"]
    total_frames = len(per_frame_results)

    logger.info(f"  Original video: {video_info['path']}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")

    # Calculate when each frame should appear (considering latency)
    frame_timings = []
    cumulative_time = 0.0
    
    for frame_result in per_frame_results:
        frame_idx = frame_result["frame_idx"]
        inference_time_ms = frame_result["inference_time_ms"]
        
        # Frame arrives at original timestamp
        arrival_time = frame_idx / fps
        
        # Inference completes after latency
        completion_time = arrival_time + (inference_time_ms / 1000.0)
        
        frame_timings.append({
            "frame_idx": frame_idx,
            "arrival_time": arrival_time,
            "inference_time_ms": inference_time_ms,
            "completion_time": completion_time,
        })
        
        cumulative_time = max(cumulative_time, completion_time)

    # Calculate total video duration (max completion time)
    total_duration = frame_timings[-1]["completion_time"]
    output_total_frames = int(total_duration * fps) + 1
    
    logger.info(f"  Output duration: {total_duration:.2f}s ({output_total_frames} frames)")
    logger.info(f"  Average latency: {results['metrics']['mean_ms']:.2f}ms")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    # Build a mapping: output_frame_idx -> annotated_frame_path
    # For each output frame at time T, show the most recent completed inference
    logger.info("  Processing frames...")
    
    last_completed_idx = None
    last_annotated_frame = None
    frames_written = 0
    
    for output_frame_idx in range(output_total_frames):
        current_time = output_frame_idx / fps
        
        # Find the most recent frame that has completed inference by current_time
        for timing in frame_timings:
            if timing["completion_time"] <= current_time:
                last_completed_idx = timing["frame_idx"]
            else:
                break
        
        # Load and write the appropriate frame
        if last_completed_idx is not None:
            # Load annotated frame if we haven't already or if it changed
            if last_annotated_frame is None or last_completed_idx != frames_written - 1:
                frame_path = frames_dir / f"frame_{last_completed_idx:05d}.jpg"
                if frame_path.exists():
                    last_annotated_frame = cv2.imread(str(frame_path))
                else:
                    logger.warning(f"Frame not found: {frame_path}")
                    # Create black frame as fallback
                    last_annotated_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            out.write(last_annotated_frame)
        else:
            # No inference completed yet, write black frame
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(black_frame)
        
        frames_written += 1
        
        if (frames_written) % 30 == 0:
            logger.info(f"    Written {frames_written}/{output_total_frames} frames")

    out.release()
    logger.info(f"  Latency video created: {output_path}")
    logger.info(f"  Total frames written: {frames_written}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create latency-delayed inference video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to original video file",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        required=True,
        help="Directory containing annotated frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for latency video",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.video.exists():
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)
    
    if not args.results.exists():
        logger.error(f"Results JSON not found: {args.results}")
        sys.exit(1)
    
    if not args.frames.exists() or not args.frames.is_dir():
        logger.error(f"Frames directory not found: {args.frames}")
        sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate latency video
    try:
        output_path = create_latency_video(
            video_path=args.video,
            results_json_path=args.results,
            frames_dir=args.frames,
            output_path=args.output,
        )
        logger.info(f"Success! Video saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create latency video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
