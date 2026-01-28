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
    video_total_frames = video_info.get("total_frames", total_frames)

    logger.info(f"  Original video: {video_info['path']}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Benchmarked frames: {total_frames}")
    if video_total_frames > total_frames:
        logger.warning(f"  WARNING: Video has {video_total_frames} frames but only {total_frames} were benchmarked")
        logger.warning(f"  Latency video will only show the first {total_frames} frames")
        logger.warning(f"  Use --runs {video_total_frames} to benchmark the entire video")

    # Calculate when each frame should appear (considering latency AND queuing)
    # This simulates real-time sequential processing:
    # - Each frame arrives at its timestamp (frame_idx / fps)
    # - Inference starts when frame arrives OR when previous inference completes (whichever is later)
    # - This accounts for frames piling up when inference is slower than frame rate
    frame_timings = []
    inference_available_at = 0.0  # When the inference engine becomes available
    
    for frame_result in per_frame_results:
        frame_idx = frame_result["frame_idx"]
        inference_time_ms = frame_result["inference_time_ms"]
        inference_time_s = inference_time_ms / 1000.0
        
        # Frame arrives at its timestamp
        arrival_time = frame_idx / fps
        
        # Inference starts when BOTH:
        # 1. Frame has arrived
        # 2. Previous inference is complete
        inference_start_time = max(arrival_time, inference_available_at)
        
        # Inference completes after processing
        completion_time = inference_start_time + inference_time_s
        
        # Update when inference engine becomes available for next frame
        inference_available_at = completion_time
        
        # Calculate delay: how long after arrival does result appear?
        delay = completion_time - arrival_time
        
        frame_timings.append({
            "frame_idx": frame_idx,
            "arrival_time": arrival_time,
            "inference_start_time": inference_start_time,
            "inference_time_ms": inference_time_ms,
            "completion_time": completion_time,
            "delay_ms": delay * 1000.0,
        })

    # Calculate total video duration (max completion time)
    # Extend to match original video duration if benchmarked frames are less
    benchmarked_duration = frame_timings[-1]["completion_time"]
    original_duration = video_total_frames / fps
    total_duration = max(benchmarked_duration, original_duration)
    output_total_frames = int(total_duration * fps) + 1
    
    # Calculate statistics about delays
    delays = [t["delay_ms"] for t in frame_timings]
    avg_delay = sum(delays) / len(delays) if delays else 0
    max_delay = max(delays) if delays else 0
    
    logger.info(f"  Benchmarked duration: {benchmarked_duration:.2f}s")
    logger.info(f"  Original video duration: {original_duration:.2f}s")
    logger.info(f"  Output duration: {total_duration:.2f}s ({output_total_frames} frames)")
    logger.info(f"  Average inference time: {results['metrics']['mean_ms']:.2f}ms")
    logger.info(f"  Average end-to-end delay: {avg_delay:.2f}ms")
    logger.info(f"  Maximum end-to-end delay: {max_delay:.2f}ms")
    
    # Check if real-time capable
    frame_interval_ms = 1000.0 / fps
    if avg_delay > frame_interval_ms:
        logger.warning(f"  ⚠ Inference is NOT real-time capable!")
        logger.warning(f"    Frame interval: {frame_interval_ms:.2f}ms, but avg delay: {avg_delay:.2f}ms")
        logger.warning(f"    Frames are piling up - output will lag increasingly behind input")

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
    
    # Create a "No More Inference Data" overlay for frames beyond benchmarked range
    def create_no_data_frame():
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add text overlay
        text = "No inference data"
        text2 = f"(only {total_frames} frames benchmarked)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        
        # Calculate text position (centered)
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        (text2_w, text2_h), _ = cv2.getTextSize(text2, font, font_scale * 0.6, 2)
        
        x1 = (width - text_w) // 2
        y1 = (height - text_h) // 2
        x2 = (width - text2_w) // 2
        y2 = y1 + text_h + 20
        
        cv2.putText(frame, text, (x1, y1), font, font_scale, (100, 100, 100), thickness)
        cv2.putText(frame, text2, (x2, y2), font, font_scale * 0.6, (80, 80, 80), 2)
        return frame
    
    no_data_frame = None
    
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
            need_load = (last_annotated_frame is None or 
                        (frames_written > 0 and last_completed_idx != last_completed_idx))
            if need_load:
                frame_path = frames_dir / f"frame_{last_completed_idx:05d}.jpg"
                if frame_path.exists():
                    last_annotated_frame = cv2.imread(str(frame_path))
                else:
                    logger.warning(f"Frame not found: {frame_path}")
                    # Create "no data" frame
                    if no_data_frame is None:
                        no_data_frame = create_no_data_frame()
                    last_annotated_frame = no_data_frame
            
            out.write(last_annotated_frame)
        else:
            # No inference completed yet - either before first frame or after last benchmarked frame
            if output_frame_idx > total_frames:
                # Beyond benchmarked frames - show "no data" message
                if no_data_frame is None:
                    no_data_frame = create_no_data_frame()
                out.write(no_data_frame)
            else:
                # Before first frame completes - show black
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
