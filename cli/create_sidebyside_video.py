#!/usr/bin/env python3
"""
Create a side-by-side comparison video showing original video alongside
latency-delayed inference results.

The left side shows the original video feed playing in real-time.
The right side shows the inference output with realistic latency delays.
Text overlays display timing information and latency metrics.

Usage:
    python -m cli.create_sidebyside_video \\
        --original original.mp4 \\
        --latency detected_latency.mp4 \\
        --results benchmark_results.json \\
        --output sidebyside_latency.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: tuple[int, int, int] | None = (0, 0, 0),
    bg_alpha: float = 0.7,
) -> np.ndarray:
    """
    Add text overlay to frame with optional background.

    Args:
        frame: Input frame (BGR)
        text: Text to display
        position: (x, y) position for text
        font_scale: Font scale
        color: Text color (BGR)
        thickness: Text thickness
        bg_color: Background color (BGR), None for no background
        bg_alpha: Background transparency (0-1)

    Returns:
        Frame with text overlay
    """
    result = frame.copy()
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle if specified
    if bg_color is not None:
        padding = 5
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - text_h - baseline - padding),
            (x + text_w + padding, y + baseline + padding),
            bg_color,
            -1,
        )
        # Blend overlay with original
        cv2.addWeighted(overlay, bg_alpha, result, 1 - bg_alpha, 0, result)
    
    # Draw text
    cv2.putText(
        result,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    
    return result


def create_sidebyside_video(
    original_video_path: Path,
    latency_video_path: Path,
    results_json_path: Path,
    output_path: Path,
) -> Path:
    """
    Create side-by-side comparison video.

    Left side: Original video at real-time speed
    Right side: Inference output with latency delays

    Args:
        original_video_path: Path to original video
        latency_video_path: Path to latency-delayed inference video
        results_json_path: Path to benchmark results JSON
        output_path: Output path for side-by-side video

    Returns:
        Path to created video file
    """
    logger.info(f"Creating side-by-side comparison video: {output_path}")

    # Load benchmark results for metrics
    with open(results_json_path) as f:
        results = json.load(f)

    metrics = results["metrics"]
    video_info = results["video_info"]
    per_frame_results = results.get("per_frame_results", [])
    
    # Calculate actual end-to-end delays (accounting for queuing)
    fps = video_info["fps"]
    inference_available_at = 0.0
    frame_delays = {}
    
    for frame_result in per_frame_results:
        frame_idx = frame_result["frame_idx"]
        inference_time_ms = frame_result["inference_time_ms"]
        inference_time_s = inference_time_ms / 1000.0
        
        arrival_time = frame_idx / fps
        inference_start_time = max(arrival_time, inference_available_at)
        completion_time = inference_start_time + inference_time_s
        inference_available_at = completion_time
        
        delay = completion_time - arrival_time
        frame_delays[frame_idx] = delay * 1000.0  # Convert to ms

    logger.info(f"  Mean inference time: {metrics['mean_ms']:.2f}ms")
    logger.info(f"  P99 inference time: {metrics['p99_ms']:.2f}ms")

    # Open both videos
    cap_original = cv2.VideoCapture(str(original_video_path))
    cap_latency = cv2.VideoCapture(str(latency_video_path))

    if not cap_original.isOpened():
        raise RuntimeError(f"Failed to open original video: {original_video_path}")
    if not cap_latency.isOpened():
        raise RuntimeError(f"Failed to open latency video: {latency_video_path}")

    # Get video properties
    fps = cap_original.get(cv2.CAP_PROP_FPS)
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use the longer of the two videos
    original_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    latency_frames = int(cap_latency.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(original_frames, latency_frames)

    logger.info(f"  Resolution per side: {width}x{height}")
    logger.info(f"  FPS: {fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")

    # Output dimensions (side-by-side)
    output_width = width * 2
    output_height = height

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    logger.info("  Processing frames...")
    
    frame_idx = 0

    while frame_idx < total_frames:
        # Read from both videos
        ret_orig, frame_orig = cap_original.read()
        ret_lat, frame_lat = cap_latency.read()

        # Handle end of videos
        if not ret_orig:
            # Create black frame for original
            frame_orig = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not ret_lat:
            # Create black frame for latency
            frame_lat = np.zeros((height, width, 3), dtype=np.uint8)

        # Create side-by-side frame
        sidebyside = np.hstack([frame_orig, frame_lat])

        # Add text overlays
        # Top-left: "Original Feed"
        sidebyside = add_text_overlay(
            sidebyside,
            "Original Feed",
            (20, 40),
            font_scale=0.8,
            color=(255, 255, 255),
            thickness=2,
        )

        # Top-right: "Inference Output (Delayed)"
        sidebyside = add_text_overlay(
            sidebyside,
            "Inference Output (Delayed)",
            (width + 20, 40),
            font_scale=0.8,
            color=(255, 255, 255),
            thickness=2,
        )

        # Bottom-left: Current time
        current_time = frame_idx / fps
        sidebyside = add_text_overlay(
            sidebyside,
            f"Time: {current_time:.2f}s",
            (20, height - 20),
            font_scale=0.6,
            color=(200, 200, 200),
            thickness=1,
        )

        # Bottom-center: Current frame delay (end-to-end, not just inference time)
        if frame_idx in frame_delays:
            current_delay = frame_delays[frame_idx]
            latency_text = f"End-to-End Delay: {current_delay:.1f}ms"
            # Green if real-time capable for this FPS, red otherwise
            frame_interval_ms = 1000.0 / video_info["fps"]
            latency_color = (100, 255, 100) if current_delay < frame_interval_ms else (100, 100, 255)
        else:
            latency_text = "Avg Inference: {:.1f}ms".format(metrics['mean_ms'])
            latency_color = (200, 200, 200)
        
        sidebyside = add_text_overlay(
            sidebyside,
            latency_text,
            (output_width // 2 - 100, height - 20),
            font_scale=0.7,
            color=latency_color,
            thickness=2,
        )

        # Bottom-right: Performance metrics
        realtime_30fps = "✓" if results["realtime_capable"]["30fps"] else "✗"
        sidebyside = add_text_overlay(
            sidebyside,
            f"30FPS: {realtime_30fps} | P99: {metrics['p99_ms']:.1f}ms",
            (width + 20, height - 20),
            font_scale=0.6,
            color=(200, 200, 200),
            thickness=1,
        )

        # Write frame
        out.write(sidebyside)
        frame_idx += 1

        if frame_idx % 30 == 0:
            logger.info(f"    Processed {frame_idx}/{total_frames} frames")

    # Release resources
    cap_original.release()
    cap_latency.release()
    out.release()

    logger.info(f"  Side-by-side video created: {output_path}")
    logger.info(f"  Total frames written: {frame_idx}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side latency comparison video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Path to original video file",
    )
    parser.add_argument(
        "--latency",
        type=Path,
        required=True,
        help="Path to latency-delayed inference video",
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for side-by-side video",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.original.exists():
        logger.error(f"Original video not found: {args.original}")
        sys.exit(1)

    if not args.latency.exists():
        logger.error(f"Latency video not found: {args.latency}")
        sys.exit(1)

    if not args.results.exists():
        logger.error(f"Results JSON not found: {args.results}")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Generate side-by-side video
    try:
        output_path = create_sidebyside_video(
            original_video_path=args.original,
            latency_video_path=args.latency,
            results_json_path=args.results,
            output_path=args.output,
        )
        logger.info(f"Success! Video saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to create side-by-side video: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
