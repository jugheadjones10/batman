#!/usr/bin/env python3
"""
Benchmark RF-DETR inference latency across different GPUs.

Measures comprehensive latency metrics including percentiles, throughput,
and real-time capability for video inference.

Usage:
    # Benchmark with dummy images (synthetic)
    python -m cli.benchmark_latency --run rfdetr_h200_20260120_105925

    # Benchmark with real video frames (realistic)
    python -m cli.benchmark_latency --run rfdetr_h200_20260120_105925 \\
        --video crane_hook_1_short.mp4

    # Benchmark with specific parameters
    python -m cli.benchmark_latency --run my_run --warmup 10 --runs 100 \\
        --video crane_hook_1_short.mp4 --output benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from src.core.inference import RFDETRInference, draw_detections


RUNS_DIR = Path("runs")
DEFAULT_VIDEO = "crane_hook_1_short.mp4"


def find_checkpoint_in_run(run_dir: Path) -> Path | None:
    """Find the best checkpoint in a run directory."""
    checkpoint_names = [
        "best.pth",
        "checkpoint_best.pth",
        "model_best.pth",
        "last.pth",
        "checkpoint_last.pth",
        "model_last.pth",
    ]

    for name in checkpoint_names:
        checkpoint = run_dir / name
        if checkpoint.exists():
            return checkpoint

    # Fallback: find any .pth file
    pth_files = list(run_dir.glob("*.pth"))
    if pth_files:
        return pth_files[0]

    return None


def find_latest_run() -> Path | None:
    """Find the most recent run directory."""
    if not RUNS_DIR.exists():
        return None

    run_dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def get_gpu_info() -> dict:
    """Get GPU information."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                "available": True,
                "name": gpu_name,
                "memory_gb": round(gpu_memory_gb, 1),
                "device": "cuda",
            }
        else:
            return {"available": False, "name": "CPU", "device": "cpu"}
    except ImportError:
        return {"available": False, "name": "CPU", "device": "cpu"}


def load_video_frames(video_path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    """
    Load frames from a video file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None = all)

    Returns:
        List of frames as numpy arrays (RGB)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps:.1f}")
    logger.info(f"  Total frames: {total_frames}")

    frames_to_load = min(max_frames, total_frames) if max_frames else total_frames

    while len(frames) < frames_to_load:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    logger.info(f"  Loaded: {len(frames)} frames")

    return frames, {"fps": fps, "width": width, "height": height, "total_frames": total_frames}


def benchmark_latency(
    checkpoint_path: Path,
    model_size: str = "base",
    image_size: int = 640,
    warmup_runs: int = 10,
    test_runs: int = 100,
    optimize: bool = True,
    video_path: Path | None = None,
    save_frames: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """
    Benchmark inference latency with comprehensive metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        model_size: Model size (base or large)
        image_size: Image size for inference (used for dummy images)
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs
        optimize: Whether to optimize model for inference
        video_path: Path to video file (None = use dummy images)
        save_frames: Whether to save annotated frames for latency visualization
        output_dir: Output directory for saving frames (required if save_frames=True)

    Returns:
        Dictionary with benchmark results
    """
    logger.info("Loading model...")
    engine = RFDETRInference(
        checkpoint=checkpoint_path,
        model_size=model_size,
    )
    # Load model onto device with optimization
    engine.load_model(optimize=optimize)

    # Get GPU info
    gpu_info = get_gpu_info()
    logger.info(f"GPU: {gpu_info['name']}")

    # Prepare test data
    video_info = None
    if video_path:
        # Load video frames
        frames, video_info = load_video_frames(video_path, max_frames=test_runs + warmup_runs)
        if len(frames) < warmup_runs + test_runs:
            logger.warning(
                f"Video has only {len(frames)} frames, adjusting warmup={min(warmup_runs, len(frames)//2)}, "
                f"runs={len(frames) - min(warmup_runs, len(frames)//2)}"
            )
            warmup_runs = min(warmup_runs, len(frames) // 2)
            test_runs = len(frames) - warmup_runs
        benchmark_mode = "video"
        test_images = frames
    else:
        # Create dummy images
        benchmark_mode = "synthetic"
        dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        test_images = [dummy_img] * (warmup_runs + test_runs)

    # Warmup phase
    logger.info(f"Warming up with {warmup_runs} runs...")
    for i in range(warmup_runs):
        img = test_images[i % len(test_images)]
        _ = engine.model.predict(img)
        if (i + 1) % 5 == 0:
            logger.info(f"  Warmup: {i + 1}/{warmup_runs}")

    # Prepare frames directory if saving annotated frames
    frames_dir = None
    if save_frames:
        if output_dir is None:
            raise ValueError("output_dir is required when save_frames=True")
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving annotated frames to: {frames_dir}")

    # Benchmark phase
    logger.info(f"Running {test_runs} benchmark iterations...")
    times = []
    per_frame_results = []

    for i in range(test_runs):
        img = test_images[(warmup_runs + i) % len(test_images)]

        start = time.perf_counter()
        detections = engine.model.predict(img)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)
        
        # Store per-frame result
        per_frame_results.append({
            "frame_idx": i,
            "inference_time_ms": round(elapsed, 2)
        })

        # Save annotated frame if requested
        if save_frames and frames_dir:
            # Convert image to BGR for OpenCV
            if isinstance(img, np.ndarray):
                if img.shape[2] == 3:  # RGB
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img
            else:
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert detections to Detection objects for drawing
            from src.core.inference import Detection
            detection_objs = []
            if hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
                for j in range(len(detections.xyxy)):
                    class_id = int(detections.class_id[j])
                    class_name = (
                        engine.class_names[class_id]
                        if class_id < len(engine.class_names)
                        else f"class_{class_id}"
                    )
                    detection_objs.append(
                        Detection(
                            bbox=tuple(detections.xyxy[j].tolist()),
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(detections.confidence[j]),
                        )
                    )
            
            # Draw detections on frame
            annotated = draw_detections(img_bgr, detection_objs, thickness=2)
            
            # Save frame
            frame_path = frames_dir / f"frame_{i:05d}.jpg"
            cv2.imwrite(str(frame_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i + 1}/{test_runs}")

    # Calculate statistics
    times_array = np.array(times)
    mean_ms = float(np.mean(times_array))
    std_ms = float(np.std(times_array))
    min_ms = float(np.min(times_array))
    max_ms = float(np.max(times_array))
    p50_ms = float(np.percentile(times_array, 50))
    p95_ms = float(np.percentile(times_array, 95))
    p99_ms = float(np.percentile(times_array, 99))
    fps = 1000.0 / mean_ms

    # Check real-time capability
    realtime_30fps = p99_ms < 33.33  # 30 FPS = 33.33ms per frame
    realtime_60fps = p99_ms < 16.67  # 60 FPS = 16.67ms per frame

    # Check against video's native FPS if available
    realtime_native = None
    if video_info:
        native_fps = video_info["fps"]
        native_frame_time = 1000.0 / native_fps if native_fps > 0 else 33.33
        realtime_native = p99_ms < native_frame_time

    results = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "checkpoint": str(checkpoint_path),
        "model_size": model_size,
        "benchmark_mode": benchmark_mode,
        "optimized": optimize,
        "gpu_info": gpu_info,
        "benchmark_config": {
            "warmup_runs": warmup_runs,
            "test_runs": test_runs,
        },
        "metrics": {
            "mean_ms": round(mean_ms, 2),
            "std_ms": round(std_ms, 2),
            "min_ms": round(min_ms, 2),
            "max_ms": round(max_ms, 2),
            "p50_ms": round(p50_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "fps": round(fps, 1),
        },
        "realtime_capable": {
            "30fps": realtime_30fps,
            "60fps": realtime_60fps,
        },
        "per_frame_results": per_frame_results,
    }

    # Add video-specific info
    if video_info:
        results["video_info"] = {
            "path": str(video_path),
            "width": video_info["width"],
            "height": video_info["height"],
            "fps": video_info["fps"],
            "total_frames": video_info["total_frames"],
        }
        results["realtime_capable"]["native_fps"] = realtime_native
        results["realtime_capable"]["native_fps_value"] = video_info["fps"]
    else:
        results["image_size"] = image_size

    return results


def print_results(results: dict) -> None:
    """Print benchmark results in a readable format."""
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nGPU: {results['gpu_info']['name']}")
    if results["gpu_info"]["available"]:
        logger.info(f"GPU Memory: {results['gpu_info']['memory_gb']} GB")

    logger.info(f"\nModel: {results['model_size']}")
    logger.info(f"Benchmark Mode: {results['benchmark_mode']}")
    logger.info(f"Optimized: {results['optimized']}")

    if "video_info" in results:
        vi = results["video_info"]
        logger.info(f"\nVideo: {Path(vi['path']).name}")
        logger.info(f"  Resolution: {vi['width']}x{vi['height']}")
        logger.info(f"  Native FPS: {vi['fps']:.1f}")
    else:
        logger.info(f"Image Size: {results.get('image_size', 'N/A')}")

    logger.info(f"\nBenchmark Config:")
    logger.info(f"  Warmup runs: {results['benchmark_config']['warmup_runs']}")
    logger.info(f"  Test runs: {results['benchmark_config']['test_runs']}")

    metrics = results["metrics"]
    logger.info(f"\nLatency Metrics:")
    logger.info(f"  Mean:   {metrics['mean_ms']:.2f} ms")
    logger.info(f"  Std:    {metrics['std_ms']:.2f} ms")
    logger.info(f"  Min:    {metrics['min_ms']:.2f} ms")
    logger.info(f"  Max:    {metrics['max_ms']:.2f} ms")
    logger.info(f"  P50:    {metrics['p50_ms']:.2f} ms")
    logger.info(f"  P95:    {metrics['p95_ms']:.2f} ms")
    logger.info(f"  P99:    {metrics['p99_ms']:.2f} ms")

    logger.info(f"\nThroughput:")
    logger.info(f"  FPS: {metrics['fps']:.1f}")

    rt = results["realtime_capable"]
    logger.info(f"\nReal-time Capability:")
    logger.info(f"  30 FPS: {'✓ YES' if rt['30fps'] else '✗ NO'} (requires P99 < 33.33ms)")
    logger.info(f"  60 FPS: {'✓ YES' if rt['60fps'] else '✗ NO'} (requires P99 < 16.67ms)")

    if "native_fps" in rt:
        native_fps = rt.get("native_fps_value", 0)
        native_frame_time = 1000.0 / native_fps if native_fps > 0 else 0
        logger.info(
            f"  Native ({native_fps:.1f} FPS): {'✓ YES' if rt['native_fps'] else '✗ NO'} "
            f"(requires P99 < {native_frame_time:.2f}ms)"
        )

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RF-DETR inference latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    model_group.add_argument("--run", type=str, help="Run name (e.g., rfdetr_h200_20260120_105925)")
    model_group.add_argument("--latest", action="store_true", help="Use the latest run")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "large"],
        default="base",
        help="Model size (default: base)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image size for synthetic benchmark (default: 640)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable model optimization (faster startup, slower inference)",
    )

    # Video benchmark
    parser.add_argument(
        "--video",
        type=str,
        help=f"Video file for realistic benchmark (default: {DEFAULT_VIDEO} if --use-video is set)",
    )
    parser.add_argument(
        "--use-video",
        action="store_true",
        help=f"Use default video ({DEFAULT_VIDEO}) for benchmark",
    )

    # Benchmark configuration
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs (default: 10)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs (default: 100)",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for results (default: benchmark_results/)",
    )
    
    # Latency visualization
    parser.add_argument(
        "--create-latency-video",
        action="store_true",
        help="Create side-by-side latency visualization video (requires --video)",
    )

    args = parser.parse_args()

    # Resolve checkpoint
    checkpoint_path: Path | None = None

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    elif args.run:
        run_dir = RUNS_DIR / args.run
        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            sys.exit(1)
        checkpoint_path = find_checkpoint_in_run(run_dir)
        if not checkpoint_path:
            logger.error(f"No checkpoint found in run directory: {run_dir}")
            sys.exit(1)
    elif args.latest:
        run_dir = find_latest_run()
        if not run_dir:
            logger.error("No runs found")
            sys.exit(1)
        checkpoint_path = find_checkpoint_in_run(run_dir)
        if not checkpoint_path:
            logger.error(f"No checkpoint found in latest run: {run_dir}")
            sys.exit(1)
        logger.info(f"Using latest run: {run_dir.name}")

    logger.info(f"Checkpoint: {checkpoint_path}")

    # Resolve video path
    video_path = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            sys.exit(1)
    elif args.use_video:
        video_path = Path(DEFAULT_VIDEO)
        if not video_path.exists():
            logger.error(f"Default video not found: {video_path}")
            logger.error("Please specify a video with --video or ensure crane_hook_1_short.mp4 exists")
            sys.exit(1)
    
    # Validate latency video requirements
    if args.create_latency_video and not video_path:
        logger.error("--create-latency-video requires a video file (use --video or --use-video)")
        sys.exit(1)
    
    # Determine output directory early for frame saving
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("benchmark_results") / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    results = benchmark_latency(
        checkpoint_path=checkpoint_path,
        model_size=args.model,
        image_size=args.image_size,
        warmup_runs=args.warmup,
        test_runs=args.runs,
        optimize=not args.no_optimize,
        video_path=video_path,
        save_frames=args.create_latency_video,
        output_dir=output_dir,
    )

    # Print results
    print_results(results)

    # Save results
    output_file = output_dir / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    
    # Create latency visualization if requested
    if args.create_latency_video:
        logger.info("\n" + "=" * 70)
        logger.info("Creating latency visualization video...")
        logger.info("=" * 70)
        
        try:
            from cli.create_latency_video import create_latency_video
            from cli.create_sidebyside_video import create_sidebyside_video
            
            # Step 1: Create latency-delayed inference video
            latency_video_path = create_latency_video(
                video_path=video_path,
                results_json_path=output_file,
                frames_dir=output_dir / "frames",
                output_path=output_dir / "detected_latency.mp4",
            )
            logger.info(f"Latency video created: {latency_video_path}")
            
            # Step 2: Create side-by-side comparison video
            sidebyside_path = create_sidebyside_video(
                original_video_path=video_path,
                latency_video_path=latency_video_path,
                results_json_path=output_file,
                output_path=output_dir / "sidebyside_latency.mp4",
            )
            logger.info(f"Side-by-side video created: {sidebyside_path}")
            
            logger.info("\n" + "=" * 70)
            logger.info("Latency visualization complete!")
            logger.info("=" * 70)
            logger.info(f"View the result: {sidebyside_path}")
            
        except Exception as e:
            logger.error(f"Failed to create latency visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
