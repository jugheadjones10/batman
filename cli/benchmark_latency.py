#!/usr/bin/env python3
"""
Benchmark RF-DETR inference latency across different GPUs.

Measures comprehensive latency metrics including percentiles, throughput,
and real-time capability for video inference.

Usage:
    # Benchmark with checkpoint
    python -m cli.benchmark_latency --checkpoint runs/my_run/best.pth

    # Benchmark with run name
    python -m cli.benchmark_latency --run rfdetr_h200_20260120_105925

    # Benchmark with specific parameters
    python -m cli.benchmark_latency --run my_run --warmup 10 --runs 100 \\
        --image-size 640 --model base --output benchmark_results/
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.core.inference import RFDETRInference


RUNS_DIR = Path("runs")


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


def benchmark_latency(
    checkpoint_path: Path,
    model_size: str = "base",
    image_size: int = 640,
    warmup_runs: int = 10,
    test_runs: int = 100,
    optimize: bool = True,
) -> dict:
    """
    Benchmark inference latency with comprehensive metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        model_size: Model size (base or large)
        image_size: Image size for inference
        warmup_runs: Number of warmup runs
        test_runs: Number of test runs
        optimize: Whether to optimize model for inference

    Returns:
        Dictionary with benchmark results
    """
    logger.info("Loading model...")
    engine = RFDETRInference(
        checkpoint=checkpoint_path,
        model_size=model_size,
        optimize_for_inference=optimize,
    )

    # Create dummy image
    dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    # Get GPU info
    gpu_info = get_gpu_info()
    logger.info(f"GPU: {gpu_info['name']}")

    # Warmup phase
    logger.info(f"Warming up with {warmup_runs} runs...")
    for i in range(warmup_runs):
        _ = engine.model.predict(dummy_img)
        if (i + 1) % 5 == 0:
            logger.info(f"  Warmup: {i + 1}/{warmup_runs}")

    # Benchmark phase
    logger.info(f"Running {test_runs} benchmark iterations...")
    times = []

    for i in range(test_runs):
        start = time.perf_counter()
        _ = engine.model.predict(dummy_img)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

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

    results = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "checkpoint": str(checkpoint_path),
        "model_size": model_size,
        "image_size": image_size,
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
    }

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
    logger.info(f"Image Size: {results['image_size']}")
    logger.info(f"Optimized: {results['optimized']}")

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
        help="Image size for inference (default: 640)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable model optimization (faster startup, slower inference)",
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

    # Run benchmark
    results = benchmark_latency(
        checkpoint_path=checkpoint_path,
        model_size=args.model,
        image_size=args.image_size,
        warmup_runs=args.warmup,
        test_runs=args.runs,
        optimize=not args.no_optimize,
    )

    # Print results
    print_results(results)

    # Save results
    output_dir = Path(args.output) if args.output else Path("benchmark_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
