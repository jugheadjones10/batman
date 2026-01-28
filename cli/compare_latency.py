#!/usr/bin/env python3
"""
Compare RF-DETR latency benchmark results across different GPUs.

Generates comparison tables and identifies which GPUs can maintain
real-time performance for different video frame rates.

Usage:
    # Compare results from a benchmark run
    python -m cli.compare_latency benchmark_results/20260128_120000/

    # Save comparison as markdown
    python -m cli.compare_latency benchmark_results/20260128_120000/ --output comparison.md

    # Compare specific GPUs only
    python -m cli.compare_latency benchmark_results/20260128_120000/ --gpus h200,a100-80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def load_benchmark_results(benchmark_dir: Path) -> dict[str, dict]:
    """
    Load all benchmark results from a directory.

    Args:
        benchmark_dir: Directory containing GPU-specific subdirectories

    Returns:
        Dictionary mapping GPU type to benchmark results
    """
    results = {}

    if not benchmark_dir.exists():
        logger.error(f"Benchmark directory not found: {benchmark_dir}")
        return results

    # Look for subdirectories (one per GPU type)
    for gpu_dir in sorted(benchmark_dir.iterdir()):
        if not gpu_dir.is_dir():
            continue

        result_file = gpu_dir / "benchmark_results.json"
        if not result_file.exists():
            logger.warning(f"No results found in {gpu_dir.name}")
            continue

        try:
            with open(result_file) as f:
                data = json.load(f)
                results[gpu_dir.name] = data
        except Exception as e:
            logger.error(f"Failed to load {result_file}: {e}")

    return results


def format_comparison_table(results: dict[str, dict]) -> str:
    """
    Generate a formatted comparison table.

    Args:
        results: Dictionary mapping GPU type to benchmark results

    Returns:
        Formatted table as string
    """
    if not results:
        return "No results to compare"

    lines = []
    lines.append("=" * 120)
    lines.append("GPU LATENCY COMPARISON")
    lines.append("=" * 120)

    # Show benchmark mode info from first result
    first_result = next(iter(results.values()))
    benchmark_mode = first_result.get("benchmark_mode", "unknown")
    lines.append(f"Benchmark Mode: {benchmark_mode}")
    if "video_info" in first_result:
        vi = first_result["video_info"]
        lines.append(f"Video: {Path(vi['path']).name} ({vi['width']}x{vi['height']} @ {vi['fps']:.1f} FPS)")
    lines.append("")

    # Table header
    GPU_NAME_WIDTH = 30
    header = f"{'GPU Type':<12} {'GPU Name':<{GPU_NAME_WIDTH}} {'Mean':<9} {'P50':<9} {'P95':<9} {'P99':<9} {'FPS':<7} {'30fps':<6} {'60fps':<6}"
    lines.append(header)
    lines.append("-" * 120)

    # Sort by mean latency (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"]["mean_ms"])

    # Table rows
    for gpu_type, data in sorted_results:
        gpu_name = data["gpu_info"]["name"]
        # Truncate long GPU names
        if len(gpu_name) > GPU_NAME_WIDTH:
            gpu_name = gpu_name[: GPU_NAME_WIDTH - 3] + "..."
        metrics = data["metrics"]
        rt = data["realtime_capable"]

        mean = f"{metrics['mean_ms']:.1f}ms"
        p50 = f"{metrics['p50_ms']:.1f}ms"
        p95 = f"{metrics['p95_ms']:.1f}ms"
        p99 = f"{metrics['p99_ms']:.1f}ms"
        fps = f"{metrics['fps']:.1f}"
        rt_30 = "✓" if rt["30fps"] else "✗"
        rt_60 = "✓" if rt["60fps"] else "✗"

        row = f"{gpu_type:<12} {gpu_name:<{GPU_NAME_WIDTH}} {mean:<9} {p50:<9} {p95:<9} {p99:<9} {fps:<7} {rt_30:<6} {rt_60:<6}"
        lines.append(row)

    lines.append("=" * 120)
    lines.append("")

    # Add detailed statistics
    lines.append("DETAILED STATISTICS")
    lines.append("-" * 120)
    lines.append("")

    for gpu_type, data in sorted_results:
        lines.append(f"{gpu_type.upper()} - {data['gpu_info']['name']}")
        lines.append(f"  Device: {data['gpu_info']['device']}")
        if data["gpu_info"]["available"]:
            lines.append(f"  Memory: {data['gpu_info']['memory_gb']:.1f} GB")

        metrics = data["metrics"]
        lines.append(f"  Latency:")
        lines.append(f"    Mean:   {metrics['mean_ms']:.2f} ms  (± {metrics['std_ms']:.2f} ms)")
        lines.append(f"    Min:    {metrics['min_ms']:.2f} ms")
        lines.append(f"    Max:    {metrics['max_ms']:.2f} ms")
        lines.append(f"    P50:    {metrics['p50_ms']:.2f} ms")
        lines.append(f"    P95:    {metrics['p95_ms']:.2f} ms")
        lines.append(f"    P99:    {metrics['p99_ms']:.2f} ms")
        lines.append(f"  Throughput: {metrics['fps']:.1f} FPS")

        rt = data["realtime_capable"]
        lines.append(f"  Real-time:")
        lines.append(f"    30 FPS: {'✓ YES' if rt['30fps'] else '✗ NO'}")
        lines.append(f"    60 FPS: {'✓ YES' if rt['60fps'] else '✗ NO'}")
        lines.append("")

    lines.append("=" * 120)

    return "\n".join(lines)


def format_comparison_markdown(results: dict[str, dict]) -> str:
    """
    Generate a markdown-formatted comparison table.

    Args:
        results: Dictionary mapping GPU type to benchmark results

    Returns:
        Formatted markdown as string
    """
    if not results:
        return "No results to compare"

    lines = []
    lines.append("# GPU Latency Comparison")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| GPU Type | GPU Name | Mean | P50 | P95 | P99 | FPS | 30fps | 60fps |")
    lines.append("|----------|----------|------|-----|-----|-----|-----|-------|-------|")

    # Sort by mean latency (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["metrics"]["mean_ms"])

    for gpu_type, data in sorted_results:
        gpu_name = data["gpu_info"]["name"]
        metrics = data["metrics"]
        rt = data["realtime_capable"]

        mean = f"{metrics['mean_ms']:.1f}ms"
        p50 = f"{metrics['p50_ms']:.1f}ms"
        p95 = f"{metrics['p95_ms']:.1f}ms"
        p99 = f"{metrics['p99_ms']:.1f}ms"
        fps = f"{metrics['fps']:.1f}"
        rt_30 = "✓" if rt["30fps"] else "✗"
        rt_60 = "✓" if rt["60fps"] else "✗"

        lines.append(f"| {gpu_type} | {gpu_name} | {mean} | {p50} | {p95} | {p99} | {fps} | {rt_30} | {rt_60} |")

    lines.append("")

    # Detailed sections
    lines.append("## Detailed Results")
    lines.append("")

    for gpu_type, data in sorted_results:
        lines.append(f"### {gpu_type.upper()} - {data['gpu_info']['name']}")
        lines.append("")

        lines.append(f"**Device:** {data['gpu_info']['device']}")
        if data["gpu_info"]["available"]:
            lines.append(f"**Memory:** {data['gpu_info']['memory_gb']:.1f} GB")
        lines.append("")

        metrics = data["metrics"]
        lines.append("**Latency Metrics:**")
        lines.append(f"- Mean: {metrics['mean_ms']:.2f} ms (± {metrics['std_ms']:.2f} ms)")
        lines.append(f"- Min: {metrics['min_ms']:.2f} ms")
        lines.append(f"- Max: {metrics['max_ms']:.2f} ms")
        lines.append(f"- P50: {metrics['p50_ms']:.2f} ms")
        lines.append(f"- P95: {metrics['p95_ms']:.2f} ms")
        lines.append(f"- P99: {metrics['p99_ms']:.2f} ms")
        lines.append("")

        lines.append(f"**Throughput:** {metrics['fps']:.1f} FPS")
        lines.append("")

        rt = data["realtime_capable"]
        lines.append("**Real-time Capability:**")
        lines.append(f"- 30 FPS: {'✓ YES' if rt['30fps'] else '✗ NO'}")
        lines.append(f"- 60 FPS: {'✓ YES' if rt['60fps'] else '✗ NO'}")
        lines.append("")

    # Add notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Real-time 30 FPS** requires P99 latency < 33.33ms")
    lines.append("- **Real-time 60 FPS** requires P99 latency < 16.67ms")
    lines.append("- Latency is measured for single-frame inference on dummy images")
    lines.append("- Actual video inference may include additional overhead from frame reading, tracking, etc.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare RF-DETR latency benchmark results across GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "benchmark_dir",
        type=str,
        help="Directory containing benchmark results (e.g., benchmark_results/20260128_120000/)",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated list of GPU types to compare (default: all)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save comparison to file (auto-detects format from extension: .md, .txt)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "markdown"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # Load results
    benchmark_dir = Path(args.benchmark_dir)
    logger.info(f"Loading benchmark results from: {benchmark_dir}")

    results = load_benchmark_results(benchmark_dir)

    if not results:
        logger.error("No benchmark results found")
        sys.exit(1)

    logger.info(f"Found results for {len(results)} GPU(s): {', '.join(results.keys())}")

    # Filter by GPU types if specified
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",")]
        results = {k: v for k, v in results.items() if k in gpu_list}
        if not results:
            logger.error(f"No results found for specified GPUs: {args.gpus}")
            sys.exit(1)

    # Detect format from output filename if provided
    output_format = args.format
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == ".md":
            output_format = "markdown"
        elif output_path.suffix == ".txt":
            output_format = "text"

    # Generate comparison
    if output_format == "markdown":
        comparison = format_comparison_markdown(results)
    else:
        comparison = format_comparison_table(results)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(comparison)
        logger.info(f"Comparison saved to: {output_path}")
    else:
        print("\n" + comparison)


if __name__ == "__main__":
    main()
