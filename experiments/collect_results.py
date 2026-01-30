#!/usr/bin/env python3
"""
Collect and compare results from all experiment runs.

Usage:
    # Collect from a specific multirun directory
    python experiments/collect_results.py experiments/multirun/2026-01-30_12-00-00/

    # Collect from the latest multirun
    python experiments/collect_results.py --latest

    # Output as markdown table
    python experiments/collect_results.py --latest --format markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_latest_multirun(base_dir: Path) -> Path | None:
    """Find the most recent multirun directory."""
    multirun_dirs = sorted(base_dir.glob("*"), reverse=True)
    for d in multirun_dirs:
        if d.is_dir() and any(d.iterdir()):
            return d
    return None


def collect_results(multirun_dir: Path) -> list[dict]:
    """
    Aggregate results from a multirun directory.

    Args:
        multirun_dir: Path to the multirun directory

    Returns:
        List of experiment result dictionaries
    """
    results = []

    for exp_dir in sorted(multirun_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                results.append(json.load(f))

    return results


def print_table(results: list[dict], format: str = "text") -> None:
    """
    Print comparison table of experiment results.

    Args:
        results: List of experiment result dictionaries
        format: Output format ('text' or 'markdown')
    """
    if not results:
        print("No results found.")
        return

    # Sort by person fraction
    results.sort(key=lambda r: r.get("frame_sample_fractions", {}).get("person", 1.0))

    if format == "markdown":
        print("\n## Experiment Results Comparison\n")
        print("| Experiment | Person % | Train Images | Train Anns | mAP@50 | mAP@75 | Time (s) |")
        print("|------------|----------|--------------|------------|--------|--------|----------|")

        for r in results:
            fractions = r.get("frame_sample_fractions", {})
            metrics = r.get("metrics", {})
            ds_stats = r.get("dataset_stats", {})
            training = r.get("training", {})

            person_pct = fractions.get("person", 1.0) * 100
            train_imgs = ds_stats.get("train_images", "N/A")
            train_anns = ds_stats.get("train_annotations", "N/A")
            map50 = metrics.get("mAP@50", "N/A")
            map75 = metrics.get("mAP@75", "N/A")
            time_s = training.get("time_seconds", "N/A")

            if isinstance(map50, float):
                map50 = f"{map50:.3f}"
            if isinstance(map75, float):
                map75 = f"{map75:.3f}"
            if isinstance(time_s, float):
                time_s = f"{time_s:.0f}"

            print(
                f"| {r['experiment']:<10} | {person_pct:>6.0f}%  | {train_imgs:>12} | "
                f"{train_anns:>10} | {map50:>6} | {map75:>6} | {time_s:>8} |"
            )
    else:
        # Text format
        print("\n" + "=" * 100)
        print("EXPERIMENT RESULTS COMPARISON")
        print("=" * 100)
        print(
            f"{'Experiment':<16} {'Person %':<10} {'Train Imgs':<12} {'Train Anns':<12} "
            f"{'mAP@50':<10} {'mAP@75':<10} {'Time (s)':<10}"
        )
        print("-" * 100)

        for r in results:
            fractions = r.get("frame_sample_fractions", {})
            metrics = r.get("metrics", {})
            ds_stats = r.get("dataset_stats", {})
            training = r.get("training", {})

            person_pct = fractions.get("person", 1.0) * 100
            train_imgs = ds_stats.get("train_images", "N/A")
            train_anns = ds_stats.get("train_annotations", "N/A")
            map50 = metrics.get("mAP@50", "N/A")
            map75 = metrics.get("mAP@75", "N/A")
            time_s = training.get("time_seconds", "N/A")

            if isinstance(map50, float):
                map50 = f"{map50:.3f}"
            if isinstance(map75, float):
                map75 = f"{map75:.3f}"
            if isinstance(time_s, float):
                time_s = f"{time_s:.0f}"

            print(
                f"{r['experiment']:<16} {person_pct:>6.0f}%{'':<4} {train_imgs:>12} "
                f"{train_anns:>12} {map50:>10} {map75:>10} {time_s:>10}"
            )

        print("=" * 100)

    # Print per-class metrics if available
    print("\n### Per-Class AP@50\n" if format == "markdown" else "\nPer-Class AP@50:\n")

    if format == "markdown":
        # Build header dynamically based on available classes
        all_classes = set()
        for r in results:
            metrics = r.get("metrics", {})
            for key in metrics:
                if key.startswith("AP@50_"):
                    all_classes.add(key.replace("AP@50_", ""))

        if all_classes:
            classes = sorted(all_classes)
            header = "| Experiment | " + " | ".join(classes) + " |"
            separator = "|------------|" + "|".join(["-" * 12 for _ in classes]) + "|"
            print(header)
            print(separator)

            for r in results:
                metrics = r.get("metrics", {})
                row = f"| {r['experiment']:<10} |"
                for cls in classes:
                    val = metrics.get(f"AP@50_{cls}", "N/A")
                    if isinstance(val, float):
                        val = f"{val:.3f}"
                    row += f" {val:>10} |"
                print(row)
    else:
        all_classes = set()
        for r in results:
            metrics = r.get("metrics", {})
            for key in metrics:
                if key.startswith("AP@50_"):
                    all_classes.add(key.replace("AP@50_", ""))

        if all_classes:
            classes = sorted(all_classes)
            header = f"{'Experiment':<16} " + " ".join([f"{c:<12}" for c in classes])
            print(header)
            print("-" * len(header))

            for r in results:
                metrics = r.get("metrics", {})
                row = f"{r['experiment']:<16} "
                for cls in classes:
                    val = metrics.get(f"AP@50_{cls}", "N/A")
                    if isinstance(val, float):
                        val = f"{val:.3f}"
                    row += f"{val:<12} "
                print(row)


def main():
    parser = argparse.ArgumentParser(description="Collect and compare experiment results")
    parser.add_argument(
        "multirun_dir",
        nargs="?",
        type=Path,
        help="Path to multirun directory",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest multirun directory",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Determine multirun directory
    if args.latest:
        base_dir = Path(__file__).parent / "multirun"
        if not base_dir.exists():
            print(f"Error: No multirun directory found at {base_dir}")
            sys.exit(1)
        multirun_dir = find_latest_multirun(base_dir)
        if not multirun_dir:
            print("Error: No multirun directories found")
            sys.exit(1)
        print(f"Using latest multirun: {multirun_dir}")
    elif args.multirun_dir:
        multirun_dir = args.multirun_dir
        if not multirun_dir.exists():
            print(f"Error: Directory not found: {multirun_dir}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Collect results
    results = collect_results(multirun_dir)

    if not results:
        print(f"No experiment_summary.json files found in {multirun_dir}")
        sys.exit(1)

    # Print comparison table
    print_table(results, args.format)

    # Save summary JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Also save to multirun directory
    summary_path = multirun_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
