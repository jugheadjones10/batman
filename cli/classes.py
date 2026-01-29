#!/usr/bin/env python3
"""
Manage classes in a Batman project.

This CLI provides tools for listing, renaming, and merging classes.

Usage:
    # List all classes with frame counts and annotations
    python -m cli.classes list --project data/projects/MyProject

    # Rename a class
    python -m cli.classes rename --project data/projects/MyProject \\
        --old-name "container-spreader" --new-name "spreader"

    # Merge multiple classes into one
    python -m cli.classes merge --project data/projects/MyProject \\
        --source container-spreader spreader-old \\
        --target spreader
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core import ClassManager, Project


def list_classes(args: argparse.Namespace) -> int:
    """List all classes in a project."""
    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project not found: {project_path}")
        return 1

    project = Project.load(project_path)
    manager = ClassManager(project)
    classes = manager.list_classes()

    if not classes:
        print("No classes defined in this project")
        return 0

    # Count frames per class
    from collections import defaultdict
    
    annotations = project.load_annotations()
    frames_by_class = defaultdict(set)
    
    for ann in annotations.values():
        class_idx = ann.get("class_label_id")
        frame_id = ann.get("frame_id")
        if class_idx is not None and frame_id is not None:
            frames_by_class[class_idx].add(frame_id)
    
    # Add frame counts to class info
    for cls in classes:
        cls['frame_count'] = len(frames_by_class[cls['index']])
        if cls['frame_count'] > 0:
            cls['avg_per_frame'] = cls['annotation_count'] / cls['frame_count']
        else:
            cls['avg_per_frame'] = 0

    print(f"Project: {project.name}")
    print(f"Total classes: {len(classes)}")
    print(f"Total frames: {project.frame_count}")
    print()
    print(f"{'Idx':<5} {'Name':<30} {'Frames':<8} {'Annotations':<12} {'Avg/Frame':<10} {'Source':<15}")
    print("-" * 95)

    for cls in classes:
        print(
            f"{cls['index']:<5} {cls['name']:<30} {cls['frame_count']:<8} "
            f"{cls['annotation_count']:<12} {cls['avg_per_frame']:<10.2f} {cls['source']:<15}"
        )

    return 0


def rename_class(args: argparse.Namespace) -> int:
    """Rename a class."""
    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project not found: {project_path}")
        return 1

    project = Project.load(project_path)
    manager = ClassManager(project)

    print(f"Project: {project.name}")
    print(f"Renaming: '{args.old_name}' -> '{args.new_name}'")
    print()

    try:
        result = manager.rename_class(args.old_name, args.new_name)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("=" * 50)
    print("Rename Complete!")
    print("=" * 50)
    print(f"  {result.message}")

    return 0


def merge_classes(args: argparse.Namespace) -> int:
    """Merge multiple classes into one."""
    project_path = Path(args.project)

    if not project_path.exists():
        print(f"Error: Project not found: {project_path}")
        return 1

    project = Project.load(project_path)
    manager = ClassManager(project)

    # Build source list: include target if not already present
    source_classes = list(args.source)
    if args.target not in source_classes:
        source_classes.append(args.target)

    print(f"Project: {project.name}")
    print(f"Merging: {', '.join(args.source)} -> '{args.target}'")
    print()

    try:
        result = manager.merge_classes(source_classes, args.target)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print("=" * 50)
    print("Merge Complete!")
    print("=" * 50)
    print(f"  {result.message}")
    print(f"  Annotations updated: {result.annotations_updated}")
    if result.classes_removed:
        print(f"  Classes removed: {', '.join(result.classes_removed)}")
    print()
    print(f"Project now has {len(project.classes)} classes")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage classes in Batman projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all classes with frame counts and annotations
  python -m cli.classes list --project data/projects/MyProject

  # Rename a class
  python -m cli.classes rename --project data/projects/MyProject \\
      --old-name "container-spreader" --new-name "spreader"

  # Merge classes (merge container-spreader into spreader)
  python -m cli.classes merge --project data/projects/MyProject \\
      --source container-spreader spreader-old --target spreader
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Class operation")

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List all classes in a project")
    list_parser.add_argument(
        "--project", type=str, required=True, help="Batman project path"
    )

    # Rename subcommand
    rename_parser = subparsers.add_parser("rename", help="Rename a class")
    rename_parser.add_argument(
        "--project", type=str, required=True, help="Batman project path"
    )
    rename_parser.add_argument(
        "--old-name", type=str, required=True, help="Current class name"
    )
    rename_parser.add_argument(
        "--new-name", type=str, required=True, help="New class name"
    )

    # Merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge multiple classes into one")
    merge_parser.add_argument(
        "--project", type=str, required=True, help="Batman project path"
    )
    merge_parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        required=True,
        help="Source class names to merge FROM",
    )
    merge_parser.add_argument(
        "--target", type=str, required=True, help="Target class name to merge INTO"
    )

    args = parser.parse_args()

    if args.command == "list":
        return list_classes(args)
    elif args.command == "rename":
        return rename_class(args)
    elif args.command == "merge":
        return merge_classes(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
