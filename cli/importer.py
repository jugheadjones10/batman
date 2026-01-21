#!/usr/bin/env python3
"""
Import data from Roboflow or COCO Zoo datasets into a Batman project.

This CLI wraps the core import logic from src.core.importer.

Usage:
    # Import from Roboflow
    python -m cli.importer roboflow \\
        --project data/projects/MyProject \\
        --api-key YOUR_API_KEY \\
        --workspace your-workspace \\
        --rf-project your-project \\
        --version 1

    # Import from COCO Zoo via FiftyOne (specific classes)
    python -m cli.importer coco \\
        --project data/projects/MyProject \\
        --classes person car \\
        --split validation \\
        --max-samples 500

    # Create a new project and import
    python -m cli.importer roboflow \\
        --project data/projects/NewProject \\
        --create \\
        --api-key YOUR_API_KEY \\
        --workspace your-workspace \\
        --rf-project your-project \\
        --version 1

    # List all projects
    python -m cli.importer list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Import core logic
from src.core.importer import DataImporter
from src.core.project import Project


def print_progress(status: str, pct: int, msg: str) -> None:
    """Print progress with a simple progress bar."""
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r[{bar}] {pct:3d}% {msg}", end="", flush=True)
    if pct >= 100:
        print()  # New line when complete


def import_roboflow(args: argparse.Namespace) -> int:
    """Import from Roboflow."""
    # Get or create project
    project_path = Path(args.project)

    if args.create:
        if project_path.exists():
            print(f"Error: Project already exists: {project_path}")
            return 1
        print(f"Creating new project: {project_path}")
        project = Project.create(
            project_path=project_path,
            name=project_path.name,
            description=f"Imported from Roboflow: {args.workspace}/{args.rf_project}",
        )
    else:
        if not project_path.exists():
            print(f"Error: Project not found: {project_path}")
            print("Use --create to create a new project")
            return 1
        project = Project.load(project_path)

    print(f"Project: {project.name}")
    print(f"Existing classes: {project.classes}")
    print(f"Existing frames: {project.frame_count}")
    print()

    # Get API key
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: Roboflow API key required")
        print("Provide via --api-key or ROBOFLOW_API_KEY environment variable")
        print("Get your API key at: https://app.roboflow.com/settings/api")
        return 1

    # Import
    print(f"Importing from Roboflow: {args.workspace}/{args.rf_project}/v{args.version}")
    print()

    importer = DataImporter(project)

    try:
        stats = importer.import_roboflow(
            api_key=api_key,
            workspace=args.workspace,
            rf_project=args.rf_project,
            version=args.version,
            format=args.format,
            on_progress=print_progress,
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    # Print results
    print()
    print("=" * 50)
    print("Import Complete!")
    print("=" * 50)
    print(f"  Images imported:     {stats.images_imported}")
    print(f"  Annotations imported: {stats.annotations_imported}")
    print(f"  Splits imported:     {', '.join(stats.splits_imported)}")
    if stats.classes_added:
        print(f"  New classes added:   {', '.join(stats.classes_added)}")
    print()
    print("Project now has:")
    print(f"  Total classes: {len(project.classes)}")
    print(f"  Total frames:  {project.frame_count}")
    print(f"  Total annotations: {project.annotation_count}")

    return 0


def import_coco(args: argparse.Namespace) -> int:
    """Import from COCO Zoo via FiftyOne."""
    # Get or create project
    project_path = Path(args.project)

    if args.create:
        if project_path.exists():
            print(f"Error: Project already exists: {project_path}")
            return 1
        print(f"Creating new project: {project_path}")
        project = Project.create(
            project_path=project_path,
            name=project_path.name,
            description=f"Imported from COCO Zoo: {', '.join(args.classes)}",
        )
    else:
        if not project_path.exists():
            print(f"Error: Project not found: {project_path}")
            print("Use --create to create a new project")
            return 1
        project = Project.load(project_path)

    print(f"Project: {project.name}")
    print(f"Existing classes: {project.classes}")
    print(f"Existing frames: {project.frame_count}")
    print()

    # Import
    print("Importing from COCO Zoo via FiftyOne")
    print(f"  Classes: {', '.join(args.classes)}")
    print(f"  Split: {args.split}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print()

    importer = DataImporter(project)

    try:
        stats = importer.import_coco_zoo(
            classes=args.classes,
            split=args.split,
            max_samples=args.max_samples,
            on_progress=print_progress,
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    # Print results
    print()
    print("=" * 50)
    print("Import Complete!")
    print("=" * 50)
    print(f"  Images imported:     {stats.images_imported}")
    print(f"  Annotations imported: {stats.annotations_imported}")
    print(f"  Splits imported:     {', '.join(stats.splits_imported)}")
    if stats.classes_added:
        print(f"  New classes added:   {', '.join(stats.classes_added)}")
    print()
    print("Project now has:")
    print(f"  Total classes: {len(project.classes)}")
    print(f"  Total frames:  {project.frame_count}")
    print(f"  Total annotations: {project.annotation_count}")

    return 0


def list_projects(args: argparse.Namespace) -> int:
    """List all projects."""
    projects_dir = Path(args.projects_dir)

    if not projects_dir.exists():
        print(f"No projects directory found at: {projects_dir}")
        return 0

    projects = []
    for p in projects_dir.iterdir():
        if p.is_dir() and (p / "project.json").exists():
            try:
                project = Project.load(p)
                projects.append(project)
            except Exception:
                pass

    if not projects:
        print("No projects found")
        return 0

    print(f"Found {len(projects)} project(s):")
    print()
    print(f"{'Name':<20} {'Classes':<10} {'Frames':<10} {'Annotations':<12}")
    print("-" * 55)
    for p in projects:
        print(f"{p.name:<20} {len(p.classes):<10} {p.frame_count:<10} {p.annotation_count:<12}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import data into Batman projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import from Roboflow
  python -m cli.importer roboflow --project data/projects/MyProject \\
      --api-key YOUR_KEY --workspace my-workspace --rf-project my-project --version 1

  # Import person class from COCO Zoo via FiftyOne
  python -m cli.importer coco --project data/projects/MyProject \\
      --classes person --split validation --max-samples 500

  # Import multiple classes from COCO Zoo
  python -m cli.importer coco --project data/projects/MyProject \\
      --classes person car dog --split train

  # Create new project and import
  python -m cli.importer coco --project data/projects/NewProject --create \\
      --classes person --split validation --max-samples 1000

  # List all projects
  python -m cli.importer list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Import source")

    # Roboflow subcommand
    rf_parser = subparsers.add_parser("roboflow", help="Import from Roboflow")
    rf_parser.add_argument("--project", type=str, required=True, help="Batman project path")
    rf_parser.add_argument(
        "--create", action="store_true", help="Create new project if it doesn't exist"
    )
    rf_parser.add_argument(
        "--api-key", type=str, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    rf_parser.add_argument("--workspace", type=str, required=True, help="Roboflow workspace name")
    rf_parser.add_argument("--rf-project", type=str, required=True, help="Roboflow project name")
    rf_parser.add_argument("--version", type=int, required=True, help="Dataset version number")
    rf_parser.add_argument(
        "--format", type=str, default="coco", help="Download format (default: coco)"
    )

    # COCO subcommand
    coco_parser = subparsers.add_parser("coco", help="Import from COCO Zoo via FiftyOne")
    coco_parser.add_argument("--project", type=str, required=True, help="Batman project path")
    coco_parser.add_argument(
        "--create", action="store_true", help="Create new project if it doesn't exist"
    )
    coco_parser.add_argument(
        "--classes", type=str, nargs="+", required=True,
        help="COCO class names to import (e.g., person car dog)"
    )
    coco_parser.add_argument(
        "--split", type=str, default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split to import (default: validation)"
    )
    coco_parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to import (default: all)"
    )

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List all projects")
    list_parser.add_argument(
        "--projects-dir", type=str, default="data/projects", help="Projects directory"
    )

    args = parser.parse_args()

    if args.command == "roboflow":
        return import_roboflow(args)
    elif args.command == "coco":
        return import_coco(args)
    elif args.command == "list":
        return list_projects(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
