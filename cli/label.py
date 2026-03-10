#!/usr/bin/env python3
"""
Run SAM3 auto-labeling on a Batman project.

Supports video frames and manual-data sources. Uses class descriptions from project
(or --descriptions JSON) as SAM prompts.

Usage:
    # Label all frames in a video
    python -m cli.label --project data/projects/MyProject --video video_1

    # Label specific frames
    python -m cli.label --project data/projects/MyProject --video video_1 --frames 0,5,10,20

    # Label only manual_data (all images in root dataset)
    python -m cli.label --project data/projects/MyProject --source manual_data

    # Label a specific manual dataset
    python -m cli.label --project data/projects/MyProject --source manual_data__mydataset

    # Custom class descriptions (JSON object)
    python -m cli.label --project data/projects/MyProject --video video_1 --descriptions '{"hook":"yellow metal crane hook"}'

    # Skip frames that already have annotations (default)
    python -m cli.label --project data/projects/MyProject --video video_1 --skip-labeled
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from src.core.labeler import auto_label_frames
from src.core.project import Project


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SAM3 auto-labeling on project frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--project",
        type=Path,
        required=True,
        help="Path to the project directory (or project name under data/projects)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video ID or source_key (e.g. video_1). If set, only frames from this video are considered.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source key to filter by (e.g. manual_data, manual_data__dataset). Mutually exclusive with --video.",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame IDs. If omitted, all frames for the video/source are used.",
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default=None,
        help='JSON object of class_name -> description for SAM prompts. Overrides project class_descriptions.',
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for SAM (default 0.25).",
    )
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        default=True,
        help="Skip frames that already have at least one annotation (default).",
    )
    parser.add_argument(
        "--no-skip-labeled",
        action="store_false",
        dest="skip_labeled",
        help="Do not skip already-labeled frames.",
    )

    args = parser.parse_args()

    if args.video and args.source:
        logger.error("Use only one of --video or --source")
        return 1

    project_path = args.project
    if not project_path.is_absolute():
        data_projects = Path("data/projects")
        if (data_projects / project_path).exists():
            project_path = data_projects / project_path
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return 1

    try:
        project = Project.load(project_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load project: {e}")
        return 1

    if not project.classes:
        logger.error("Project has no classes defined")
        return 1

    frame_ids: list[str | int] = []
    if args.frames:
        for part in args.frames.split(","):
            part = part.strip()
            if part.isdigit():
                frame_ids.append(int(part))
            else:
                frame_ids.append(part)
        if args.video or args.source:
            source_key = args.video or args.source
            meta = project.load_frames_meta(source_key)
            valid_ids = set(meta.keys())
            frame_ids = [fid for fid in frame_ids if str(fid) in valid_ids]
    else:
        if args.video:
            meta = project.load_frames_meta(args.video)
            frame_ids = list(meta.keys())
        elif args.source:
            meta = project.load_frames_meta(args.source)
            frame_ids = list(meta.keys())
        else:
            logger.error("Specify --video, --source, or --frames")
            return 1

    if not frame_ids:
        logger.warning("No frames to process")
        return 0

    class_descriptions = dict(project.class_descriptions) if project.class_descriptions else {}
    if args.descriptions:
        try:
            class_descriptions.update(json.loads(args.descriptions))
        except json.JSONDecodeError as e:
            logger.error(f"Invalid --descriptions JSON: {e}")
            return 1

    logger.info(f"Project: {project.name}, frames: {len(frame_ids)}, skip_labeled: {args.skip_labeled}")
    created = auto_label_frames(
        project,
        frame_ids,
        class_descriptions=class_descriptions or None,
        confidence=args.confidence,
        skip_labeled=args.skip_labeled,
    )
    print(f"Created {len(created)} annotations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
