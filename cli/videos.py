#!/usr/bin/env python3
"""
Manage videos in a Batman project from the command line.

Usage:
    # Add a video to a project
    python -m cli.videos add --project data/projects/CraneHook /path/to/video.mp4

    # Add as test-only (excluded from training)
    python -m cli.videos add --project data/projects/CraneHook --test-only /path/to/video.mp4

    # List videos
    python -m cli.videos list --project data/projects/CraneHook

    # Remove a video
    python -m cli.videos remove --project data/projects/CraneHook video_2

    # Toggle test-only flag
    python -m cli.videos set-test --project data/projects/CraneHook video_2 --on
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
from loguru import logger

from src.core.project import Project


def get_video_info(video_path: Path) -> dict:
    """Probe a video file for metadata using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def cmd_add(args):
    """Add a video to a project."""
    project = Project.load(args.project)

    for video_file in args.files:
        if not video_file.exists():
            logger.error(f"File not found: {video_file}")
            continue

        suffix = video_file.suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
            logger.error(f"Unsupported video format: {suffix}")
            continue

        videos_meta = project.load_videos_meta()
        video_id = project.get_next_video_source_key()

        dest = project.videos_dir / f"{video_id}_{video_file.name}"
        project.videos_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Copying {video_file.name} -> {dest.name}")
        shutil.copy2(video_file, dest)

        logger.info("Probing video metadata...")
        try:
            info = get_video_info(dest)
        except Exception as e:
            logger.error(f"Failed to read video: {e}")
            dest.unlink()
            continue

        now = datetime.utcnow()
        videos_meta[video_id] = {
            "filename": video_file.name,
            "original_path": str(dest),
            "proxy_path": None,
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "duration": info["duration"],
            "total_frames": info["total_frames"],
            "has_proxy": False,
            "frame_count": 0,
            "exclude_from_training": args.test_only,
            "created_at": now.isoformat(),
        }

        project.save_videos_meta(videos_meta)

        project.video_count = len(videos_meta)
        project.save()

        flag = " (test-only)" if args.test_only else ""
        logger.info(f"Added: {video_id} -> {video_file.name}{flag}")
        logger.info(f"  {info['width']}x{info['height']} @ {info['fps']:.1f}fps, {info['duration']:.1f}s, {info['total_frames']} frames")


def cmd_list(args):
    """List videos in a project."""
    project = Project.load(args.project)
    videos_meta = project.load_videos_meta()

    if not videos_meta:
        print("No videos in project.")
        return

    print(f"\nVideos in {project.name} ({len(videos_meta)} total):\n")
    for vid_id, data in sorted(videos_meta.items()):
        test_flag = " [TEST-ONLY]" if data.get("exclude_from_training", False) else ""
        frames = data.get("frame_count", 0)
        print(f"  {vid_id:15s}  {data['filename']:30s}  {data['width']}x{data['height']}  {data.get('fps', 0):.0f}fps  {data.get('duration', 0):.1f}s  frames={frames}{test_flag}")


def cmd_remove(args):
    """Remove a video from a project."""
    project = Project.load(args.project)
    videos_meta = project.load_videos_meta()

    for video_id in args.video_ids:
        if video_id not in videos_meta:
            logger.error(f"Video not found: {video_id}")
            continue

        data = videos_meta[video_id]

        video_path = Path(data["original_path"])
        if video_path.exists():
            video_path.unlink()
            logger.info(f"Deleted file: {video_path}")

        if data.get("proxy_path"):
            proxy = Path(data["proxy_path"])
            if proxy.exists():
                proxy.unlink()

        frames_dir = project.frames_dir / video_id
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.info(f"Deleted frames: {frames_dir}")

        del videos_meta[video_id]
        project.save_videos_meta(videos_meta)

        project.video_count = len(videos_meta)
        project.save()

        logger.info(f"Removed: {video_id}")


def cmd_set_test(args):
    """Toggle exclude_from_training flag."""
    project = Project.load(args.project)
    videos_meta = project.load_videos_meta()

    for video_id in args.video_ids:
        if video_id not in videos_meta:
            logger.error(f"Video not found: {video_id}")
            continue

        value = args.on
        videos_meta[video_id]["exclude_from_training"] = value
        state = "test-only" if value else "training-included"
        logger.info(f"{video_id}: set to {state}")

    project.save_videos_meta(videos_meta)


def main():
    parser = argparse.ArgumentParser(
        description="Manage videos in a Batman project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project", "-p", type=Path, required=True, help="Path to Batman project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add
    add_parser = subparsers.add_parser("add", help="Add video(s) to the project")
    add_parser.add_argument("files", type=Path, nargs="+", help="Video file(s) to add")
    add_parser.add_argument("--test-only", action="store_true", help="Mark as test-only (exclude from training)")

    # list
    subparsers.add_parser("list", help="List videos in the project")

    # remove
    rm_parser = subparsers.add_parser("remove", help="Remove video(s) from the project")
    rm_parser.add_argument("video_ids", type=str, nargs="+", help="Video source_key(s) to remove")

    # set-test
    st_parser = subparsers.add_parser("set-test", help="Toggle test-only flag on video(s)")
    st_parser.add_argument("video_ids", type=str, nargs="+", help="Video source_key(s)")
    st_parser.add_argument("--on", action="store_true", default=True, help="Set exclude_from_training=true (default)")
    st_parser.add_argument("--off", dest="on", action="store_false", help="Set exclude_from_training=false")

    args = parser.parse_args()

    if not args.project.exists():
        logger.error(f"Project not found: {args.project}")
        sys.exit(1)

    if args.command == "add":
        cmd_add(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "set-test":
        cmd_set_test(args)


if __name__ == "__main__":
    main()
