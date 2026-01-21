#!/usr/bin/env python3
"""
Launch FiftyOne viewer for a Batman project.

Usage:
    python scripts/fiftyone_viewer.py <project_name>
    python scripts/fiftyone_viewer.py Test
    python scripts/fiftyone_viewer.py Test --port 5152

This will:
1. Load all images and annotations from the project
2. Convert to FiftyOne format
3. Launch the FiftyOne web UI for visual exploration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_project_to_fiftyone(project_name: str):
    """Load a Batman project into FiftyOne dataset."""
    import fiftyone as fo

    from src.core.project import Project

    # Find project
    data_dir = Path("data/projects")
    project_path = data_dir / project_name

    if not project_path.exists():
        print(f"Error: Project '{project_name}' not found at {project_path}")
        print(f"\nAvailable projects:")
        for p in data_dir.iterdir():
            if p.is_dir() and (p / "project.json").exists():
                print(f"  - {p.name}")
        sys.exit(1)

    print(f"Loading project: {project_name}")
    project = Project.load(project_path)

    # Load annotations
    annotations = project.load_annotations()
    print(f"  Found {len(annotations)} annotations")

    # Get class names
    class_names = project.classes
    print(f"  Classes: {class_names}")

    # Create FiftyOne dataset
    dataset_name = f"batman-{project_name}"

    # Delete existing dataset with same name
    if dataset_name in fo.list_datasets():
        print(f"  Deleting existing FiftyOne dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = False  # Don't save to disk

    # Group annotations by frame_id
    annotations_by_frame: dict[int, list[dict]] = {}
    for ann_id, ann in annotations.items():
        frame_id = ann.get("frame_id")
        if frame_id not in annotations_by_frame:
            annotations_by_frame[frame_id] = []
        annotations_by_frame[frame_id].append(ann)

    # Find all frames
    frames_dir = project_path / "frames"
    samples = []

    for video_dir in frames_dir.iterdir():
        if not video_dir.is_dir():
            continue

        try:
            video_id = int(video_dir.name)
        except ValueError:
            continue

        # Load frames metadata
        meta_path = video_dir / "frames.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            frames_meta = json.load(f)

        for frame_id_str, frame_data in frames_meta.items():
            frame_id = int(frame_id_str)
            image_path = Path(frame_data.get("image_path", ""))

            if not image_path.exists():
                # Try relative path
                image_path = video_dir / image_path.name
                if not image_path.exists():
                    continue

            # Create sample
            sample = fo.Sample(filepath=str(image_path.absolute()))

            # Add metadata
            sample["video_id"] = video_id
            sample["frame_id"] = frame_id
            sample["source"] = frame_data.get("source", "unknown")
            sample["split"] = frame_data.get("split", "")
            sample["original_filename"] = frame_data.get("original_filename", "")

            # Add detections
            frame_annotations = annotations_by_frame.get(frame_id, [])
            detections = []

            for ann in frame_annotations:
                # Convert from normalized center format to FiftyOne format
                # FiftyOne uses [x, y, w, h] where x,y is top-left corner (normalized)
                cx = ann.get("x", 0)
                cy = ann.get("y", 0)
                w = ann.get("width", 0)
                h = ann.get("height", 0)

                # Convert to top-left corner
                x = cx - w / 2
                y = cy - h / 2

                # Get class name
                class_idx = ann.get("class_label_id", 0)
                label = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"

                detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=[x, y, w, h],
                        confidence=ann.get("confidence", 1.0),
                        source=ann.get("source", "unknown"),
                    )
                )

            if detections:
                sample["ground_truth"] = fo.Detections(detections=detections)

            samples.append(sample)

    print(f"  Loaded {len(samples)} images")
    dataset.add_samples(samples)

    # Add dataset info
    dataset.info = {
        "project_name": project_name,
        "classes": class_names,
        "total_annotations": len(annotations),
    }

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Launch FiftyOne viewer for a Batman project"
    )
    parser.add_argument("project_name", help="Name of the project to view")
    parser.add_argument("--port", type=int, default=5151, help="Port for FiftyOne UI")
    parser.add_argument(
        "--remote", action="store_true", help="Enable remote access (0.0.0.0)"
    )

    args = parser.parse_args()

    # Check FiftyOne is installed
    try:
        import fiftyone as fo
    except ImportError:
        print("Error: FiftyOne is not installed.")
        print("Install with: pip install fiftyone")
        sys.exit(1)

    # Load project
    dataset = load_project_to_fiftyone(args.project_name)

    print(f"\n{'='*60}")
    print(f"FiftyOne Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {dataset.info.get('classes', [])}")
    print(f"{'='*60}")

    # Launch UI
    print(f"\nLaunching FiftyOne UI on port {args.port}...")
    print(f"Open http://localhost:{args.port} in your browser")
    print("Press Ctrl+C to stop\n")

    address = "0.0.0.0" if args.remote else "localhost"
    session = fo.launch_app(dataset, port=args.port, address=address)

    # Keep running until interrupted
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
