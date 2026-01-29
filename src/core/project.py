"""Project management utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ProjectConfig:
    """Project configuration settings."""

    sample_mode: str = "seconds"
    sample_interval: float = 0.5
    tracking_mode: str = "visible_only"
    max_age: int = 30
    iou_threshold: float = 0.3
    min_hits: int = 3
    use_appearance_embedding: bool = False


@dataclass
class Project:
    """
    Represents a Batman project.
    
    A project contains videos, extracted frames, annotations, and training runs.
    This class provides utilities for loading, saving, and managing project data.
    """

    path: Path
    name: str = ""
    description: str = ""
    classes: list[str] = field(default_factory=list)
    class_sources: dict[str, str] = field(default_factory=dict)
    config: ProjectConfig = field(default_factory=ProjectConfig)
    video_count: int = 0
    frame_count: int = 0
    annotation_count: int = 0
    current_iteration: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        self.path = Path(self.path)

    # -------------------------------------------------------------------------
    # Directory paths
    # -------------------------------------------------------------------------

    @property
    def videos_dir(self) -> Path:
        return self.path / "videos"

    @property
    def frames_dir(self) -> Path:
        return self.path / "frames"

    @property
    def labels_dir(self) -> Path:
        return self.path / "labels" / "current"

    @property
    def exports_dir(self) -> Path:
        return self.path / "exports"

    @property
    def runs_dir(self) -> Path:
        return self.path / "runs"

    @property
    def imports_dir(self) -> Path:
        return self.path / "imports"

    @property
    def config_path(self) -> Path:
        return self.path / "project.json"

    @property
    def annotations_path(self) -> Path:
        return self.labels_dir / "annotations.json"

    @property
    def tracks_path(self) -> Path:
        return self.labels_dir / "tracks.json"

    @property
    def imports_metadata_path(self) -> Path:
        return self.path / "imports" / "imports.json"

    def get_next_import_video_id(self) -> int:
        """
        Get the next available negative video ID for a new import.
        
        Returns:
            Negative integer for the new import (-1, -2, -3, etc.)
        """
        # Find all existing video IDs (both positive and negative)
        existing_ids = []
        if self.frames_dir.exists():
            for video_dir in self.frames_dir.iterdir():
                if video_dir.is_dir() and video_dir.name.lstrip('-').isdigit():
                    existing_ids.append(int(video_dir.name))
        
        # Find the lowest negative ID (most negative)
        negative_ids = [vid for vid in existing_ids if vid < 0]
        if negative_ids:
            return min(negative_ids) - 1  # Go one more negative
        else:
            return -1  # First import

    # -------------------------------------------------------------------------
    # Class methods for loading/creating
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, project_path: Path | str) -> "Project":
        """Load a project from disk."""
        project_path = Path(project_path)

        if not project_path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")

        config_path = project_path / "project.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

        with open(config_path) as f:
            data = json.load(f)

        # Parse config
        config_data = data.get("config", {})
        config = ProjectConfig(**config_data)

        # Parse dates
        created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        updated_at = datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))

        return cls(
            path=project_path,
            name=data.get("name", project_path.name),
            description=data.get("description", ""),
            classes=data.get("classes", []),
            class_sources=data.get("class_sources", {}),
            config=config,
            video_count=data.get("video_count", 0),
            frame_count=data.get("frame_count", 0),
            annotation_count=data.get("annotation_count", 0),
            current_iteration=data.get("current_iteration", 0),
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    def create(
        cls,
        project_path: Path | str,
        name: str,
        description: str = "",
        classes: list[str] | None = None,
    ) -> "Project":
        """Create a new project."""
        project_path = Path(project_path)

        if project_path.exists():
            raise FileExistsError(f"Project already exists: {project_path}")

        # Create directory structure
        project_path.mkdir(parents=True)
        (project_path / "videos").mkdir()
        (project_path / "frames").mkdir()
        (project_path / "labels" / "current").mkdir(parents=True)
        (project_path / "exports").mkdir()
        (project_path / "runs").mkdir()
        (project_path / "imports").mkdir()

        # Initialize empty annotations
        annotations_path = project_path / "labels" / "current" / "annotations.json"
        with open(annotations_path, "w") as f:
            json.dump({}, f)

        tracks_path = project_path / "labels" / "current" / "tracks.json"
        with open(tracks_path, "w") as f:
            json.dump({}, f)

        project = cls(
            path=project_path,
            name=name,
            description=description,
            classes=classes or [],
        )

        project.save()

        return project

    # -------------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Save project configuration to disk."""
        self.updated_at = datetime.utcnow()

        data = {
            "name": self.name,
            "description": self.description,
            "classes": self.classes,
            "class_sources": self.class_sources,
            "config": {
                "sample_mode": self.config.sample_mode,
                "sample_interval": self.config.sample_interval,
                "tracking_mode": self.config.tracking_mode,
                "max_age": self.config.max_age,
                "iou_threshold": self.config.iou_threshold,
                "min_hits": self.config.min_hits,
                "use_appearance_embedding": self.config.use_appearance_embedding,
            },
            "video_count": self.video_count,
            "frame_count": self.frame_count,
            "annotation_count": self.annotation_count,
            "current_iteration": self.current_iteration,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_class(self, class_name: str, source: str = "manual") -> int:
        """
        Add a class to the project.
        
        Returns the class index.
        """
        if class_name not in self.classes:
            self.classes.append(class_name)
            self.class_sources[class_name] = source
        return self.classes.index(class_name)

    def get_class_index(self, class_name: str) -> int | None:
        """Get the index of a class, or None if not found."""
        try:
            return self.classes.index(class_name)
        except ValueError:
            return None

    def load_annotations(self) -> dict[str, Any]:
        """Load annotations from disk."""
        if self.annotations_path.exists():
            with open(self.annotations_path) as f:
                return json.load(f)
        return {}

    def save_annotations(self, annotations: dict[str, Any]) -> None:
        """Save annotations to disk."""
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        with open(self.annotations_path, "w") as f:
            json.dump(annotations, f, indent=2)
        self.annotation_count = len(annotations)

    def load_frames_meta(self, video_id: int) -> dict[str, Any]:
        """Load frames metadata for a video."""
        frames_dir = self.frames_dir / str(video_id)
        frames_meta_path = frames_dir / "frames.json"

        if frames_meta_path.exists():
            with open(frames_meta_path) as f:
                return json.load(f)
        return {}

    def save_frames_meta(self, video_id: int, frames_meta: dict[str, Any]) -> None:
        """Save frames metadata for a video."""
        frames_dir = self.frames_dir / str(video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames_meta_path = frames_dir / "frames.json"

        with open(frames_meta_path, "w") as f:
            json.dump(frames_meta, f, indent=2)

    def load_imports_metadata(self) -> dict[str, Any]:
        """Load imports metadata."""
        if self.imports_metadata_path.exists():
            with open(self.imports_metadata_path) as f:
                return json.load(f)
        return {}

    def save_imports_metadata(self, imports_meta: dict[str, Any]) -> None:
        """Save imports metadata."""
        self.imports_dir.mkdir(parents=True, exist_ok=True)
        with open(self.imports_metadata_path, "w") as f:
            json.dump(imports_meta, f, indent=2)

    def register_import(self, import_metadata: dict[str, Any]) -> str:
        """
        Register a new import and return its ID.
        
        Args:
            import_metadata: Metadata about the import (type, source details, etc.)
            
        Returns:
            Import ID that can be referenced by frames
        """
        imports_meta = self.load_imports_metadata()
        
        # Generate import ID based on timestamp
        import_id = f"import_{len(imports_meta) + 1}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        imports_meta[import_id] = import_metadata
        self.save_imports_metadata(imports_meta)
        
        return import_id

    def __repr__(self) -> str:
        return f"Project(name={self.name!r}, path={self.path}, classes={len(self.classes)}, frames={self.frame_count})"
