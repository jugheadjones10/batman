"""Project-related Pydantic models."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    """Request to create a new project."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    classes: list[str] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Project configuration settings."""

    # Sampling settings
    sample_mode: Literal["frames", "seconds"] = "seconds"
    sample_interval: float = 0.5  # N frames or T seconds

    # Tracking settings
    tracking_mode: Literal["visible_only", "occlusion_tolerant"] = "visible_only"
    max_age: int = 30
    iou_threshold: float = 0.3
    min_hits: int = 3
    use_appearance_embedding: bool = False


class ProjectInfo(BaseModel):
    """Project information response."""

    name: str
    path: str
    description: Optional[str] = None
    classes: list[str] = []
    config: ProjectConfig
    video_count: int = 0
    frame_count: int = 0
    annotation_count: int = 0
    current_iteration: int = 0
    created_at: datetime
    updated_at: datetime


class ClassLabelCreate(BaseModel):
    """Request to create a class label."""

    name: str = Field(..., min_length=1, max_length=255)
    color: str = Field(default="#00FF00", pattern=r"^#[0-9A-Fa-f]{6}$")
    description: Optional[str] = None


class ClassLabelInfo(BaseModel):
    """Class label response."""

    id: int
    name: str
    color: str
    description: Optional[str] = None


class LabelIterationInfo(BaseModel):
    """Label iteration information."""

    id: int
    version: int
    description: Optional[str] = None
    total_annotations: int
    total_tracks: int
    approved_frames: int
    is_active: bool
    created_at: datetime

