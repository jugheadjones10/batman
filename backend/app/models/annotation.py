"""Annotation-related Pydantic models."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 coordinates)."""

    x: float = Field(..., ge=0, le=1)  # Center x
    y: float = Field(..., ge=0, le=1)  # Center y
    width: float = Field(..., ge=0, le=1)
    height: float = Field(..., ge=0, le=1)


class AnnotationCreate(BaseModel):
    """Request to create an annotation."""

    frame_id: int
    class_label_id: int
    box: BoundingBox
    track_id: Optional[int] = None
    source: Literal["auto", "manual", "corrected"] = "manual"
    is_exemplar: bool = False
    exemplar_type: Optional[Literal["anchor", "correction"]] = None


class AnnotationUpdate(BaseModel):
    """Request to update an annotation."""

    class_label_id: Optional[int] = None
    box: Optional[BoundingBox] = None
    track_id: Optional[int] = None
    source: Optional[Literal["auto", "manual", "corrected"]] = None
    is_exemplar: Optional[bool] = None
    exemplar_type: Optional[Literal["anchor", "correction"]] = None


class AnnotationInfo(BaseModel):
    """Annotation information response."""

    id: int
    frame_id: int
    class_label_id: int
    class_name: str
    class_color: str
    box: BoundingBox
    track_id: Optional[int] = None
    confidence: float
    source: str
    is_exemplar: bool
    exemplar_type: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TrackInfo(BaseModel):
    """Track information response."""

    id: int
    track_id: int
    class_label_id: int
    class_name: str
    class_color: str
    video_id: int
    start_frame: int
    end_frame: int
    annotation_count: int
    is_approved: bool
    needs_review: bool


class TrackUpdate(BaseModel):
    """Request to update a track."""

    class_label_id: Optional[int] = None
    is_approved: Optional[bool] = None
    needs_review: Optional[bool] = None


class TrackSplitRequest(BaseModel):
    """Request to split a track at a specific frame."""

    track_id: int
    split_frame: int


class TrackMergeRequest(BaseModel):
    """Request to merge two tracks."""

    source_track_id: int
    target_track_id: int


class ExemplarCreate(BaseModel):
    """Request to create an exemplar."""

    class_label_id: int
    frame_id: int
    box: BoundingBox
    exemplar_type: Literal["anchor", "correction"] = "anchor"


class ExemplarInfo(BaseModel):
    """Exemplar information response."""

    id: int
    class_label_id: int
    class_name: str
    frame_id: int
    box: BoundingBox
    exemplar_type: str
    created_at: datetime


class ProblemQueueItem(BaseModel):
    """An item in the problem queue for review."""

    frame_id: int
    frame_number: int
    timestamp: float
    video_id: int
    problem_type: str  # 'box_jump', 'track_switch', 'rapid_appear_disappear'
    severity: float  # 0-1, higher = more likely a problem
    description: str
    affected_track_ids: list[int] = []

