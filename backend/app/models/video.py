"""Video-related Pydantic models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Video information response."""

    id: int
    filename: str
    width: int
    height: int
    fps: float
    duration: float
    total_frames: int
    has_proxy: bool = False
    frame_count: int = 0  # Number of sampled frames
    annotation_count: int = 0  # Number of annotations
    created_at: datetime


class VideoUploadResponse(BaseModel):
    """Response after video upload."""

    video: VideoInfo
    message: str = "Video uploaded successfully"


class FrameInfo(BaseModel):
    """Frame information response."""

    id: int
    video_id: int
    frame_number: int
    timestamp: float
    image_path: str
    is_approved: bool
    needs_review: bool
    annotation_count: int = 0


class SamplingConfig(BaseModel):
    """Configuration for frame sampling."""

    mode: str = Field("seconds", pattern=r"^(frames|seconds)$")
    interval: float = Field(0.5, gt=0)


class FrameExtractionRequest(BaseModel):
    """Request to extract frames from video."""

    video_id: int
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    start_time: Optional[float] = None  # Start timestamp
    end_time: Optional[float] = None  # End timestamp

