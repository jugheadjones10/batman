"""SQLAlchemy ORM models for project database."""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Video(Base):
    """Video file in a project."""

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    proxy_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    total_frames: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    frames: Mapped[list["Frame"]] = relationship("Frame", back_populates="video", cascade="all, delete-orphan")


class Frame(Base):
    """Sampled frame from a video."""

    __tablename__ = "frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Review status
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="frames")
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="frame", cascade="all, delete-orphan"
    )


class ClassLabel(Base):
    """Object class definition."""

    __tablename__ = "class_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    color: Mapped[str] = mapped_column(String(7), default="#00FF00")  # Hex color
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Track(Base):
    """Object track across frames."""

    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)  # External track ID
    class_label_id: Mapped[int] = mapped_column(Integer, ForeignKey("class_labels.id"), nullable=False)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)

    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)

    # Review status
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    class_label: Mapped["ClassLabel"] = relationship("ClassLabel")
    annotations: Mapped[list["Annotation"]] = relationship("Annotation", back_populates="track")


class Annotation(Base):
    """Bounding box annotation on a frame."""

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    frame_id: Mapped[int] = mapped_column(Integer, ForeignKey("frames.id"), nullable=False)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("tracks.id"), nullable=True)
    class_label_id: Mapped[int] = mapped_column(Integer, ForeignKey("class_labels.id"), nullable=False)

    # Bounding box (normalized 0-1)
    x: Mapped[float] = mapped_column(Float, nullable=False)  # Center x
    y: Mapped[float] = mapped_column(Float, nullable=False)  # Center y
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)

    # Confidence from auto-labeling
    confidence: Mapped[float] = mapped_column(Float, default=1.0)

    # Source: 'auto' (SAM), 'manual', 'corrected'
    source: Mapped[str] = mapped_column(String(50), default="auto")

    # Is this an exemplar for prompting?
    is_exemplar: Mapped[bool] = mapped_column(Boolean, default=False)
    exemplar_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # 'anchor' or 'correction'

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    frame: Mapped["Frame"] = relationship("Frame", back_populates="annotations")
    track: Mapped[Optional["Track"]] = relationship("Track", back_populates="annotations")
    class_label: Mapped["ClassLabel"] = relationship("ClassLabel")


class LabelIteration(Base):
    """Immutable snapshot of labeling state."""

    __tablename__ = "label_iterations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration used for this iteration
    config_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Statistics
    total_annotations: Mapped[int] = mapped_column(Integer, default=0)
    total_tracks: Mapped[int] = mapped_column(Integer, default=0)
    approved_frames: Mapped[int] = mapped_column(Integer, default=0)

    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Path to exported labels
    export_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)


class TrainingRun(Base):
    """Training run configuration and results."""

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    label_iteration_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("label_iterations.id"), nullable=False
    )

    # Model config
    base_model: Mapped[str] = mapped_column(String(100), nullable=False)  # 'yolo12', 'rfdetr'
    config_json: Mapped[str] = mapped_column(Text, nullable=False)

    # Status: 'pending', 'running', 'completed', 'failed'
    status: Mapped[str] = mapped_column(String(50), default="pending")
    progress: Mapped[float] = mapped_column(Float, default=0.0)

    # Results
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    label_iteration: Mapped["LabelIteration"] = relationship("LabelIteration")


class Exemplar(Base):
    """Exemplar for guiding SAM prompting."""

    __tablename__ = "exemplars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_label_id: Mapped[int] = mapped_column(Integer, ForeignKey("class_labels.id"), nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, ForeignKey("frames.id"), nullable=False)

    # Bounding box (normalized 0-1)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)

    # Type: 'anchor' (strong example) or 'correction'
    exemplar_type: Mapped[str] = mapped_column(String(50), default="anchor")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    class_label: Mapped["ClassLabel"] = relationship("ClassLabel")
    frame: Mapped["Frame"] = relationship("Frame")

