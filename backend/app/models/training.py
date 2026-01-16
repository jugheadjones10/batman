"""Training-related Pydantic models."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration."""

    # Base model selection
    base_model: Literal["yolo11n", "yolo11s", "yolo11m", "rfdetr-b", "rfdetr-l"] = "yolo11s"

    # Training parameters
    image_size: int = Field(640, ge=320, le=1280)
    batch_size: int = Field(16, ge=1, le=128)
    epochs: int = Field(100, ge=1, le=1000)
    lr_preset: Literal["small", "medium", "large"] = "medium"

    # Augmentation
    augmentation_preset: Literal["none", "light", "standard", "heavy"] = "standard"

    # Data split
    val_split: float = Field(0.2, ge=0.1, le=0.5)
    test_split: float = Field(0.1, ge=0, le=0.3)

    # Advanced options
    freeze_backbone: bool = False
    mixed_precision: bool = True
    early_stopping_patience: int = 20


class TrainingRequest(BaseModel):
    """Request to start training."""

    name: str = Field(..., min_length=1, max_length=255)
    label_iteration_id: int
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingProgress(BaseModel):
    """Training progress update."""

    run_id: int
    status: str
    progress: float  # 0-1
    current_epoch: int
    total_epochs: int
    metrics: Optional[dict] = None
    eta_seconds: Optional[float] = None


class TrainingRunInfo(BaseModel):
    """Training run information."""

    id: int
    name: str
    label_iteration_id: int
    base_model: str
    status: str
    progress: float
    metrics: Optional[dict] = None
    checkpoint_path: Optional[str] = None
    latency_ms: Optional[float] = None
    tensorboard_url: Optional[str] = None  # TensorBoard URL when running
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class InferenceConfig(BaseModel):
    """Configuration for running inference."""

    model_run_id: int
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    iou_threshold: float = Field(0.45, ge=0, le=1)
    max_detections: int = Field(100, ge=1, le=1000)

    # Tracking settings (for video)
    enable_tracking: bool = True
    tracking_mode: Literal["visible_only", "occlusion_tolerant"] = "visible_only"
    
    # Detection interval - run detection every N frames (1 = every frame, 5 = every 5th frame)
    # Higher values = faster but less accurate on fast-moving objects
    detection_interval: int = Field(1, ge=1, le=30)


class InferenceResult(BaseModel):
    """Single frame inference result."""

    frame_number: int
    timestamp: float
    detections: list[dict]  # List of {box, class_id, class_name, confidence, track_id}
    inference_time_ms: float


class DatasetExportConfig(BaseModel):
    """Configuration for dataset export."""

    format: Literal["yolo", "coco", "both"] = "both"
    include_unapproved: bool = False
    split_by_video: bool = True  # Train/val/test by video, not by frame


class DatasetExportResult(BaseModel):
    """Result of dataset export."""

    format: str
    output_path: str
    train_images: int
    val_images: int
    test_images: int
    total_annotations: int
    classes: list[str]

