"""Training and inference Pydantic models — RF-DETR only, GPU cluster execution."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Training ─────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    """RF-DETR training hyper-parameters (mirrors cli.train defaults)."""

    model: Literal["nano", "small", "base", "medium", "large"] = "base"
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int | None = None  # None = auto based on GPU type
    image_size: int = 640
    lr: float = 1e-4
    patience: int = 10
    grad_accum: int = 1


class GPUConfig(BaseModel):
    """SLURM GPU resource configuration."""

    gpu_type: Literal["h200", "h100-96", "h100-47", "a100-80", "a100-40", "nv"] = "a100-80"
    num_gpus: int = Field(1, ge=1, le=4)
    time_limit: str = "24:00:00"


class DataConfig(BaseModel):
    """Dataset preparation options (mirrors cli.train data flags)."""

    sources: Optional[list[Literal["manual_data", "imports"]]] = None
    manual_split_strategy: Literal[
        "proportional", "val_only", "train_only", "all_splits"
    ] = "train_only"
    manual_datasets: Optional[list[str]] = None
    exclude_manual_datasets: Optional[list[str]] = None
    filter_classes: Optional[list[str]] = None
    max_frames_per_class: Optional[int] = None
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15


class TrainingSubmitRequest(BaseModel):
    """Request to submit a training job to the GPU cluster."""

    label: Optional[str] = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    infer_after: bool = False
    infer_test_only: bool = False


class TrainingRunInfo(BaseModel):
    """Training run information read from meta.json."""

    id: int
    name: str
    status: str
    model: str
    gpu_type: Optional[str] = None
    slurm_job_id: Optional[str] = None
    progress: float = 0.0
    metrics: Optional[dict] = None
    checkpoint_path: Optional[str] = None
    latency_ms: Optional[float] = None
    tensorboard_url: Optional[str] = None
    config: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class GPUJobInfo(BaseModel):
    """Status of a tracked GPU cluster job."""

    job_id: str
    run_name: str
    job_type: Literal["training", "inference"]
    status: Literal["queued", "running", "completed", "failed", "cancelled", "timeout", "unknown"]
    gpu_type: str
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metrics: Optional[dict] = None
    checkpoint_path: Optional[str] = None


# ── Inference (GPU submission) ───────────────────────────────────────────

class InferenceGPUSubmitRequest(BaseModel):
    """Request to submit an inference job to the GPU cluster."""

    run_name: Optional[str] = None  # training run to use; None → latest
    video_ids: Optional[list[str]] = None  # None → all project videos
    test_only: bool = False
    model: Literal["nano", "small", "base", "medium", "large"] = "base"
    confidence: float = 0.5
    frame_interval: int = 1
    track: bool = False
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    no_video: bool = False
    gpu: GPUConfig = Field(default_factory=GPUConfig)


# ── Inference (local, kept for backward compat) ─────────────────────────

class InferenceConfig(BaseModel):
    """Configuration for running local inference."""

    model_run_id: int
    confidence_threshold: float = Field(0.0, ge=0, le=1)
    iou_threshold: float = Field(0.45, ge=0, le=1)
    max_detections: int = Field(100, ge=1, le=1000)
    enable_tracking: bool = True
    tracking_mode: Literal["visible_only", "occlusion_tolerant"] = "visible_only"
    detection_interval: int = Field(1, ge=1, le=30)


class InferenceResult(BaseModel):
    """Single frame inference result."""

    frame_number: int
    timestamp: float
    detections: list[dict]
    inference_time_ms: float


# ── Dataset export ───────────────────────────────────────────────────────

class DatasetExportConfig(BaseModel):
    """Configuration for dataset export."""

    format: Literal["yolo", "coco", "both"] = "both"
    include_unapproved: bool = False
    split_by_video: bool = True
    data_sources: Optional[list[Literal["manual_data", "imports"]]] = None
    manual_data_split_strategy: Literal[
        "proportional", "val_only", "train_only", "all_splits"
    ] = "train_only"
    manual_datasets: Optional[list[str]] = None
    exclude_manual_datasets: Optional[list[str]] = None


class DatasetExportResult(BaseModel):
    """Result of dataset export."""

    format: str
    output_path: str
    train_images: int
    val_images: int
    test_images: int
    total_annotations: int
    classes: list[str]
