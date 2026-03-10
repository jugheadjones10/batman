"""Application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App settings
    app_name: str = "Batman"
    app_version: str = "0.1.0"
    debug: bool = True

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000

    # Paths
    data_dir: Path = Path("./data")
    projects_dir: Path = Path("./data/projects")

    # Video processing
    default_sample_interval_seconds: float = 0.5
    proxy_video_scale: float = 0.5  # Scale factor for proxy videos
    proxy_video_crf: int = 28  # Quality for proxy videos

    # SAM3 settings
    sam_model_path: Path = Path("./sam3.pt")  # Local SAM3 model file
    sam_model_name: str = "sam3-hiera-large"  # Ultralytics model name if local not found
    sam_device: str = "auto"  # Device for SAM3 (auto, cuda, mps, cpu). Use cpu to avoid double-free on WSL2.
    sam_in_process: bool = False  # If True, run SAM3 in the API process. If False, use subprocess worker (avoids double-free crash killing server).

    # Tracking settings
    default_tracking_mode: Literal["visible_only", "occlusion_tolerant"] = "visible_only"
    default_max_age: int = 30  # Frames before track is lost
    default_iou_threshold: float = 0.3
    default_min_hits: int = 3
    use_appearance_embedding: bool = False

    # Training settings
    default_image_size: int = 640
    default_batch_size: int = 16
    default_epochs: int = 100

    # Device settings
    device: str = "auto"  # auto (CUDA/MPS/CPU), cuda, mps, or cpu

    # GPU cluster SSH settings
    ssh_host: str = "xlogin.comp.nus.edu.sg"
    ssh_user: str = "youngjin"
    remote_dir: str = "/home/y/youngjin/batman"

    @field_validator("sam_in_process", mode="before")
    @classmethod
    def parse_sam_in_process(cls, v):
        if isinstance(v, str):
            if v.strip().lower() in ("1", "true", "yes"):
                return True
            if v.strip().lower() in ("0", "false", "no", ""):
                return False
        return v

    class Config:
        env_file = ".env"
        env_prefix = "BATMAN_"
        extra = "ignore"  # Allow extra env vars (e.g. ROBOFLOW_API_KEY) without validation error


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.projects_dir.mkdir(parents=True, exist_ok=True)

