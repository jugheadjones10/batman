"""Core logic shared between backend, frontend, and CLI tools."""

from src.core.importer import DataImporter
from src.core.inference import (
    Detection,
    FrameResult,
    InferenceConfig,
    InferenceStats,
    RFDETRInference,
    create_tracker,
    draw_detections,
    save_results_json,
)
from src.core.project import Project, ProjectConfig
from src.core.trainer import (
    DatasetStats,
    RFDETRTrainer,
    TrainingConfig,
    TrainingResult,
    prepare_coco_dataset,
)

__all__ = [
    "DataImporter",
    "DatasetStats",
    "Detection",
    "FrameResult",
    "InferenceConfig",
    "InferenceStats",
    "Project",
    "ProjectConfig",
    "RFDETRInference",
    "RFDETRTrainer",
    "TrainingConfig",
    "TrainingResult",
    "create_tracker",
    "draw_detections",
    "prepare_coco_dataset",
    "save_results_json",
]
