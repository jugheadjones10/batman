"""Core logic shared between backend, frontend, and CLI tools."""

from src.core.classes import ClassManager, MergeResult, RenameResult
from src.core.importer import COCO_CLASSES, DataImporter
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
    "ClassManager",
    "COCO_CLASSES",
    "DataImporter",
    "DatasetStats",
    "Detection",
    "FrameResult",
    "InferenceConfig",
    "InferenceStats",
    "MergeResult",
    "Project",
    "ProjectConfig",
    "RenameResult",
    "RFDETRInference",
    "RFDETRTrainer",
    "TrainingConfig",
    "TrainingResult",
    "create_tracker",
    "draw_detections",
    "prepare_coco_dataset",
    "save_results_json",
]
