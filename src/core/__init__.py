"""Core logic shared between backend, frontend, and CLI tools."""

from src.core.importer import DataImporter
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
    "Project",
    "ProjectConfig",
    "RFDETRTrainer",
    "TrainingConfig",
    "TrainingResult",
    "prepare_coco_dataset",
]
