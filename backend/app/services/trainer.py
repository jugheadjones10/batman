"""Model training service for YOLO and RF-DETR."""

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from backend.app.config import settings

# Enable MPS fallback for unsupported PyTorch operations on Apple Silicon
# This allows RF-DETR to train on Mac by falling back to CPU for specific ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class TrainingCallback:
    """Callback for tracking training progress."""

    def __init__(self, on_progress: Callable | None = None):
        self.on_progress = on_progress
        self.current_epoch = 0
        self.total_epochs = 0
        self.metrics = {}

    def __call__(self, epoch: int, total: int, metrics: dict):
        self.current_epoch = epoch
        self.total_epochs = total
        self.metrics = metrics
        if self.on_progress:
            self.on_progress(epoch / total, epoch, total, metrics)


class ModelTrainer:
    """Handles model training for different architectures."""

    # Learning rate presets
    LR_PRESETS = {
        "small": 0.001,
        "medium": 0.01,
        "large": 0.1,
    }

    # Augmentation presets
    AUG_PRESETS = {
        "none": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        },
        "light": {
            "hsv_h": 0.01,
            "hsv_s": 0.3,
            "hsv_v": 0.3,
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.1,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
        },
        "standard": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.3,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.1,
        },
        "heavy": {
            "hsv_h": 0.02,
            "hsv_s": 0.9,
            "hsv_v": 0.5,
            "degrees": 20.0,
            "translate": 0.2,
            "scale": 0.5,
            "flipud": 0.2,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.3,
            # Video-specific augmentations
            "blur": 0.01,
            "jpeg_quality": (70, 95),
        },
    }

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.runs_dir = project_path / "runs"
        self.runs_dir.mkdir(exist_ok=True)

    async def train(
        self,
        run_name: str,
        dataset_path: Path,
        base_model: str,
        config: dict,
        callback: TrainingCallback | None = None,
    ) -> dict:
        """
        Train a model on the dataset.

        Args:
            run_name: Name for this training run
            dataset_path: Path to YOLO-format dataset
            base_model: Model to fine-tune ('yolo11n', 'yolo11s', 'rfdetr-b', etc.)
            config: Training configuration
            callback: Optional callback for progress updates

        Returns:
            Training results dictionary
        """
        run_dir = self.runs_dir / run_name
        run_dir.mkdir(exist_ok=True)

        # Save config
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        if base_model.startswith("yolo"):
            return await self._train_yolo(run_dir, dataset_path, base_model, config, callback)
        elif base_model.startswith("rfdetr"):
            return await self._train_rfdetr(run_dir, dataset_path, base_model, config, callback)
        else:
            raise ValueError(f"Unknown base model: {base_model}")

    async def _train_yolo(
        self,
        run_dir: Path,
        dataset_path: Path,
        base_model: str,
        config: dict,
        callback: TrainingCallback | None = None,
    ) -> dict:
        """Train YOLO model."""
        from ultralytics import YOLO

        # Map model names
        model_map = {
            "yolo11n": "yolo11n.pt",
            "yolo11s": "yolo11s.pt",
            "yolo11m": "yolo11m.pt",
            "yolo11l": "yolo11l.pt",
            "yolo11x": "yolo11x.pt",
        }

        model_weights = model_map.get(base_model, "yolo11s.pt")
        model = YOLO(model_weights)

        # Get augmentation settings
        aug_preset = self.AUG_PRESETS.get(config.get("augmentation_preset", "standard"), {})

        # Training arguments
        train_args = {
            "data": str(dataset_path / "data.yaml"),
            "epochs": config.get("epochs", 100),
            "imgsz": config.get("image_size", 640),
            "batch": config.get("batch_size", 16),
            "lr0": self.LR_PRESETS.get(config.get("lr_preset", "medium"), 0.01),
            "project": str(run_dir),
            "name": "train",
            "exist_ok": True,
            "device": settings.device,
            "patience": config.get("early_stopping_patience", 20),
            "amp": config.get("mixed_precision", True),
            "freeze": config.get("freeze_backbone", False),
            **aug_preset,
        }

        logger.info(f"Starting YOLO training: {train_args}")

        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        start_time = time.time()

        def run_training():
            return model.train(**train_args)

        results = await loop.run_in_executor(None, run_training)

        training_time = time.time() - start_time

        # Get best weights path
        best_weights = run_dir / "train" / "weights" / "best.pt"

        # Measure latency
        latency_ms = await self._measure_latency(best_weights, config.get("image_size", 640))

        return {
            "status": "completed",
            "checkpoint_path": str(best_weights),
            "training_time_seconds": training_time,
            "latency_ms": latency_ms,
            "metrics": {
                "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
                "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
                "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
            },
        }

    async def _train_rfdetr(
        self,
        run_dir: Path,
        dataset_path: Path,
        base_model: str,
        config: dict,
        callback: TrainingCallback | None = None,
    ) -> dict:
        """Train RF-DETR model (COCO format dataset required)."""
        # Import based on model variant
        if base_model == "rfdetr-l":
            from rfdetr import RFDETRLarge as RFDETRModel
        elif base_model == "rfdetr-m":
            from rfdetr import RFDETRMedium as RFDETRModel
        elif base_model == "rfdetr-s":
            from rfdetr import RFDETRSmall as RFDETRModel
        elif base_model == "rfdetr-n":
            from rfdetr import RFDETRNano as RFDETRModel
        else:
            from rfdetr import RFDETRBase as RFDETRModel

        model = RFDETRModel()

        # Get batch size and calculate grad accumulation for effective batch of 16
        batch_size = config.get("batch_size", 4)
        grad_accum = max(1, 16 // batch_size)

        # Resolution must be divisible by 56 for RF-DETR
        resolution = config.get("image_size", 560)
        resolution = (resolution // 56) * 56  # Round down to nearest multiple of 56
        if resolution < 280:
            resolution = 280

        # Training arguments per RF-DETR docs
        train_args = {
            "dataset_dir": str(dataset_path),
            "epochs": config.get("epochs", 10),
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum,
            "lr": self.LR_PRESETS.get(config.get("lr_preset", "medium"), 1e-4),
            "resolution": resolution,
            "output_dir": str(run_dir),
            "early_stopping": True,
            "early_stopping_patience": config.get("early_stopping_patience", 5),
            "tensorboard": True,  # Enable TensorBoard logging
        }

        # Only add device if not mps (RF-DETR may not support MPS yet)
        if settings.device != "mps":
            train_args["device"] = settings.device
        else:
            logger.info("Using MPS with CPU fallback for unsupported operations")

        logger.info(f"Starting RF-DETR training: {train_args}")

        loop = asyncio.get_event_loop()
        start_time = time.time()

        def run_training():
            model.train(**train_args)
            return model

        await loop.run_in_executor(None, run_training)

        training_time = time.time() - start_time

        # Find best checkpoint (RF-DETR saves as checkpoint_best_total.pth)
        checkpoint_path = run_dir / "checkpoint_best_total.pth"
        if not checkpoint_path.exists():
            # Try other possible names
            for name in [
                "checkpoint_best_ema.pth",
                "checkpoint_best_regular.pth",
                "checkpoint.pth",
            ]:
                alt_path = run_dir / name
                if alt_path.exists():
                    checkpoint_path = alt_path
                    break

        # Measure latency
        latency_ms = await self._measure_latency(checkpoint_path, resolution, model_type="rfdetr")

        return {
            "status": "completed",
            "checkpoint_path": str(checkpoint_path),
            "training_time_seconds": training_time,
            "latency_ms": latency_ms,
            "metrics": {},  # RF-DETR metrics would be extracted from training logs
        }

    async def _measure_latency(
        self,
        checkpoint_path: Path,
        image_size: int,
        model_type: str = "yolo",
        warmup_runs: int = 5,
        test_runs: int = 20,
    ) -> float:
        """Measure inference latency on a dummy image."""
        import numpy as np

        try:
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                return 0.0

            if model_type == "yolo":
                from ultralytics import YOLO

                model = YOLO(str(checkpoint_path))

                # Create dummy image
                dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

                # Warmup
                for _ in range(warmup_runs):
                    model(dummy_img, verbose=False)

                # Measure
                times = []
                for _ in range(test_runs):
                    start = time.perf_counter()
                    model(dummy_img, verbose=False)
                    times.append((time.perf_counter() - start) * 1000)

                return float(np.mean(times))

            else:
                # RF-DETR latency measurement
                from rfdetr import RFDETRBase

                model = RFDETRBase(pretrain_weights=str(checkpoint_path))

                # Create dummy image
                dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

                # Warmup
                for _ in range(warmup_runs):
                    model.predict(dummy_img)

                # Measure
                times = []
                for _ in range(test_runs):
                    start = time.perf_counter()
                    model.predict(dummy_img)
                    times.append((time.perf_counter() - start) * 1000)

                return float(np.mean(times))

        except Exception as e:
            logger.warning(f"Failed to measure latency: {e}")
            return 0.0
