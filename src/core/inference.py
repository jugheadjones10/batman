"""
Core inference logic for RF-DETR models.

Supports:
- Single image inference
- Batch image inference
- Video inference (all frames or every Nth frame)
- Video inference with tracking between detection frames (using ByteTrack)
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image


@dataclass
class Detection:
    """Single detection result."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None


@dataclass
class FrameResult:
    """Inference result for a single frame."""

    frame_idx: int
    timestamp: float  # seconds
    detections: list[Detection]
    inference_time_ms: float
    is_keyframe: bool = True  # True if inference was run, False if interpolated/tracked


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    device: str = "auto"
    # Optimization settings
    optimize: bool = True  # Call optimize_for_inference() on model load
    optimize_compile: bool = False  # Use JIT compilation (may fail on some systems)
    # Video settings
    frame_interval: int = 1  # Run inference every N frames
    use_tracking: bool = False  # Track objects between keyframes
    use_kalman_prediction: bool = True  # Use Kalman filter to predict positions on non-keyframes
    # ByteTrack settings
    track_thresh: float = 0.25  # Detection threshold for tracking
    track_buffer: int = 30  # Frames to keep lost tracks
    match_thresh: float = 0.8  # IoU threshold for matching
    # Output settings
    save_visualizations: bool = True
    save_json: bool = True
    visualization_thickness: int = 2


@dataclass
class InferenceStats:
    """Statistics from inference run."""

    total_frames: int
    keyframes: int
    total_detections: int
    avg_inference_time_ms: float
    total_time_seconds: float
    fps: float


def create_tracker(config: InferenceConfig):
    """
    Create a ByteTrack tracker from supervision library.

    Args:
        config: Inference configuration with tracking parameters

    Returns:
        sv.ByteTrack tracker instance
    """
    import supervision as sv

    return sv.ByteTrack(
        track_activation_threshold=config.track_thresh,
        lost_track_buffer=config.track_buffer,
        minimum_matching_threshold=config.match_thresh,
        frame_rate=30,  # Will be updated when processing video
    )


class RFDETRInference:
    """
    RF-DETR inference engine.

    Handles loading models and running inference on images/videos.
    """

    def __init__(
        self,
        checkpoint: Path,
        class_names: list[str] | None = None,
        model_size: str = "base",
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint: Path to model checkpoint
            class_names: List of class names (loaded from class_info.json if not provided)
            model_size: Model size ('base', 'large', etc.)
        """
        self.checkpoint = Path(checkpoint)
        self.model_size = model_size

        # Load class names
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = self._load_class_names()

        self.model = None
        self._device = None

    def _load_class_names(self) -> list[str]:
        """Load class names from class_info.json next to checkpoint."""
        info_path = self.checkpoint.parent / "class_info.json"
        if info_path.exists():
            with open(info_path) as f:
                return json.load(f).get("classes", [])

        # Fallback
        logger.warning(f"class_info.json not found at {info_path}, using generic class names")
        return [f"class_{i}" for i in range(100)]

    def _predict_tracks_kalman(
        self,
        tracker,
        track_metadata: dict[int, dict],
    ) -> list[Detection]:
        """
        Use Kalman filter to predict track positions for non-keyframes.

        This advances each track's Kalman filter state and returns predicted
        bounding boxes, providing smooth motion interpolation between keyframes.

        Args:
            tracker: ByteTrack tracker instance
            track_metadata: Dict mapping track_id to {class_id, class_name, confidence}

        Returns:
            List of Detection objects with predicted positions
        """
        detections = []

        # Get all active tracks (tracked + lost but not yet removed)
        all_tracks = list(tracker.tracked_tracks) + list(tracker.lost_tracks)

        for track in all_tracks:
            # Advance Kalman filter state (predict next position)
            track.predict()

            # Get predicted bounding box (tlbr = top-left, bottom-right = x1,y1,x2,y2)
            bbox = track.tlbr

            # Get track metadata (class info, confidence from last detection)
            track_id = track.external_track_id
            meta = track_metadata.get(track_id, {})

            detections.append(
                Detection(
                    bbox=tuple(bbox.tolist()),
                    class_id=meta.get("class_id", 0),
                    class_name=meta.get("class_name", "unknown"),
                    confidence=meta.get("confidence", 0.0),
                    track_id=track_id,
                )
            )

        return detections

    def load_model(
        self,
        device: str = "auto",
        optimize: bool = True,
        optimize_compile: bool = False,
    ) -> None:
        """
        Load the model onto the specified device.

        Args:
            device: Device to load model on ('auto', 'cuda', 'mps', 'cpu')
            optimize: Whether to call optimize_for_inference()
            optimize_compile: Whether to use JIT compilation (may fail on some systems)
        """
        import torch
        from rfdetr import RFDETRBase, RFDETRLarge

        load_start = time.time()

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        logger.info(f"Loading model on {device}")

        # Load model
        weights_start = time.time()
        ModelClass = RFDETRLarge if self.model_size == "large" else RFDETRBase
        self.model = ModelClass(pretrain_weights=str(self.checkpoint))
        weights_time = time.time() - weights_start
        logger.info(f"Loaded pretrained weights in {weights_time:.2f}s")

        # Optimize for inference if requested
        if optimize:
            logger.info("Optimizing model for inference...")
            opt_start = time.time()
            try:
                self.model.optimize_for_inference(compile=optimize_compile)
                opt_time = time.time() - opt_start
                logger.info(f"Model optimization complete in {opt_time:.2f}s")
            except Exception as e:
                logger.warning(f"Model optimization failed (non-fatal): {e}")
                logger.warning("Continuing with non-optimized model")

        total_load_time = time.time() - load_start
        logger.info(f"Model loaded: RF-DETR {self.model_size} (total: {total_load_time:.2f}s)")

    def predict_image(
        self,
        image: Image.Image | np.ndarray | Path | str,
        config: InferenceConfig | None = None,
    ) -> list[Detection]:
        """
        Run inference on a single image.

        Args:
            image: PIL Image, numpy array, or path to image
            config: Inference configuration

        Returns:
            List of Detection objects
        """
        if config is None:
            config = InferenceConfig()

        if self.model is None:
            self.load_model(
                device=config.device,
                optimize=config.optimize,
                optimize_compile=config.optimize_compile,
            )

        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Run inference
        detections = self.model.predict(image, threshold=config.confidence_threshold)

        # Convert to Detection objects
        results = []
        if hasattr(detections, "xyxy") and len(detections.xyxy) > 0:
            for i in range(len(detections.xyxy)):
                class_id = int(detections.class_id[i])
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"class_{class_id}"
                )
                results.append(
                    Detection(
                        bbox=tuple(detections.xyxy[i].tolist()),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(detections.confidence[i]),
                    )
                )

        return results

    def predict_video(
        self,
        video_path: Path | str,
        config: InferenceConfig | None = None,
        progress_callback: callable = None,
    ) -> Generator[FrameResult, None, InferenceStats]:
        """
        Run inference on a video, yielding results frame by frame.

        Uses ByteTrack from supervision library for tracking when enabled.

        Args:
            video_path: Path to video file
            config: Inference configuration
            progress_callback: Optional callback(current_frame, total_frames)

        Yields:
            FrameResult for each frame

        Returns:
            InferenceStats when complete
        """
        import cv2
        import supervision as sv

        if config is None:
            config = InferenceConfig()

        if self.model is None:
            self.load_model(
                device=config.device,
                optimize=config.optimize,
                optimize_compile=config.optimize_compile,
            )

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"  Frames: {total_frames}, FPS: {fps:.2f}")
        logger.info(f"  Frame interval: {config.frame_interval}, Tracking: {config.use_tracking}")
        if config.use_tracking and config.frame_interval > 1:
            logger.info(f"  Kalman prediction: {config.use_kalman_prediction}")

        # Initialize ByteTrack tracker if needed
        tracker = None
        # Store track metadata for Kalman prediction on non-keyframes
        track_metadata: dict[int, dict] = {}  # track_id -> {class_id, class_name, confidence}

        if config.use_tracking:
            tracker = sv.ByteTrack(
                track_activation_threshold=config.track_thresh,
                lost_track_buffer=config.track_buffer,
                minimum_matching_threshold=config.match_thresh,
                frame_rate=int(fps) if fps > 0 else 30,
            )
            logger.info(f"  ByteTrack: thresh={config.track_thresh}, buffer={config.track_buffer}")

        stats = {
            "total_frames": 0,
            "keyframes": 0,
            "total_detections": 0,
            "inference_times": [],
        }

        start_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps if fps > 0 else 0
            is_keyframe = frame_idx % config.frame_interval == 0

            if is_keyframe:
                # Run inference
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                t0 = time.time()
                raw_detections = self.predict_image(pil_image, config)
                inference_time = (time.time() - t0) * 1000

                stats["inference_times"].append(inference_time)
                stats["keyframes"] += 1

                # Apply ByteTrack if enabled
                if tracker and raw_detections:
                    # Convert to supervision Detections format
                    sv_detections = sv.Detections(
                        xyxy=np.array([d.bbox for d in raw_detections]),
                        confidence=np.array([d.confidence for d in raw_detections]),
                        class_id=np.array([d.class_id for d in raw_detections]),
                    )

                    # Update tracker
                    tracked = tracker.update_with_detections(sv_detections)

                    # Convert back to Detection objects with track IDs
                    detections = []
                    for i in range(len(tracked.xyxy)):
                        class_id = int(tracked.class_id[i])
                        track_id = (
                            int(tracked.tracker_id[i]) if tracked.tracker_id is not None else None
                        )
                        class_name = (
                            self.class_names[class_id]
                            if class_id < len(self.class_names)
                            else f"class_{class_id}"
                        )
                        detections.append(
                            Detection(
                                bbox=tuple(tracked.xyxy[i].tolist()),
                                class_id=class_id,
                                class_name=class_name,
                                confidence=float(tracked.confidence[i]),
                                track_id=track_id,
                            )
                        )

                        # Store metadata for Kalman prediction on non-keyframes
                        if track_id is not None:
                            track_metadata[track_id] = {
                                "class_id": class_id,
                                "class_name": class_name,
                                "confidence": float(tracked.confidence[i]),
                            }
                elif tracker:
                    # No detections but tracker exists - still need to update tracker state
                    detections = []
                else:
                    detections = raw_detections
            else:
                # Non-keyframe: use Kalman prediction if enabled
                if tracker and config.use_kalman_prediction:
                    detections = self._predict_tracks_kalman(tracker, track_metadata)
                else:
                    detections = []
                inference_time = 0

            stats["total_frames"] += 1
            stats["total_detections"] += len(detections)

            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)

            yield FrameResult(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detections=detections,
                inference_time_ms=inference_time,
                is_keyframe=is_keyframe,
            )

            frame_idx += 1

        cap.release()

        total_time = time.time() - start_time
        avg_inference = np.mean(stats["inference_times"]) if stats["inference_times"] else 0
        total_inference = sum(stats["inference_times"]) if stats["inference_times"] else 0

        logger.info(f"Video processing complete:")
        logger.info(f"  Total frames: {stats['total_frames']}")
        logger.info(f"  Keyframes (inference): {stats['keyframes']}")
        logger.info(f"  Avg inference time per keyframe: {avg_inference:.1f}ms")
        logger.info(f"  Total inference time: {total_inference/1000:.2f}s")
        logger.info(f"  Total processing time: {total_time:.2f}s")
        logger.info(f"  Processing FPS: {stats['total_frames'] / total_time:.1f}")

        return InferenceStats(
            total_frames=stats["total_frames"],
            keyframes=stats["keyframes"],
            total_detections=stats["total_detections"],
            avg_inference_time_ms=avg_inference,
            total_time_seconds=total_time,
            fps=stats["total_frames"] / total_time if total_time > 0 else 0,
        )


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw detection boxes and labels on an image.

    Args:
        image: BGR image (numpy array)
        detections: List of Detection objects
        thickness: Line thickness
        font_scale: Font scale for labels

    Returns:
        Annotated image
    """
    import cv2

    # Color palette
    colors = [
        (255, 107, 107),
        (78, 205, 196),
        (69, 183, 209),
        (150, 206, 180),
        (255, 234, 167),
        (221, 160, 221),
        (152, 216, 200),
        (255, 159, 243),
    ]

    result = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        
        # Calculate center coordinates
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        color = colors[det.class_id % len(colors)]

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Prepare label with center coordinates
        if det.track_id is not None:
            label = f"{det.class_name} #{det.track_id} {det.confidence:.2f} ({cx},{cy})"
        else:
            label = f"{det.class_name} {det.confidence:.2f} ({cx},{cy})"

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(result, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(
            result,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    return result


def save_results_json(
    results: list[FrameResult],
    output_path: Path,
    stats: InferenceStats | None = None,
    metadata: dict | None = None,
) -> None:
    """Save inference results to JSON file."""
    data = {
        "metadata": metadata or {},
        "stats": {
            "total_frames": stats.total_frames,
            "keyframes": stats.keyframes,
            "total_detections": stats.total_detections,
            "avg_inference_time_ms": stats.avg_inference_time_ms,
            "total_time_seconds": stats.total_time_seconds,
            "fps": stats.fps,
        }
        if stats
        else {},
        "frames": [],
    }

    for frame in results:
        frame_data = {
            "frame_idx": frame.frame_idx,
            "timestamp": frame.timestamp,
            "is_keyframe": frame.is_keyframe,
            "inference_time_ms": frame.inference_time_ms,
            "detections": [
                {
                    "bbox": list(det.bbox),
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "track_id": det.track_id,
                }
                for det in frame.detections
            ],
        }
        data["frames"].append(frame_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
