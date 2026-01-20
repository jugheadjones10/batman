"""
Core inference logic for RF-DETR models.

Supports:
- Single image inference
- Batch image inference  
- Video inference (all frames or every Nth frame)
- Video inference with tracking between detection frames
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import numpy as np
from PIL import Image
from loguru import logger


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
    # Video settings
    frame_interval: int = 1  # Run inference every N frames
    use_tracking: bool = False  # Track objects between keyframes
    max_track_age: int = 30  # Max frames to keep a track without detection
    iou_threshold: float = 0.3  # IoU threshold for track matching
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


class SimpleTracker:
    """
    Simple IoU-based tracker for maintaining object identities between frames.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[int, dict] = {}  # track_id -> {bbox, class_id, age, ...}
        self.next_track_id = 1
    
    def _iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate IoU between two boxes (x1, y1, x2, y2)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, detections: list[Detection]) -> list[Detection]:
        """
        Update tracks with new detections.
        
        Returns detections with track_id assigned.
        """
        # Age all existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]
        
        if not detections:
            return []
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        results = []
        
        # Calculate IoU matrix
        for det_idx, det in enumerate(detections):
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                if track["class_id"] != det.class_id:
                    continue
                    
                iou = self._iou(det.bbox, track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
                self.tracks[best_track_id]["bbox"] = det.bbox
                self.tracks[best_track_id]["age"] = 0
                
                results.append(Detection(
                    bbox=det.bbox,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    track_id=best_track_id,
                ))
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
                
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id] = {
                "bbox": det.bbox,
                "class_id": det.class_id,
                "age": 0,
            }
            
            results.append(Detection(
                bbox=det.bbox,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                track_id=track_id,
            ))
        
        return results
    
    def get_interpolated_detections(self, class_names: list[str]) -> list[Detection]:
        """
        Get current track positions for frames without inference.
        Used for interpolation between keyframes.
        """
        results = []
        for track_id, track in self.tracks.items():
            if track["age"] <= self.max_age:
                results.append(Detection(
                    bbox=track["bbox"],
                    class_id=track["class_id"],
                    class_name=class_names[track["class_id"]] if track["class_id"] < len(class_names) else f"class_{track['class_id']}",
                    confidence=0.5,  # Reduced confidence for interpolated
                    track_id=track_id,
                ))
        return results


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
    
    def load_model(self, device: str = "auto") -> None:
        """Load the model onto the specified device."""
        from rfdetr import RFDETRBase, RFDETRLarge
        
        # Determine device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self._device = device
        logger.info(f"Loading model on {device}")
        
        # Load model
        ModelClass = RFDETRLarge if self.model_size == "large" else RFDETRBase
        self.model = ModelClass(pretrain_weights=str(self.checkpoint))
        
        logger.info(f"Model loaded: RF-DETR {self.model_size}")
    
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
        if self.model is None:
            self.load_model(config.device if config else "auto")
        
        if config is None:
            config = InferenceConfig()
        
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
                results.append(Detection(
                    bbox=tuple(detections.xyxy[i].tolist()),
                    class_id=class_id,
                    class_name=self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                    confidence=float(detections.confidence[i]),
                ))
        
        return results
    
    def predict_video(
        self,
        video_path: Path | str,
        config: InferenceConfig | None = None,
        progress_callback: callable = None,
    ) -> Generator[FrameResult, None, InferenceStats]:
        """
        Run inference on a video, yielding results frame by frame.
        
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
        
        if self.model is None:
            self.load_model(config.device if config else "auto")
        
        if config is None:
            config = InferenceConfig()
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"  Frames: {total_frames}, FPS: {fps:.2f}")
        logger.info(f"  Frame interval: {config.frame_interval}, Tracking: {config.use_tracking}")
        
        # Initialize tracker if needed
        tracker = None
        if config.use_tracking:
            tracker = SimpleTracker(
                iou_threshold=config.iou_threshold,
                max_age=config.max_track_age,
            )
        
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
            is_keyframe = (frame_idx % config.frame_interval == 0)
            
            if is_keyframe:
                # Run inference
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                t0 = time.time()
                detections = self.predict_image(pil_image, config)
                inference_time = (time.time() - t0) * 1000
                
                stats["inference_times"].append(inference_time)
                stats["keyframes"] += 1
                
                # Update tracker if enabled
                if tracker:
                    detections = tracker.update(detections)
            else:
                # Use tracked/interpolated detections
                if tracker:
                    detections = tracker.get_interpolated_detections(self.class_names)
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
        (255, 107, 107), (78, 205, 196), (69, 183, 209), (150, 206, 180),
        (255, 234, 167), (221, 160, 221), (152, 216, 200), (255, 159, 243),
    ]
    
    result = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        color = colors[det.class_id % len(colors)]
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        if det.track_id is not None:
            label = f"{det.class_name} #{det.track_id} {det.confidence:.2f}"
        else:
            label = f"{det.class_name} {det.confidence:.2f}"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            result,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
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
        } if stats else {},
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
