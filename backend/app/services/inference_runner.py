"""Inference service for running trained models on videos."""

import time
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

import cv2
import numpy as np
from loguru import logger

from backend.app.config import settings
from backend.app.services.tracker import Tracker, TrackingConfig


class InferenceRunner:
    """Runs inference on videos using trained models."""

    def __init__(self):
        self.model = None
        self.model_path: Optional[Path] = None
        self.model_type: Optional[str] = None
        self.class_names: list[str] = []

    async def load_model(
        self,
        checkpoint_path: Path,
        class_names: list[str],
        model_type: str = "yolo",
    ):
        """Load a trained model."""
        self.model_path = checkpoint_path
        self.model_type = model_type
        self.class_names = class_names

        if model_type == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(str(checkpoint_path))
        elif model_type == "rfdetr":
            from rfdetr import RFDETRBase
            self.model = RFDETRBase()
            self.model.load(str(checkpoint_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Loaded {model_type} model from {checkpoint_path}")

    async def run_on_image(
        self,
        image_path: Path,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> dict:
        """Run inference on a single image."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.perf_counter()

        if self.model_type == "yolo":
            results = self.model(
                str(image_path),
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            detections = self._parse_yolo_results(results[0])
        else:
            results = self.model.predict(str(image_path))
            detections = self._parse_rfdetr_results(results)

        inference_time = (time.perf_counter() - start_time) * 1000

        return {
            "detections": detections,
            "inference_time_ms": inference_time,
        }

    async def run_on_video(
        self,
        video_path: Path,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        enable_tracking: bool = True,
        tracking_config: Optional[TrackingConfig] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run inference on a video, yielding results per frame.

        Yields:
            Frame results with detections and timing info
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total_frames

        tracker = Tracker(tracking_config) if enable_tracking else None

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_times = []

        try:
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.perf_counter()

                # Run detection
                if self.model_type == "yolo":
                    results = self.model(
                        frame,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False,
                    )
                    detections = self._parse_yolo_results(results[0])
                else:
                    results = self.model.predict(frame)
                    detections = self._parse_rfdetr_results(results)

                # Apply tracking
                if tracker:
                    detections = tracker.update(detections, frame_num)

                inference_time = (time.perf_counter() - start_time) * 1000
                frame_times.append(inference_time)

                yield {
                    "frame_number": frame_num,
                    "timestamp": frame_num / fps,
                    "detections": detections,
                    "inference_time_ms": inference_time,
                    "avg_fps": 1000 / (sum(frame_times[-30:]) / len(frame_times[-30:])) if frame_times else 0,
                }

        finally:
            cap.release()

    async def run_on_video_full(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        enable_tracking: bool = True,
        tracking_config: Optional[TrackingConfig] = None,
        detection_interval: int = 1,  # Run detection every N frames (1 = every frame)
    ) -> dict:
        """
        Run inference on entire video and optionally save annotated output.

        Returns:
            Summary statistics and results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        tracker = Tracker(tracking_config) if enable_tracking else None

        all_results = []
        frame_times = []

        if detection_interval > 1:
            logger.info(f"Starting inference on {total_frames} frames (detecting every {detection_interval} frames)...")
        else:
            logger.info(f"Starting inference on {total_frames} frames...")
        
        try:
            frame_num = 0
            last_log_time = time.time()
            last_detections = []  # Cache last detections for interpolation
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.perf_counter()
                
                # Only run detection on keyframes (every N frames)
                is_keyframe = (frame_num % detection_interval == 0)
                
                if is_keyframe:
                    # Run detection
                    if self.model_type == "yolo":
                        results = self.model(
                            frame,
                            conf=confidence_threshold,
                            iou=iou_threshold,
                            verbose=False,
                        )
                        detections = self._parse_yolo_results(results[0])
                    else:
                        results = self.model.predict(frame)
                        detections = self._parse_rfdetr_results(results)
                    
                    last_detections = detections
                else:
                    # Use cached detections (tracking will update positions)
                    detections = [det.copy() for det in last_detections]

                # Apply tracking (updates positions even on non-keyframes)
                if tracker:
                    detections = tracker.update(detections, frame_num)

                inference_time = (time.perf_counter() - start_time) * 1000
                frame_times.append(inference_time)

                all_results.append({
                    "frame_number": frame_num,
                    "timestamp": frame_num / fps,
                    "detections": detections,
                    "is_keyframe": is_keyframe,
                })

                # Draw annotations if saving
                if writer:
                    annotated = self._draw_annotations(frame, detections)
                    writer.write(annotated)

                frame_num += 1
                
                # Log progress every 2 seconds
                current_time = time.time()
                if current_time - last_log_time >= 2.0:
                    progress = (frame_num / total_frames) * 100
                    avg_ms = sum(frame_times[-30:]) / len(frame_times[-30:]) if frame_times else 0
                    est_remaining = ((total_frames - frame_num) * avg_ms) / 1000
                    logger.info(f"Inference progress: {frame_num}/{total_frames} ({progress:.1f}%) - {avg_ms:.1f}ms/frame - ETA: {est_remaining:.0f}s")
                    last_log_time = current_time

        finally:
            cap.release()
            if writer:
                writer.release()

        # Compute statistics
        avg_time = sum(frame_times) / len(frame_times) if frame_times else 0
        avg_fps = 1000 / avg_time if avg_time > 0 else 0
        
        logger.info(f"Inference complete: {len(all_results)} frames processed at {avg_fps:.1f} FPS")

        return {
            "total_frames": total_frames,
            "processed_frames": len(all_results),
            "avg_inference_time_ms": avg_time,
            "avg_fps": avg_fps,
            "output_path": str(output_path) if output_path else None,
            "results": all_results,
        }

    def _parse_yolo_results(self, result) -> list[dict]:
        """Parse YOLO results to common format."""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        img_h, img_w = result.orig_shape

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            detections.append({
                "box": {"x": cx, "y": cy, "width": w, "height": h},
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
            })

        return detections

    def _parse_rfdetr_results(self, results) -> list[dict]:
        """Parse RF-DETR results to common format."""
        # RF-DETR result parsing would go here
        # This is a placeholder implementation
        return []

    def _draw_annotations(
        self,
        frame: np.ndarray,
        detections: list[dict],
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        frame = frame.copy()
        height, width = frame.shape[:2]

        # Color palette for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]

        for det in detections:
            box = det["box"]
            cls_id = det.get("class_id", 0)
            class_name = det.get("class_name", f"class_{cls_id}")
            conf = det.get("confidence", 1.0)
            track_id = det.get("track_id")

            # Convert normalized coords to pixel coords
            x1 = int((box["x"] - box["width"] / 2) * width)
            y1 = int((box["y"] - box["height"] / 2) * height)
            x2 = int((box["x"] + box["width"] / 2) * width)
            y2 = int((box["y"] + box["height"] / 2) * height)

            # Calculate center coordinates
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            color = colors[cls_id % len(colors)]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with center coordinates
            label = f"{class_name} {conf:.2f} ({cx},{cy})"
            if track_id is not None:
                label = f"[{track_id}] " + label

            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - label_h - 4), (x1 + label_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame


# Global instance
inference_runner = InferenceRunner()

