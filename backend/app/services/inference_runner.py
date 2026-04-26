"""Inference service for running trained models on videos."""

import asyncio
import time
from pathlib import Path
from typing import AsyncGenerator, Callable, Literal, Optional

import cv2
import numpy as np
from loguru import logger

# Type alias for the streaming progress callback used by the API layer to
# report granular progress to the client. Callback is invoked synchronously
# from the inference worker thread with a plain dict event, e.g.
#   {"type": "stage", "stage": "encoding_video"}
#   {"type": "progress", "current": 120, "total": 450,
#    "avg_ms": 42.1, "avg_fps": 23.7, "eta_s": 14}
# The callback must be fast and non-blocking; event-loop bridging (e.g.
# asyncio.Queue.put_nowait via call_soon_threadsafe) should happen inside it.
ProgressCallback = Callable[[dict], None]

from backend.app.config import settings
from backend.app.services.tracker import Tracker, TrackingConfig
from src.core.inference import Detection, draw_detections
from src.core.trainer import get_device, get_device_info, resolve_rfdetr_class


class InferenceRunner:
    """Runs inference on videos using trained models."""

    def __init__(self):
        self.model = None
        self.model_path: Optional[Path] = None
        self.model_type: Optional[str] = None
        self.model_task: str = "detection"
        self.class_names: list[str] = []
        self.current_run_name: Optional[str] = None
        self._device: Optional[str] = None

    @staticmethod
    def _load_rfdetr_model(
        checkpoint_path: Path,
        model_size: str = "base",
        task: str = "detection",
    ):
        """Instantiate the correct RF-DETR variant for (task, size)."""
        Model = resolve_rfdetr_class(task, model_size)
        return Model(pretrain_weights=str(checkpoint_path))

    async def load_model(
        self,
        checkpoint_path: Path,
        class_names: list[str],
        model_type: str = "yolo",
        device: str = "auto",
        model_size: str = "base",
        task: str = "detection",
    ):
        """Load a trained model onto the given device (auto, cuda, mps, cpu)."""
        self.model_path = checkpoint_path
        self.model_type = model_type
        self.model_task = task
        self.class_names = class_names
        resolved = get_device(device)
        self._device = resolved
        info = get_device_info(resolved)
        logger.info(f"Loading model on {info.get('name', resolved)} (task={task})")

        if model_type == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(str(checkpoint_path))
        elif model_type == "rfdetr":
            self.model = self._load_rfdetr_model(checkpoint_path, model_size, task=task)
            if hasattr(self.model, "to") and resolved != "cpu":
                try:
                    import torch

                    dev = torch.device(resolved)
                    self.model.to(dev)
                except Exception as e:
                    logger.warning(f"Could not move RF-DETR to {resolved}: {e}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Loaded {model_type} model from {checkpoint_path}")

    async def run_on_image(
        self,
        image_path: Path,
        confidence_threshold: float = 0.0,
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
            # Load image to get shape for RF-DETR parser
            img = cv2.imread(str(image_path))
            shape = img.shape[:2] if img is not None else None
            results = self.model.predict(str(image_path), threshold=confidence_threshold)
            detections = self._parse_rfdetr_results(results, img_shape=shape)

        inference_time = (time.perf_counter() - start_time) * 1000

        return {
            "detections": detections,
            "inference_time_ms": inference_time,
        }

    async def run_on_video(
        self,
        video_path: Path,
        confidence_threshold: float = 0.0,
        iou_threshold: float = 0.45,
        enable_tracking: bool = False,
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
                    results = self.model.predict(frame, threshold=confidence_threshold)
                    detections = self._parse_rfdetr_results(results, img_shape=frame.shape[:2])

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
                    "avg_fps": 1000 / (sum(frame_times[-30:]) / len(frame_times[-30:]))
                    if frame_times
                    else 0,
                }

        finally:
            cap.release()

    async def run_on_video_full(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        confidence_threshold: float = 0.0,
        iou_threshold: float = 0.45,
        enable_tracking: bool = False,
        tracking_config: Optional[TrackingConfig] = None,
        detection_interval: int = 1,  # Run detection every N frames (1 = every frame)
        progress_callback: Optional[ProgressCallback] = None,
    ) -> dict:
        """Run inference on entire video and optionally save annotated output.

        The actual heavy work is offloaded to a thread so the event loop stays
        responsive (e.g. for SSE streaming clients polling other endpoints).
        """
        return await asyncio.to_thread(
            self._run_on_video_full_sync,
            video_path,
            output_path,
            confidence_threshold,
            iou_threshold,
            enable_tracking,
            tracking_config,
            detection_interval,
            progress_callback,
        )

    def _run_on_video_full_sync(
        self,
        video_path: Path,
        output_path: Optional[Path],
        confidence_threshold: float,
        iou_threshold: float,
        enable_tracking: bool,
        tracking_config: Optional[TrackingConfig],
        detection_interval: int,
        progress_callback: Optional[ProgressCallback],
    ) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        def emit(event: dict) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(event)
                except Exception as e:
                    logger.debug(f"progress_callback failed: {e}")

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
            logger.info(
                f"Starting inference on {total_frames} frames (detecting every {detection_interval} frames)..."
            )
        else:
            logger.info(f"Starting inference on {total_frames} frames...")

        emit({"type": "stage", "stage": "running_inference", "total_frames": total_frames})
        # Push an initial 0% tick so the UI shows "starting" rather than staying
        # indeterminate while the first frame is being read / warmed up.
        emit(
            {
                "type": "progress",
                "current": 0,
                "total": total_frames,
                "avg_ms": 0.0,
                "avg_fps": 0.0,
                "eta_s": None,
            }
        )

        try:
            frame_num = 0
            last_log_time = time.time()
            last_progress_emit = 0.0
            last_detections = []  # Cache last detections for interpolation

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.perf_counter()

                # Only run detection on keyframes (every N frames)
                is_keyframe = frame_num % detection_interval == 0

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
                        results = self.model.predict(frame, threshold=confidence_threshold)
                        detections = self._parse_rfdetr_results(results, img_shape=frame.shape[:2])

                    last_detections = detections
                else:
                    # Use cached detections (tracking will update positions)
                    detections = [det.copy() for det in last_detections]

                # Apply tracking (updates positions even on non-keyframes)
                if tracker:
                    detections = tracker.update(detections, frame_num)

                inference_time = (time.perf_counter() - start_time) * 1000
                frame_times.append(inference_time)

                all_results.append(
                    {
                        "frame_number": frame_num,
                        "timestamp": frame_num / fps,
                        "detections": detections,
                        "is_keyframe": is_keyframe,
                    }
                )

                # Draw annotations if saving
                if writer:
                    height, width = frame.shape[:2]
                    det_objects = [
                        Detection(
                            bbox=(
                                (d["box"]["x"] - d["box"]["width"] / 2) * width,
                                (d["box"]["y"] - d["box"]["height"] / 2) * height,
                                (d["box"]["x"] + d["box"]["width"] / 2) * width,
                                (d["box"]["y"] + d["box"]["height"] / 2) * height,
                            ),
                            class_id=d.get("class_id", 0),
                            class_name=d.get("class_name", f"class_{d.get('class_id', 0)}"),
                            confidence=d.get("confidence", 1.0),
                            track_id=d.get("track_id"),
                            z_mm=d.get("z_mm"),
                            mask=d.get("mask"),
                        )
                        for d in detections
                    ]
                    annotated = draw_detections(frame, det_objects)
                    writer.write(annotated)

                frame_num += 1

                current_time = time.time()
                # Emit progress to the streaming client ~5 Hz. This is cheap
                # (a dict push onto a bounded queue) and gives a responsive bar.
                if current_time - last_progress_emit >= 0.2 or frame_num >= total_frames:
                    avg_ms = (
                        sum(frame_times[-30:]) / len(frame_times[-30:]) if frame_times else 0
                    )
                    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                    remaining = max(total_frames - frame_num, 0)
                    eta_s = (remaining * avg_ms) / 1000 if avg_ms > 0 else None
                    emit(
                        {
                            "type": "progress",
                            "current": frame_num,
                            "total": total_frames,
                            "avg_ms": avg_ms,
                            "avg_fps": avg_fps,
                            "eta_s": eta_s,
                        }
                    )
                    last_progress_emit = current_time

                # Log progress every 2 seconds
                if current_time - last_log_time >= 2.0:
                    progress = (frame_num / total_frames) * 100
                    avg_ms = sum(frame_times[-30:]) / len(frame_times[-30:]) if frame_times else 0
                    est_remaining = ((total_frames - frame_num) * avg_ms) / 1000
                    logger.info(
                        f"Inference progress: {frame_num}/{total_frames} ({progress:.1f}%) - {avg_ms:.1f}ms/frame - ETA: {est_remaining:.0f}s"
                    )
                    last_log_time = current_time

        finally:
            cap.release()
            if writer:
                writer.release()

        # Re-encode to H.264 so browsers can play the video
        if output_path and output_path.exists():
            emit({"type": "stage", "stage": "encoding_video"})
            # #region agent log
            import json as _json

            open("/home/batman/batman/.cursor/debug-b2be69.log", "a").write(
                _json.dumps(
                    {
                        "sessionId": "b2be69",
                        "hypothesisId": "H1",
                        "location": "inference_runner.py:remux",
                        "message": "re-encoding to h264",
                        "data": {"output_path": str(output_path)},
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
            # #endregion
            tmp_path = output_path.with_suffix(".tmp.mp4")
            try:
                import subprocess as _sp

                result_ffmpeg = _sp.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(output_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        "-an",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    timeout=300,
                )
                if result_ffmpeg.returncode == 0 and tmp_path.exists():
                    tmp_path.replace(output_path)
                    # #region agent log
                    open("/home/batman/batman/.cursor/debug-b2be69.log", "a").write(
                        _json.dumps(
                            {
                                "sessionId": "b2be69",
                                "hypothesisId": "H1",
                                "location": "inference_runner.py:remux_done",
                                "message": "h264 remux succeeded",
                                "data": {"size": output_path.stat().st_size},
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
                    # #endregion
                else:
                    # #region agent log
                    open("/home/batman/batman/.cursor/debug-b2be69.log", "a").write(
                        _json.dumps(
                            {
                                "sessionId": "b2be69",
                                "hypothesisId": "H1",
                                "location": "inference_runner.py:remux_fail",
                                "message": "ffmpeg failed",
                                "data": {
                                    "rc": result_ffmpeg.returncode,
                                    "stderr": result_ffmpeg.stderr.decode()[-500:],
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
                    # #endregion
                    logger.warning(
                        f"ffmpeg re-encode failed (rc={result_ffmpeg.returncode}), keeping mp4v file"
                    )
            except FileNotFoundError:
                logger.warning(
                    "ffmpeg not found; video will remain in mp4v codec (may not play in browsers)"
                )
            except Exception as e:
                logger.warning(f"ffmpeg re-encode error: {e}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)

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

            detections.append(
                {
                    "box": {"x": cx, "y": cy, "width": w, "height": h},
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f"class_{cls_id}",
                }
            )

        return detections

    def _parse_rfdetr_results(self, results, img_shape=None) -> list[dict]:
        """Parse RF-DETR model output to common format (normalized center box, class_name, etc.).
        img_shape: (height, width) of the image; required to normalize pixel coords from the model.

        For segmentation models (results.mask present), extracts the largest contour from each
        instance mask, simplifies it with approxPolyDP, and attaches it as a normalised polygon
        under the "mask" field on each detection.
        """
        out = []
        if results is None:
            return out
        if (
            not hasattr(results, "xyxy")
            or not hasattr(results, "class_id")
            or not hasattr(results, "confidence")
        ):
            return out
        n = len(results.xyxy)
        if n == 0:
            return out
        if img_shape is not None:
            img_h, img_w = int(img_shape[0]), int(img_shape[1])
            if img_w <= 0 or img_h <= 0:
                return out
        else:
            img_w = img_h = 1

        masks = getattr(results, "mask", None)
        has_masks = masks is not None and len(masks) == n

        for i in range(n):
            xyxy = results.xyxy[i]
            if hasattr(xyxy, "tolist"):
                xyxy = xyxy.tolist()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            w_px = x2 - x1
            h_px = y2 - y1
            if w_px <= 0 or h_px <= 0:
                continue
            cx_norm = (x1 + x2) / 2 / img_w
            cy_norm = (y1 + y2) / 2 / img_h
            w_norm = w_px / img_w
            h_norm = h_px / img_h
            class_id = int(results.class_id[i])
            conf = float(results.confidence[i])
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"class_{class_id}"
            )
            det = {
                "box": {"x": cx_norm, "y": cy_norm, "width": w_norm, "height": h_norm},
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
            }
            if has_masks:
                poly = _mask_to_polygon_norm(masks[i], img_w, img_h)
                if poly is not None:
                    det["mask"] = poly
            out.append(det)
        return out


def _mask_to_polygon_norm(mask, img_w: int, img_h: int) -> Optional[list[list[float]]]:
    """Extract the largest contour of a binary mask and return as a normalised polygon.

    Accepts HxW numpy arrays / torch tensors; any non-zero value is treated as inside.
    The contour is simplified with cv2.approxPolyDP at epsilon=1.5px to keep result.json small.
    Returns None if the mask is empty, degenerate, or any step fails.
    """
    try:
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        elif hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr.squeeze()
        if arr.ndim != 2:
            return None
        bin_mask = (arr > 0).astype(np.uint8)
        if bin_mask.max() == 0:
            return None

        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) <= 0:
            return None

        simplified = cv2.approxPolyDP(biggest, epsilon=1.5, closed=True)
        pts = simplified.reshape(-1, 2)
        if len(pts) < 3:
            return None

        return [
            [float(x) / img_w, float(y) / img_h]
            for x, y in pts
        ]
    except Exception as e:
        logger.debug(f"mask_to_polygon_norm failed: {e}")
        return None


# Global instance
inference_runner = InferenceRunner()
