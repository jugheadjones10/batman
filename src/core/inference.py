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
    z_mm: float | None = None
    # Normalised instance segmentation polygon, list of [x, y] in [0, 1] image coords.
    # Only populated when running a segmentation model (RF-DETR-Seg).
    mask: list[list[float]] | None = None


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
        task: str = "detection",
    ):
        """
        Initialize inference engine.

        Args:
            checkpoint: Path to model checkpoint
            class_names: List of class names (loaded from class_info.json if not provided)
            model_size: Model size ('base', 'large', etc.)
            task: "detection" or "segmentation"
        """
        self.checkpoint = Path(checkpoint)
        self.model_size = model_size
        self.task = task

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

        from src.core.trainer import resolve_rfdetr_class

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
        logger.info(f"Loading model on {device} (task={self.task}, size={self.model_size})")

        # Load model
        weights_start = time.time()
        ModelClass = resolve_rfdetr_class(self.task, self.model_size)
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

        # Grab image size for mask normalisation (PIL → (w, h))
        img_w, img_h = image.size

        masks = getattr(detections, "mask", None)

        results = []
        if hasattr(detections, "xyxy") and len(detections.xyxy) > 0:
            n = len(detections.xyxy)
            has_masks = masks is not None and len(masks) == n
            for i in range(n):
                class_id = int(detections.class_id[i])
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"class_{class_id}"
                )
                poly = None
                if has_masks:
                    poly = _mask_to_polygon_norm_core(masks[i], img_w, img_h)
                results.append(
                    Detection(
                        bbox=tuple(detections.xyxy[i].tolist()),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(detections.confidence[i]),
                        mask=poly,
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


COLOR_PALETTE = [
    (255, 107, 107),  # Red
    (78, 205, 196),  # Teal
    (69, 183, 209),  # Blue
    (150, 206, 180),  # Green
    (255, 234, 167),  # Yellow
    (221, 160, 221),  # Purple
    (152, 216, 200),  # Mint
    (255, 159, 243),  # Pink
]


def _mask_to_polygon_norm_core(mask, img_w: int, img_h: int) -> list[list[float]] | None:
    """Extract largest-contour polygon from a binary mask, normalised to [0,1].

    Returns None if the mask is empty, degenerate, or any conversion step fails.
    Used by RFDETRInference.predict_image for segmentation models.
    """
    try:
        import cv2

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

        return [[float(x) / img_w, float(y) / img_h] for x, y in pts]
    except Exception:
        return None


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    thickness: int = 2,
    font_scale: float = 0.6,
    render_mode: str = "polygon",
) -> np.ndarray:
    """
    Draw detection bounding boxes with class-name-only labels on an image.

    Args:
        image: BGR image (numpy array)
        detections: List of Detection objects
        thickness: Line thickness
        font_scale: Font scale for box labels
        render_mode: One of:
            - "polygon": draw polygon if mask present, else fall back to bbox (default)
            - "bbox":    always draw bbox, never polygon

    Returns:
        Annotated image
    """
    import cv2

    if render_mode not in ("polygon", "bbox"):
        render_mode = "polygon"

    result = image.copy()
    h, w = result.shape[:2]
    overlay = None  # Allocate lazily only if we actually have masks to draw

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        color = COLOR_PALETTE[det.class_id % len(COLOR_PALETTE)]

        drew_polygon = False
        if render_mode != "bbox" and det.mask:
            try:
                pts_px = np.array(
                    [[int(round(px * w)), int(round(py * h))] for px, py in det.mask],
                    dtype=np.int32,
                )
                if len(pts_px) >= 3:
                    if overlay is None:
                        overlay = result.copy()
                    cv2.fillPoly(overlay, [pts_px], color)
                    cv2.polylines(result, [pts_px], True, color, max(1, thickness - 1))
                    drew_polygon = True
            except Exception:
                pass

        if not drew_polygon:
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = det.class_name
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(result, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(
            result,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    if overlay is not None:
        cv2.addWeighted(overlay, 0.35, result, 0.65, 0, dst=result)

    return result


def rerender_detected_video(
    source_video_path: Path,
    result_json_path: Path,
    output_path: Path,
    render_mode: str = "polygon",
) -> int:
    """Re-bake the overlay video from a saved result.json + the source video.

    Reads per-frame detections from `result_json_path` and writes an mp4 with
    overlays drawn according to `render_mode` (see draw_detections for values).
    The output is written first in mp4v and then transcoded to H.264 via ffmpeg
    so browsers can stream it with seeking.

    Args:
        source_video_path: Path to the original source video.
        result_json_path: Path to result.json produced by inference.
        output_path: Destination mp4 (e.g. .../detected.mp4). Will be overwritten.
        render_mode: One of "polygon" | "bbox".

    Returns:
        Number of frames written.
    """
    import cv2
    import json as _json
    import subprocess as _sp

    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_json_path}")

    with open(result_json_path) as f:
        data = _json.load(f)

    frames_list = data.get("frames", [])
    frame_map: dict[int, dict] = {f["frame_number"]: f for f in frames_list if "frame_number" in f}

    cap = cv2.VideoCapture(str(source_video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (vid_w, vid_h))
        try:
            frames_written = 0
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_data = frame_map.get(frame_num)
                if frame_data:
                    det_objects = [
                        Detection(
                            bbox=(
                                (d["box"]["x"] - d["box"]["width"] / 2) * vid_w,
                                (d["box"]["y"] - d["box"]["height"] / 2) * vid_h,
                                (d["box"]["x"] + d["box"]["width"] / 2) * vid_w,
                                (d["box"]["y"] + d["box"]["height"] / 2) * vid_h,
                            ),
                            class_id=d.get("class_id", 0),
                            class_name=d.get("class_name", ""),
                            confidence=d.get("confidence", 1.0),
                            track_id=d.get("track_id"),
                            z_mm=d.get("z_mm"),
                            mask=d.get("mask"),
                        )
                        for d in frame_data.get("detections", [])
                    ]
                    frame = draw_detections(frame, det_objects, render_mode=render_mode)

                writer.write(frame)
                frames_written += 1
        finally:
            writer.release()
    finally:
        cap.release()

    # Re-encode to H.264 so the browser can stream it with Range seeking.
    if output_path.exists():
        tmp_path = output_path.with_suffix(".tmp.mp4")
        try:
            proc = _sp.run(
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
                timeout=600,
            )
            if proc.returncode == 0 and tmp_path.exists():
                tmp_path.replace(output_path)
        except FileNotFoundError:
            # ffmpeg not installed — leave the mp4v file as-is.
            pass
        except Exception:
            pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return frames_written


def render_comparison_videos(
    source_video_path: Path,
    result_json_path: Path,
    out_raw_path: Path,
    out_bytetrack_path: Path,
    render_mode: str = "polygon",
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
) -> int:
    """Render a pair of annotated MP4s from an existing result.json for A/B
    comparison of raw per-frame detections vs sv.ByteTrack.

    - out_raw_path: per-frame detections drawn as-is, with no tracker at all.
      Shows the native flicker / missed frames of the detector.
    - out_bytetrack_path: the same per-frame detections piped through
      sv.ByteTrack. Both matched detections and currently-lost tracks (still
      within lost_track_buffer) are rendered, so Kalman predictions visibly
      fill in missed frames. ByteTrack is an online / causal tracker, so the
      behaviour here is identical to what it would produce live.

    Both videos are written in a single pass over the source video and then
    remuxed to H.264 via ffmpeg so browsers can stream them with Range
    seeking. The function is safe to call multiple times; outputs are
    overwritten deterministically.

    Args:
        source_video_path: Path to the original source video.
        result_json_path: Path to result.json produced by a prior inference run.
        out_raw_path: Destination mp4 for the untracked baseline.
        out_bytetrack_path: Destination mp4 for the ByteTrack variant.
        render_mode: One of "polygon" | "bbox". Applies to the raw side;
            the ByteTrack side always draws bboxes because segmentation masks
            are not preserved through the tracker's association step.
        track_activation_threshold: ByteTrack high-confidence gating threshold.
        lost_track_buffer: Frames a lost track is kept alive with Kalman
            predictions before being removed.
        minimum_matching_threshold: IoU association threshold.

    Returns:
        Number of frames written per output.
    """
    import cv2
    import json as _json
    import subprocess as _sp

    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not result_json_path.exists():
        raise FileNotFoundError(f"result.json not found: {result_json_path}")

    with open(result_json_path) as f:
        data = _json.load(f)

    frames_list = data.get("frames", [])
    frame_map: dict[int, dict] = {
        f["frame_number"]: f for f in frames_list if "frame_number" in f
    }

    cap = cv2.VideoCapture(str(source_video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Precompute class-aware ByteTrack output so the video overlay and the
        # schematic comparison show exactly the same tracked data (see the
        # docstring on compute_bytetrack_frames for why per-class tracking
        # matters — class-agnostic IoU association makes spreader/container
        # steal each other's tracks).
        bt_frames = compute_bytetrack_frames(
            data,
            fps=fps,
            vid_w=vid_w,
            vid_h=vid_h,
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
        )
        bt_frame_map: dict[int, dict] = {
            f["frame_number"]: f for f in bt_frames if "frame_number" in f
        }

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        raw_writer = cv2.VideoWriter(str(out_raw_path), fourcc, fps, (vid_w, vid_h))
        bt_writer = cv2.VideoWriter(
            str(out_bytetrack_path), fourcc, fps, (vid_w, vid_h)
        )

        def _denorm_dets(frame_data: dict | None) -> list[Detection]:
            if not frame_data:
                return []
            dets: list[Detection] = []
            for d in frame_data.get("detections", []):
                b = d["box"]
                dets.append(
                    Detection(
                        bbox=(
                            (b["x"] - b["width"] / 2) * vid_w,
                            (b["y"] - b["height"] / 2) * vid_h,
                            (b["x"] + b["width"] / 2) * vid_w,
                            (b["y"] + b["height"] / 2) * vid_h,
                        ),
                        class_id=d.get("class_id", 0),
                        class_name=d.get("class_name", ""),
                        confidence=d.get("confidence", 1.0),
                        track_id=d.get("track_id"),
                        z_mm=d.get("z_mm"),
                        mask=d.get("mask"),
                    )
                )
            return dets

        try:
            frames_written = 0
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                raw_dets = _denorm_dets(frame_map.get(frame_num))
                bt_dets = _denorm_dets(bt_frame_map.get(frame_num))

                raw_writer.write(
                    draw_detections(frame, raw_dets, render_mode=render_mode)
                )
                bt_writer.write(draw_detections(frame, bt_dets, render_mode="bbox"))
                frames_written += 1
        finally:
            raw_writer.release()
            bt_writer.release()
    finally:
        cap.release()

    for out_path in (out_raw_path, out_bytetrack_path):
        if not out_path.exists():
            continue
        tmp_path = out_path.with_suffix(".tmp.mp4")
        try:
            proc = _sp.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(out_path),
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
                timeout=600,
            )
            if proc.returncode == 0 and tmp_path.exists():
                tmp_path.replace(out_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return frames_written


def compute_bytetrack_frames(
    result_data: dict,
    fps: float,
    vid_w: int,
    vid_h: int,
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
) -> list[dict]:
    """Re-run sv.ByteTrack over an existing result.json's per-frame detections
    and return a `frames[]` list in the same normalized schema.

    The returned frames preserve `frame_number`, `timestamp`, and
    `is_keyframe`, but replace `detections` with ByteTrack's output:

      - Matched tracks use the tracker's Kalman-posterior box (the KF state
        after fusing the prior prediction with the new measurement), tagged
        with a stable per-class `track_id` and `track_source="matched"`.
        This is what supervision
        computes internally but discards in `update_with_detections`; we
        recover it via `tracker.tracked_tracks[*].tlbr`. Surfacing the
        posterior instead of the raw input bbox is the cheapest way to
        reduce RF-DETR's frame-to-frame box jitter (which compounds badly
        through `z ∝ 1 / max(w_px, h_px)` on the schematic).
      - Tracks currently inside `lost_track_buffer` without a matching
        detection are emitted with their Kalman-predicted box (prior only)
        and `track_source="lost"`.
        Both matched and lost boxes therefore come from the same KF state
        vector, so there is no visual discontinuity between measured and
        extrapolated frames.
      - No `z_mm` is carried on the output. The persisted `z_mm` in
        `result.json` was computed from the *raw* bbox, so pairing it with
        a smoothed bbox would re-introduce the jitter on the only signal
        that's consumed for z. The UI recomputes z on the fly from the
        emitted box via the project calibration (same `k / max(w_px, h_px)`
        formula the backend uses), keeping box and z strictly consistent.

    The tracker runs CLASS-AWARE: one independent `sv.ByteTrack` per class.
    This matters because supervision's ByteTrack does pure IoU association
    and happily matches a track of class A to a detection of class B if
    their boxes overlap — which they always do for e.g. a spreader sitting
    on top of a container. Empirically that causes ~30% of long-running
    tracks to swap classes between frames, making the raw vs. tracked
    schematic flicker worse than the raw trace. Per-class trackers
    eliminate that failure mode entirely.

    No video I/O is performed; this is a pure transformation of the JSON.
    """
    import supervision as sv

    raw_frames = sorted(
        result_data.get("frames", []),
        key=lambda fr: fr.get("frame_number", 0),
    )

    def _to_norm_box(x1: float, y1: float, x2: float, y2: float) -> dict:
        return {
            "x": float(((x1 + x2) / 2.0) / vid_w),
            "y": float(((y1 + y2) / 2.0) / vid_h),
            "width": float((x2 - x1) / vid_w),
            "height": float((y2 - y1) / vid_h),
        }

    def _new_tracker() -> "sv.ByteTrack":
        return sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=int(round(fps)) if fps > 0 else 30,
        )

    # Per-class state. Each class_name gets its own ByteTrack instance, and
    # we remap the tracker's (local, per-class) ids onto globally-unique ids
    # so the consumer never sees a collision between classes.
    class_trackers: dict[str, sv.ByteTrack] = {}
    # (class_name, local_tid) -> (global_tid, class_id)
    tid_map: dict[tuple[str, int], tuple[int, int]] = {}
    global_tid_counter = 0

    out: list[dict] = []
    for frame in raw_frames:
        raw_dets: list[dict] = list(frame.get("detections", []))

        # Group detections by class_name so each tracker only sees its own
        # class. Retain original indices to merge z_mm back in.
        by_cls: dict[str, list[tuple[int, dict]]] = {}
        for i, d in enumerate(raw_dets):
            cname = d.get("class_name", "")
            by_cls.setdefault(cname, []).append((i, d))

        # Ensure every class we've ever seen gets .update() called this
        # frame (even with zero detections) so its Kalman predictions and
        # lost-buffer bookkeeping advance in sync with the video clock.
        all_classes = set(class_trackers.keys()) | set(by_cls.keys())

        det_out: list[dict] = []
        for cname in all_classes:
            tracker = class_trackers.get(cname)
            if tracker is None:
                tracker = _new_tracker()
                class_trackers[cname] = tracker

            cls_dets = by_cls.get(cname, [])
            if cls_dets:
                xyxy = np.array(
                    [
                        [
                            (d["box"]["x"] - d["box"]["width"] / 2) * vid_w,
                            (d["box"]["y"] - d["box"]["height"] / 2) * vid_h,
                            (d["box"]["x"] + d["box"]["width"] / 2) * vid_w,
                            (d["box"]["y"] + d["box"]["height"] / 2) * vid_h,
                        ]
                        for _, d in cls_dets
                    ],
                    dtype=np.float32,
                )
                confidence = np.array(
                    [d.get("confidence", 1.0) for _, d in cls_dets],
                    dtype=np.float32,
                )
                class_id = np.array(
                    [d.get("class_id", 0) for _, d in cls_dets],
                    dtype=int,
                )
                sv_in = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id,
                    data={"local_idx": np.arange(len(cls_dets), dtype=int)},
                )
            else:
                sv_in = sv.Detections(
                    xyxy=np.empty((0, 4), dtype=np.float32),
                    confidence=np.empty((0,), dtype=np.float32),
                    class_id=np.empty((0,), dtype=int),
                    data={"local_idx": np.empty((0,), dtype=int)},
                )

            tracked = tracker.update_with_detections(sv_in)

            # supervision's `update_with_detections` runs the Kalman update
            # internally (so every matched STrack's `mean` is the posterior)
            # but then returns the *raw input* detections with only a
            # `tracker_id` stapled on — i.e. the smoothing is computed and
            # thrown away. Build an external_track_id → STrack lookup off
            # the tracker's private state so we can emit the posterior bbox
            # (`strack.tlbr`) instead of `tracked.xyxy[i]`. This removes the
            # matched-vs-lost asymmetry: every box we emit, measured or
            # extrapolated, now comes from the same KF state vector.
            strack_by_tid: dict[int, object] = {
                int(t.external_track_id): t
                for t in tracker.tracked_tracks
                if getattr(t, "external_track_id", -1) >= 0
            }

            for i in range(len(tracked)):
                local_tid = int(tracked.tracker_id[i])
                cid = int(tracked.class_id[i])

                key = (cname, local_tid)
                mapping = tid_map.get(key)
                if mapping is None:
                    global_tid_counter += 1
                    mapping = (global_tid_counter, cid)
                    tid_map[key] = mapping
                gtid, _ = mapping

                strack = strack_by_tid.get(local_tid)
                if strack is not None:
                    tlbr = strack.tlbr
                    x1, y1, x2, y2 = (
                        float(tlbr[0]),
                        float(tlbr[1]),
                        float(tlbr[2]),
                        float(tlbr[3]),
                    )
                else:
                    # Defensive fallback — shouldn't happen, because every
                    # tracker_id in `tracked` came from an STrack that was
                    # just joined into `self.tracked_tracks`.
                    x1, y1, x2, y2 = (float(c) for c in tracked.xyxy[i])

                det: dict = {
                    "box": _to_norm_box(x1, y1, x2, y2),
                    "confidence": float(tracked.confidence[i]),
                    "class_id": cid,
                    "class_name": cname,
                    "track_id": gtid,
                    "track_source": "matched",
                }
                # Deliberately NO `z_mm` here; see the function docstring.
                # The emitted box is Kalman-smoothed, so the raw-bbox-based
                # `z_mm` from result.json would be inconsistent; the UI
                # recomputes z from this box via the project calibration.
                det_out.append(det)

            # Kalman fills from this class's lost buffer (no z_mm — UI
            # re-estimates z from the predicted box via calibration).
            for strack in tracker.lost_tracks:
                local_tid = int(getattr(strack, "external_track_id", -1))
                if local_tid < 0:
                    continue
                mapping = tid_map.get((cname, local_tid))
                if mapping is None:
                    # Never had a matched detection for this track → no class
                    # metadata to attach; skip.
                    continue
                gtid, cid = mapping
                tlbr = strack.tlbr
                x1, y1, x2, y2 = (
                    float(tlbr[0]),
                    float(tlbr[1]),
                    float(tlbr[2]),
                    float(tlbr[3]),
                )
                det_out.append(
                    {
                        "box": _to_norm_box(x1, y1, x2, y2),
                        "confidence": float(getattr(strack, "score", 0.0)),
                        "class_id": cid,
                        "class_name": cname,
                        "track_id": gtid,
                        "track_source": "lost",
                    }
                )

        out.append(
            {
                "frame_number": int(frame.get("frame_number", 0)),
                "timestamp": float(frame.get("timestamp", 0.0)),
                "is_keyframe": bool(frame.get("is_keyframe", False)),
                "detections": det_out,
            }
        )

    return out


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
                    **({"z_mm": det.z_mm} if det.z_mm is not None else {}),
                    **({"mask": det.mask} if det.mask is not None else {}),
                }
                for det in frame.detections
            ],
        }
        data["frames"].append(frame_data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
