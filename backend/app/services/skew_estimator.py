"""
Skew-angle estimation between the spreader and the container, derived from
segmentation masks produced by RF-DETR-Seg.

Given the normalised polygons stored on each detection (``det["mask"]``), we
fit a ``cv2.minAreaRect`` to each polygon, canonicalise the long-axis angle
into ``(-90, 90]`` degrees, and express the skew as
``skew_deg = container_deg - spreader_deg`` (wrapped back into ``(-90, 90]``).

The typical use case is to call :func:`apply_skew_to_result` after inference
has written ``result.json``; it mutates frames in place to add ``skew_deg`` /
``spreader_deg`` / ``container_deg`` entries and writes a top-level ``skew``
block summarising the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

try:
    import cv2
except Exception as e:  # pragma: no cover - cv2 is a runtime dep
    cv2 = None  # type: ignore[assignment]
    logger.warning(f"cv2 not available, skew estimation will be disabled: {e}")


SPREADER_CLASSES: set[str] = {"spreader"}
CONTAINER_CLASSES: set[str] = {"container"}


def _polygon_norm_to_pixels(
    polygon_norm: list[list[float]],
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Convert a normalised [[x, y], ...] polygon to an ``Nx1x2`` int32 array.

    Returns ``None`` if the polygon is missing, malformed, or has fewer than
    three valid vertices (``cv2.minAreaRect`` needs at least three points).
    """
    if not polygon_norm or len(polygon_norm) < 3:
        return None
    pts: list[list[int]] = []
    for pt in polygon_norm:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        try:
            x = int(round(float(pt[0]) * img_w))
            y = int(round(float(pt[1]) * img_h))
        except (TypeError, ValueError):
            return None
        pts.append([x, y])
    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


def _canonical_angle(angle_deg: float, width: float, height: float) -> float:
    """Map ``cv2.minAreaRect`` output to the long-axis angle in ``(-90, 90]``.

    ``cv2.minAreaRect`` returns ``(center, (w, h), angle)`` where the angle is
    the rotation of the edge measured against the ``w`` axis. When ``w < h``
    the long axis is perpendicular to that edge, so we rotate by 90 degrees
    before normalising.
    """
    if width < height:
        long_axis_angle = angle_deg + 90.0
    else:
        long_axis_angle = angle_deg
    # Wrap into (-90, 90] via modulo 180.
    wrapped = ((long_axis_angle + 90.0) % 180.0) - 90.0
    if wrapped <= -90.0:
        wrapped += 180.0
    return wrapped


def orientation_deg(
    polygon_norm: list[list[float]],
    img_w: int,
    img_h: int,
) -> Optional[float]:
    """Return the long-axis orientation of a polygon in degrees, or ``None``.

    Angle is in ``(-90, 90]`` where 0 is horizontal. Fails gracefully (returns
    ``None``) for degenerate inputs so callers can skip that frame.
    """
    if cv2 is None:
        return None
    pts = _polygon_norm_to_pixels(polygon_norm, img_w, img_h)
    if pts is None:
        return None
    try:
        (_, (w, h), angle) = cv2.minAreaRect(pts)
    except Exception as e:
        logger.debug(f"minAreaRect failed: {e}")
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    return _canonical_angle(float(angle), float(w), float(h))


def _wrap_signed_angle(delta: float) -> float:
    """Wrap a signed angle difference into ``(-90, 90]`` (lines, not rays)."""
    wrapped = ((delta + 90.0) % 180.0) - 90.0
    if wrapped <= -90.0:
        wrapped += 180.0
    return wrapped


def _get_polygon(det: dict) -> Optional[list[list[float]]]:
    """Detections use ``mask`` at inference time and ``polygon`` at label time."""
    poly = det.get("mask")
    if poly is None:
        poly = det.get("polygon")
    return poly  # type: ignore[return-value]


def _class_name(det: dict) -> str:
    return str(det.get("class_name") or "").strip().lower()


def compute_skew(
    detections: list[dict],
    img_w: int,
    img_h: int,
) -> Optional[dict[str, float]]:
    """Return ``{spreader_deg, container_deg, skew_deg}`` for a single frame.

    If either a spreader or a container detection with a usable polygon is
    missing, returns ``None``. When multiple detections of one class exist the
    one with the largest bbox area (which proxies for nearest / most visible)
    is picked.
    """
    if not detections:
        return None

    def _pick_best(class_set: set[str]) -> Optional[dict]:
        candidates = [
            d for d in detections
            if _class_name(d) in class_set and _get_polygon(d)
        ]
        if not candidates:
            return None
        # Largest bbox area wins. Supports both the normalised "box" schema
        # used by the backend and the pixel xyxy "bbox" tuple emitted by
        # src.core.inference.save_results_json.
        def _area(d: dict) -> float:
            box = d.get("box")
            if isinstance(box, dict):
                return float(box.get("width", 0.0)) * float(box.get("height", 0.0))
            bbox = d.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
            return 0.0
        return max(candidates, key=_area)

    spreader = _pick_best(SPREADER_CLASSES)
    container = _pick_best(CONTAINER_CLASSES)
    if spreader is None or container is None:
        return None

    spreader_deg = orientation_deg(_get_polygon(spreader), img_w, img_h)
    container_deg = orientation_deg(_get_polygon(container), img_w, img_h)
    if spreader_deg is None or container_deg is None:
        return None

    skew = _wrap_signed_angle(container_deg - spreader_deg)
    return {
        "spreader_deg": round(spreader_deg, 2),
        "container_deg": round(container_deg, 2),
        "skew_deg": round(skew, 2),
    }


def apply_skew_to_result(
    result_dir: Path,
    video_resolution: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Populate ``skew_deg`` on each frame in ``result.json`` plus a summary block.

    Args:
        result_dir: Directory containing ``result.json``.
        video_resolution: Optional ``{"width", "height"}``. If omitted we try
            ``data["video_resolution"]`` in the file.

    Returns:
        The summary block written under ``data["skew"]``.
    """
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found at {result_dir}")

    with open(result_path) as f:
        data = json.load(f)

    if video_resolution is None:
        video_resolution = data.get("video_resolution")
    if not video_resolution or "width" not in video_resolution or "height" not in video_resolution:
        raise ValueError("video_resolution with width/height required for skew estimation")

    img_w = int(video_resolution["width"])
    img_h = int(video_resolution["height"])

    frames = data.get("frames") or []
    measured: list[float] = []
    for frame in frames:
        skew = compute_skew(frame.get("detections") or [], img_w, img_h)
        if skew is None:
            # Make sure stale values don't linger from a previous run.
            frame.pop("skew_deg", None)
            frame.pop("spreader_deg", None)
            frame.pop("container_deg", None)
            continue
        frame["skew_deg"] = skew["skew_deg"]
        frame["spreader_deg"] = skew["spreader_deg"]
        frame["container_deg"] = skew["container_deg"]
        measured.append(skew["skew_deg"])

    if measured:
        abs_vals = [abs(v) for v in measured]
        summary: dict[str, Any] = {
            "frames_with_skew": len(measured),
            "total_frames": len(frames),
            "mean_deg": round(float(np.mean(measured)), 2),
            "median_deg": round(float(np.median(measured)), 2),
            "std_deg": round(float(np.std(measured)), 2),
            "min_deg": round(float(np.min(measured)), 2),
            "max_deg": round(float(np.max(measured)), 2),
            "mean_abs_deg": round(float(np.mean(abs_vals)), 2),
            "max_abs_deg": round(float(np.max(abs_vals)), 2),
        }
    else:
        summary = {
            "frames_with_skew": 0,
            "total_frames": len(frames),
        }

    data["skew"] = summary
    data["frames"] = frames

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"Skew estimation: {summary.get('frames_with_skew', 0)}/{summary.get('total_frames', 0)} "
        f"frames scored, saved to {result_path}"
    )
    return summary
