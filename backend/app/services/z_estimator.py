"""
Z-axis height estimation from bounding box sizes.

Supports two calibration models:
- 1 label:  Z = k / s   where k = Z_cal * s_cal
- 2+ labels: Z = a / s + b   (least-squares fit of Z on 1/s)

Size metric: h_px (bbox height in pixels) — most stable for swinging crane hooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def _h_px(det: dict, video_height: int) -> float:
    """Extract bbox height in pixels from a normalized detection dict."""
    return det["box"]["height"] * video_height


def calibrate(
    labels: list[dict],
    frames: list[dict],
    video_resolution: dict,
    class_name: str = "crane hook",
    size_metric: str = "h_px",
) -> dict:
    """
    Build a calibration model from user-provided ground-truth labels.

    Args:
        labels: List of {"frame_number": int, "z_mm": float, "detection_index": int}
        frames: Full frame list from result.json (each has "frame_number" and "detections")
        video_resolution: {"width": int, "height": int}
        class_name: Only estimate Z for this class
        size_metric: "h_px" (only supported metric for now)

    Returns:
        Calibration dict: {"type": str, "k"?: float, "a"?: float, "b"?: float}
    """
    vid_h = video_resolution["height"]

    frame_map: dict[int, dict] = {f["frame_number"]: f for f in frames}

    pairs: list[tuple[float, float]] = []
    for label in labels:
        fn = label["frame_number"]
        z_mm = label["z_mm"]
        det_idx = label.get("detection_index", 0)

        frame = frame_map.get(fn)
        if frame is None:
            logger.warning(f"Z calibration: frame {fn} not found in results, skipping")
            continue

        dets = frame.get("detections", [])
        if det_idx >= len(dets):
            logger.warning(f"Z calibration: detection index {det_idx} out of range for frame {fn}")
            continue

        det = dets[det_idx]
        s = _h_px(det, vid_h)
        if s <= 0:
            logger.warning(f"Z calibration: zero/negative size for frame {fn}, skipping")
            continue

        pairs.append((s, z_mm))

    if len(pairs) == 0:
        raise ValueError("No valid calibration pairs found")

    if len(pairs) == 1:
        # 1-point: Z = k / s
        s_cal, z_cal = pairs[0]
        k = z_cal * s_cal
        logger.info(f"Z calibration: 1-point model, k={k:.1f} (Z={z_cal}mm, s={s_cal:.1f}px)")
        return {"type": "k_over_s", "k": k}
    else:
        # 2+ points: Z = a / s + b  → linear regression of Z on 1/s
        # y = a * x + b  where x = 1/s, y = Z
        n = len(pairs)
        xs = [1.0 / s for s, _ in pairs]
        ys = [z for _, z in pairs]

        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xx = sum(x * x for x in xs)
        sum_xy = sum(x * y for x, y in zip(xs, ys))

        denom = n * sum_xx - sum_x * sum_x
        if abs(denom) < 1e-12:
            # Degenerate — all same size, fall back to mean k
            k_mean = sum(z * s for s, z in pairs) / n
            logger.warning(f"Z calibration: degenerate 2-point, falling back to k={k_mean:.1f}")
            return {"type": "k_over_s", "k": k_mean}

        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n

        logger.info(f"Z calibration: {n}-point linear model, a={a:.1f}, b={b:.1f}")
        return {"type": "linear_inv", "a": a, "b": b}


def estimate(
    model: dict,
    frames: list[dict],
    video_resolution: dict,
    class_name: str = "crane hook",
) -> list[dict]:
    """
    Apply a calibration model to estimate Z for all frames.

    Modifies detection dicts in-place by adding "z_mm" field to matching detections.
    Returns the modified frames list.
    """
    vid_h = video_resolution["height"]
    model_type = model["type"]

    estimated_count = 0

    for frame in frames:
        for det in frame.get("detections", []):
            if det.get("class_name") != class_name:
                continue

            s = _h_px(det, vid_h)
            if s <= 0:
                continue

            if model_type == "k_over_s":
                z = model["k"] / s
            elif model_type == "linear_inv":
                z = model["a"] / s + model["b"]
            else:
                continue

            det["z_mm"] = round(z, 1)
            estimated_count += 1

    logger.info(
        f"Z estimation: applied to {estimated_count} detections across {len(frames)} frames"
    )
    return frames


def load_z_calibration(result_dir: Path) -> dict | None:
    """Load Z calibration from result.json if it exists."""
    result_path = result_dir / "result.json"
    if not result_path.exists():
        return None

    with open(result_path) as f:
        data = json.load(f)

    return data.get("z_calibration")


def save_z_calibration(result_dir: Path, calibration: dict) -> None:
    """Save Z calibration data into result.json."""
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found at {result_dir}")

    with open(result_path) as f:
        data = json.load(f)

    data["z_calibration"] = calibration

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Z calibration saved to {result_path}")


def apply_z_to_result(result_dir: Path, class_name: str = "crane hook") -> dict:
    """
    Full pipeline: load calibration from result.json, estimate Z for all frames,
    save updated frames back to result.json.

    Returns the calibration model used.
    """
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found at {result_dir}")

    with open(result_path) as f:
        data = json.load(f)

    z_cal = data.get("z_calibration")
    if z_cal is None:
        raise ValueError("No z_calibration found in result.json — calibrate first")

    labels = z_cal.get("labels", [])
    frames = data.get("frames", [])

    video_resolution = z_cal.get("video_resolution")
    if video_resolution is None:
        raise ValueError("No video_resolution found in z_calibration")

    model = calibrate(labels, frames, video_resolution, class_name=class_name)

    z_cal["model"] = model
    z_cal["class_name"] = class_name

    frames = estimate(model, frames, video_resolution, class_name=class_name)

    data["z_calibration"] = z_cal
    data["frames"] = frames

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Z estimation complete, saved to {result_path}")
    return model
