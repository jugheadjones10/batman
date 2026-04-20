"""
Z-axis height estimation from bounding box sizes.

Supports calibration models:
- 1 label:  Z = k / s   where k = Z_cal * s_cal
- 2+ labels: Z = a / s + b   (least-squares fit of Z on 1/s)
- multi_target: focal-length calibration from a reference object, then
  per-target k derived via k = f * real_width_mm.

Size metrics: h_px (bbox height) or w_px (bbox width).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def _h_px(det: dict, video_height: int) -> float:
    """Extract bbox height in pixels from a normalized detection dict."""
    return det["box"]["height"] * video_height


def _w_px(det: dict, video_width: int) -> float:
    """Extract bbox width in pixels from a normalized detection dict."""
    return det["box"]["width"] * video_width


def _get_size(det: dict, video_resolution: dict, size_metric: str) -> float:
    """Extract the configured size signal from a detection."""
    if size_metric == "w_px":
        return _w_px(det, video_resolution["width"])
    return _h_px(det, video_resolution["height"])


def _fit_single_class(pairs: list[tuple[float, float]]) -> dict:
    """Fit a k_over_s or linear_inv model from (s, z_mm) pairs."""
    if len(pairs) == 1:
        s_cal, z_cal = pairs[0]
        k = z_cal * s_cal
        return {"type": "k_over_s", "k": k}

    n = len(pairs)
    xs = [1.0 / s for s, _ in pairs]
    ys = [z for _, z in pairs]

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        k_mean = sum(z * s for s, z in pairs) / n
        logger.warning(f"Z calibration: degenerate regression, falling back to k={k_mean:.1f}")
        return {"type": "k_over_s", "k": k_mean}

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n
    return {"type": "linear_inv", "a": a, "b": b}


def calibrate(
    labels: list[dict],
    frames: list[dict],
    video_resolution: dict,
    class_name: str = "crane hook",
    size_metric: str = "h_px",
    targets: list[dict] | None = None,
    reference_real_width_mm: float | None = None,
) -> dict:
    """
    Build a calibration model from user-provided ground-truth labels.

    Args:
        labels: List of {"frame_number": int, "z_mm": float, "detection_index": int}
        frames: Full frame list from result.json
        video_resolution: {"width": int, "height": int}
        class_name: Reference class name (used for label detection lookup)
        size_metric: "h_px" or "w_px"
        targets: Optional list of {"class_name": str, "real_width_mm": float}.
            When provided together with reference_real_width_mm, enables
            multi-target focal-length calibration.
        reference_real_width_mm: Real-world size of the reference object (mm).
            Required when targets is set.

    Returns:
        Single-class: {"type": "k_over_s"|"linear_inv", ...}
        Multi-target:  {"type": "multi_target", "focal_length_px": f,
                        "targets": [{"class_name": str, "model": {...}}, ...]}
    """
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
        s = _get_size(det, video_resolution, size_metric)
        if s <= 0:
            logger.warning(f"Z calibration: zero/negative size for frame {fn}, skipping")
            continue

        pairs.append((s, z_mm))

    if len(pairs) == 0:
        raise ValueError("No valid calibration pairs found")

    # --- Multi-target focal-length mode ---
    if targets and reference_real_width_mm:
        target_classes = {t["class_name"] for t in targets}
        if class_name not in target_classes:
            targets = [*targets, {"class_name": class_name, "real_width_mm": reference_real_width_mm}]
            logger.info(f"Z calibration: auto-added reference class '{class_name}' as target")

        if len(pairs) == 1:
            s_cal, z_cal = pairs[0]
            f = z_cal * s_cal / reference_real_width_mm
        else:
            ref_model = _fit_single_class(pairs)
            if ref_model["type"] == "k_over_s":
                f = ref_model["k"] / reference_real_width_mm
            else:
                f = ref_model["a"] / reference_real_width_mm

        target_models = []
        for tgt in targets:
            cn = tgt["class_name"]
            w = tgt["real_width_mm"]
            scale = reference_real_width_mm / w
            scaled_pairs = [(s * scale, z) for s, z in pairs]
            tgt_model = _fit_single_class(scaled_pairs)
            target_models.append({"class_name": cn, "model": tgt_model})
            logger.info(
                f"Z calibration: target '{cn}' (w={w}mm) → {tgt_model['type']}"
            )

        logger.info(f"Z calibration: multi-target, f={f:.1f}px, {len(target_models)} target(s)")
        return {
            "type": "multi_target",
            "focal_length_px": round(f, 2),
            "targets": target_models,
        }

    # --- Legacy single-class mode ---
    model = _fit_single_class(pairs)
    if model["type"] == "k_over_s":
        logger.info(f"Z calibration: 1-point model, k={model['k']:.1f}")
    else:
        logger.info(f"Z calibration: {len(pairs)}-point linear model, a={model['a']:.1f}, b={model['b']:.1f}")
    return model


def estimate(
    model: dict,
    frames: list[dict],
    video_resolution: dict,
    class_name: str = "crane hook",
    size_metric: str = "h_px",
) -> list[dict]:
    """
    Apply a calibration model to estimate Z for all frames.

    Modifies detection dicts in-place by adding "z_mm" field to matching detections.
    Returns the modified frames list.
    """
    model_type = model["type"]

    if model_type == "multi_target":
        total = 0
        for tgt in model["targets"]:
            count = _estimate_single(
                tgt["model"], frames, video_resolution,
                tgt["class_name"], size_metric,
            )
            total += count
        logger.info(
            f"Z estimation: applied to {total} detections across {len(frames)} frames "
            f"({len(model['targets'])} target classes)"
        )
        return frames

    count = _estimate_single(model, frames, video_resolution, class_name, size_metric)
    logger.info(
        f"Z estimation: applied to {count} detections across {len(frames)} frames"
    )
    return frames


def _estimate_single(
    model: dict,
    frames: list[dict],
    video_resolution: dict,
    class_name: str,
    size_metric: str,
) -> int:
    """Apply a k_over_s or linear_inv model to detections of one class. Returns count."""
    model_type = model["type"]
    estimated_count = 0

    for frame in frames:
        for det in frame.get("detections", []):
            if det.get("class_name") != class_name:
                continue

            s = _get_size(det, video_resolution, size_metric)
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

    return estimated_count


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

    size_metric = z_cal.get("size_metric", "h_px")
    targets = z_cal.get("targets")
    reference_real_width_mm = z_cal.get("reference_real_width_mm")

    model = calibrate(
        labels, frames, video_resolution,
        class_name=class_name,
        size_metric=size_metric,
        targets=targets,
        reference_real_width_mm=reference_real_width_mm,
    )

    z_cal["model"] = model
    z_cal["class_name"] = class_name

    frames = estimate(
        model, frames, video_resolution,
        class_name=class_name,
        size_metric=size_metric,
    )

    data["z_calibration"] = z_cal
    data["frames"] = frames

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Z estimation complete, saved to {result_path}")
    return model
