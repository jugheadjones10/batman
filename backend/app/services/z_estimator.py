"""
Z-axis distance estimation from bounding box sizes.

Pinhole model (see docs/guides/z-axis-height-estimation.md for the full
derivation): a detection with pixel-size ``s`` and a real-world size ``ℓ`` along
the camera's optical axis-aligned dimension satisfies ``Z = δ · ℓ / s`` with δ
the camera's focal length. Fold ``δ · ℓ`` into a single constant ``k`` and the
model is ``Z = k / s``. With 2+ labels we fit a bias-correcting line in 1/s,
``Z = m · (1/s) + c``.

Because Batman's use case lifts spreaders whose telescoping length matches the
container being picked, every class the user cares about shares the same ``ℓ``
and therefore the same fit. There is no per-target rescaling — one flat model
is broadcast across every class in ``targets``.

``s`` is always the **longer side** of the axis-aligned bbox (in pixels). That
side is the cleaner signal (largest pixel extent, insensitive to short-side
rotation noise) and removes the axis-picking step from the UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


def _longer_side_px(det: dict, video_resolution: dict) -> float:
    """Return the longer bbox side in pixels (max of width, height)."""
    w = det["box"]["width"] * video_resolution["width"]
    h = det["box"]["height"] * video_resolution["height"]
    return max(w, h)


def _fit_single_class(pairs: list[tuple[float, float]]) -> dict:
    """Fit a k_over_s or linear_inv model from (s, z_mm) pairs.

    1 pair  -> {type: k_over_s, k}
    2+ pairs -> closed-form OLS of z on 1/s -> {type: linear_inv, m, c}

    Falls back to the 1-point form when the 2+ labels land at nearly the same
    ``s`` (ill-conditioned linear system).
    """
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

    m = (n * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - m * sum_x) / n
    return {"type": "linear_inv", "m": m, "c": c}


def calibrate(
    labels: list[dict],
    frames: list[dict],
    video_resolution: dict,
    reference_class: str,
    target_classes: list[str] | None = None,  # noqa: ARG001 (broadcast happens in estimate())
) -> dict:
    """Fit a flat pinhole model from the reference-class labels.

    Args:
        labels: List of {"frame_number": int, "z_mm": float, "detection_index": int}
        frames: Full frame list from result.json
        video_resolution: {"width": int, "height": int}
        reference_class: Class name whose detections provide the (s, z) pairs.
        target_classes: Class names the model will be applied to downstream.
            Accepted here for API symmetry; the fit itself only depends on the
            reference labels.

    Returns:
        Flat model: {"type": "k_over_s", "k": ...}
                 or {"type": "linear_inv", "m": ..., "c": ...}
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
        det: dict | None = None

        if 0 <= det_idx < len(dets):
            indexed = dets[det_idx]
            if indexed.get("class_name") == reference_class:
                det = indexed
            else:
                logger.warning(
                    "Z calibration: frame "
                    f"{fn} detection_index={det_idx} is class '{indexed.get('class_name')}' "
                    f"(expected '{reference_class}'); falling back to first matching "
                    "reference detection"
                )
        else:
            logger.warning(f"Z calibration: detection index {det_idx} out of range for frame {fn}")

        if det is None:
            det = next((d for d in dets if d.get("class_name") == reference_class), None)
            if det is None:
                logger.warning(
                    f"Z calibration: no '{reference_class}' detection in frame {fn}, skipping"
                )
                continue

        s = _longer_side_px(det, video_resolution)
        if s <= 0:
            logger.warning(f"Z calibration: zero/negative size for frame {fn}, skipping")
            continue

        pairs.append((s, z_mm))

    if len(pairs) == 0:
        raise ValueError("No valid calibration pairs found")

    model = _fit_single_class(pairs)
    if model["type"] == "k_over_s":
        logger.info(f"Z calibration: 1-point model, k={model['k']:.1f}")
    else:
        logger.info(
            f"Z calibration: {len(pairs)}-point linear model, m={model['m']:.1f}, c={model['c']:.1f}"
        )
    return model


def estimate(
    model: dict,
    frames: list[dict],
    video_resolution: dict,
    target_classes: list[str],
) -> list[dict]:
    """Apply ``model`` to every detection whose class is in ``target_classes``.

    Modifies detection dicts in-place by adding a rounded ``z_mm`` field.
    Returns the modified frames list.
    """
    if not target_classes:
        logger.warning("Z estimation: empty target_classes, nothing to apply")
        return frames

    target_set = set(target_classes)
    model_type = model.get("type")
    estimated_count = 0

    for frame in frames:
        for det in frame.get("detections", []):
            if det.get("class_name") not in target_set:
                continue

            s = _longer_side_px(det, video_resolution)
            if s <= 0:
                continue

            if model_type == "k_over_s":
                z = model["k"] / s
            elif model_type == "linear_inv":
                z = model["m"] / s + model["c"]
            else:
                continue

            det["z_mm"] = round(z, 1)
            estimated_count += 1

    logger.info(
        f"Z estimation: applied to {estimated_count} detections across {len(frames)} frames "
        f"({len(target_set)} target class(es))"
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


_LEGACY_KEYS = ("size_metric", "reference_real_width_mm", "class_name")


def apply_z_to_result(result_dir: Path) -> dict:
    """Load the calibration from ``result.json``, fit, estimate, save.

    Returns the fitted model. Raises a clear ValueError on legacy schema so the
    user knows to re-run the (now simpler) calibration flow.
    """
    result_path = result_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"result.json not found at {result_dir}")

    with open(result_path) as f:
        data = json.load(f)

    z_cal = data.get("z_calibration")
    if z_cal is None:
        raise ValueError("No z_calibration found in result.json — calibrate first")

    if any(key in z_cal for key in _LEGACY_KEYS):
        raise ValueError(
            "Legacy z_calibration schema detected — please re-calibrate using the "
            "updated UI (length dropdown + longer-side fit)."
        )

    labels = z_cal.get("labels", [])
    frames = data.get("frames", [])

    video_resolution = z_cal.get("video_resolution")
    if video_resolution is None:
        raise ValueError("No video_resolution found in z_calibration")

    reference_class = z_cal.get("reference_class")
    if not reference_class:
        raise ValueError("No reference_class found in z_calibration")

    targets: list[str] = list(z_cal.get("targets") or [])
    if reference_class not in targets:
        targets.append(reference_class)
        logger.info(f"Z calibration: auto-added reference class '{reference_class}' as target")

    model = calibrate(
        labels,
        frames,
        video_resolution,
        reference_class=reference_class,
        target_classes=targets,
    )

    z_cal["model"] = model
    z_cal["reference_class"] = reference_class
    z_cal["targets"] = targets

    frames = estimate(model, frames, video_resolution, target_classes=targets)

    data["z_calibration"] = z_cal
    data["frames"] = frames

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Z estimation complete, saved to {result_path}")
    return model
