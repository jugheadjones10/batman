"""
SAM3 worker process: load model once, read JSON lines from stdin, write JSON lines to stdout.

Run with: uv run python -m backend.app.services.sam_worker

Protocol:
- Parent sets env: BATMAN_SAM_DEVICE, BATMAN_SAM_MODEL_PATH
- stdout line 1: {"ready": true, "device": "0"} on success
- stdin: one JSON object per line: {"image_path": "...", "class_prompts": [...]}
- stdout: one JSON object per line: {"detections": [...], "error": null}
- On fatal error the process exits non-zero.

IMPORTANT: This module deliberately avoids importing backend.app.config or any
heavy batman modules at top level. The LD_PRELOAD allocator must be active before
any CUDA / PyTorch code runs.
"""

import json
import os
import sys
from pathlib import Path


def _mask_to_bbox(mask, img_width: int, img_height: int) -> dict | None:
    """Convert binary mask to normalized bounding box (center xywh)."""
    import numpy as np
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    cx = (x1 + x2) / 2 / img_width
    cy = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return {"x": float(cx), "y": float(cy), "width": float(w), "height": float(h)}


def _mask_to_polygon_norm(mask, img_width: int, img_height: int):
    """Extract the largest contour of a binary mask and normalise to [0,1]."""
    try:
        import cv2
        import numpy as np
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
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) <= 0:
            return None
        simplified = cv2.approxPolyDP(biggest, epsilon=1.5, closed=True)
        pts = simplified.reshape(-1, 2)
        if len(pts) < 3:
            return None
        return [[float(x) / img_width, float(y) / img_height] for x, y in pts]
    except Exception:
        return None


def _mask_bbox_norm(mask, img_width: int, img_height: int) -> dict | None:
    """Binary mask → normalised center xywh bbox (used when SAM3 returns mask but no box)."""
    return _mask_to_bbox(mask, img_width, img_height)


def run_one(predictor, image_path: Path, class_prompts: list[str]) -> list[dict]:
    """Run SAM3 on one image; return list of detections."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    detections = []

    predictor.set_image(str(image_path))

    for class_id, prompt in enumerate(class_prompts):
        try:
            results = predictor(text=[prompt])
            if not results or len(results) == 0:
                continue
            result = results[0]

            mask_arrays = None
            if result.masks is not None and len(result.masks) > 0:
                mask_arrays = result.masks.data.cpu().numpy()

            if result.boxes is not None and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    xyxy = result.boxes.xyxy[i].cpu().numpy()
                    conf = (
                        float(result.boxes.conf[i].cpu().numpy())
                        if result.boxes.conf is not None
                        else 1.0
                    )
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2 / width
                    cy = (y1 + y2) / 2 / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    det = {
                        "box": {"x": float(cx), "y": float(cy), "width": float(w), "height": float(h)},
                        "confidence": conf,
                        "class_id": class_id,
                    }
                    # Attach polygon (derived from the matching mask) when SAM3 returned masks.
                    if mask_arrays is not None and i < len(mask_arrays):
                        poly = _mask_to_polygon_norm(mask_arrays[i], width, height)
                        if poly is not None:
                            det["polygon"] = poly
                    detections.append(det)
            elif mask_arrays is not None:
                for mask in mask_arrays:
                    bbox = _mask_to_bbox(mask, width, height)
                    if bbox:
                        det = {"box": bbox, "confidence": 1.0, "class_id": class_id}
                        poly = _mask_to_polygon_norm(mask, width, height)
                        if poly is not None:
                            det["polygon"] = poly
                        detections.append(det)
        except Exception:
            continue

    return detections


def main() -> int:
    # Read config from env vars directly (avoid importing backend.app.config which
    # can trigger CUDA init before the LD_PRELOAD allocator is fully active).
    model_path = os.environ.get("BATMAN_SAM_MODEL_PATH", "sam3.pt")
    device = os.environ.get("BATMAN_SAM_DEVICE", "0")

    sys.stderr.write(f"sam_worker: device={device} model={model_path} LD_PRELOAD={os.environ.get('LD_PRELOAD', 'none')}\n")
    sys.stderr.flush()

    # Capture the real stdout for our JSON protocol. Redirect sys.stdout to stderr
    # so Ultralytics' banner/warning prints don't pollute the JSON channel.
    _real_stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    sys.stdout = sys.stderr

    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = {
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "model": model_path,
        "half": False,
        "save": False,
        "device": device,
        "verbose": False,
    }

    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        sys.stderr.write(f"sam_worker: failed to load model: {e}\n")
        return 1

    sys.stderr.write("sam_worker: model loaded, signaling ready\n")
    sys.stderr.flush()
    _real_stdout.write(json.dumps({"ready": True, "device": device}) + "\n")
    _real_stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            image_path = Path(req["image_path"])
            class_prompts = req.get("class_prompts") or []
        except (KeyError, json.JSONDecodeError) as e:
            _real_stdout.write(json.dumps({"detections": None, "error": str(e)}) + "\n")
            _real_stdout.flush()
            continue

        if not image_path.exists():
            _real_stdout.write(json.dumps({"detections": None, "error": f"image not found: {image_path}"}) + "\n")
            _real_stdout.flush()
            continue

        try:
            detections = run_one(predictor, image_path, class_prompts)
            _real_stdout.write(json.dumps({"detections": detections, "error": None}) + "\n")
            _real_stdout.flush()
        except Exception as e:
            _real_stdout.write(json.dumps({"detections": None, "error": str(e)}) + "\n")
            _real_stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
