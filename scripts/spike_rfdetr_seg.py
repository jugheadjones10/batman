"""Spike: run a pretrained RFDETRSeg model on a video to validate mask plumbing.

No fine-tuning, no labelling — this is purely a smoke test for:
  1. The (task, size) dispatch in ``src.core.trainer.resolve_rfdetr_class``
  2. RF-DETR-Seg returning masks via ``supervision.Detections.mask``
  3. ``_mask_to_polygon_norm_core`` turning those masks into polygons
  4. Polygons landing on the ``Detection.mask`` field and ``result.json``
  5. ``skew_estimator.apply_skew_to_result`` scoring any frames that happen
     to contain both a ``spreader`` and a ``container`` polygon

Usage:
    uv run python scripts/spike_rfdetr_seg.py \\
        --video /path/to/sample.mp4 \\
        --output ./spike_out \\
        [--model medium] \\
        [--confidence 0.5]

The pretrained RFDETRSeg checkpoint ships with generic COCO classes, so the
``skew`` block will usually be empty unless the video happens to contain
objects COCO labels as "spreader" / "container". In that case this script
still validates that polygons are emitted; you should see
``len(det["mask"]) > 0`` on at least some detections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.core.inference import (
    InferenceConfig,
    RFDETRInference,
    save_results_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Path to input video")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for detected.mp4 + result.json",
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["nano", "small", "medium", "large", "xlarge"],
        help="RFDETRSeg size (default: medium)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cuda | cpu | cuda:0 etc.",
    )
    parser.add_argument(
        "--skip-skew",
        action="store_true",
        help="Do not run skew_estimator after inference",
    )
    args = parser.parse_args()

    video: Path = args.video
    out_dir: Path = args.output
    if not video.exists():
        logger.error(f"Video not found: {video}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading RFDETRSeg{args.model.capitalize()} (pretrained, no checkpoint) …")
    inference = RFDETRInference(
        model_size=args.model,
        task="segmentation",
    )
    inference.load_model(device=args.device)

    config = InferenceConfig(
        confidence_threshold=args.confidence,
        enable_tracking=False,
    )

    logger.info(f"Running inference on {video} …")
    results, stats = [], None
    gen = inference.predict_video(video, config=config)
    try:
        while True:
            results.append(next(gen))
    except StopIteration as e:
        stats = e.value

    n_with_mask = sum(1 for r in results for d in r.detections if d.mask)
    n_total = sum(len(r.detections) for r in results)
    logger.info(
        f"Inference complete: {len(results)} frames, "
        f"{n_total} detections, {n_with_mask} with mask polygons."
    )
    if n_with_mask == 0:
        logger.warning(
            "No masks were populated. Check that RFDETRSeg is installed and that "
            "the checkpoint is returning supervision.Detections with .mask set."
        )

    result_path = out_dir / "result.json"
    save_results_json(
        results,
        result_path,
        stats=stats,
        metadata={"spike": True, "video": str(video), "model": f"rfdetr-seg-{args.model}"},
    )

    # Add a fake video_resolution so skew_estimator can run over the file.
    import cv2
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    with open(result_path) as f:
        payload = json.load(f)
    payload["video_resolution"] = {"width": w, "height": h}
    with open(result_path, "w") as f:
        json.dump(payload, f, indent=2)

    if not args.skip_skew:
        from backend.app.services import skew_estimator
        summary = skew_estimator.apply_skew_to_result(
            out_dir, video_resolution={"width": w, "height": h}
        )
        logger.info(f"Skew summary: {json.dumps(summary, indent=2)}")

    logger.info(f"Wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
