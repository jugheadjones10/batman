# Segmentation & Skew Angle (RF-DETR-Seg)

Batman can optionally swap its detector for **RF-DETR-Seg**, the instance-segmentation sibling of the default RF-DETR detector. When segmentation is enabled, each detection carries a pixel-level mask in addition to the familiar axis-aligned bounding box. That mask unlocks one feature in particular: measuring the **skew angle** between the spreader and the container underneath it — something an axis-aligned bbox cannot express, because bboxes are, by construction, always orthogonal to the image axes.

This guide explains the pipeline end-to-end, the polygon schema that flows through it, and the math behind `skew_deg`. If you are coming from the [Z-axis guide](z-axis-height-estimation.md), think of this as the angular counterpart: `z_mm` answers *how far*, `skew_deg` answers *how twisted*.

## Why masks

A rectangular bounding box has four degrees of freedom (cx, cy, w, h). Rotation is not one of them. Two containers stacked perfectly vs. rotated 20° give the same bbox. To recover rotation we need a shape that is *not* constrained to the image axes; the simplest such shape is a polygon derived from an instance mask. RF-DETR-Seg produces that polygon natively.

Only the `spreader` and `container` classes get masks through the pipeline — every other class stays detection-only. This keeps labelling cheap and the dataset small, while still enabling the one geometric measurement we actually care about.

## The pipeline

```
 Video ──► RF-DETR-Seg ──► supervision.Detections(xyxy, mask, class_id) ─┐
                                                                          │
 For each detection:                                                      │
   mask ─► largest contour ─► cv2.approxPolyDP ─► normalised polygon ─────┤
                                                                          ▼
                              Detection(box, mask=[[x,y], ...])
                                                                          │
                              save_results_json → result.json
                                                                          │
                              backend.app.services.skew_estimator ─► adds
                                  skew_deg / spreader_deg / container_deg
                                  per frame + top-level "skew" summary
                                                                          ▼
                              Frontend LiveDetectionReadout renders °
```

The key components:

| Stage                         | File                                                   | What it does                                                                                 |
| ----------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Model loader                  | `src/core/trainer.py` (`resolve_rfdetr_class`)         | Dispatches on `(task, size)` to either the RF-DETR family or the RFDETR-Seg family.          |
| Training config               | `TrainingConfig.task`                                  | New field: `"detection"` (default) or `"segmentation"`. Threaded through `cli/train.py`.     |
| Label polygon capture         | `backend/app/services/sam_worker.py`, `sam_labeler.py` | SAM3 already returns masks; we simplify the largest contour into a normalised polygon.       |
| Annotation storage            | `backend/app/api/annotations.py`                       | `AnnotationCreate / Update / Info` carry an optional `polygon: list[list[float]]`.           |
| COCO export                   | `src/core/trainer.py` and `backend/app/services/dataset_exporter.py` | Emit `segmentation` for every annotation. Real polygon when present, rectangular fallback otherwise. |
| Inference mask extraction     | `backend/app/services/inference_runner.py`, `src/core/inference.py` | `Detection.mask` is set from `sv.Detections.mask` via the same simplification routine.       |
| Rendering                     | `src/core/inference.draw_detections`                   | Draws a translucent fill + polygon outline alongside the bbox.                               |
| Skew estimation               | `backend/app/services/skew_estimator.py`               | Computes per-frame `skew_deg` and a run-level summary.                                       |
| Overlay                       | `frontend/src/components/LiveDetectionReadout.tsx`     | Adds a "Spreader ↔ Container Skew" card below the class readouts.                            |

## Polygon schema

Wherever a polygon travels, it is **normalised to `[0,1]`** as a list of `[x, y]` vertices — never pixels, never absolute coordinates. This makes polygons resolution-independent and safe to serialise through JSON.

```json
// Stored on AnnotationInfo (user-drawn or SAM3-auto)
{
  "id": 42,
  "class_name": "container",
  "box":   { "x": 0.51, "y": 0.47, "width": 0.32, "height": 0.29 },
  "polygon": [
    [0.37, 0.33], [0.64, 0.34], [0.66, 0.59], [0.38, 0.58]
  ]
}

// Stored on Detection (at inference time, same shape but named "mask")
{
  "bbox": [/* xyxy in pixels */],
  "class_name": "spreader",
  "mask": [[0.41, 0.28], [0.62, 0.29], [0.63, 0.54], [0.40, 0.55]]
}
```

Two minor differences between the two carriers:

- User annotations use the key `polygon`; inference results use `mask`. `skew_estimator._get_polygon` accepts either.
- Polygons have **at least 3 vertices**. The mask-to-polygon helpers refuse to emit anything shorter; you will never see a 0- or 2-vertex polygon in a valid result.

## Mask → polygon

Given a binary mask `H×W`, the normalisation pipeline is:

1. Convert to `uint8` (non-zero ⇒ 1).
2. `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)` — pick the **largest** contour by area. An instance mask can, in rare occlusion cases, produce multiple disconnected components; we keep the dominant blob.
3. `cv2.approxPolyDP(epsilon=1.5)` — simplify to keep `result.json` small. 1.5 pixels is chosen to preserve container corners (which are sharp) while discarding scan-line noise along straight edges.
4. Normalise every vertex: `[x / img_w, y / img_h]`.

If any step fails — empty mask, degenerate contour, less than 3 vertices after simplification — the polygon is dropped and the detection keeps just its bounding box. Nothing else in the pipeline is disrupted.

## Skew math

### Orientation of one polygon

We use `cv2.minAreaRect`, which fits the **smallest rotated rectangle** enclosing a set of points. It returns `((cx, cy), (w, h), angle)` where `angle` is the rotation of the `w` side in degrees. In OpenCV's convention, `angle ∈ [-90, 0)` (pre-4.5) or `[0, 90)` (4.5+). We normalise both conventions to a single canonical form: the **long-axis angle**, wrapped into `(-90, 90]` with 0 being horizontal.

```python
def _canonical_angle(angle, w, h):
    long_axis = angle + 90 if w < h else angle        # long axis, not min-edge
    return ((long_axis + 90) % 180) - 90              # wrap to (-90, 90]
```

Why `(-90, 90]` and not `[0, 360)`? A spreader and a container are **symmetric under a 180° flip** — they don't have a "head" and "tail". Their orientation is a line, not an arrow. Wrapping modulo 180 removes the false 180° jumps that would otherwise dominate the skew reading as the spreader rotates past horizontal.

### Skew between two polygons

Given `spreader_deg` and `container_deg`, the skew is just their signed difference, wrapped back into `(-90, 90]`:

```python
skew_deg = ((container_deg - spreader_deg + 90) % 180) - 90
```

Positive values mean the container is rotated counter-clockwise relative to the spreader (following OpenCV's image-coordinate convention where y grows downward). Landing a TEU typically needs `|skew_deg|` below about 1–2°; anything larger means operator intervention.

### Picking a detection per class

A single frame can contain multiple spreaders / containers (e.g. a stacker with twin hooks, or several containers in the yard). `compute_skew` resolves the ambiguity by picking the **largest-area** detection of each class as the canonical one — area is a reasonable proxy for "nearest to camera" / "most visible". If either class is missing from the frame, or its polygon is degenerate, no skew is reported for that frame.

## Training a segmentation model

1. Flip the **Task** toggle on the Training page to **Segmentation**. The model grid greys out sizes that don't have a seg variant (currently just "base"); you'll usually pick `medium`.
2. Label polygons on ≥~200 frames of spreader + container. The auto-label button uses SAM3 and already stores polygons for these two classes; for the rest of your classes it still falls back to bboxes. You can tune prompts from the class-description dialog.
3. Submit training (local or GPU cluster). The COCO exporter will attach a `segmentation` field to every annotation — the real polygon when you have one, a rectangular fallback (from the bbox) when you don't. This keeps the COCO file valid for RF-DETR-Seg even on detection-only classes.
4. After training, run inference from the Inference page. `result.json` will contain per-detection `mask` fields plus per-frame `skew_deg`. The live readout will show the skew as soon as a frame contains both a spreader and a container.

## Sanity check without training

`scripts/spike_rfdetr_seg.py` runs the **pretrained** RFDETRSeg checkpoint on a video and saves a `result.json` with masks + a `skew` summary. COCO-pretrained weights won't know about the `spreader`/`container` classes specifically, so the skew block will often be empty — but you'll still see `"mask": [...]` populated on at least some detections, which confirms the plumbing.

```bash
uv run python scripts/spike_rfdetr_seg.py \
    --video path/to/sample.mp4 \
    --output ./spike_out \
    --model medium
```

## Related reading

- [Distance Calibration (Z-axis)](z-axis-height-estimation.md) — the `z_mm` counterpart to `skew_deg`. Both live side-by-side in `result.json` and the overlay.
- [Training Workflow](training.md) — where the task toggle and image-size rounding rules live.
- [Inference Workflow](inference.md) — end-to-end inference, including the `result.json` schema.
