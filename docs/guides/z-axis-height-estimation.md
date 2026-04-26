# Distance Calibration (Z-axis)

Batman turns bounding boxes into **real-world distance** (`z_mm`). Given a detection with a pixel-sized box, the system reports how far the object is from the camera — typically used to read crane-hook position along the optical axis, or to measure how far away a shipping container is in a top-down yard view.

This guide explains the intuition, the math, and the three calibration modes the system actually ships.

> For the angular counterpart — how *twisted* the container is relative to the spreader — see [Segmentation & Skew Angle](segmentation-and-skew.md). `z_mm` and `skew_deg` live in the same `result.json` and render side-by-side in the live overlay.

## The idea in one picture

A camera is a pinhole. Light from a real object of size `S` passes through the pinhole and lands on the image plane at size `s` pixels. Two similar triangles share the pinhole:

```
       real object
       (S wide/tall)
        ┌─────┐
        │      `.
        S        `.
        │          `.   image plane
  ──────┴────────────●─────┐
                   pinhole │
                           s  ← bbox size in pixels
                           │
                           └────
        |────── Z ──────|──δ──|
        distance to object    focal length (pixels)
```

Similar triangles ⇒ `S / Z = s / δ`, which rearranges to

```
Z = (δ · S) / s
```

For a given camera + object, `δ` (focal length) and `S` (the object's real size) are **constants**. The only thing that changes frame-to-frame is `s`, the bbox size in pixels. So distance is always a constant over a per-detection measurement:

```
Z = k / s          where k = δ · S
```

Every calibration mode in Batman is a variation on this one equation. `**k` is fit once, `s` is measured per detection.**

## What `s` actually is, in code

### First, what's a bbox?

A **bounding box** (bbox) is what the object detector outputs for each thing it finds in a frame: the axis-aligned rectangle that tightly encloses the object. Every detection Batman produces has one, along with a class label and a confidence score:

```
┌──────────────── video frame (1920 × 1080 px) ─────────────────┐
│                                                                │
│         ┌─────────────┐ ← bounding box                         │
│         │   🪝         │                                        │
│         │  crane_hook │   the detector returns a rectangle    │
│         │   0.94      │   around the object it detected        │
│         └─────────────┘                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

Batman stores every bbox in **normalised coordinates** — four numbers in the `[0, 1]` range:

```json
{"class_name": "crane_hook", "confidence": 0.94,
 "box": {"x": 0.49, "y": 0.27, "width": 0.06, "height": 0.15}}
```

- `x`, `y` — centre of the box as a fraction of frame width/height
- `width`, `height` — box dimensions as a fraction of frame width/height

Normalising by frame size means the same bbox numbers describe the same real region whether the video is played back at 1920 × 1080 or 960 × 540. It's resolution-independent.

### Converting back to pixels

The pinhole math in the previous section works in pixels — similar triangles relate the pixel size on the sensor to the real size in the world. So to use a bbox in the equation `Z = k/s`, we multiply the normalised dimension by the video's actual pixel resolution to turn the fraction back into pixels:

```
s = bbox.height × video_height_px   # e.g. 0.15 × 1080 = 162 px
s = bbox.width  × video_width_px    # e.g. 0.06 × 1920 = 115 px
```

That's all "multiplying by `video_height_px`" is doing: **undoing the normalisation** so `s` has units the pinhole model expects (pixels on the image sensor).

### `s` is per-detection; `k` is fit once

`s` is the **bbox size of a single detection, measured in pixels**. It's a per-detection number — big when the object is close, small when it's far — and it's the only runtime input the calibration model sees. Everything else (`k`, or later `m` and `c`) is fit once during calibration and then held fixed.

Batman always takes `s` to be the **longer side** of the bbox:

```
s = max(bbox.width × video_width_px,
        bbox.height × video_height_px)
```

The longer side has the largest pixel extent (so quantisation noise is the smallest fraction of the signal) and it's insensitive to rotations around the short axis, which makes it by far the cleaner pinhole input for spreaders and containers. There's no axis selector in the UI — the system picks for you on every detection.

The table below makes the equation concrete for a `Z = k/s` model with `k = 750 000 mm·px`:


| Detection           | bbox size `s` | Estimated distance `Z = k/s` |
| ------------------- | ------------- | ---------------------------- |
| object far away     | 50 px         | 15 000 mm                    |
| object at mid-range | 100 px        | 7 500 mm                     |
| object close        | 250 px        | 3 000 mm                     |


Same `k`, different `s`, different `Z` — exactly what you'd expect from similar triangles.

## Calibration labels

To fit `k` (or `m, c`) we need **examples** — frames where the true distance is already known. In the UI you provide each example as a pair `(frame_number, z_mm)`: the frame you're pointing at, and the distance from the camera to the reference object in that frame, in millimetres (measured with a tape, laser, PLC readout, etc.).

Internally the system turns each label into an `(s, z)` pair by looking up the reference detection's bbox in that frame and computing `s` from it (same formula as above). `s` is the bbox size in pixels, `z` is your `z_mm`. These `(s, z)` pairs are what every calibration mode below actually fits against.

The number of pairs picks the mode:

| Pairs | Fit |
|---|---|
| 1 | `Z = k / s` |
| 2+ | `Z = m / s + c` |
| any, with a target list | the same fit, applied to every target class |

---

## Mode 1 — Single class, 1 point

**Model:** `Z = k / s`

With a single `(s_cal, z_cal)`, there's exactly one free parameter:

```
k = z_cal · s_cal
```

The fit passes precisely through your one calibration point and assumes the pinhole relation is clean. This works well when the object operates near the calibration distance.

Use this for quick "is it roughly right?" checks, or when you only have one frame you're confident about.

## Mode 2 — Single class, 2+ points

**Model:** `Z = m · (1/s) + c`

With two or more points, the estimator fits a **line** through the points `(x, y) = (1/s, z)` using ordinary least squares (closed form). In `1/s` space the pinhole relationship is linear, so OLS gives you the right answer.

```
denom = n·Σx² − (Σx)²
m = (n·Σxy − Σx·Σy) / denom
c = (Σy − m·Σx) / n
```

### Why the intercept `c` matters

Real cameras, real detectors, and real labelling always carry small systematic biases:

- the detector consistently cuts off a few pixels at the hook tip
- the point you measured with a tape measure isn't the geometric centre of the bbox
- there's a fixed offset between the camera's optical centre and the reference plane

A pure `Z = k/s` cannot say "everything is shifted by 20 cm". `Z = m/s + c` can, and the intercept `c` is exactly that shift. In practice the 2+ point fit noticeably outperforms 1-point anywhere away from the calibration distance — often by a factor of 2–3×.

### Degenerate case

If all your labels land at near-identical `s` values (`denom < 1e-12`), the linear system is ill-conditioned. The estimator logs a warning and falls back to a 1-point model with `k = mean(z · s)` over all labels. Spread your calibration points across the object's operating range and this won't happen.

## Mode 3 — Multi-target (shared fit across spreader and containers)

The case: **the object you can calibrate on isn't the only object you want distances for.**

Canonical example. A camera mounted on a crane trolley looks down at both a **spreader** and the **shipping container** it's picking. You can measure the distance to the spreader from the PLC hoist readout. You can't easily measure distance to a container mid-lift. Luckily, you don't need to — a modern telescoping spreader extends to **match the length of the container it's engaged with**, so along the long axis of the bbox they share the exact same real-world dimension.

### The key simplification

For Batman's setup, **the reference class and every target class share the same real-world length** `ℓ`. The spreader telescopes to 20 / 40 / 45 ft to lock onto the matching container, so both objects genuinely have the same `ℓ` frame after frame. Combined with the longer-bbox-side rule for `s`, the pinhole constant `k = δ · ℓ` is identical for the spreader and every container class.

Whatever fit you run on the spreader (Mode 1 or Mode 2) transfers **directly** to every target — no per-target rescaling, no per-class model.

### The fit

Run Mode 1 or Mode 2 exactly as described above on the reference class (the one whose distance you can measure directly, i.e. the spreader). Then apply the resulting model to every detection whose class is in the target list:

- 1 label → every target uses `Z = k / s`
- 2+ labels → every target uses `Z = m / s + c`

There is no separate `(k_target, m_target, c_target)`. One model drives all classes — the same `k`, or the same `(m, c)` — because they all share `ℓ`.

The bias intercept `c` from Mode 2 carries across classes for free, too: `c` encodes a camera/detector bias (the few-pixel bbox clip, the tape-measure offset) that has nothing to do with which class is in the frame, so it applies identically to every target.

### Picking `ℓ` from the UI

In the calibration panel, `ℓ` is a dropdown of the three ISO container lengths:


| Container | `ℓ` (dropdown value) |
| --------- | -------------------- |
| 20 ft     | 6058 mm              |
| 40 ft     | 12192 mm             |
| 45 ft     | 13716 mm             |


You pick the length of the container being lifted; Batman assumes the spreader has telescoped to match and applies the resulting fit to every class in the target list. `ℓ` itself never appears in the math at estimation time — it's baked into `k` (or `m`) during calibration. The stored value is informational: it tells you which container size this calibration was derived on.

### Fallback behaviour

If no length is selected and the target list is empty, the system fits a plain single-class model on the reference class only. The simple case stays simple.

---

## Using it from the UI

Open a finished inference run on the **Inference** page and expand the **Z-Axis Calibration** panel.

1. **Container length (ℓ)** — dropdown of 20 / 40 / 45 ft. Sets the shared real-world size for the reference and every target. Leave blank to run a plain single-class fit on the reference with no targets.
2. **Reference Class** — the class you'll calibrate with (the one whose distance you can actually measure; typically the spreader).
3. **Estimation Targets** — one row per additional class you want distances for. They all inherit the same fit as the reference; there is no per-target size field. The reference class is auto-added if missing.
4. **Calibration Points** — for each calibration frame, enter the frame number and ground-truth distance in mm. 1 works; 2+ is better.
5. **Calibrate & Estimate** — fits the model, writes `z_mm` onto every matching detection in every frame, and persists into the run's `result.json`.
6. **Re-export Video with Z** — re-encodes the annotated video with the distance overlays baked in.

!!! warning "Length-sharing assumption"
    Every class in the target list must genuinely share the same real-world length `ℓ` as the reference. A telescoping spreader locked onto a container satisfies this by construction. A free-floating container at a different ISO length, or a bare spreader not yet engaged, does not — exclude those frames or re-calibrate with the correct `ℓ`.

!!! tip "Which mode should I pick?"
    | Situation | Mode |
    |---|---|
    | You only have one confident reference frame | 1 point |
    | You can label 2+ frames spanning the operating range | 2+ points |
    | The object you want distances for is not the one you labelled | Multi-target |

### Side-view schematic (debugging visual)

The right-hand column of a calibrated inference run also renders a **Side-View Schematic** card: a live elevation diagram of camera → spreader → container that updates as the video plays. Pick the spreader and container classes with the two dropdowns; the container's length is read straight from the calibration's `length_mm` (falling back to an aspect-ratio inference if the calibration is absent), the vertical height is the ISO-standard 2591 mm, and the card reports the four distances (camera→spreader, spreader→container-top, container height, camera→container-bottom). Use it as a quick sanity check on whether the calibrated z values produce physically plausible stacking.

---

## Persisted shape (`result.json`)

Once calibrated, a run's `result.json` gains a `z_calibration` block and every matching detection gains a `z_mm` field. For a multi-target run it looks like:

```json
{
  "z_calibration": {
    "labels": [
      {"frame_number": 142, "z_mm": 12000, "detection_index": 0},
      {"frame_number": 487, "z_mm": 4500,  "detection_index": 0}
    ],
    "reference_class": "spreader",
    "length_mm": 12192,
    "targets": ["spreader", "container"],
    "video_resolution": {"width": 1920, "height": 1080},
    "model": {"type": "linear_inv", "m": 7606790.0, "c": -30.2}
  },
  "frames": [
    {
      "frame_number": 100,
      "detections": [
        {"class_name": "container", "box": {...}, "confidence": 0.94, "z_mm": 8210.3}
      ]
    }
  ]
}
```

Single-class runs have the same `model` shape (`k_over_s` for 1-label, `linear_inv` for 2+); they just leave `targets` empty (or equal to `[reference_class]`) and estimate only the reference class. `length_mm` is informational — it documents which container size `k` / `m` were derived on, and is not used at estimation time.

---

## Source map

- `backend/app/services/z_estimator.py`
  - `_longer_side_px()` — the single source of truth for `s` (max of bbox width and height in pixels).
  - `calibrate()` — builds `(s, z)` pairs on the reference class and returns one flat model.
  - `_fit_single_class()` — closed-form OLS on `(1/s, z)`, with the 1-label shortcut and degenerate fallback.
  - `estimate()` — applies the flat model to every detection whose class is in `target_classes`, writing `z_mm` in-place.
  - `apply_z_to_result()` — end-to-end: read `result.json`, fit, estimate, write back; raises on legacy schemas.
- `backend/app/api/inference.py` — REST endpoints: save calibration, apply estimation, re-export video.
- `frontend/src/components/ZCalibrationPanel.tsx` — the calibration UI.
- `frontend/src/pages/ZCalibrationPage.tsx` — the full-screen frame picker for selecting calibration frames.
- `frontend/src/components/SideViewSchematic.tsx` — the live elevation diagram of camera / spreader / container on the inference detail page.

---

## Practical tips

- **Spread your calibration points.** Two labels at nearly the same distance are effectively one label — and trigger the degenerate fallback. Pick the shortest and longest distances you care about.
- **Prefer 2+ points.** The intercept `c` is where most of the real-world accuracy comes from.
- **Don't extrapolate far beyond your labels.** `Z = m/s + c` is a linear fit in `1/s`. Well outside the calibration range the linearisation drifts and errors grow.
- **Pick the container length you're lifting.** 20 / 40 / 45 ft are the three standard ISO options. The spreader telescopes to match, so a single selection covers both.
- **Reference class as a target.** If you want distances for the reference class itself, include it — but the system adds it automatically if you forget.
- **Re-calibrate if the camera moves or the container size changes.** The model bakes the camera geometry *and* the chosen `ℓ` into `k` / `m` / `c`; any physical change to mounting, zoom, or ISO length invalidates the fit.

