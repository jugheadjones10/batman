# Distance Calibration (Z-axis)

Batman turns bounding boxes into **real-world distance** (`z_mm`). Given a detection with a pixel-sized box, the system reports how far the object is from the camera — typically used to read crane-hook position along the optical axis, or to measure how far away a shipping container is in a top-down yard view.

This guide explains the intuition, the math, the measurement sources, and the calibration modes the system actually ships.


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

Run Mode 1 or Mode 2 exactly as described above on the reference class (the one whose distance you can measure directly, i.e. the spreader). Then apply the resulting model to the detections whose class is in the target list:

- 1 label → every target uses `Z = k / s`
- 2+ labels → every target uses `Z = m / s + c`

There is no separate `(k_target, m_target, c_target)`. One model drives all classes — the same `k`, or the same `(m, c)` — because they all share `ℓ`.

The bias intercept `c` from Mode 2 carries across classes for free, too: `c` encodes a camera/detector bias (the few-pixel bbox clip, the tape-measure offset) that has nothing to do with which class is in the frame, so it applies identically to every target.

### Center container selection

When a detector sees every container in the frame, Batman keeps those detections visible but only uses the **container closest to the image center** for container Z and spreader-to-container gap estimates. The chosen container is the target whose bbox center is nearest `(0.5, 0.5)` in normalized frame coordinates.

This matches the crane-camera setup: the intended pickup container should be framed near the center of the video. Side containers still appear in overlays and exports, but they do not receive `z_mm` in a spreader→container calibration run, and they do not drive the side-view schematic, live readout, or Z graph.

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

## Measurement sources

The model always fits against a pixel size `s`, but `s` can now come from either of two sources.

### Whole bbox longer side

This is the historical default:

```
s = max(bbox.width × video_width_px,
        bbox.height × video_height_px)
```

Use it when the whole spreader or reference object is visible and the detector's bbox is stable.

### Round feature → equivalent length

If the whole spreader is partially missing but the small round top feature is visible, the system can detect that feature and scale its pixel diameter into the equivalent whole-spreader/container length:

```
diameter_px = max(round_feature_bbox.width × video_width_px,
                  round_feature_bbox.height × video_height_px)

equivalent_size_ratio = length_mm / round_feature_diameter_mm
s = diameter_px × equivalent_size_ratio
```

`length_mm` is the same 20 / 40 / 45 ft spreader/container length used by the multi-target model. `round_feature_diameter_mm` is the measured physical diameter of the round feature. The ratio is computed and stored in `result.json` for auditability; it is not hardcoded in the app.

This keeps the calibration on the same physical scale as whole container bboxes. During estimation, the round feature class uses the scaled `s`, while container targets still use their own whole-bbox longer side.

---

## Using it from the UI

Open a finished inference run on the **Inference** page and open its calibration page.

The main form has two steps:

1. **Round feature class** — confirm `round` (selected automatically when available). If your model names the feature differently, select that class. Without a detected round feature, run inference with a model trained to detect it.
2. **Known distances** — browse to a frame, click **Add Frame** (or press Space), and enter the camera-to-reference distance in millimetres. One point is the minimum; a second at a different height lets the fit account for a constant offset. These are camera distances, **not** spreader-to-target gaps.

Click **Apply calibration**, then return to inference to view the gap. New calibrations always use round-feature measurement and automatically include the reference and spreader/container classes. Whole-object targets must share the specified container/spreader length. Frame sampling only changes browsing: entered and saved points are retained, including points outside the sampled filmstrip. Clicking one of those points switches to every-frame browsing.

**Advanced settings** contains container length, round-feature diameter, and classes sharing the calibration. There is no measurement-method selector: the page always uses round-feature diameter scaled to equivalent spreader length. Both physical dimensions are required by the current scale-transfer model. Missing dimensions open Advanced and prevent applying an incomplete calibration. Existing round-feature settings, label detection indices, and feature offsets are preserved when recalibrating.

Previously saved whole-object calibrations remain usable in inference. Opening one in this page starts a fresh round-feature setup, retaining its container length when available. Its old distance labels and reference are not reused because they may describe a different measurement plane. The old calibration stays active until new round-feature points and dimensions are submitted with **Apply calibration**. Whole-object math below remains documented for those historical results.

Calibration supplies camera distances; tracking supplies load state and the target plane. For an overhead view, empty clearance is `target top − spreader`, and loaded clearance is `target top − spreader − load height`. The current tracking implementation assumes a **2,591 mm** load height. Occluded targets may be inferred from touchdown and are marked as inferred; calibration alone does not locate them or guarantee correct load classification.

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

## Stacking distance across the crane cycle

A crane move is a repeating cycle: the empty spreader descends onto a pickup container, hoists it, travels, lowers it onto a stack, releases it, and hoists back up empty. Each phase needs a different physical target for the spreader↔container distance — and only one of them (the loaded travel) matches the naive "container at screen center" pick. Batman handles the whole cycle with a frontend analysis (`frontend/src/lib/stackingDistance.ts`) that runs over the smoothed ByteTrack frames on the inference detail page and moves through four states:

```
idle ──(descent starts)──▶ pickup ──(load acquired)──▶ carrying
  ▲                                                        │
  │                                              (descent starts, lock)
  └──────────(put-down detach)──────────── locked ◀────────┘
```

Two spreader roles are resolved independently. The **Z class** is the calibration reference (possibly a small proxy feature such as a round casting, whose bbox stays unclipped and yields trustworthy pinhole Z). The **geometry class** (`resolveGeometrySpreaderClass`) is a detection class literally named like "spreader", used for all *spatial* reasoning; it falls back to the Z class when absent. Every pickup/placement candidate must **overlap the spreader's geometry box** (≥ 20 % of the smaller box): a container that is not under the spreader can never be the physical target, no matter how close it is to the frame center — when nothing under the spreader is detected, the target is honestly `null` instead of a wrong neighbor.

The geometry box is **anchored to the Z-reference feature** (`pickGeometrySpreader`): detectors emit spurious extra spreader boxes on lookalike structures (container stacks seen end-on) whose confidence momentarily exceeds the real spreader's, and picking by confidence alone makes the body box jump across the frame between consecutive frames. Since the reference feature is physically mounted *on* the spreader, the body is the candidate whose box contains it, and containment outranks any confidence margin.

### Merged (coincident) blobs

Detectors routinely emit **one box** covering the spreader and whatever is in line with it, and report that same box under both the spreader and container classes — observed identical to four decimal places on real footage. Such a `container` track is not an independent observation: its box *is* the spreader's box, so it says nothing about whether a load is attached, and every attachment cue passes it trivially. `isMergedWithSpreader` (≥ 80 % overlap of the smaller box, size ratio ≤ 1.3) identifies them, and they are then treated as follows:

- **Never attachment evidence, never detach evidence.** Whether a merged blob is a load or a static container below is decided by the Z-profile sequence alone.
- **Never static background.** A merged blob's apparent motion during a descent is large; counting it as background reads as "the trolley moved" and aborts every lock episode.
- **Never a placement target while carrying** — it is the load, so measuring against it would compare the load to itself and pin the gap near zero for the whole descent.
- **Kept as a target candidate while empty**, where the blob is the best available handle on the container directly beneath the spreader.
- **Adopted as the carried track for display** while carrying, so the overlay labels it as the load.

1. **Idle (empty spreader).** The reference container is the **under-spreader track nearest the spreader**. Selection is sticky (a challenger must be ~20 % closer to steal it) so it doesn't flicker between stacks, and right after a put-down it is seeded with the just-placed container, so the schematic keeps measuring spreader ↔ placed box while the spreader hoists away. The container's Z prefers a **contact-learned plane** (see below) over the live pinhole read; live reads from edge-clipped bboxes are never used.

2. **Pickup lock (empty spreader descending).** When the background container tracks have been still for ~1 s (trolley stopped) and the spreader has been *descending* for ~0.5 s (signed bbox-scale rate: shrinking = moving away from the overhead camera) **and has actually travelled ≥ 100 mm in depth since that streak began**, the reference container is **locked** as the pickup target and its Z is frozen (see below). The absolute-depth requirement matters because the scale-rate test is relative and a hovering spreader fakes sustained streaks from bbox noise, which produced phantom locks during travel. The lock may be **blind** (`targetTrackId = null`) when no container is detected under the spreader — the descent is real even when the detector misses the target — and a visible target is adopted later if one appears.

    The episode ends when the load is acquired (→ carrying), or aborts if the background starts moving again. There is deliberately **no separate ascent-timer abort**: the Z-profile branch below resolves both outcomes from the same trigger (real rise off the deepest plane), marking the episode `aborted` when the approach never really descended (≥ 300 mm). A timer-only rule fired on scale noise during landing plateaus — twist-lock engagement can hold the spreader at the touchdown plane for tens of seconds — and killed otherwise valid episodes.

### Load state: the one thing boxes cannot tell you

Whether the spreader carries a container is **not observable from bounding boxes** in this camera geometry. The detector emits a `container` box coincident with the spreader body in nearly every frame regardless of load — measured on "Stacking 2", the container track sat at overlap 1.00 with the spreader box both at t = 3 s (visibly loaded) and at t = 125 s (visibly empty, spreader frame see-through). The distinction is purely one of appearance, which a box cannot express.

Everything below therefore *infers* the carry state from the crane's motion, and that inference has one irreducible blind spot: a clip that **starts mid-cycle with a load already attached** has no preceding pickup to infer from, so the first placement is mis-read as a pickup (no carried container drawn, and the target plane one ISO height too shallow because the placement offset is not applied).

The fix is at the model: label the spreader body as `spreader_loaded` / `spreader_empty` (see [Training](training.md#spreader-load-state-required-for-stacking-distance)). When such classes are present the analysis treats them as **authoritative** — a direct observation always beats an indirect inference:

- The first reading of the clip is accepted immediately, anchoring the cycle. Nothing else can do this, and the state cannot have changed yet.
- Later disagreements flip the state only after `LOAD_CLASS_CONFIRM_SECONDS` (1 s) of sustained evidence, since per-frame classification flickers around the transition.
- `loaded` while believed empty starts a carry (and completes any open pickup); `empty` while believed carrying releases the load through the normal put-down path. A carry declared this way is treated as already hoisted, since its hoist happened before the clip began — otherwise the put-down could never fire.
- All spreader-body classes are used interchangeably for *spatial* reasoning, so splitting the class does not affect the geometry gates.

`StackingAnalysis.loadStateSource` reports `'class'` or `'inferred'` so it is always clear which regime produced a result.

3. **Carrying — a state, not a track.** In nadir camera views the carried container hangs directly under the spreader and is largely invisible, so carrying can outlive (or never have) a visible carried track. Without load-state classes it is entered two ways:
    - **Cue-based acquisition.** A container track qualifies when its bbox overlaps the spreader's geometry box, it moves in lockstep with the spreader's **geometry (body) box** (velocity *and* signed bbox-scale-rate agreement) — judging lockstep against the small off-center Z-class proxy instead makes a rigidly attached load look detached, since the proxy's image motion during a descent differs from the body's — and it reads at the spreader's depth (skipped for edge-clipped boxes, whose pinhole Z is meaningless). Qualification time only accumulates while the spreader is moving **and not descending** — a load is physically acquired by *hoisting*, and during an empty descent the merged under-spreader blob tracks the spreader frame-perfectly, which must never count as evidence.
    - **Sequence (Z-profile) inference.** During a pickup episode, touchdown followed by sustained re-ascent (the spreader rises ≥ 300 mm above the episode's deepest plane) completes the pickup: the spreader now carries the load even if no container track ever qualified.

    Losing the carried track does **not** end the carry; it only clears the overlay reference.

4. **Placement lock (loaded spreader descending) and put-down.** While carrying, the same still-background + sustained-descent signature locks the placement target: the under-spreader container nearest the spreader that is **not moving with the spreader**, or a blind lock when none is visible. A blind placement lock is the *normal* outcome, not a failure mode: the target sits directly under the carried container and is genuinely occluded by it, so no detector can see it. A **put-down** additionally requires the carry to have physically happened: the spreader must have hoisted ≥ 800 mm above its pickup plane and re-descended ≥ 800 mm before either signal — sustained re-ascent off the new touchdown plane, or the visible carried track failing the carried cues for ~1 s — is allowed to release. This makes the false put-down during the post-pickup hoist (where the same cue-failure signature occurs) impossible. On release the placed container becomes the idle reference target and the cycle returns to state 1.

**Contact-plane knowledge.** Touchdowns are physical measurements: at a pickup touchdown the spreader plane *is* the target's top; at a put-down the deepest carry plane *is* the placed container's top. On "Stacking 1" this inference was independently confirmed: a contact-inferred placement plane of 8922 mm sat 7 mm from the 8915 mm pinhole read of an unclipped box that briefly resolved for the target container. These planes are remembered per track (`knownTopZByTrack`) with before/after semantics — a completed placement leaves its new top at the contact plane, a completed pickup exposes the container below (one ISO height deeper). In camera geometries where every container bbox is clipped by the frame edge (containers longer than the field of view), these contact planes are the **only** valid container depths; all pinhole reads off clipped boxes are excluded throughout.

**Frozen target Z and continuous constrained descent.** For both lock kinds the target's Z is frozen per episode, preferring the most physical source: the episode's own touchdown contact plane (when the episode completed rather than aborted) → a previously contact-learned plane for the track, **snapshotted at lock/adopt time** so a later episode's contact cannot retroactively rewrite the plane an earlier one was measured against → the median pinhole Z of *unclipped* boxes over a window around the lock frame (2 s before to 0.5 s after, preferring matched over Kalman-extrapolated boxes). Freezing is deliberate — the target is static and the camera has stopped translating, so its true Z no longer changes, and the growing occlusion from the descending spreader/load would otherwise corrupt the live bbox measurement. The measured spreader trajectory is mapped between two physical boundary conditions:

```
displayed_spreader(lock)    = measured_spreader(lock)
displayed_spreader(contact) = target_top − 2591 mm   (placement)
displayed_spreader(contact) = target_top             (pickup)
```

An affine mapping between those endpoints preserves continuity at lock and preserves the timing and progress of the complete measured descent, while compensating for offset or scale disagreement between the independently inferred target and spreader depths. The remaining gap is derived from that same displayed geometry, so it reaches zero exactly when the deepest observed in-episode spreader position reaches the contact plane.

In the UI, the tracked-video overlay marks the carried container (sky blue, `carried #id`), the locked pickup or placement target (purple, `target #id · N mm`), and the idle nearest-to-spreader reference (lime, `nearest #id`). A target whose plane came from contact inference rather than its own bbox is labelled `(inferred)` in the schematic, and the status line reads "not detected — plane from touchdown", so a target correctly drawn at a known depth with no bounding box does not look like a bug. The side-view schematic renders the carried container under the spreader as soon as carrying is detected, then adds the target at its frozen Z when lock occurs; the carried shape therefore does not switch coordinate models or jump at that transition. The vertical axis also reserves the frozen target's full extent throughout the loaded sequence, preventing an auto-zoom jump when the target appears. The amber bracket measures carried-bottom → target-top when loaded (spreader → target-top when empty), and a status line reports the state and live remaining drop.

All thresholds (lockstep velocity and scale-rate epsilons, stillness epsilon, movement/descent thresholds, acquire/detach/sustain durations, empty-target hysteresis and overlap gate, hoist/touchdown margins) are exported constants at the top of `frontend/src/lib/stackingDistance.ts`.

**Offline debugging harness.** The exact frontend pipeline can be reproduced outside the browser to close the feedback loop against a real video: dump ByteTrack frames with `src.core.inference.compute_bytetrack_frames` over a run's `result.json`, then run `frontend/scripts/debugStacking.ts` (bundle with `node_modules/.bin/esbuild scripts/debugStacking.ts --bundle --platform=node --format=esm --alias:@=./src`) to apply the real One Euro smoothing + `analyzeStacking` and print the state timeline, and `frontend/scripts/render_stacking_debug.py` to draw the detections and analysis roles onto the actual video frames.

The harness also asserts **physical invariants** and prints a pass/fail block, each encoding a failure observed on real footage: a placement target may never be the spreader's own merged blob; a completed episode's remaining drop must actually close (< 250 mm); a placement lock requires a carrying state; and the carried identity must not flap frame-to-frame. Run it after any change to the state machine.

`frontend/scripts/simulate_load_classes.py` relabels an existing tracked-frames dump into `spreader_loaded` / `spreader_empty` against a known release time, so the load-state consumption path can be exercised on today's runs before a load-state-aware model exists.

---

## Persisted shape (`result.json`)

Once calibrated, a run's `result.json` gains a `z_calibration` block. Every matching non-container target gains a `z_mm` field; container targets gain `z_mm` only on the per-frame center-selected container. For a multi-target run it looks like:

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
    "measurement_source": "bbox_longer_side",
    "model": {"type": "linear_inv", "m": 7606790.0, "c": -30.2}
  },
  "frames": [
    {
      "frame_number": 100,
      "detections": [
        {"class_name": "container", "box": {...}, "confidence": 0.94, "z_mm": 8210.3, "z_selected": true},
        {"class_name": "container", "box": {...}, "confidence": 0.91}
      ]
    }
  ]
}
```

Round-feature calibrations add the physical feature fields:

```json
{
  "measurement_source": "round_feature_equivalent_length",
  "reference_class": "spreader_round_top",
  "length_mm": 12192,
  "round_feature_diameter_mm": 250,
  "equivalent_size_ratio": 48.768
}
```

Single-class runs have the same `model` shape (`k_over_s` for 1-label, `linear_inv` for 2+); they just leave `targets` empty (or equal to `[reference_class]`) and estimate only the reference class. For whole-bbox mode, `length_mm` is informational at estimation time. For round-feature mode, `length_mm` and `round_feature_diameter_mm` define the stored `equivalent_size_ratio`.

---

## Source map

- `backend/app/services/z_estimator.py`
  - `_longer_side_px()` — the single source of truth for `s` (max of bbox width and height in pixels).
  - `_measurement_size_px()` — chooses between raw bbox size and round-feature equivalent length.
  - `calibrate()` — builds `(s, z)` pairs on the reference class and returns one flat model.
  - `_fit_single_class()` — closed-form OLS on `(1/s, z)`, with the 1-label shortcut and degenerate fallback.
  - `_pick_center_detection()` — picks the target container closest to the normalized frame center.
  - `estimate()` — applies the flat model to target detections, writing `z_mm` in-place; container targets are center-selected per frame.
  - `apply_z_to_result()` — end-to-end: read `result.json`, fit, estimate, write back; raises on legacy schemas.
- `backend/app/api/inference.py` — REST endpoints: save calibration, apply estimation, re-export video.
- `frontend/src/pages/ZCalibrationPage.tsx` — the active frame browser and simplified calibration UI.
- `frontend/src/pages/ZCalibrationPage.tsx` — the full-screen frame picker for selecting calibration frames.
- `frontend/src/components/SideViewSchematic.tsx` — the live elevation diagram of camera / spreader / container on the inference detail page, including the loaded-spreader stacked mode.
- `frontend/src/lib/zCalibration.ts` — shared frontend fallback calculation for bbox and round-feature measurement sources.
- `frontend/src/lib/stackingDistance.ts` — crane-cycle stacking analysis: under-spreader target gating, merged-blob handling, pickup and placement locks on descent (including blind locks), cue- and Z-profile-based carrying, hoist-gated put-down, contact-plane knowledge, frozen target Z, per-frame remaining drop.
- `frontend/scripts/debugStacking.ts` / `frontend/scripts/render_stacking_debug.py` — offline reproduction harness: run the exact analysis pipeline on a stored run, assert physical invariants, and render the roles onto real video frames.
- `frontend/scripts/simulate_load_classes.py` / `frontend/scripts/inspect_blob.py` — relabel a run with load-state spreader classes to exercise that path, and compare the spreader's merged container box between loaded and empty frames.

---

## Practical tips

- **Spread your calibration points.** Two labels at nearly the same distance are effectively one label — and trigger the degenerate fallback. Pick the shortest and longest distances you care about.
- **Prefer 2+ points.** The intercept `c` is where most of the real-world accuracy comes from.
- **Don't extrapolate far beyond your labels.** `Z = m/s + c` is a linear fit in `1/s`. Well outside the calibration range the linearisation drifts and errors grow.
- **Pick the container length you're lifting.** 20 / 40 / 45 ft are the three standard ISO options. The spreader telescopes to match, so a single selection covers both.
- **Store physical dimensions, not magic ratios.** Round-feature mode records both the spreader/container length and the round feature diameter, then computes the ratio from those values.
- **Reference class as a target.** If you want distances for the reference class itself, include it — but the system adds it automatically if you forget.
- **Re-calibrate if the camera moves or the container size changes.** The model bakes the camera geometry *and* the chosen `ℓ` into `k` / `m` / `c`; any physical change to mounting, zoom, or ISO length invalidates the fit.

