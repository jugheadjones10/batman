# Detection Tracking & Jitter Reduction

Raw per-frame RF-DETR output has two problems that show up the moment you try to use it for anything downstream (measuring `z_mm`, reading container heights, judging whether a spreader has docked):

1. **Missed frames.** The detector occasionally drops an object for a frame or two — the spreader vanishes mid-descent even though nothing physically changed.
2. **Bounding-box jitter.** Even when the detector *does* fire every frame, the box wobbles by a few pixels in random directions. Because distance is computed as `Z = k / s` with `s` the longer bbox side (see [Distance Calibration](z-axis-height-estimation.md)), that pixel jitter shows up on the Side-View Schematic as a nervous flicker on the z reading.

The Inference detail page applies tracking and smoothing automatically to playback overlays, position graphs, and the Side-View Schematic. These are transformations of the saved `result.json`; no re-inference is needed. Parameters use the defaults in `frontend/src/lib/trackingPresentation.ts`.

This guide explains the three techniques that power inference playback:

1. **Gap-fill tracking** — an `sv.ByteTrack` instance re-associates frame-by-frame detections into stable tracks, and keeps predicting the box for a few extra frames when the detector loses its object.
2. **Kalman-posterior emission** — instead of shipping the raw detector bbox on matched frames, we surface ByteTrack's internal Kalman state (the posterior after fusing prediction + measurement). Same infrastructure; different output. Cheapest jitter reduction available.
3. **One Euro filter** — a second-stage adaptive low-pass filter applied per-track on top of the Kalman posterior. Heavy smoothing at rest, quick release on motion; the parameters control the trade-off between jitter and lag.

Techniques 2 and 3 compose cleanly: the Kalman posterior makes matched and extrapolated frames come from a single KF state vector (removing a structural discontinuity), and the One Euro filter scrubs the residual detector noise out of whatever the backend emits.

---

## The problem, made concrete

Run inference on a typical spreader-descent clip and look at a single tracked object's bbox over ~30 consecutive frames:

```
  frame-to-frame Δ (pixels) on bbox.width   ── jittery
  ┌────────────────────────────────────────────┐
  │  ▇ ▅ ▆ █ ▄ ▇ ▆ ▃ █ ▅ ▆ █ ▄ ▇ ▅ ▇ ▄ ▆ █ ▅  │   stdev ≈ 4 px
  │                                            │
  └────────────────────────────────────────────┘
  t →
```

The detector is not broken — it's just that each frame is inferred independently, and the bbox position it settles on is a function of feature maps that themselves have pixel-level noise. A 4 px wobble on a 120 px spreader side translates through `Z = k / s` into about a ±3% wobble on `z_mm`. On a 10 m working distance that's 30 cm of false motion per frame. Fine for object counting; not fine for reading crane-hook altitude at a glance.

---

## Part 1 — Gap-fill tracking (ByteTrack)

The first thing we want is *continuity*: when the detector misses the spreader on frame 42 but has it on 41 and 43, we don't want the schematic to flicker blank. We want a best-guess box on 42.

This is what a **tracker** gives you. Batman reuses `supervision.ByteTrack` (the MOT algorithm from Zhang et al., ECCV 2022 — developed at ByteDance — which pairs a linear Kalman filter with a two-pass IoU association on high- and low-confidence detections). The tracker assigns every detection a `track_id` that is stable across frames, and when a detection is missing it continues predicting the box from the Kalman filter for up to `lost_track_buffer` frames before giving up.

Inference playback renders tracked boxes as follows:

- **Solid green boxes** — a *measured* track: a raw detection of the same class overlaps the tracked box with IoU ≥ 0.3 on this frame.
- **Dashed amber boxes** — an *extrapolated* track: the detection was missed on this frame but the track is still inside `lost_track_buffer`, so we're showing the Kalman prediction.

### Why a class-aware tracker

`supervision.ByteTrack` does pure IoU association — it doesn't care about class labels. That's fine for pedestrians, but for our workload the spreader *sits on top of* the container for most of a lift: two different classes, nearly the same bounding box. A class-agnostic tracker happily swaps the two tracks' IDs from one frame to the next, which makes the schematic comparison worse than the raw output.

We work around this by running **one independent ByteTrack instance per class** and remapping the per-class track IDs to a globally-unique ID space. The design is visible in `compute_bytetrack_frames`:

```python
# src/core/inference.py
class_trackers: dict[str, sv.ByteTrack] = {}
tid_map: dict[tuple[str, int], tuple[int, int]] = {}
...
for cname in all_classes:
    tracker = class_trackers.get(cname) or _new_tracker()
    tracked = tracker.update_with_detections(sv_in)   # only this class's dets
```

Every class's tracker advances every frame (including empty ones), so its Kalman predictions and lost-buffer bookkeeping stay synchronised with the video clock even when one class briefly has no detections.

### The three tunable ByteTrack knobs

The defaults in `DEFAULT_TRACKER_PARAMS` map directly to `sv.ByteTrack` constructor arguments:

| Parameter                      | What it gates                                                                                                                                                                      | Default |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `track_activation_threshold`   | Minimum detection confidence for a detection to *spawn* a new track. Lower = flickerier new tracks from marginal detections; higher = misses legitimate objects with shy confidence. | 0.25    |
| `lost_track_buffer`            | Frames a track stays alive with Kalman predictions after its detection disappears. Above this, the track is killed. At 30 fps, 30 frames ≈ 1 s of gap fill.                         | 30      |
| `minimum_matching_threshold`   | IoU gate for associating a current detection to an existing track's Kalman-predicted box. Lower = more lenient matching through jitter.                                             | 0.8     |

The inference page requests `/bytetrack-frames` with these defaults and caches the response with React Query.

---

## Part 2 — Emit the Kalman posterior, not the raw measurement

The first jitter win costs zero new parameters and no new dependencies. It comes from fixing a quirk of `supervision`'s ByteTrack implementation.

### The quirk

When you call `tracker.update_with_detections(raw_dets)`, `supervision` does this internally:

1. **Predict** — advance every track's Kalman state forward one time step.
2. **Match** — IoU-associate raw detections to predicted tracks (high-conf pass → low-conf pass).
3. **Update** — for each matched track, call `track.update(raw_det)` which runs the Kalman measurement update and stores the posterior in `track.mean`.
4. **Return** — hand back the **raw input detections** with only a `tracker_id` stapled on. The posterior it just computed is discarded.

Look at `supervision/tracker/byte_tracker/core.py` and you can see it throw away the smoothed state:

```python
track_bounding_boxes = np.asarray([track.tlbr for track in tracks])  # posterior
...
return detections[detections.tracker_id != -1]                       # raw input
```

Meanwhile, our own `compute_bytetrack_frames` was already reading `strack.tlbr` (the KF state) on the lost-track branch. So before this fix, the output stream looked like this:

```
matched frame   → raw detection bbox        (measurement, unsmoothed)
lost frame      → strack.tlbr               (Kalman prediction, smoothed)
matched frame   → raw detection bbox        (measurement, unsmoothed)
```

Every time a frame flipped between matched and extrapolated, the emitted box source flipped with it. That discontinuity is *visible* on the schematic — a clean amber prediction, then a jumpy green measurement, then another clean amber.

### The fix

Reach into `tracker.tracked_tracks` after the update, build an `external_track_id → STrack` lookup, and use `strack.tlbr` for matched tracks too. Same KF state we were already using for the lost branch. One source for every emitted box:

```python
# src/core/inference.py (excerpt)
strack_by_tid = {
    int(t.external_track_id): t
    for t in tracker.tracked_tracks
    if getattr(t, "external_track_id", -1) >= 0
}
for i in range(len(tracked)):
    strack = strack_by_tid.get(tracker_id)
    tlbr = strack.tlbr if strack is not None else tracked.xyxy[i]   # fallback
    ...
```

### Why `z_mm` has to be dropped on the tracked side

The persisted `z_mm` in `result.json` was computed from the *raw* detector bbox (see [Distance Calibration](z-axis-height-estimation.md) — it's baked into the detection at the time the calibration was applied). Pairing a smoothed bbox with the raw-bbox `z_mm` re-introduces the exact jitter we're trying to kill on the only signal that feeds the schematic.

So `compute_bytetrack_frames` deliberately drops `z_mm` on tracked-side detections. Both matched and extrapolated frames come out without it. The frontend's `SideViewSchematic` already has a path for this: if `z_mm` is absent, it recomputes `Z = k / s` from the emitted bbox via the project's calibration, using the same pinhole formula the backend would have used. Internally consistent: box and z come from the same source.

The calibration panel itself still reads `z_mm` from `result.json` (the raw side). The tracked side never writes back.

### Expected result

Running the smoke test in `src/core/inference.py` on a synthetic stream with 6 px detector jitter:

| Signal          | Raw  | Matched-frame KF posterior |
| --------------- | ---- | -------------------------- |
| stdev(Δ cx)     | 5.7 px | 3.6 px (−37%)              |
| stdev(Δ width)  | 4.0 px | 2.3 px (−42%)              |

~40% reduction, no new parameters, no new dependencies, no new UI.

---

## Part 3 — One Euro filter, per track

The Kalman posterior is smoother than the raw measurement, but supervision's ByteTrack uses its stock process-noise / measurement-noise matrices (tuned for MOT17-style pedestrian motion). For a static spreader it's still noisier than we want. The second stage is a dedicated, tunable low-pass filter.

### The filter, intuitively

The [One Euro filter](https://gery.casiez.net/1euro/) (Casiez, Roussel & Vogel, CHI 2012) is an adaptive low-pass with two knobs:

- **`min_cutoff`** (Hz) — cutoff frequency at rest. Lower = heavier smoothing on stationary signals.
- **`beta`** — speed coefficient. Higher = the cutoff *rises* quickly with velocity, so the filter releases when the object actually moves.

The central idea: you can't have a fixed low-pass that's both quiet at rest and fast on motion — whichever cutoff you pick, you're trading one for the other. One Euro sidesteps the trade by **letting the cutoff itself be a function of the signal's current speed**. At rest, motion ≈ 0, cutoff stays at `min_cutoff`, lots of smoothing. Once the signal starts moving, cutoff rises as `min_cutoff + beta · |velocity|`, and the filter releases.

### The math

At each new sample `(x, t)` for a given signal:

```
dt    = t - t_prev
dx    = (x - x_prev) / dt                    # raw velocity
v̂     = LPF(dx, α(dt, d_cutoff), v̂_prev)     # smooth the velocity
cutoff = min_cutoff + beta · |v̂|             # adaptive cutoff
x̂     = LPF(x, α(dt, cutoff), x̂_prev)        # smooth the signal
```

where the single-pole low-pass and its smoothing factor are:

```
LPF(x, α, s_prev)  = α · x + (1 − α) · s_prev
α(dt, cutoff)      = 1 / (1 + τ/dt)    where τ = 1 / (2π · cutoff)
```

`d_cutoff` (the velocity smoothing's own cutoff) is fixed at 1.0 Hz, which is the paper's recommended constant.

### Applied per track

We run four independent One Euro filters — one for `cx`, one for `cy`, one for `width`, one for `height` — per `track_id`, across the chronologically-ordered frame list. Because the filter is causal (it only looks at past samples), running it end-to-end at page-load produces exactly the stream it would have produced live, frame-by-frame.

```ts
// frontend/src/lib/oneEuroFilter.ts
export function smoothFramesPerTrack<F extends Frame>(
  frames: readonly F[], params: OneEuroParams,
): F[] {
  const state = new Map<number, { cx, cy, w, h: OneEuroFilter }>()
  return frames.map((f) => ({
    ...f,
    detections: f.detections.map((d) => {
      if (d.track_id == null) return d
      const s = state.get(d.track_id) ?? mkState(params)
      state.set(d.track_id, s)
      return { ...d, box: {
        x: s.cx.filter(d.box.x, f.timestamp),
        y: s.cy.filter(d.box.y, f.timestamp),
        width: s.w.filter(d.box.width, f.timestamp),
        height: s.h.filter(d.box.height, f.timestamp),
      } }
    }),
  }))
}
```

Detections without a `track_id` (shouldn't happen on the tracked side, but defensive) are passed through unchanged.

### Smoothing defaults

Inference playback applies `DEFAULT_OEF.minCutoff = 1.0` Hz and
`DEFAULT_OEF.beta = 0.007` through `smoothFramesPerTrack`. These parameters
are configured in `frontend/src/lib/trackingPresentation.ts`.

Lower `minCutoff` for stronger smoothing at rest; raise `beta` to reduce lag
on fast motion. Parameter changes require a code change and should be checked
against representative footage.

### Expected result

Same 6 px-jitter synthetic stream, after the One Euro filter (on top of the Kalman posterior):

| Setting                          | stdev(Δ cx) | stdev(Δ width) |
| -------------------------------- | ----------- | -------------- |
| raw detector                     | 8.6 px      | 4.4 px         |
| ByteTrack KF posterior (Part 2)  | ≈5.0 px     | ≈2.6 px        |
| + One Euro at defaults (1.0, 0.007) | **1.1 px** | **0.6 px**     |
| + One Euro aggressive (0.5, 0.01)   | 0.6 px     | 0.3 px         |
| + One Euro conservative (2.0, 0.05) | 1.9 px     | 1.0 px         |

Roughly ~87% reduction vs. the raw signal at defaults, and you have a continuous knob to push it further (at the cost of lag) or back off (to chase transients faster).

---

## How Parts 2 and 3 compose

The two filters are in series on the same signal, but they solve different problems:

```
                  Part 1                    Part 2                       Part 3
 raw dets  →  ByteTrack assoc  →  Kalman update (measured)  →  One Euro (per-track)  →  overlay / schematic
            (gap-fill + IDs)     or Kalman predict (lost)     (tunable jitter scrub)
```

- **Part 1** hands each object a stable `track_id` and fills short gaps with Kalman predictions.
- **Part 2** makes every emitted box come from a single, consistent Kalman state — no discontinuity between measured and extrapolated frames. ByteTrack's process/measurement noise is fixed, tuned generically.
- **Part 3** is a dedicated, tunable smoother that only has to worry about the *residual* noise after Part 2. Because it runs per `track_id` over an ordered timeline, it doesn't need to know anything about measured vs. extrapolated — Part 2 already unified them.

This is why stacking them behaves well: the knobs on Part 3 expose a clean trade-off between residual jitter and lag, without having to care about the stateful interactions inside ByteTrack. If you turn Part 3 off, you get a less-jittery version of what you had before; turn it on, and you can push the quiet-at-rest behavior as far as your lag tolerance allows.

!!! warning "IoU-based green/amber labeling caveat"
    The shared overlay renderer labels a tracked box **measured** (solid green) if any same-class raw detection overlaps it with IoU ≥ 0.3, otherwise **extrapolated** (dashed amber). Under aggressive One Euro smoothing the tracked box can drift a few pixels from the raw detection it was matched to, and for small objects that's enough to drop the IoU under 0.3 and flip the label to amber even though the detection was actually present. 0.3 is a loose threshold and sane defaults won't cross it, but if you see a green→amber flash under `min_cutoff ≈ 0.3` / `beta ≈ 0.05` on a small bbox, that's the cause. It's a cosmetic quirk of the badging, not a tracking failure.

---

## Using it from the UI

Open a finished inference run on the **Inference** page. Tracking and smoothing
are applied to playback automatically. Scrub or play the video to inspect the
overlays, position graphs, and schematic.

---

## Persistence model

| Artifact                            | Contains                                                                   | Updated by                             |
| ----------------------------------- | -------------------------------------------------------------------------- | -------------------------------------- |
| `result.json`                       | Raw per-frame detections with `z_mm` computed from **raw** bboxes.         | Original inference; not modified.      |
| `/bytetrack-frames` JSON (endpoint) | Tracked frames with Kalman-posterior bboxes and no `z_mm`. Recomputed on every request from the three ByteTrack knobs. | The `get_bytetrack_frames` endpoint.   |
| One Euro state                      | Lives entirely in the browser tab. Never persisted.                        | Tracked-frame changes.                 |

The raw ground truth always stays in `result.json`. Tracking and smoothing are overlays you can layer on top non-destructively.

---

## Source map

- `src/core/inference.py`
  - `compute_bytetrack_frames()` — runs one `sv.ByteTrack` per class over a `result.json`; emits tracked frames with Kalman-posterior bboxes and no `z_mm`. Docstring enumerates the invariants.
- `backend/app/api/inference.py`
  - `GET /results/.../bytetrack-frames` — thin wrapper around `compute_bytetrack_frames`; takes the three ByteTrack knobs as query params; no server-side cache.
- `frontend/src/lib/oneEuroFilter.ts` — the `OneEuroFilter` class and `smoothFramesPerTrack()` helper.
- `frontend/src/pages/InferencePage.tsx` — requests tracked frames and applies One Euro smoothing for playback, graphs, and the schematic.
- `frontend/src/lib/trackingPresentation.ts` — shared tracker and smoothing defaults, track selection, and overlay helpers.
- `frontend/src/components/DetectionOverlaySvg.tsx` — SVG bbox overlay on top of each video pane.
- `frontend/src/components/SideViewSchematic.tsx` — the elevation schematic. Recomputes z from the bbox via calibration when `z_mm` is absent (the path the tracked side uses).

---

## Practical tips

- **Fix structural problems before reaching for One Euro.** If the spreader is flickering because the *detector* is dropping it every 5 frames, bumping `lost_track_buffer` to cover the gap is a better fix than smoothing harder. One Euro works best on residual noise, not on missing data.
- **Don't stack smoothers if you're going to persist the output.** Everything here is non-destructive on purpose — `result.json` stays raw, and the smoothed boxes only live in the inference playback transforms. Persisting smoothed bboxes back into `result.json` would make Part 2's `z_mm`-dropping decision a much bigger commitment.
- **Prefer the defaults until you see a real problem.** `min_cutoff = 1.0` / `beta = 0.007` is the MediaPipe-style default and removes ~87% of detector jitter without visible lag at 25–30 fps. Only deviate if you have a specific failure mode you can name.
- **Match `lost_track_buffer` to the gap you actually observe.** 30 frames ≈ 1 s at 30 fps. If the detector's worst gap on your footage is 3 frames, setting the buffer to 300 just extends phantom tracks further into empty frames without helping.

Related reading: [Distance Calibration (Z-axis)](z-axis-height-estimation.md).
