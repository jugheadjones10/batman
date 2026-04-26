---
name: pdf-pinhole-calibration-refactor
overview: "Rewrite the Z-calibration stack to match the simplified PDF-style pinhole math: one shared fit (`k` or `m/s + c`) broadcast across spreader and every container class, `s = longer bbox side` (no axis selector), and a single `length_mm` dropdown (ISO container lengths) instead of per-class widths. Old result.json calibrations become invalid and require a quick re-run."
todos:
  - id: backend-estimator
    content: "Rewrite backend/app/services/z_estimator.py: _longer_side_px, new calibrate() / estimate() signatures, flat model (k or m/c), legacy schema error in apply_z_to_result"
    status: completed
  - id: backend-api
    content: Update ZCalibrationRequest in backend/app/api/inference.py (reference_class, length_mm, target_classes); drop ZCalibrationTarget; rewrite save_z_calibration body
    status: completed
  - id: frontend-types
    content: Rewrite ZCalibration / ZCalibrationModel types in frontend/src/types/index.ts; delete ZCalibrationTarget
    status: completed
  - id: frontend-client
    content: Update saveZCalibration in frontend/src/api/client.ts with the new body keys and parameter shape
    status: completed
  - id: frontend-panel
    content: "Refactor ZCalibrationPanel.tsx: drop axis select, swap width Input for ISO-length dropdown, simplify targets to class-name list, update existing-model summary and info popup"
    status: completed
  - id: frontend-page
    content: Mirror the same UI changes in ZCalibrationPage.tsx
    status: completed
  - id: frontend-schematic
    content: "Simplify SideViewSchematic.tsx: drop multi_target / per-target / size_metric / focal-length derivation; use flat model + longer-side rule + calibration.length_mm"
    status: completed
  - id: docs-guide
    content: "Update docs/guides/z-axis-height-estimation.md: remove axis-choice subsection, reframe Mode 3 around container length + telescoping spreader, refresh UI walkthrough and result.json example"
    status: completed
isProject: false
---

## Scope

- Backend: collapse `multi_target` + `size_metric` + per-target widths into one flat model broadcast by class name.
- Frontend: drop axis selector, drop per-target width inputs, add ISO-length dropdown.
- Docs: mirror the axis removal + length-selection changes.
- Hard break on old `z_calibration` schema (per user confirmation): old runs raise a clear error on estimate; user re-calibrates.

## New data model

On-disk shape inside `result.json`:

```json
{
  "z_calibration": {
    "labels": [{"frame_number": 142, "z_mm": 12000, "detection_index": 0}],
    "reference_class": "spreader",
    "length_mm": 12192,
    "targets": ["spreader", "container"],
    "video_resolution": {"width": 1920, "height": 1080},
    "model": {"type": "linear_inv", "m": 7606790.0, "c": -30.2}
  }
}
```

Key changes vs. today:
- `class_name` -> `reference_class`
- `size_metric` removed (always longer bbox side)
- `reference_real_width_mm` -> `length_mm` (selected from dropdown)
- `targets: [{class_name, real_width_mm}]` -> `targets: [string, ...]`
- Model: no more nested `multi_target`; one flat `{k_over_s, k}` or `{linear_inv, m, c}` broadcast to every class in `targets`

`length_mm` is informational (documents what `k` / `m` bake in); the math never needs it at estimation time.

## Backend: [backend/app/services/z_estimator.py](backend/app/services/z_estimator.py)

Full rewrite of the math surface:

- Replace `_h_px`, `_w_px`, `_get_size` with a single helper:

```python
def _longer_side_px(det: dict, video_resolution: dict) -> float:
    w = det["box"]["width"] * video_resolution["width"]
    h = det["box"]["height"] * video_resolution["height"]
    return max(w, h)
```

- `_fit_single_class(pairs)` keeps the same OLS closed form, but rename output keys: `a`->`m`, `b`->`c` (so `{"type": "linear_inv", "m": ..., "c": ...}`). `k_over_s` unchanged.
- `calibrate(labels, frames, video_resolution, reference_class, target_classes=None)` — new signature. No `size_metric`, no widths. Builds `(s, z)` pairs on the reference class using `_longer_side_px`, calls `_fit_single_class`, returns one flat model. No `multi_target` branch.
- `estimate(model, frames, video_resolution, target_classes)` — iterate detections, apply the same `k/s` or `m/s + c` to every detection whose `class_name` is in `target_classes`.
- `apply_z_to_result()` — read `reference_class` / `length_mm` / `targets`, refuse to run if any legacy field (`size_metric`, `reference_real_width_mm`) is present (`raise ValueError("legacy z_calibration schema — please re-calibrate")`), auto-add `reference_class` to `targets` if missing.

## Backend: [backend/app/api/inference.py](backend/app/api/inference.py)

Redefine `ZCalibrationRequest` (around line 918):

```python
class ZCalibrationRequest(BaseModel):
    labels: list[ZCalibrationLabel]
    reference_class: str
    length_mm: float | None = None
    target_classes: list[str] = []
```

Drop `ZCalibrationTarget` (class no longer needed). `save_z_calibration` persists the new keys verbatim. `get_z_calibration` and `export_z_video` are unchanged (they only read `z_mm` off detections).

## Frontend types: [frontend/src/types/index.ts](frontend/src/types/index.ts)

Replace the three interfaces at lines 346-374 with:

```ts
export interface ZCalibrationLabel { frame_number: number; z_mm: number; detection_index: number }

export interface ZCalibrationModel {
  type: 'k_over_s' | 'linear_inv'
  k?: number
  m?: number
  c?: number
}

export interface ZCalibration {
  labels: ZCalibrationLabel[]
  model: ZCalibrationModel | null
  reference_class: string
  length_mm?: number | null
  targets?: string[] | null
  video_resolution?: { width: number; height: number }
}
```

Delete `ZCalibrationTarget`.

## Frontend API client: [frontend/src/api/client.ts](frontend/src/api/client.ts)

`saveZCalibration` new signature (around line 470):

```ts
saveZCalibration: (
  projectName, runName, videoId, inferenceId,
  labels: ZCalibrationLabel[], referenceClass: string,
  opts?: { lengthMm?: number | null; targetClasses?: string[] },
)
```

Body keys: `labels`, `reference_class`, `length_mm`, `target_classes`.

## Frontend panels: [frontend/src/components/ZCalibrationPanel.tsx](frontend/src/components/ZCalibrationPanel.tsx) and [frontend/src/pages/ZCalibrationPage.tsx](frontend/src/pages/ZCalibrationPage.tsx)

Shared UI changes (apply to both files; they carry parallel state):

- Remove `sizeMetric` state + the Axis `<select>`.
- Remove `referenceRealWidth` Input; add a length dropdown:

```ts
const ISO_LENGTHS = [
  { mm: 6058,  label: '20 ft (6058 mm)' },
  { mm: 12192, label: '40 ft (12192 mm)' },
  { mm: 13716, label: '45 ft (13716 mm)' },
] as const
```

- Simplify targets state to `string[]` — each row is a class-name `<select>` plus a trash button. Drop per-row mm Input. "Add target" seeds with first unused class name.
- Auto-seed the reference class into `targets` when `length_mm` is set and `targets` is empty (mirror existing behaviour).
- Update the existing-model summary strip (around line 189-206 in the panel): drop the `multi_target` branch; always render `Z = k/s` or `Z = m/s + c`.
- Rewrite the info popup "How Distance Calibration Works" to match the new math (remove axis-mixing warning, explain the length-sharing assumption, refer to the guide).
- `calibrateMutation` passes `{ lengthMm, targetClasses }` to the new `saveZCalibration`.

## Frontend schematic: [frontend/src/components/SideViewSchematic.tsx](frontend/src/components/SideViewSchematic.tsx)

Under the new model there is only one `k` / `(m, c)` and every class shares it. Simplifications:

- Delete `deriveFocalLengthPx`, `pickTargetModel`, the `multi_target` branches inside `computeZForTarget`, and the `DEFAULT_SPREADER_LENGTH_MM` fallback.
- `computeZForTarget(cal, box, vw, vh)` -> compute `s = max(box.width*vw, box.height*vh)`, then `k/s` or `m/s + c` from the single flat model. Signature drops `className` and `realWidthMm`.
- `spreaderRealLengthMm` and `containerRealWidthMm` both collapse to `calibration.length_mm ?? <aspect-ratio fallback>`. Spreader and container render at the same horizontal footprint (since `ℓ_spreader = ℓ_container` by assumption).
- The aspect-ratio-based `containerLengthMm` inference can remain as the display fallback when no calibration exists (keeps the schematic useful pre-calibration), but defer to `calibration.length_mm` whenever present.

## Longer-side rule everywhere

The single invariant `s = max(box.width * video_w, box.height * video_h)` replaces every `h_px`/`w_px` switch:
- `_longer_side_px` in [z_estimator.py](backend/app/services/z_estimator.py)
- inline in `computeZForTarget` in [SideViewSchematic.tsx](frontend/src/components/SideViewSchematic.tsx)
- no axis-aware dimension logic anywhere in the calibration UIs

## Docs: [docs/guides/z-axis-height-estimation.md](docs/guides/z-axis-height-estimation.md)

Follow-up pass to match the new code surface:
- Remove the whole "What `s` actually is, in code > Why the choice exists at all" subsection; replace with a short "Batman always uses `s = max(bbox_width_px, bbox_height_px)` — the longer side is the cleaner signal and doesn't require the user to pick" paragraph.
- Drop the `h_px` / `w_px` table in that section.
- Rework Mode 3 framing: `ℓ` is now the **container length** (selected from 20/40/45 ft dropdown), shared with the spreader because it telescopes to match. Remove the "2438 mm width" emphasis; keep the length table but reframe it as the three dropdown options.
- Update the UI walkthrough: fields are now {Length dropdown, Reference Class, Estimation Targets, Calibration Points}; drop the Axis item and the per-target size column.
- Update the `result.json` example to the new schema (`reference_class`, `length_mm`, `targets: [string]`, flat model).
- Update practical tips: replace the "2438 mm for `ℓ`" bullet with "pick the container length you're lifting; 20/40/45 ft are standard".

## Out of scope

- Back-compat for old `z_calibration` blocks (hard break per decision).
- Per-class length mapping (user confirmed single length per calibration is fine).
- Consolidating `ZCalibrationPanel` + `ZCalibrationPage` (duplicated UI is pre-existing and not touched beyond the mirrored edits).

## Test plan (manual)

1. Open a run with an old `z_calibration` block -> apply-estimate returns a clear "legacy schema, please re-calibrate" error; UI still renders.
2. Fresh calibration, 1 label, 40 ft selected -> model is `{k_over_s, k}`; every detection in `targets` gets a `z_mm`.
3. Fresh calibration, 3 labels -> model is `{linear_inv, m, c}`; estimation works across all classes in `targets`.
4. SideViewSchematic renders spreader and container at the same footprint, with the selected length in the label, using only the flat model.
5. Re-export video with Z overlays succeeds and burns in the new `z_mm` values.