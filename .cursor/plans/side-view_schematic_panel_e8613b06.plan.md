---
name: Side-view schematic panel
overview: Add a live side-view schematic card to the right-hand column of `InferencePage.tsx` that shows, for the currently-playing frame, camera → spreader → container stack with annotated z-distances. Two dropdowns pick which classes drive the spreader and container slots. Container length class is snapped from bbox aspect ratio; vertical height is fixed at 2591 mm (ISO standard).
todos:
  - id: component
    content: Create frontend/src/components/SideViewSchematic.tsx with dropdowns, frame lookup, SVG layout, and container length classifier
    status: completed
  - id: integrate
    content: Wire SideViewSchematic into InferencePage.tsx's right-column stack
    status: completed
  - id: typecheck
    content: npx tsc --noEmit and ReadLints on changed files
    status: completed
  - id: docs
    content: (Optional) Add a one-line callout to docs/guides/z-axis-height-estimation.md pointing at the schematic
    status: completed
isProject: false
---

## What the panel shows

An SVG elevation diagram (camera on top, z axis drawn downward):

- Camera icon/marker at y = 0.
- Spreader rectangle centered at y proportional to `z_spreader_mm`.
- Container rectangle with top at y proportional to `z_container_top_mm`, vertical extent equal to the fixed ISO height (`2591 mm`), horizontal extent set to the classified length (6058 / 12192 / 13716 mm) scaled to fit.
- Dashed guide lines and labels on the right-hand side of the SVG:
  - `Camera → Spreader: z_spreader mm`
  - `Spreader → Container top: (z_container_top − z_spreader) mm`
  - `Container height: 2591 mm`
  - `Camera → Container bottom: (z_container_top + 2591) mm`

The y-axis scale auto-fits: min = 0, max = `z_container_top + 2591 + padding`, using the per-frame value. A CSS transition on SVG `<g transform="translateY(...)">` elements gives smooth motion as the video plays.

## Files to change

### New: [`frontend/src/components/SideViewSchematic.tsx`](frontend/src/components/SideViewSchematic.tsx)

Props (mirrors `ZGapReadout` + video resolution so we can do bbox pixel math):

```ts
interface Props {
  frames: InferenceResult[]
  currentTime: number
  videoWidth: number
  videoHeight: number
}
```

Internal state:
- `classA` (spreader) and `classB` (container) — two selects in the header, same pattern as `ZGapReadout` (default to first two z-capable classes).

Per-frame derivation (same `findClosestFrameIndex` pattern as the other two readouts):
- Pick the highest-confidence detection of each chosen class that has `z_mm != null`.
- Spreader: `z_spreader = det.z_mm`; horizontal length in the diagram = spreader's real width in mm if available (via `existingCal?.z_calibration?.reference_real_width_mm`), otherwise a sensible default (e.g., 2500 mm).
- Container:
  - `z_container_top = det.z_mm`.
  - Classify length via aspect ratio: compute `long_px / short_px` from the normalized bbox (accounting for `videoWidth` / `videoHeight`), then snap the implied long-side length (assuming short side = 2438 mm) to the nearest of `{6058, 12192, 13716}` mm. Unknown → use the snapped value directly (graceful even if ISO).
  - `z_container_bottom = z_container_top + 2591`.

SVG layout:
- `viewBox="0 0 400 320"` (or responsive via `preserveAspectRatio`).
- Central vertical axis at `x = 200` with tick marks every 1000 mm.
- Spreader/container rectangles centered on the axis, with their on-screen horizontal extents = `real_length_mm * horizontal_scale` (horizontal scale chosen so the widest element fits in ~70% of the SVG width).
- Smooth motion: wrap spreader and container in `<g>` with `style={{ transform: 'translateY(...)px', transition: 'transform 160ms ease-out' }}` — no heavy animation lib needed.
- Missing detection for a slot → dim the rectangle + show "--" in its label row.

### Edit: [`frontend/src/pages/InferencePage.tsx`](frontend/src/pages/InferencePage.tsx)

- New import alongside `LiveDetectionReadout` / `ZGapReadout`:

```tsx
import SideViewSchematic from '@/components/SideViewSchematic'
```

- Inside the right-column scrollable stack (currently `[LiveDetectionReadout, ZGapReadout]` around line 557-568), append:

```tsx
<SideViewSchematic
  frames={detailResult.frames}
  currentTime={videoTime}
  videoWidth={vw}
  videoHeight={vh}
/>
```

No other edits here.

### Small helper (collocated in the new component file)

Pure function for length classification:

```ts
const ISO_CONTAINER_SHORT_SIDE_MM = 2438
const ISO_CONTAINER_HEIGHT_MM = 2591
const ISO_CONTAINER_LENGTHS_MM = [6058, 12192, 13716]

function classifyContainerLengthMm(box: {width:number; height:number}, vw:number, vh:number): number {
  const w = box.width * vw
  const h = box.height * vh
  const ratio = Math.max(w, h) / Math.min(w, h)
  const estimatedLength = ISO_CONTAINER_SHORT_SIDE_MM * ratio
  return ISO_CONTAINER_LENGTHS_MM
    .reduce((best, candidate) =>
      Math.abs(candidate - estimatedLength) < Math.abs(best - estimatedLength) ? candidate : best,
      ISO_CONTAINER_LENGTHS_MM[0])
}
```

This makes the classifier unit-testable and easy to swap if HC handling arrives later.

## Deliberately not in scope

- No backend changes. Everything runs from already-persisted `z_mm` + bbox data on `InferenceResult`.
- No high-cube detection. A later PR can add a second ISO height (2896 mm) and a heuristic/toggle.
- No new UI in `ZCalibrationPanel` / `ZCalibrationPage`. The schematic is read-only.
- Docs update optional — I'll add one short callout in `docs/guides/z-axis-height-estimation.md` pointing at the schematic as a debugging visual, but can skip if you'd rather keep this PR minimal.

## Verification

- `npx tsc --noEmit` on the frontend.
- `ReadLints` on both changed files.
- Manual smoke: load a multi-target-calibrated run, check that the schematic updates smoothly during playback and that the container length snaps stably.
