/**
 * One Euro Filter — Casiez, Roussel & Vogel (CHI 2012).
 *
 * Low-lag, adaptive low-pass filter for noisy 1-D signals. Holds its cutoff
 * low when the signal is stationary (kills jitter) and raises it quickly
 * when the signal starts moving (avoids lag). Two tunable knobs:
 *   - `minCutoff` (Hz): cutoff at rest. Smaller ⇒ heavier smoothing at rest.
 *   - `beta`:           speed coefficient. Larger ⇒ faster unclamping on motion.
 *
 * We use it here as a second-stage post-processor on top of ByteTrack's own
 * Kalman filter: ByteTrack provides the track_id and a causally-smoothed
 * bbox (option 1), and this filter scrubs the residual jitter out of each
 * of (cx, cy, w, h) independently. Because this filter is causal and only
 * looks at current + previous samples per track, running it across the full
 * ordered frame list produces exactly the same output it would emit live.
 */

class LowPassFilter {
  private s: number | null = null

  filter(x: number, a: number): number {
    this.s = this.s == null ? x : a * x + (1 - a) * this.s
    return this.s
  }
}

function smoothingFactor(dt: number, cutoff: number): number {
  const tau = 1 / (2 * Math.PI * cutoff)
  return 1 / (1 + tau / dt)
}

export class OneEuroFilter {
  private readonly xFilt = new LowPassFilter()
  private readonly dxFilt = new LowPassFilter()
  private lastT: number | null = null
  private lastX: number | null = null

  constructor(
    private readonly minCutoff: number,
    private readonly beta: number,
    private readonly dCutoff: number = 1.0,
  ) {}

  /** Feed one sample `(x, t)` in (value, seconds); returns the filtered value. */
  filter(x: number, t: number): number {
    if (this.lastT == null || t <= this.lastT) {
      // Bootstrap or out-of-order sample: initialise and pass through.
      this.lastT = t
      this.lastX = x
      return this.xFilt.filter(x, 1.0)
    }
    const dt = t - this.lastT
    const dx = (x - (this.lastX ?? x)) / dt
    const edx = this.dxFilt.filter(dx, smoothingFactor(dt, this.dCutoff))
    const cutoff = this.minCutoff + this.beta * Math.abs(edx)
    const xHat = this.xFilt.filter(x, smoothingFactor(dt, cutoff))
    this.lastT = t
    this.lastX = x
    return xHat
  }
}

// ---------------------------------------------------------------------------
// Per-track frame-level applicator
// ---------------------------------------------------------------------------

interface Box {
  x: number
  y: number
  width: number
  height: number
}

// Deliberately loose structural types — we only touch `box` and `track_id`
// on detections and `timestamp` / `detections` on frames. Any additional
// fields are preserved untouched via `{ ...d, ... }` spreads.
interface Det {
  box: Box
  track_id?: number | null
}

interface Frame {
  timestamp: number
  detections: readonly Det[]
}

export interface OneEuroParams {
  minCutoff: number
  beta: number
  dCutoff?: number
}

interface TrackState {
  cx: OneEuroFilter
  cy: OneEuroFilter
  w: OneEuroFilter
  h: OneEuroFilter
}

/**
 * Apply One Euro filtering to (cx, cy, w, h) per `track_id` across a
 * chronologically-ordered list of frames. Detections without a `track_id`
 * are passed through unchanged. Returns a new frames array with new
 * detection objects whose `box` field has been replaced; every other field
 * (class_name, confidence, z_mm, etc.) is preserved.
 *
 * O(total_detections). Designed to run inside a `useMemo`.
 */
export function smoothFramesPerTrack<F extends Frame>(
  frames: readonly F[],
  params: OneEuroParams,
): F[] {
  const state = new Map<number, TrackState>()
  const mk = (): TrackState => ({
    cx: new OneEuroFilter(params.minCutoff, params.beta, params.dCutoff),
    cy: new OneEuroFilter(params.minCutoff, params.beta, params.dCutoff),
    w: new OneEuroFilter(params.minCutoff, params.beta, params.dCutoff),
    h: new OneEuroFilter(params.minCutoff, params.beta, params.dCutoff),
  })

  return frames.map((f) => {
    const t = f.timestamp
    const nextDets = f.detections.map((d) => {
      const tid = d.track_id
      if (tid == null) return d
      let s = state.get(tid)
      if (!s) {
        s = mk()
        state.set(tid, s)
      }
      const cx = s.cx.filter(d.box.x, t)
      const cy = s.cy.filter(d.box.y, t)
      const w = s.w.filter(d.box.width, t)
      const h = s.h.filter(d.box.height, t)
      return { ...d, box: { x: cx, y: cy, width: w, height: h } }
    })
    return { ...f, detections: nextDets }
  })
}
