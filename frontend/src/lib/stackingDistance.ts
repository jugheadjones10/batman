/**
 * Stacking-distance estimation over a full crane cycle.
 *
 * The analysis runs over the smoothed ByteTrack frames and models the crane's
 * repeating cycle as a state machine:
 *
 *   idle ──(descent starts)──▶ pickup ──(load acquired)──▶ carrying
 *     ▲                                                        │
 *     │                                              (descent starts, lock)
 *     └──────────(put-down detach)──────────── locked ◀────────┘
 *
 *   idle     — spreader is empty. The reference container is the track nearest
 *              the spreader (NOT the frame center: with an offset camera the
 *              pickup container sits under the spreader, not under the image
 *              center). Selection is sticky with hysteresis; right after a
 *              put-down it is seeded with the just-placed container so the
 *              schematic keeps measuring spreader ↔ placed box on the way up.
 *   pickup   — empty spreader descending onto the reference container. The
 *              target is locked, its Z frozen (median around the lock frame),
 *              and the constrained trajectory maps the descent to the contact
 *              plane at the target's top.
 *   carrying — a container track is CARRIED: it overlaps the spreader box,
 *              moves in lockstep with it (screen velocity AND bbox-scale rate:
 *              rigid attachment scales both boxes identically), and reads at
 *              the spreader's depth. Qualification time only accumulates while
 *              the spreader is moving — attachment must be proven by joint
 *              motion, so a container hovering a few feet below a static empty
 *              spreader can never acquire the flag. The depth cue is waived
 *              for boxes clipped by the frame edge, whose pixel size — and
 *              therefore Z — cannot be trusted.
 *   locked   — loaded spreader descending onto a stack: background container
 *              tracks are still (trolley stopped) while the spreader descends.
 *              The placement target (non-carried track nearest the spreader)
 *              is locked, its Z frozen, and the constrained trajectory maps
 *              the descent to the contact plane one container height above
 *              the target's top. The episode ends when the carried track stays
 *              visible but fails the carried cues for a sustained period (the
 *              spreader detached and hoisted away) — a put-down.
 *
 * Locks require sustained DESCENT (signed bbox-scale rate: shrinking = moving
 * away from the overhead camera), so the post-pickup hoist cannot fire a
 * phantom placement lock.
 *
 * Freezing the target Z makes the estimate immune to the growing occlusion
 * from the descending spreader/load. The constrained trajectory maps the
 * measured spreader Z affinely from its value at lock to the physical contact
 * plane at its deepest observed in-episode Z, so locking cannot cause a jump
 * and rigid bodies cannot intersect.
 *
 * This is an offline playback analysis: the frozen-Z window peeks 0.5 s past
 * lock and the trajectory constraint uses the deepest subsequent spreader Z.
 * A live port would replace those look-aheads with delayed target confirmation
 * and an explicit touchdown detector.
 */

import type { Detection, InferenceResult, ZCalibration } from '@/types'
import { computeZForBox } from '@/lib/zCalibration'
import { distanceToFrameCenter } from '@/lib/trackingPresentation'

// ISO dry-box container height (shared with SideViewSchematic).
export const ISO_CONTAINER_HEIGHT_MM = 2591

// ---------------------------------------------------------------------------
// Tunable thresholds (speeds are in normalized image units per second).
// ---------------------------------------------------------------------------

/** Max |v_container − v_spreader| for the "moves in lockstep" test. */
export const CARRIED_VELOCITY_EPS = 0.04
/**
 * Max |signed relative scale rate difference| (1/s) for the lockstep test.
 * A rigidly attached container's bbox scales exactly with the spreader's;
 * a static container under a descending spreader does not.
 */
export const CARRIED_SCALE_RATE_EPS = 0.02
/**
 * Min fraction of the spreader (proxy) bbox covered by a container bbox for
 * the overlap cue. Intersection is used instead of a center-in-box test
 * because at close range the offset camera's parallax pushes the proxy (e.g.
 * the round feature) toward the edge of — or just outside — the carried
 * container's box.
 */
export const CARRIED_OVERLAP_MIN_FRAC = 0.25
/**
 * Normalized margin for "bbox touches the frame edge". A container clipped by
 * the frame boundary has an unreliable pixel size, so its pinhole Z reads too
 * far; the depth cue is skipped for such boxes.
 */
export const EDGE_CLIP_MARGIN = 0.01
/**
 * Max |z_container − z_spreader| (mm) for a container to plausibly be engaged
 * with the spreader. A carried box hangs at the spreader's depth (offset by
 * roughly the spreader thickness); a pickup target below is farther. Kept
 * tighter than a typical hover gap ("a few feet") so an empty spreader
 * hovering above its pickup container fails the cue.
 */
export const CARRIED_MAX_Z_DELTA_MM = 800
/**
 * Uncalibrated fallback for the depth test: a carried container's bbox is the
 * same real length as the spreader at (almost) the same depth, so their pixel
 * sizes should agree within this ratio. Containers farther below read smaller.
 */
export const CARRIED_SIZE_RATIO_MAX = 1.3
/** Seconds of accumulated carried evidence (while moving) before acquiring. */
export const CARRIED_ACQUIRE_SECONDS = 0.7
/** Seconds without seeing the carried track before releasing the flag. */
export const CARRIED_RELEASE_SECONDS = 1.5
/**
 * Seconds the carried track must stay visible while FAILING the carried cues
 * before it is considered detached (put-down / false acquisition).
 */
export const CARRIED_DETACH_SECONDS = 1.0

/** Background (non-carried containers) median speed below this = "still". */
export const BACKGROUND_STILL_EPS = 0.012
/** Background must be still for this long before a lock can fire. */
export const BACKGROUND_STILL_SECONDS = 1.0
/** Spreader center speed above this counts as "spreader moving". */
export const SPREADER_MOVING_SPEED = 0.008
/** Spreader relative bbox-scale rate (1/s) above this counts as scaling. */
export const SPREADER_SCALE_RATE = 0.01
/** Spreader must be descending (while background is still) this long to lock. */
export const SPREADER_MOVING_SECONDS = 0.5
/** Direction timers hold through neutral (noisy) frames for this long. */
export const DIRECTION_HOLD_SECONDS = 0.6
/**
 * Absolute Z the spreader must have descended since the descent streak began
 * before a lock may fire. The bbox-scale-rate direction test alone is a
 * relative measure and a noisy hovering spreader can fake a sustained streak;
 * requiring real depth travel keeps locks off the hover/travel phases.
 */
export const LOCK_MIN_DESCENT_MM = 100

/** Empty-target hysteresis: switch only when the challenger is this much closer. */
export const EMPTY_TARGET_SWITCH_RATIO = 0.8
/** Seconds without seeing the empty target before switching to the nearest. */
export const EMPTY_TARGET_LOST_SECONDS = 1.0
/**
 * Min overlap (fraction of the smaller box) between a container and the
 * spreader's GEOMETRY box for the container to be a pickup/placement target
 * candidate. A container that is not under the spreader can never be the
 * physical target, no matter how close to the frame center it sits.
 */
export const TARGET_OVERLAP_MIN_FRAC = 0.2

// --- Merged (coincident) blobs ------------------------------------------------
// Detectors regularly emit ONE box covering the spreader and the container
// directly in line with it, and report it under both classes. Such a
// `container` track is not an independent observation: its box is the
// spreader's box, so it says nothing about whether a load is attached. It must
// never be used as attachment evidence, nor as a placement target, nor as
// static background — but while carrying it is the best visual handle on the
// load, and while empty it is the best handle on the target below.
/** Min overlap (fraction of the smaller box) for a container to count as merged. */
export const COINCIDENT_OVERLAP_FRAC = 0.8
/** Max size ratio (either direction) for a container to count as merged. */
export const COINCIDENT_SIZE_RATIO = 1.3

/** Sustained background motion aborts an active lock episode. */
export const EPISODE_ABORT_BACKGROUND_SECONDS = 1.0
/**
 * How long a load-state class must disagree with the current carry state
 * before the state flips. Per-frame classification of "loaded" vs "empty"
 * flickers around the transition (twist locks engaging, load just clearing the
 * stack); the first reading of a video is accepted immediately instead, since
 * nothing else can establish the state and it cannot have changed yet.
 */
export const LOAD_CLASS_CONFIRM_SECONDS = 1.0

// --- Sequence (Z-profile) cycle inference ------------------------------------
// In nadir camera geometries the carried container hangs directly under the
// spreader and is invisible, so the crane cycle must be inferred from the
// spreader's own Z profile: descend → plateau (touchdown) → hoist.
/** Min descent below the lock plane for an episode to count as a real approach. */
export const PICKUP_MIN_DESCENT_MM = 300
/** Rise above the episode's deepest plane that confirms touchdown-then-hoist. */
export const TOUCHDOWN_BACKOFF_MM = 300
/**
 * Min hoist above the pickup plane before a put-down becomes possible: a real
 * carry always lifts the load first, so cue-based "detach" while the spreader
 * is still at (or above, but never re-descended from) its pickup plane is a
 * false release, not a put-down.
 */
export const CARRY_HOIST_MIN_MM = 800

/** Max age of a track's previous sample used for velocity estimation. */
const MAX_VELOCITY_GAP_SECONDS = 0.5
/** Frozen-Z sampling window around the lock frame. */
const TARGET_Z_WINDOW_BEFORE_SECONDS = 2.0
const TARGET_Z_WINDOW_AFTER_SECONDS = 0.5

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

export type StackingState = 'idle' | 'pickup' | 'carrying' | 'locked'

export type StackingEpisodeKind = 'pickup' | 'place'

export interface StackingFrameInfo {
  state: StackingState
  /** Track id of the container currently held by the spreader, if detected. */
  carriedTrackId: number | null
  /** Locked target of the active episode (pickup or placement). */
  targetTrackId: number | null
  /** Frozen Z (mm) of the locked target's top (pickup/locked frames). */
  targetZMm: number | null
  /**
   * True when `targetZMm` came from physical contact inference rather than
   * from the target's own bbox. This is the normal case for a container the
   * spreader is descending onto: it is occluded by the load, so it often has
   * no detection at all and no box can be drawn for it.
   */
  targetZInferred: boolean
  /** Collision-constrained spreader Z used by the schematic after lock. */
  displayedSpreaderZMm: number | null
  /**
   * Remaining drop (mm). Placement: carried-container bottom → target top.
   * Pickup: spreader plane → target top. Locked/pickup frames only.
   */
  gapMm: number | null
  /**
   * Empty-spreader reference container: the track nearest the spreader
   * (idle frames only). Seeded with the just-placed container after put-down.
   */
  emptyTargetTrackId: number | null
  /** Live Z (mm) of the empty-spreader reference container's top. */
  emptyTargetZMm: number | null
  /** True when emptyTargetZMm came from the bbox flat-model fallback. */
  emptyTargetZEstimated: boolean
}

export interface StackingEpisode {
  kind: StackingEpisodeKind
  /** Frame index where the target was locked. */
  lockFrameIndex: number
  /** Frame the episode ended (put-down / pickup complete / abort), or null. */
  endFrameIndex: number | null
  /**
   * How the episode ended: 'completed' = physical touchdown happened (pickup
   * grabbed / placement put down), 'aborted' = lock premise broke before
   * contact, null = still running at the end of the video.
   */
  endReason: 'completed' | 'aborted' | null
  /**
   * Track id of the locked target container, or null for a BLIND lock (the
   * descent is real but no container detection exists under the spreader;
   * the target plane comes from touchdown contact inference).
   */
  targetTrackId: number | null
  /**
   * Contact-learned top plane (mm) for the target, snapshotted when the
   * target was locked or adopted. Snapshotting is what keeps a later episode
   * from rewriting the plane an earlier one was measured against.
   */
  targetKnownZMm: number | null
  /** Frozen Z (mm) of the locked target container's top. */
  targetZMm: number | null
}

export interface StackingAnalysis {
  /** Per-frame info, index-aligned with the input frames array. */
  frames: StackingFrameInfo[]
  /** Lock episodes (pickups and placements) in chronological order. */
  episodes: StackingEpisode[]
  /**
   * Where the carry state came from. `'class'` means the model reported it
   * directly (load-state-aware spreader classes); `'inferred'` means it was
   * derived from the crane's motion, which cannot resolve a video that starts
   * with a load already attached.
   */
  loadStateSource: 'class' | 'inferred'
}

export interface StackingAnalysisInput {
  frames: InferenceResult[]
  spreaderClass: string
  containerClass: string
  videoWidth: number
  videoHeight: number
  calibration: ZCalibration | null
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

interface TrackSample {
  cx: number
  cy: number
  size: number
  t: number
}

interface Velocity {
  vx: number
  vy: number
  speed: number
  /** Signed relative scale rate, d(size)/dt / size, in 1/s (negative = shrinking). */
  relScaleRateSigned: number
  /** |relScaleRateSigned|. */
  relScaleRate: number
}

function sampleOf(det: Detection, t: number): TrackSample {
  return {
    cx: det.box.x,
    cy: det.box.y,
    size: Math.max(det.box.width, det.box.height),
    t,
  }
}

function velocityBetween(prev: TrackSample, cur: TrackSample): Velocity | null {
  const dt = cur.t - prev.t
  if (dt <= 0 || dt > MAX_VELOCITY_GAP_SECONDS) return null
  const vx = (cur.cx - prev.cx) / dt
  const vy = (cur.cy - prev.cy) / dt
  const meanSize = (cur.size + prev.size) / 2
  const relScaleRateSigned = meanSize > 0 ? (cur.size - prev.size) / dt / meanSize : 0
  return {
    vx,
    vy,
    speed: Math.hypot(vx, vy),
    relScaleRateSigned,
    relScaleRate: Math.abs(relScaleRateSigned),
  }
}

function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

/** Fraction of box `a`'s area covered by its intersection with box `b`. */
function intersectionFracOfA(a: Detection, b: Detection): number {
  const areaA = a.box.width * a.box.height
  if (areaA <= 0) return 0
  const ix =
    Math.min(a.box.x + a.box.width / 2, b.box.x + b.box.width / 2) -
    Math.max(a.box.x - a.box.width / 2, b.box.x - b.box.width / 2)
  const iy =
    Math.min(a.box.y + a.box.height / 2, b.box.y + b.box.height / 2) -
    Math.max(a.box.y - a.box.height / 2, b.box.y - b.box.height / 2)
  if (ix <= 0 || iy <= 0) return 0
  return (ix * iy) / areaA
}

/**
 * True when a container box is effectively the same box as the spreader's —
 * i.e. the detector merged the two objects and reported the blob under both
 * classes. Such a track carries no independent information (see
 * `COINCIDENT_OVERLAP_FRAC`).
 */
function isMergedWithSpreader(spreader: Detection, container: Detection): boolean {
  if (overlapFracOfSmaller(spreader, container) < COINCIDENT_OVERLAP_FRAC) return false
  const sSize = Math.max(spreader.box.width, spreader.box.height)
  const cSize = Math.max(container.box.width, container.box.height)
  if (sSize <= 0 || cSize <= 0) return false
  return Math.max(sSize, cSize) / Math.min(sSize, cSize) <= COINCIDENT_SIZE_RATIO
}

/** Intersection area as a fraction of the smaller of the two boxes. */
function overlapFracOfSmaller(a: Detection, b: Detection): number {
  const areaA = a.box.width * a.box.height
  const areaB = b.box.width * b.box.height
  if (areaA <= 0 || areaB <= 0) return 0
  return areaA <= areaB ? intersectionFracOfA(a, b) : intersectionFracOfA(b, a)
}

/** True when the bbox is (possibly) clipped by the frame boundary. */
function touchesFrameEdge(det: Detection): boolean {
  return (
    det.box.x - det.box.width / 2 <= EDGE_CLIP_MARGIN ||
    det.box.x + det.box.width / 2 >= 1 - EDGE_CLIP_MARGIN ||
    det.box.y - det.box.height / 2 <= EDGE_CLIP_MARGIN ||
    det.box.y + det.box.height / 2 >= 1 - EDGE_CLIP_MARGIN
  )
}

/** Best spreader detection this frame: prefer matched tracks, then confidence. */
function pickSpreader(frame: InferenceResult, spreaderClass: string): Detection | undefined {
  let best: Detection | undefined
  for (const d of frame.detections) {
    if (d.class_name !== spreaderClass) continue
    if (!best) {
      best = d
      continue
    }
    const bestLost = best.track_source === 'lost'
    const dLost = d.track_source === 'lost'
    if (bestLost !== dLost) {
      if (bestLost) best = d
      continue
    }
    if (d.confidence > best.confidence) best = d
  }
  return best
}

/**
 * The spreader BODY detection, anchored to the Z-reference feature.
 *
 * Detectors emit spurious extra `spreader` boxes on lookalike structures
 * (stacked containers seen end-on), and their confidence can momentarily
 * exceed the real spreader's — picking by confidence alone makes the body box
 * jump across the frame between consecutive frames. The calibration reference
 * feature (e.g. a round casting) is physically mounted ON the spreader, so the
 * body is the candidate whose box contains it.
 */
function pickGeometrySpreader(
  frame: InferenceResult,
  geometryClasses: string[],
  anchor: Detection | undefined,
): Detection | undefined {
  let best: Detection | undefined
  let bestScore = -Infinity
  for (const d of frame.detections) {
    if (!geometryClasses.includes(d.class_name)) continue
    let score = d.confidence
    if (anchor) {
      const containsAnchor =
        Math.abs(d.box.x - anchor.box.x) < d.box.width / 2 &&
        Math.abs(d.box.y - anchor.box.y) < d.box.height / 2
      // Containing the reference feature outranks any confidence margin.
      if (containsAnchor) score += 10
    }
    if (d.track_source === 'lost') score -= 1
    if (score > bestScore) {
      best = d
      bestScore = score
    }
  }
  return best
}

function zForDetection(
  det: Detection,
  input: StackingAnalysisInput,
): number | null {
  if (det.z_mm != null && Number.isFinite(det.z_mm) && det.z_mm > 0) return det.z_mm
  const z = computeZForBox(
    input.calibration,
    det.box,
    input.videoWidth,
    input.videoHeight,
    det.class_name,
  )
  return z != null && Number.isFinite(z) && z > 0 ? z : null
}

/** Like zForDetection but reports whether the flat-model fallback was used. */
function zWithSource(
  det: Detection,
  input: StackingAnalysisInput,
): { z: number; estimated: boolean } | null {
  if (det.z_mm != null && Number.isFinite(det.z_mm) && det.z_mm > 0) {
    return { z: det.z_mm, estimated: false }
  }
  const z = computeZForBox(
    input.calibration,
    det.box,
    input.videoWidth,
    input.videoHeight,
    det.class_name,
  )
  return z != null && Number.isFinite(z) && z > 0 ? { z, estimated: true } : null
}

/** Squared center-to-center distance between two detections (normalized units). */
function centerDistanceSq(a: Detection, b: Detection): number {
  const dx = a.box.x - b.box.x
  const dy = a.box.y - b.box.y
  return dx * dx + dy * dy
}

/**
 * Container nearest the spreader (center-to-center). The spreader descends
 * vertically, so the container on its line of movement — the pickup or
 * placement target — is the one whose box sits closest to (and typically
 * under) the spreader box. Falls back to frame-center distance when no
 * spreader detection is available. Prefers matched tracks over Kalman ghosts.
 */
function pickNearestToSpreader(
  candidates: Detection[],
  spreaderDet: Detection | undefined,
): Detection | undefined {
  const pool = candidates.filter((d) => d.track_source !== 'lost')
  const usable = pool.length > 0 ? pool : candidates
  const score = (d: Detection) =>
    spreaderDet ? centerDistanceSq(d, spreaderDet) : distanceToFrameCenter(d)
  let best: Detection | undefined
  for (const d of usable) {
    if (!best || score(d) < score(best)) best = d
  }
  return best
}

/**
 * The spreader class used for Z (the calibration reference) can be a small
 * PROXY feature (e.g. a round casting) that sits off-center on the spreader.
 * For SPATIAL reasoning — which container is under the spreader — the actual
 * full spreader body box is much better when the detector provides one.
 * Returns a class literally named like "spreader" that differs from the Z
 * class, or null when there is none.
 */
function resolveGeometrySpreaderClasses(
  frames: InferenceResult[],
  spreaderClass: string,
  containerClass: string,
): string[] {
  const names = new Set<string>()
  for (const f of frames) for (const d of f.detections) names.add(d.class_name)
  const out: string[] = []
  for (const c of Array.from(names).sort()) {
    if (c === spreaderClass || c === containerClass) continue
    if (/spreader/i.test(c) && !/container/i.test(c)) out.push(c)
  }
  return out
}

/**
 * Load state encoded in a spreader-body class name, for models trained to
 * distinguish the two (e.g. `spreader_loaded` / `spreader_empty`).
 *
 * This is the ONLY direct observation of whether the spreader carries a load.
 * Everything else in this module infers it indirectly from the crane's motion,
 * which cannot resolve a video that starts mid-cycle with a load already
 * attached: detectors emit a `container` box coincident with the spreader
 * whether or not something hangs from it, so no bbox cue distinguishes the
 * states. Returns null for a plain `spreader` class (load state unknown).
 */
function loadStateOfClass(className: string): 'loaded' | 'empty' | null {
  if (!/spreader/i.test(className)) return null
  if (/loaded|laden|full/i.test(className) && !/unladen/i.test(className)) return 'loaded'
  if (/empty|unladen|bare/i.test(className)) return 'empty'
  return null
}

// ---------------------------------------------------------------------------
// Main analysis
// ---------------------------------------------------------------------------

export function analyzeStacking(input: StackingAnalysisInput): StackingAnalysis {
  const { frames, spreaderClass, containerClass } = input

  const geometryClasses = resolveGeometrySpreaderClasses(frames, spreaderClass, containerClass)
  // A load-state-aware model splits the body class (`spreader_loaded` /
  // `spreader_empty`). When present it is authoritative for the carry state;
  // otherwise the carry is inferred from the crane's motion.
  const hasLoadStateClasses = geometryClasses.some((c) => loadStateOfClass(c) != null)

  const frameInfos: StackingFrameInfo[] = []
  const episodes: StackingEpisode[] = []

  // Container top planes learned from physical contact: when the spreader
  // puts a container down, its top IS the deepest spreader plane of that
  // carry. Used when the container's own bbox Z is unusable (edge-clipped).
  const knownTopZByTrack = new Map<number, number>()

  // Velocity bookkeeping.
  const lastContainerSample = new Map<number, TrackSample>()
  let lastSpreaderSample: TrackSample | null = null
  let lastGeometrySample: TrackSample | null = null

  // Carrying is a STATE, not a track: in nadir geometries the carried
  // container hangs under the spreader and is often invisible, so the state
  // can outlive (or never have) a visible carried track.
  let carrying = false
  const carriedQualifyTime = new Map<number, number>()
  let carriedTrackId: number | null = null
  let carriedLastSeenT: number | null = null
  let carriedDetachTime = 0
  // Spreader Z profile bookkeeping for sequence inference.
  let carryStartZMm: number | null = null
  let carryMinZMm: number | null = null
  let carryRedescendDeepestZMm: number | null = null
  let episodeDeepestZMm: number | null = null
  // Whether this carry has been hoisted clear of its pickup plane. Tracked
  // rather than derived, because a carry declared by the load-state class was
  // hoisted before the video started and has no pickup plane to compare to.
  let carryHoisted = false

  // Confirmed load state from the body class, when the model provides one.
  // Transitions need sustained evidence (per-frame classification flickers),
  // but the FIRST reading seeds immediately: it is the only thing that can
  // resolve a video starting mid-cycle, and the state cannot have changed yet.
  let classLoadState: 'loaded' | 'empty' | null = null
  let loadedEvidenceTime = 0
  let emptyEvidenceTime = 0

  // Lock-trigger timers.
  let backgroundStillTime = 0
  let backgroundMovingTime = 0
  let spreaderDescendTime = 0
  let spreaderAscendTime = 0
  let descendStartZMm: number | null = null
  let lastDirectionalT: number | null = null
  let prevT: number | null = null

  // Empty-spreader reference target.
  let emptyTargetTrackId: number | null = null
  let emptyTargetLastSeenT: number | null = null

  let activeEpisode: StackingEpisode | null = null

  const endEpisode = (frameIndex: number, reason: 'completed' | 'aborted') => {
    if (activeEpisode && activeEpisode.endFrameIndex == null) {
      activeEpisode.endFrameIndex = frameIndex
      activeEpisode.endReason = reason
    }
    activeEpisode = null
    episodeDeepestZMm = null
  }

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i]
    const t = frame.timestamp
    const dt = prevT != null && t > prevT ? t - prevT : 0
    prevT = t

    const spreaderDet = pickSpreader(frame, spreaderClass)
    // Spatial reference for "under the spreader" tests: the full spreader
    // body box when a dedicated class exists, else the Z-class box.
    const geometryDet =
      (geometryClasses.length > 0
        ? pickGeometrySpreader(frame, geometryClasses, spreaderDet)
        : undefined) ?? spreaderDet
    const containerDets = frame.detections.filter(
      (d) => d.class_name === containerClass && d.track_id != null,
    )
    // Containers physically under the spreader: the only valid pickup or
    // placement targets. An empty pool means "we cannot see the target",
    // which must never be papered over by picking some other container.
    const underSpreaderDets =
      geometryDet != null
        ? containerDets.filter(
            (d) => overlapFracOfSmaller(geometryDet, d) >= TARGET_OVERLAP_MIN_FRAC,
          )
        : []
    // Merged blobs: the detector's box for these IS the spreader's box.
    const mergedThisFrame = new Set<number>()
    if (geometryDet != null) {
      for (const d of containerDets) {
        if (isMergedWithSpreader(geometryDet, d)) mergedThisFrame.add(d.track_id as number)
      }
    }

    // --- Velocities from the previous sighting of each track ---------------
    const spreaderVel =
      spreaderDet && lastSpreaderSample
        ? velocityBetween(lastSpreaderSample, sampleOf(spreaderDet, t))
        : null
    // Lockstep must be judged against the spreader BODY: the Z-class box can
    // be a small off-center proxy (a round casting) whose image motion during
    // a descent differs from the body's, which would make a rigidly attached
    // load look detached.
    const geometryVel =
      geometryDet && lastGeometrySample
        ? velocityBetween(lastGeometrySample, sampleOf(geometryDet, t))
        : null
    const attachVel = geometryVel ?? spreaderVel

    const containerVels = new Map<number, Velocity>()
    for (const d of containerDets) {
      const tid = d.track_id as number
      const prev = lastContainerSample.get(tid)
      if (prev) {
        const v = velocityBetween(prev, sampleOf(d, t))
        if (v) containerVels.set(tid, v)
      }
    }

    const spreaderMoving =
      spreaderVel != null &&
      (spreaderVel.speed > SPREADER_MOVING_SPEED ||
        spreaderVel.relScaleRate > SPREADER_SCALE_RATE)
    // Overhead camera: bbox shrinking = moving away = descending.
    const spreaderDescendingNow =
      spreaderVel != null && spreaderVel.relScaleRateSigned < -SPREADER_SCALE_RATE
    const zSpreaderNow = spreaderDet ? zForDetection(spreaderDet, input) : null

    // --- Spreader-attached containers, and carried-container acquisition ------
    // `attachedThisFrame` = container tracks moving rigidly WITH the spreader
    // in THIS frame. Two distinct uses downstream: as acquisition evidence
    // (with the timing gates below), and as an exclusion set — an attached
    // track is neither static background nor a valid placement target.
    // Cues, all required:
    //   overlap  — the container box covers enough of the spreader box
    //              (intersection, not center-in-box, to survive close-range
    //              parallax from the offset camera).
    //   lockstep — screen velocities AND signed bbox-scale rates agree. A
    //              rigidly attached load translates and scales exactly with
    //              the spreader; a static container under a descending
    //              spreader fails the scale-rate half of the test.
    //   depth    — the container reads at the spreader's Z. This separates
    //              the carried box from a target/hover container below it.
    //              Skipped when the container bbox touches the frame edge:
    //              a clipped box under-measures pixel size, so its pinhole Z
    //              reads meters too far (typical when the load is close to
    //              the camera and larger than the field of view).
    // Qualification time only accumulates while the spreader is MOVING and
    // NOT DESCENDING: attachment must be proven by joint motion, and a load
    // is physically acquired by HOISTING it. During an empty descent the
    // container below can track the spreader's merged blob frame-perfectly
    // (offset nadir cameras with clipped boxes defeat every depth cue), so
    // descent evidence is never allowed to count.
    const attachedThisFrame = new Set<number>()
    if (spreaderDet) {
      const zSpreader = zSpreaderNow
      const spreaderSize = Math.max(
        spreaderDet.box.width * input.videoWidth,
        spreaderDet.box.height * input.videoHeight,
      )
      for (const d of containerDets) {
        const tid = d.track_id as number
        // A merged blob is the spreader's own box relabelled: it passes every
        // cue trivially whether or not a load is attached, so it can never be
        // evidence. Attachment for these comes from the Z-profile sequence.
        if (mergedThisFrame.has(tid)) continue
        const overlaps =
          intersectionFracOfA(geometryDet ?? spreaderDet, d) >= CARRIED_OVERLAP_MIN_FRAC
        if (!overlaps) continue

        const cv = containerVels.get(tid)
        const lockstep =
          attachVel == null ||
          cv == null ||
          (Math.hypot(cv.vx - attachVel.vx, cv.vy - attachVel.vy) < CARRIED_VELOCITY_EPS &&
            Math.abs(cv.relScaleRateSigned - attachVel.relScaleRateSigned) <
              CARRIED_SCALE_RATE_EPS)
        if (!lockstep) continue

        let atSpreaderDepth: boolean
        const zContainer = zForDetection(d, input)
        if (touchesFrameEdge(d)) {
          // Clipped bbox → size (and Z) unreliable; depth is inconclusive,
          // let overlap + lockstep decide.
          atSpreaderDepth = true
        } else if (zSpreader != null && zContainer != null) {
          atSpreaderDepth = Math.abs(zContainer - zSpreader) < CARRIED_MAX_Z_DELTA_MM
        } else {
          // No calibration: spreader and carried container share the same
          // real length at the same depth, so their pixel sizes must agree.
          const containerSize = Math.max(
            d.box.width * input.videoWidth,
            d.box.height * input.videoHeight,
          )
          const ratio =
            containerSize > 0 && spreaderSize > 0
              ? Math.max(containerSize, spreaderSize) / Math.min(containerSize, spreaderSize)
              : Infinity
          atSpreaderDepth = ratio < CARRIED_SIZE_RATIO_MAX
        }
        if (!atSpreaderDepth) continue

        attachedThisFrame.add(tid)
        if (spreaderMoving && !spreaderDescendingNow) {
          carriedQualifyTime.set(tid, (carriedQualifyTime.get(tid) ?? 0) + dt)
        }
        // Cues pass but spreader static/descending: hold without accumulating.
      }
      // Reset the acquire streak for tracks that failed the cues this frame.
      for (const tid of Array.from(carriedQualifyTime.keys())) {
        if (!attachedThisFrame.has(tid)) carriedQualifyTime.delete(tid)
      }
      // Acquire (or adopt, when carrying was inferred without a visible
      // track): longest-qualifying track past the threshold wins.
      if (carriedTrackId == null || !containerDets.some((d) => d.track_id === carriedTrackId)) {
        let bestTid: number | null = null
        let bestTime = 0
        for (const [tid, time] of carriedQualifyTime) {
          if (time >= CARRIED_ACQUIRE_SECONDS && time > bestTime) {
            bestTid = tid
            bestTime = time
          }
        }
        if (bestTid != null && bestTid !== carriedTrackId) {
          carriedTrackId = bestTid
          carriedLastSeenT = t
          carriedDetachTime = 0
          if (!carrying) {
            // Fresh cue-based acquisition: the carry starts here, still on the
            // pickup plane — the hoist is yet to come.
            carrying = true
            carryStartZMm = zSpreaderNow
            carryMinZMm = zSpreaderNow
            carryRedescendDeepestZMm = null
            carryHoisted = false
          }
        }
      }
    }

    // Everything that is not part of the static yard this frame: cue-proven
    // attachments plus merged blobs.
    const movesWithSpreader = new Set<number>(mergedThisFrame)
    for (const tid of attachedThisFrame) movesWithSpreader.add(tid)

    // Valid target containers for a lock, state-dependent:
    //   carrying — the container BELOW the load. Anything moving with the
    //              spreader is excluded: it is the load itself, so measuring
    //              against it would compare the load to itself (gap stuck
    //              near 0) — the bug that made placement descents unusable.
    //   empty    — merged blobs are kept. During a pickup descent the
    //              detector merges the spreader with the container directly
    //              beneath it, and that blob is the best available handle on
    //              the target.
    const targetCandidates = carrying
      ? underSpreaderDets.filter(
          (d) =>
            !movesWithSpreader.has(d.track_id as number) && d.track_id !== carriedTrackId,
        )
      : underSpreaderDets

    // --- Carried track visibility --------------------------------------------
    // Losing the track does NOT end the carry (the load is typically hidden
    // under the spreader); it only clears the visible-track reference.
    if (carriedTrackId != null) {
      if (containerDets.some((d) => d.track_id === carriedTrackId)) {
        carriedLastSeenT = t
      } else if (carriedLastSeenT != null && t - carriedLastSeenT > CARRIED_RELEASE_SECONDS) {
        carriedTrackId = null
        carriedLastSeenT = null
        carriedDetachTime = 0
      }
    }
    // Cue-based detach evidence: the carried track is visible, has SEPARATED
    // from the spreader's box, and no longer moves with it — the geometry the
    // detector reports after a release. While the track is still merged with
    // the spreader box there is nothing to detach from, so no evidence
    // accrues. Only a PUT-DOWN can act on this (see below); during the
    // post-pickup hoist the same signature is a false release.
    if (carriedTrackId != null && spreaderDet) {
      const visibleMatched = containerDets.some(
        (d) => d.track_id === carriedTrackId && d.track_source !== 'lost',
      )
      if (movesWithSpreader.has(carriedTrackId)) {
        carriedDetachTime = 0
      } else if (visibleMatched) {
        carriedDetachTime += dt
      }
    }

    // --- Load state from the body class (authoritative when available) --------
    if (hasLoadStateClasses && geometryDet) {
      const observed = loadStateOfClass(geometryDet.class_name)
      if (observed === 'loaded') {
        loadedEvidenceTime += dt
        emptyEvidenceTime = 0
      } else if (observed === 'empty') {
        emptyEvidenceTime += dt
        loadedEvidenceTime = 0
      }
      if (classLoadState == null) {
        classLoadState = observed
      } else if (loadedEvidenceTime >= LOAD_CLASS_CONFIRM_SECONDS) {
        classLoadState = 'loaded'
      } else if (emptyEvidenceTime >= LOAD_CLASS_CONFIRM_SECONDS) {
        classLoadState = 'empty'
      }
      // Begin a carry the motion-based inference cannot see: either the video
      // started mid-cycle, or the pickup happened off-camera.
      if (classLoadState === 'loaded' && !carrying) {
        carrying = true
        carryStartZMm = zSpreaderNow
        carryMinZMm = zSpreaderNow
        carryRedescendDeepestZMm = null
        // The hoist is a fact of the past, not something to wait for.
        carryHoisted = true
        if (activeEpisode?.kind === 'pickup') endEpisode(i, 'completed')
      }
    }

    // --- Spreader Z profile bookkeeping ---------------------------------------
    if (carrying && zSpreaderNow != null) {
      if (carryStartZMm == null) carryStartZMm = zSpreaderNow
      carryMinZMm = Math.min(carryMinZMm ?? zSpreaderNow, zSpreaderNow)
      if (
        carryStartZMm != null &&
        carryMinZMm != null &&
        carryMinZMm <= carryStartZMm - CARRY_HOIST_MIN_MM
      ) {
        carryHoisted = true
      }
      if (carryHoisted) {
        carryRedescendDeepestZMm = Math.max(carryRedescendDeepestZMm ?? -Infinity, zSpreaderNow)
      }
    }
    if (activeEpisode != null && zSpreaderNow != null) {
      episodeDeepestZMm = Math.max(episodeDeepestZMm ?? -Infinity, zSpreaderNow)
    }

    // --- Background stillness and spreader direction timers ------------------
    // "Background" means the static yard. Anything moving rigidly WITH the
    // spreader right now is excluded: detectors routinely emit the
    // spreader+load blob as a `container` track (identical bbox), and its
    // large apparent motion during a descent would otherwise read as the
    // trolley moving and abort every lock episode.
    const backgroundSpeeds: number[] = []
    for (const d of containerDets) {
      const tid = d.track_id as number
      if (tid === carriedTrackId || movesWithSpreader.has(tid)) continue
      if (d.track_source === 'lost') continue // Kalman ghosts fake motion/stillness
      const v = containerVels.get(tid)
      if (v) backgroundSpeeds.push(v.speed)
    }
    const backgroundSpeed = median(backgroundSpeeds)
    if (backgroundSpeed != null) {
      const still = backgroundSpeed < BACKGROUND_STILL_EPS
      backgroundStillTime = still ? backgroundStillTime + dt : 0
      backgroundMovingTime = still ? 0 : backgroundMovingTime + dt
    }
    // With no background velocity samples this frame, hold both timers.

    // Descent/ascent from the signed spreader bbox-scale rate: the camera is
    // overhead, so shrinking = moving away = descending. Neutral (noisy)
    // frames hold the running timer briefly instead of resetting it.
    const rs = spreaderVel?.relScaleRateSigned ?? null
    if (rs != null && rs < -SPREADER_SCALE_RATE) {
      if (spreaderDescendTime === 0) descendStartZMm = zSpreaderNow
      spreaderDescendTime += dt
      spreaderAscendTime = 0
      lastDirectionalT = t
    } else if (rs != null && rs > SPREADER_SCALE_RATE) {
      spreaderAscendTime += dt
      spreaderDescendTime = 0
      descendStartZMm = null
      lastDirectionalT = t
    } else if (lastDirectionalT == null || t - lastDirectionalT > DIRECTION_HOLD_SECONDS) {
      spreaderDescendTime = 0
      spreaderAscendTime = 0
      descendStartZMm = null
    }
    // Real depth travel since the streak began, not just a shrinking bbox.
    const descentSoFarMm =
      descendStartZMm != null && zSpreaderNow != null ? zSpreaderNow - descendStartZMm : null
    const descendingForReal =
      spreaderDescendTime >= SPREADER_MOVING_SECONDS &&
      descentSoFarMm != null &&
      descentSoFarMm >= LOCK_MIN_DESCENT_MM

    // --- Episode maintenance and sequence (Z-profile) transitions -------------
    // While carrying, adopt the visible spreader-attached blob as the carried
    // track so the overlay labels it as the load. This is display bookkeeping
    // only: it never sets `carrying` (that needs the hoist evidence above).
    if (carrying && (carriedTrackId == null || !movesWithSpreader.has(carriedTrackId))) {
      const loadDet = pickNearestToSpreader(
        containerDets.filter((d) => movesWithSpreader.has(d.track_id as number)),
        geometryDet,
      )
      if (loadDet) {
        carriedTrackId = loadDet.track_id as number
        carriedLastSeenT = t
        carriedDetachTime = 0
      }
    }
    // Adopt a target for a blind lock as soon as a genuine one becomes
    // visible. Attached blobs are excluded: while carrying, a container box
    // that tracks the spreader is the load, not the container below it.
    if (activeEpisode != null && activeEpisode.targetTrackId == null) {
      const adopted = pickNearestToSpreader(targetCandidates, geometryDet)
      if (adopted) {
        const tid = adopted.track_id as number
        activeEpisode.targetTrackId = tid
        activeEpisode.targetKnownZMm = knownTopZByTrack.get(tid) ?? null
      }
    }
    // Sustained background motion means the trolley/camera moved again: the
    // lock premise is void, abort the episode.
    if (activeEpisode && backgroundMovingTime >= EPISODE_ABORT_BACKGROUND_SECONDS) {
      endEpisode(i, 'aborted')
    }
    // Pickup completed via cues: a track now moves in joint lockstep.
    if (activeEpisode?.kind === 'pickup' && carrying) {
      endEpisode(i, 'completed')
    }
    // Pickup resolved via the Z profile: the spreader descended onto the
    // target, touched down, and is hoisting again — it now carries the load
    // even if no container track ever qualified (the load is hidden under
    // the spreader in nadir views).
    if (
      activeEpisode?.kind === 'pickup' &&
      spreaderAscendTime >= SPREADER_MOVING_SECONDS &&
      episodeDeepestZMm != null &&
      zSpreaderNow != null &&
      episodeDeepestZMm - zSpreaderNow > TOUCHDOWN_BACKOFF_MM
    ) {
      const ep = activeEpisode as StackingEpisode
      const contactZ = episodeDeepestZMm
      const lockDet = pickSpreader(frames[ep.lockFrameIndex], spreaderClass)
      const lockZ = lockDet ? zForDetection(lockDet, input) : null
      const reallyDescended = lockZ != null && contactZ - lockZ >= PICKUP_MIN_DESCENT_MM
      if (reallyDescended) {
        endEpisode(i, 'completed')
        carrying = true
        if (ep.targetTrackId != null) {
          carriedTrackId = ep.targetTrackId
          carriedLastSeenT = t
          // Future-facing plane: with the target lifted away, the track's
          // area now exposes the container below (one height deeper).
          knownTopZByTrack.set(ep.targetTrackId, contactZ + ISO_CONTAINER_HEIGHT_MM)
        }
        carriedDetachTime = 0
        carryStartZMm = contactZ
        carryMinZMm = zSpreaderNow
        carryRedescendDeepestZMm = null
        carryHoisted = false
        backgroundStillTime = 0
      } else {
        // Barely descended before pulling back up: an aborted approach.
        endEpisode(i, 'aborted')
      }
    }
    // NOTE: there is deliberately no separate "aborted on ascent" rule. The
    // Z-profile branch above already resolves BOTH outcomes off the same
    // trigger (real rise off the deepest plane), routing to 'aborted' when
    // the approach never really descended. A timer-only ascent rule fired on
    // scale noise during landing plateaus — twist-lock engagement can hold
    // the spreader at the touchdown plane for tens of seconds — and killed
    // otherwise valid episodes.

    // Put-down: only possible after the carry actually HOISTED the load and
    // re-descended to a new plane (touchdown), and the spreader is rising
    // again. Confirmed by either the Z profile alone or cue-based detach.
    if (carrying) {
      const putDownEligible =
        carryHoisted &&
        carryRedescendDeepestZMm != null &&
        carryMinZMm != null &&
        carryRedescendDeepestZMm >= carryMinZMm + CARRY_HOIST_MIN_MM
      const zProfilePutDown =
        putDownEligible &&
        zSpreaderNow != null &&
        carryRedescendDeepestZMm! - zSpreaderNow > TOUCHDOWN_BACKOFF_MM &&
        spreaderAscendTime >= SPREADER_MOVING_SECONDS
      const cuePutDown = putDownEligible && carriedDetachTime >= CARRIED_DETACH_SECONDS
      // Direct observation beats inference: if the model reports an empty
      // spreader, the load is gone, whatever the motion profile suggests.
      const classPutDown = classLoadState === 'empty'
      if (zProfilePutDown || cuePutDown || classPutDown) {
        // Cast: TS's narrowing of `activeEpisode` is stale here (it cannot
        // see that endEpisode() mutates it through the closure).
        const epNow = activeEpisode as StackingEpisode | null
        const placeTargetTrackId: number | null =
          epNow?.kind === 'place' ? epNow.targetTrackId : null
        if (epNow?.kind === 'place') endEpisode(i, 'completed')
        // Physical contact knowledge: the placed container's top is the
        // deepest spreader plane of the carry. After the put-down that plane
        // is the stack's NEW top, for the carried track and the slot's
        // (target) track alike. A class-driven release can fire without any
        // measured descent, in which case there is no contact plane to learn.
        const contact = carryRedescendDeepestZMm
        if (contact != null) {
          if (carriedTrackId != null) knownTopZByTrack.set(carriedTrackId, contact)
          if (placeTargetTrackId != null) knownTopZByTrack.set(placeTargetTrackId, contact)
        }
        // The released box becomes the empty-spreader reference target, so
        // the schematic measures spreader ↔ just-placed container on the
        // way up.
        emptyTargetTrackId = carriedTrackId ?? placeTargetTrackId
        emptyTargetLastSeenT = emptyTargetTrackId != null ? t : null
        carrying = false
        carriedTrackId = null
        carriedLastSeenT = null
        carriedDetachTime = 0
        carriedQualifyTime.clear()
        carryStartZMm = null
        carryMinZMm = null
        carryRedescendDeepestZMm = null
        carryHoisted = false
        backgroundStillTime = 0
      }
    }

    // --- Empty-spreader reference target (idle only) --------------------------
    // Nearest UNDER-SPREADER container, sticky with hysteresis. While a
    // pickup episode is active the target is frozen — no re-selection. When
    // nothing is under the spreader (target undetected / spreader between
    // bays), the reference goes null rather than latching a wrong neighbor.
    if (!carrying && activeEpisode == null && geometryDet) {
      const current = underSpreaderDets.find((d) => d.track_id === emptyTargetTrackId)
      if (current && current.track_source !== 'lost') emptyTargetLastSeenT = t
      const best = pickNearestToSpreader(underSpreaderDets, geometryDet)
      if (best != null) {
        if (emptyTargetTrackId == null) {
          emptyTargetTrackId = best.track_id as number
          emptyTargetLastSeenT = t
        } else if (current) {
          const bestDist = centerDistanceSq(best, geometryDet)
          const currentDist = centerDistanceSq(current, geometryDet)
          if (bestDist < currentDist * EMPTY_TARGET_SWITCH_RATIO * EMPTY_TARGET_SWITCH_RATIO) {
            emptyTargetTrackId = best.track_id as number
            emptyTargetLastSeenT = t
          }
        } else if (
          emptyTargetLastSeenT == null ||
          t - emptyTargetLastSeenT > EMPTY_TARGET_LOST_SECONDS
        ) {
          emptyTargetTrackId = best.track_id as number
          emptyTargetLastSeenT = t
        }
      } else if (
        emptyTargetTrackId != null &&
        (emptyTargetLastSeenT == null || t - emptyTargetLastSeenT > EMPTY_TARGET_LOST_SECONDS)
      ) {
        emptyTargetTrackId = null
        emptyTargetLastSeenT = null
      }
    }

    // --- Lock triggers: background still, spreader descending -----------------
    // A lock may be BLIND (targetTrackId null): the descent is real even when
    // the detector shows no container under the spreader (target hidden by
    // the spreader/load, or simply missed). The target plane then comes from
    // touchdown contact inference; a visible target is adopted later if one
    // appears.
    if (
      activeEpisode == null &&
      backgroundStillTime >= BACKGROUND_STILL_SECONDS &&
      descendingForReal
    ) {
      if (carrying && carriedDetachTime === 0) {
        // Placement: loaded spreader descending onto a stack. Prefer the
        // UNDER-SPREADER container nearest the spreader (NOT the frame
        // center: parallax puts the true target under the spreader's line of
        // movement, not under the image center).
        const target = pickNearestToSpreader(targetCandidates, geometryDet)
        activeEpisode = {
          kind: 'place',
          lockFrameIndex: i,
          endFrameIndex: null,
          endReason: null,
          targetTrackId: target != null ? (target.track_id as number) : null,
          targetKnownZMm:
            target != null ? knownTopZByTrack.get(target.track_id as number) ?? null : null,
          targetZMm: null,
        }
        episodes.push(activeEpisode)
        episodeDeepestZMm = zSpreaderNow
        emptyTargetTrackId = null
        emptyTargetLastSeenT = null
      } else if (!carrying) {
        // Pickup: empty spreader descending onto its reference container
        // (or blindly, when the target is not detected).
        const targetVisible: boolean =
          emptyTargetTrackId != null &&
          underSpreaderDets.some((d) => d.track_id === emptyTargetTrackId)
        activeEpisode = {
          kind: 'pickup',
          lockFrameIndex: i,
          endFrameIndex: null,
          endReason: null,
          targetTrackId: targetVisible ? emptyTargetTrackId : null,
          targetKnownZMm:
            targetVisible && emptyTargetTrackId != null
              ? knownTopZByTrack.get(emptyTargetTrackId) ?? null
              : null,
          targetZMm: null,
        }
        episodes.push(activeEpisode)
        episodeDeepestZMm = zSpreaderNow
      }
    }

    // --- Bookkeeping for next frame -----------------------------------------
    if (spreaderDet) lastSpreaderSample = sampleOf(spreaderDet, t)
    if (geometryDet) lastGeometrySample = sampleOf(geometryDet, t)
    for (const d of containerDets) {
      lastContainerSample.set(d.track_id as number, sampleOf(d, t))
    }

    const state: StackingState =
      activeEpisode != null
        ? activeEpisode.kind === 'place'
          ? 'locked'
          : 'pickup'
        : carrying
          ? 'carrying'
          : 'idle'

    let emptyTargetZMm: number | null = null
    let emptyTargetZEstimated = false
    if (state === 'idle' && emptyTargetTrackId != null) {
      const det = containerDets.find((d) => d.track_id === emptyTargetTrackId)
      const knownTopZ = knownTopZByTrack.get(emptyTargetTrackId)
      if (knownTopZ != null) {
        // A plane learned from physical contact (touchdown/put-down) beats
        // any pinhole read: the container is static, so its Z cannot change
        // until the spreader touches it again.
        emptyTargetZMm = knownTopZ
        emptyTargetZEstimated = false
      } else if (det && !touchesFrameEdge(det)) {
        const zs = zWithSource(det, input)
        if (zs) {
          emptyTargetZMm = zs.z
          emptyTargetZEstimated = zs.estimated
        }
      }
    }

    frameInfos.push({
      state,
      carriedTrackId,
      targetTrackId: activeEpisode?.targetTrackId ?? null,
      targetZMm: null,
      displayedSpreaderZMm: null,
      gapMm: null,
      targetZInferred: false,
      emptyTargetTrackId: state === 'idle' ? emptyTargetTrackId : null,
      emptyTargetZMm,
      emptyTargetZEstimated,
    })
  }

  // --- Per-episode post-processing -------------------------------------------
  for (const episode of episodes) {
    const lockIndex = episode.lockFrameIndex
    const endIndex = episode.endFrameIndex ?? frames.length - 1

    // Measured spreader trajectory over the episode; the deepest in-episode
    // spreader plane is the physical contact plane (touchdown).
    const contactOffsetMm = episode.kind === 'place' ? ISO_CONTAINER_HEIGHT_MM : 0
    const spreaderZByFrame = new Map<number, number>()
    let contactSpreaderZMm: number | null = null
    for (let i = lockIndex; i <= endIndex; i++) {
      const spreaderDet = pickSpreader(frames[i], input.spreaderClass)
      if (!spreaderDet) continue
      const zSpreader = zForDetection(spreaderDet, input)
      if (zSpreader == null) continue
      spreaderZByFrame.set(i, zSpreader)
      contactSpreaderZMm = Math.max(contactSpreaderZMm ?? -Infinity, zSpreader)
    }

    // Frozen target Z: median around the lock frame, from UNCLIPPED boxes
    // only — an edge-clipped bbox under-measures pixel size and its pinhole Z
    // is meters off. When no clean sample exists, fall back to physical
    // knowledge: the top plane learned from an earlier put-down of this exact
    // track, else the touchdown contact plane of this very episode (offline
    // look-ahead, consistent with the module's design).
    const tLock = frames[lockIndex].timestamp
    const zSamples: number[] = []
    const zSamplesLoose: number[] = []
    for (const frame of frames) {
      if (frame.timestamp < tLock - TARGET_Z_WINDOW_BEFORE_SECONDS) continue
      if (frame.timestamp > tLock + TARGET_Z_WINDOW_AFTER_SECONDS) break
      for (const d of frame.detections) {
        if (d.class_name !== containerClass || d.track_id !== episode.targetTrackId) continue
        if (touchesFrameEdge(d)) continue
        const z = zForDetection(d, input)
        if (z == null) continue
        zSamplesLoose.push(z)
        if (d.track_source !== 'lost') zSamples.push(z)
      }
    }
    // Frozen target Z, most-physical source first:
    //   1. This episode's own touchdown contact (only when the descent really
    //      finished; an aborted episode never touched down).
    //   2. A plane learned from an earlier physical contact on this track.
    //   3. Pinhole Z medians from unclipped bboxes (weakest: survives only
    //      when no contact knowledge exists).
    const touchedDown = episode.endReason !== 'aborted'
    const contactZMm =
      touchedDown && contactSpreaderZMm != null ? contactSpreaderZMm + contactOffsetMm : null
    episode.targetZMm =
      contactZMm ?? episode.targetKnownZMm ?? median(zSamples) ?? median(zSamplesLoose)
    if (episode.targetZMm == null) continue
    const targetZMm = episode.targetZMm
    const targetZInferred = contactZMm != null || episode.targetKnownZMm != null

    // Future-facing contact knowledge for later episodes/idle phases on the
    // same track: a completed placement leaves its NEW top at the contact
    // plane; a completed pickup removes the target and exposes the container
    // below (one height deeper).
    if (
      episode.targetTrackId != null &&
      episode.endReason === 'completed' &&
      contactSpreaderZMm != null
    ) {
      knownTopZByTrack.set(
        episode.targetTrackId,
        episode.kind === 'place'
          ? contactSpreaderZMm
          : contactSpreaderZMm + ISO_CONTAINER_HEIGHT_MM,
      )
    }

    // Continuous constrained descent between two fixed points:
    //   displayed(lock)    = measured(lock)          (continuity at lock)
    //   displayed(contact) = contact plane           (rigid-body contact)
    // Placement contact plane: targetTop − containerHeight (the carried box
    // rests on the target). Pickup contact plane: targetTop (the spreader
    // plane meets the container's top).

    const lockSpreaderZMm = spreaderZByFrame.get(lockIndex) ?? null
    if (lockSpreaderZMm != null && contactSpreaderZMm != null) {
      const measuredDescentMm = contactSpreaderZMm - lockSpreaderZMm
      const displayedContactZMm = targetZMm - contactOffsetMm
      const displayedDescentMm = displayedContactZMm - lockSpreaderZMm

      for (const [i, zSpreader] of spreaderZByFrame) {
        let displayedSpreaderZMm: number
        if (measuredDescentMm > 0 && displayedDescentMm >= 0) {
          const descentProgress = (zSpreader - lockSpreaderZMm) / measuredDescentMm
          displayedSpreaderZMm = lockSpreaderZMm + descentProgress * displayedDescentMm
        } else {
          // Degenerate/inconsistent calibration: preserve continuity and report
          // the physically valid part of the direct geometry.
          displayedSpreaderZMm = zSpreader
        }
        frameInfos[i].displayedSpreaderZMm = displayedSpreaderZMm
        frameInfos[i].gapMm = Math.max(
          0,
          targetZMm - (displayedSpreaderZMm + contactOffsetMm),
        )
      }
    }

    // Fill the frozen target into every frame of the episode, and backfill it
    // through the carrying phase preceding a placement lock so the schematic
    // can reserve the target's vertical extent (no auto-zoom jump at lock).
    for (let i = lockIndex; i <= endIndex; i++) {
      frameInfos[i].targetZMm = targetZMm
      frameInfos[i].targetTrackId = episode.targetTrackId
      frameInfos[i].targetZInferred = targetZInferred
    }
    if (episode.kind === 'place') {
      for (let j = lockIndex - 1; j >= 0 && frameInfos[j].state === 'carrying'; j--) {
        frameInfos[j].targetZMm = targetZMm
        frameInfos[j].targetTrackId = episode.targetTrackId
        frameInfos[j].targetZInferred = targetZInferred
      }
    }
  }

  return {
    frames: frameInfos,
    episodes,
    loadStateSource: hasLoadStateClasses ? 'class' : 'inferred',
  }
}
