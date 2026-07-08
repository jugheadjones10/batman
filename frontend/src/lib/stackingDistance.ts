/**
 * Stacking-distance estimation for a loaded spreader.
 *
 * When the spreader is carrying a container and lowering it onto a stack, the
 * target container below is partially occluded, so the usual "center container"
 * pick is wrong (the carried container is what sits at screen center). This
 * module runs over the smoothed ByteTrack frames and:
 *
 *   1. Detects which container track is being CARRIED: it overlaps the
 *      spreader box, moves in lockstep with it, and reads at the spreader's
 *      depth (pinhole Z, or matching pixel size without a calibration). The
 *      depth cue is waived for boxes clipped by the frame edge, whose pixel
 *      size — and therefore Z — cannot be trusted.
 *   2. Detects the moment vertical movement starts: the surrounding
 *      (background) container tracks stop moving on screen — the trolley/camera
 *      is stationary — while the spreader itself keeps moving (bbox growing as
 *      it descends). At that moment we LOCK the target.
 *   3. Locks the TARGET container: the non-carried container track nearest the
 *      frame center at the lock frame.
 *   4. Freezes the target's Z (median over a small window around the lock).
 *      The target is static and the camera has stopped translating, so its true
 *      Z no longer changes; freezing makes the estimate immune to the growing
 *      occlusion from the descending carried container.
 *   5. Emits a per-frame remaining drop:
 *        gap = zTargetTop − (zSpreader + ISO_CONTAINER_HEIGHT_MM)
 *      i.e. distance from the carried container's bottom to the target's top.
 *
 * Everything is causal except the frozen-Z window, which peeks 0.5 s past the
 * lock frame for robustness (a live port would just delay the readout by that
 * much).
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
 * roughly the spreader thickness); a pickup target below is meters farther.
 */
export const CARRIED_MAX_Z_DELTA_MM = 1000
/**
 * Uncalibrated fallback for the depth test: a carried container's bbox is the
 * same real length as the spreader at (almost) the same depth, so their pixel
 * sizes should agree within this ratio. Containers farther below read smaller.
 */
export const CARRIED_SIZE_RATIO_MAX = 1.3
/** Seconds of accumulated carried evidence before a candidate is acquired. */
export const CARRIED_ACQUIRE_SECONDS = 0.7
/** Seconds without seeing the carried track before releasing the flag. */
export const CARRIED_RELEASE_SECONDS = 1.5

/** Background (non-carried containers) median speed below this = "still". */
export const BACKGROUND_STILL_EPS = 0.012
/** Background must be still for this long before a lock can fire. */
export const BACKGROUND_STILL_SECONDS = 1.0
/** Spreader center speed above this counts as "spreader moving". */
export const SPREADER_MOVING_SPEED = 0.008
/** Spreader relative bbox-scale rate (1/s) above this counts as moving (descending). */
export const SPREADER_SCALE_RATE = 0.01
/** Spreader must be moving (while background is still) for this long to lock. */
export const SPREADER_MOVING_SECONDS = 0.5

/** Max age of a track's previous sample used for velocity estimation. */
const MAX_VELOCITY_GAP_SECONDS = 0.5
/** Frozen-Z sampling window around the lock frame. */
const TARGET_Z_WINDOW_BEFORE_SECONDS = 2.0
const TARGET_Z_WINDOW_AFTER_SECONDS = 0.5

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

export type StackingState = 'idle' | 'carrying' | 'locked'

export interface StackingFrameInfo {
  state: StackingState
  /** Track id of the container currently held by the spreader, if detected. */
  carriedTrackId: number | null
  /** Remaining drop (mm) from carried-container bottom to target top; locked frames only. */
  gapMm: number | null
}

export interface StackingAnalysis {
  /** Per-frame info, index-aligned with the input frames array. */
  frames: StackingFrameInfo[]
  /** Frame index where the target was locked, or null if never locked. */
  lockFrameIndex: number | null
  /** Track id of the locked target container. */
  targetTrackId: number | null
  /** Frozen Z (mm) of the locked target container's top. */
  targetZMm: number | null
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
  /** Relative scale rate, |d(size)/dt| / size, in 1/s. */
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
  return {
    vx,
    vy,
    speed: Math.hypot(vx, vy),
    relScaleRate: meanSize > 0 ? Math.abs(cur.size - prev.size) / dt / meanSize : 0,
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

// ---------------------------------------------------------------------------
// Main analysis
// ---------------------------------------------------------------------------

export function analyzeStacking(input: StackingAnalysisInput): StackingAnalysis {
  const { frames, spreaderClass, containerClass } = input

  const frameInfos: StackingFrameInfo[] = []
  let lockFrameIndex: number | null = null
  let targetTrackId: number | null = null

  // Velocity bookkeeping.
  const lastContainerSample = new Map<number, TrackSample>()
  let lastSpreaderSample: TrackSample | null = null

  // Carried-container hysteresis.
  const carriedQualifyTime = new Map<number, number>()
  let carriedTrackId: number | null = null
  let carriedLastSeenT: number | null = null

  // Lock-trigger timers.
  let backgroundStillTime = 0
  let spreaderMovingTime = 0
  let prevT: number | null = null

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i]
    const t = frame.timestamp
    const dt = prevT != null && t > prevT ? t - prevT : 0
    prevT = t

    const spreaderDet = pickSpreader(frame, spreaderClass)
    const containerDets = frame.detections.filter(
      (d) => d.class_name === containerClass && d.track_id != null,
    )

    // --- Velocities from the previous sighting of each track ---------------
    const spreaderVel =
      spreaderDet && lastSpreaderSample
        ? velocityBetween(lastSpreaderSample, sampleOf(spreaderDet, t))
        : null

    const containerVels = new Map<number, Velocity>()
    for (const d of containerDets) {
      const tid = d.track_id as number
      const prev = lastContainerSample.get(tid)
      if (prev) {
        const v = velocityBetween(prev, sampleOf(d, t))
        if (v) containerVels.set(tid, v)
      }
    }

    // --- Carried-container detection ---------------------------------------
    // Three cues, all required:
    //   overlap  — the container box covers enough of the spreader box
    //              (intersection, not center-in-box, to survive close-range
    //              parallax from the offset camera).
    //   lockstep — screen velocities agree (the camera rides the trolley, so
    //              both are near-static during travel and grow together while
    //              hoisting; a mismatch rules a candidate out).
    //   depth    — the container reads at the spreader's Z. This is what
    //              separates the carried box from the target directly below
    //              it, which also overlaps and is also static on screen but
    //              is at least a container-height farther from the camera.
    //              Skipped when the container bbox touches the frame edge:
    //              a clipped box under-measures pixel size, so its pinhole Z
    //              reads meters too far (typical when the load is close to
    //              the camera and larger than the field of view).
    if (spreaderDet) {
      const zSpreader = zForDetection(spreaderDet, input)
      const spreaderSize = Math.max(
        spreaderDet.box.width * input.videoWidth,
        spreaderDet.box.height * input.videoHeight,
      )
      const seenThisFrame = new Set<number>()
      for (const d of containerDets) {
        const tid = d.track_id as number
        const overlaps = intersectionFracOfA(spreaderDet, d) >= CARRIED_OVERLAP_MIN_FRAC
        if (!overlaps) continue

        const cv = containerVels.get(tid)
        const lockstep =
          spreaderVel == null ||
          cv == null ||
          Math.hypot(cv.vx - spreaderVel.vx, cv.vy - spreaderVel.vy) < CARRIED_VELOCITY_EPS
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

        carriedQualifyTime.set(tid, (carriedQualifyTime.get(tid) ?? 0) + dt)
        seenThisFrame.add(tid)
      }
      // Reset the qualify streak for tracks that failed this frame.
      for (const tid of Array.from(carriedQualifyTime.keys())) {
        if (!seenThisFrame.has(tid)) carriedQualifyTime.delete(tid)
      }
      // Acquire: longest-qualifying track past the threshold wins.
      if (carriedTrackId == null || !containerDets.some((d) => d.track_id === carriedTrackId)) {
        let bestTid: number | null = null
        let bestTime = 0
        for (const [tid, time] of carriedQualifyTime) {
          if (time >= CARRIED_ACQUIRE_SECONDS && time > bestTime) {
            bestTid = tid
            bestTime = time
          }
        }
        if (bestTid != null) carriedTrackId = bestTid
      }
    }
    if (carriedTrackId != null) {
      if (containerDets.some((d) => d.track_id === carriedTrackId)) {
        carriedLastSeenT = t
      } else if (carriedLastSeenT != null && t - carriedLastSeenT > CARRIED_RELEASE_SECONDS) {
        carriedTrackId = null
        carriedLastSeenT = null
      }
    }

    // --- Lock trigger: background still, spreader moving --------------------
    if (lockFrameIndex == null) {
      const backgroundSpeeds: number[] = []
      for (const d of containerDets) {
        const tid = d.track_id as number
        if (tid === carriedTrackId) continue
        if (d.track_source === 'lost') continue // Kalman ghosts fake motion/stillness
        const v = containerVels.get(tid)
        if (v) backgroundSpeeds.push(v.speed)
      }
      const backgroundSpeed = median(backgroundSpeeds)
      if (backgroundSpeed != null) {
        backgroundStillTime = backgroundSpeed < BACKGROUND_STILL_EPS ? backgroundStillTime + dt : 0
      }
      // With no background velocity samples this frame, hold the timer.

      const spreaderMoving =
        spreaderVel != null &&
        (spreaderVel.speed > SPREADER_MOVING_SPEED ||
          spreaderVel.relScaleRate > SPREADER_SCALE_RATE)
      spreaderMovingTime = spreaderMoving ? spreaderMovingTime + dt : 0

      if (
        carriedTrackId != null &&
        backgroundStillTime >= BACKGROUND_STILL_SECONDS &&
        spreaderMovingTime >= SPREADER_MOVING_SECONDS
      ) {
        // Vertical movement has started: lock the target container now.
        const candidates = containerDets.filter(
          (d) => d.track_id !== carriedTrackId && d.track_source !== 'lost',
        )
        const pool = candidates.length > 0
          ? candidates
          : containerDets.filter((d) => d.track_id !== carriedTrackId)
        let target: Detection | undefined
        for (const d of pool) {
          if (!target || distanceToFrameCenter(d) < distanceToFrameCenter(target)) target = d
        }
        if (target) {
          lockFrameIndex = i
          targetTrackId = target.track_id as number
        }
      }
    }

    // --- Bookkeeping for next frame -----------------------------------------
    if (spreaderDet) lastSpreaderSample = sampleOf(spreaderDet, t)
    for (const d of containerDets) {
      lastContainerSample.set(d.track_id as number, sampleOf(d, t))
    }

    frameInfos.push({
      state: lockFrameIndex != null ? 'locked' : carriedTrackId != null ? 'carrying' : 'idle',
      carriedTrackId,
      gapMm: null,
    })
  }

  // --- Frozen target Z: median around the lock frame -------------------------
  let targetZMm: number | null = null
  if (lockFrameIndex != null && targetTrackId != null) {
    const tLock = frames[lockFrameIndex].timestamp
    const zSamples: number[] = []
    const zSamplesLoose: number[] = []
    for (const frame of frames) {
      if (frame.timestamp < tLock - TARGET_Z_WINDOW_BEFORE_SECONDS) continue
      if (frame.timestamp > tLock + TARGET_Z_WINDOW_AFTER_SECONDS) break
      for (const d of frame.detections) {
        if (d.class_name !== containerClass || d.track_id !== targetTrackId) continue
        const z = zForDetection(d, input)
        if (z == null) continue
        zSamplesLoose.push(z)
        if (d.track_source !== 'lost') zSamples.push(z)
      }
    }
    targetZMm = median(zSamples) ?? median(zSamplesLoose)
  }

  // --- Per-frame remaining drop after lock -----------------------------------
  if (lockFrameIndex != null && targetZMm != null) {
    for (let i = lockFrameIndex; i < frames.length; i++) {
      const spreaderDet = pickSpreader(frames[i], input.spreaderClass)
      if (!spreaderDet) continue
      const zSpreader = zForDetection(spreaderDet, input)
      if (zSpreader == null) continue
      frameInfos[i].gapMm = targetZMm - (zSpreader + ISO_CONTAINER_HEIGHT_MM)
    }
  }

  return {
    frames: frameInfos,
    lockFrameIndex,
    targetTrackId,
    targetZMm,
  }
}
