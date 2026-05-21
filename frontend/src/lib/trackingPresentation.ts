import type { Detection, InferenceResult } from '@/types'
import type { OverlayBox } from '@/components/DetectionOverlaySvg'

export const DEFAULT_TRACKER_PARAMS = {
  track_activation_threshold: 0.25,
  lost_track_buffer: 30,
  minimum_matching_threshold: 0.8,
} as const

export const DEFAULT_OEF = {
  enabled: true,
  minCutoff: 1.0,
  beta: 0.007,
} as const

export const COLOR_MEASURED = '#34d399'
export const COLOR_EXTRAPOLATED = '#fbbf24'

export function findClosestFrameIndex(frames: InferenceResult[], time: number): number {
  if (frames.length === 0) return -1
  let lo = 0
  let hi = frames.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].timestamp < time) lo = mid + 1
    else hi = mid
  }
  if (lo > 0 && Math.abs(frames[lo - 1].timestamp - time) <= Math.abs(frames[lo].timestamp - time)) {
    return lo - 1
  }
  return lo
}

export function bestByConfidence(detections: Detection[]): Detection | undefined {
  return detections.reduce<Detection | undefined>(
    (best, d) => (!best || d.confidence > best.confidence ? d : best),
    undefined,
  )
}

export function isContainerClass(className: string | null | undefined): boolean {
  return /container/i.test(className ?? '')
}

export function distanceToFrameCenter(det: Detection): number {
  const dx = det.box.x - 0.5
  const dy = det.box.y - 0.5
  return dx * dx + dy * dy
}

export function pickCenterDetection(detections: Detection[]): Detection | undefined {
  return detections.reduce<Detection | undefined>(
    (best, d) => (!best || distanceToFrameCenter(d) < distanceToFrameCenter(best) ? d : best),
    undefined,
  )
}

/**
 * Presentation policy for the single-target use case: keep ByteTrack running on
 * every detection, then show one stable primary track per class. We preserve a
 * matched current primary when possible to avoid confidence-based ID flicker;
 * if it is only a lost prediction and ByteTrack has a fresh matched track, we
 * switch to that matched track so reacquisition beats stale extrapolation.
 */
export function pickPrimaryTrackPerClassFrames(frames: InferenceResult[]): InferenceResult[] {
  const primaryTrackByClass = new Map<string, number>()

  return frames.map((frame) => {
    const detectionsByClass = new Map<string, Detection[]>()
    for (const d of frame.detections) {
      const arr = detectionsByClass.get(d.class_name)
      if (arr) arr.push(d)
      else detectionsByClass.set(d.class_name, [d])
    }

    const detections: Detection[] = []
    for (const [className, classDetections] of detectionsByClass) {
      if (isContainerClass(className)) {
        const chosen = pickCenterDetection(classDetections)
        if (chosen) detections.push(chosen)
        continue
      }

      const primaryTrackId = primaryTrackByClass.get(className)
      const currentPrimary =
        primaryTrackId != null
          ? classDetections.find((d) => d.track_id === primaryTrackId)
          : undefined
      const matchedDetections = classDetections.filter((d) => d.track_source !== 'lost')

      const chosen =
        currentPrimary && currentPrimary.track_source !== 'lost'
          ? currentPrimary
          : bestByConfidence(matchedDetections) ?? currentPrimary ?? bestByConfidence(classDetections)

      if (!chosen) continue
      if (chosen.track_id != null) primaryTrackByClass.set(className, chosen.track_id)
      detections.push(chosen)
    }

    return { ...frame, detections }
  })
}

/**
 * Tracked-side overlay: the selected primary ByteTrack detections, coloured from
 * backend provenance. "matched" means ByteTrack fused a detector measurement on
 * this frame; "lost" means this box is a Kalman-only prediction.
 */
export function buildTrackedOverlayBoxes(
  trackedFrame: InferenceResult | null,
): { boxes: OverlayBox[]; measured: number; extrapolated: number } {
  if (!trackedFrame) return { boxes: [], measured: 0, extrapolated: 0 }
  let measured = 0
  let extrapolated = 0
  const boxes: OverlayBox[] = []
  trackedFrame.detections.forEach((d, i) => {
    const isMeasured = d.track_source !== 'lost'
    if (isMeasured) measured++
    else extrapolated++
    const label = d.track_id != null ? `#${d.track_id} ${d.class_name}` : d.class_name
    boxes.push({
      key: `trk-${d.track_id ?? `${d.class_name}-${i}`}`,
      box: d.box,
      color: isMeasured ? COLOR_MEASURED : COLOR_EXTRAPOLATED,
      label,
      dashed: !isMeasured,
    })
  })
  return { boxes, measured, extrapolated }
}
