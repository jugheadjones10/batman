import { useMemo, useState } from 'react'
import type { BoundingBox, InferenceResult, ZCalibration } from '@/types'

const DETECTION_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

const SPEED_OPTIONS = [
  { label: '1x', skip: 1 },
  { label: '2x', skip: 2 },
  { label: '5x', skip: 5 },
  { label: '10x', skip: 10 },
]

interface LiveDetectionReadoutProps {
  frames: InferenceResult[]
  currentTime: number
  videoWidth: number
  videoHeight: number
  zCalibration?: ZCalibration | null
}

interface ClassReadout {
  className: string
  color: string
  confidence: number | null
  trackId: number | null
  trackSource: 'matched' | 'lost' | null
  x: number | null
  y: number | null
  z: number | null
}

function findClosestFrameIndex(frames: InferenceResult[], time: number): number {
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

function computeZForBox(
  cal: ZCalibration | null | undefined,
  box: BoundingBox,
  vw: number,
  vh: number,
): number | null {
  if (!cal || !cal.model) return null
  const s = Math.max(box.width * vw, box.height * vh)
  if (s <= 0) return null
  if (cal.model.type === 'k_over_s' && cal.model.k != null) return cal.model.k / s
  if (cal.model.type === 'linear_inv' && cal.model.m != null && cal.model.c != null) {
    return cal.model.m / s + cal.model.c
  }
  return null
}

export default function LiveDetectionReadout({
  frames,
  currentTime,
  videoWidth,
  videoHeight,
  zCalibration,
}: LiveDetectionReadoutProps) {
  const [skipFrames, setSkipFrames] = useState(1)

  const classNames = useMemo(() => {
    const names = new Set<string>()
    for (const f of frames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [frames])

  const classColorMap = useMemo(() => {
    const map: Record<string, string> = {}
    classNames.forEach((name, i) => {
      map[name] = DETECTION_COLORS[i % DETECTION_COLORS.length]
    })
    return map
  }, [classNames])

  const rawFrameIndex = findClosestFrameIndex(frames, currentTime)
  const quantizedIndex = skipFrames > 1
    ? Math.floor(rawFrameIndex / skipFrames) * skipFrames
    : rawFrameIndex

  const currentFrame = quantizedIndex >= 0 && quantizedIndex < frames.length
    ? frames[Math.min(quantizedIndex, frames.length - 1)]
    : null

  const readouts = useMemo<ClassReadout[]>(() => {
    const idx = Math.min(quantizedIndex, frames.length - 1)
    const frame = idx >= 0 ? frames[idx] : null
    const result = classNames.map((cls) => {
      if (!frame) {
        return {
          className: cls,
          color: classColorMap[cls],
          confidence: null,
          trackId: null,
          trackSource: null,
          x: null,
          y: null,
          z: null,
        }
      }
      const dets = frame.detections.filter((d) => d.class_name === cls)
      if (dets.length === 0) {
        return {
          className: cls,
          color: classColorMap[cls],
          confidence: null,
          trackId: null,
          trackSource: null,
          x: null,
          y: null,
          z: null,
        }
      }
      const best = dets.reduce((a, b) => (a.confidence > b.confidence ? a : b))
      return {
        className: cls,
        color: classColorMap[cls],
        confidence: best.confidence,
        trackId: best.track_id ?? null,
        trackSource: best.track_source ?? null,
        x: best.box.x * videoWidth,
        y: best.box.y * videoHeight,
        z: best.z_mm ?? computeZForBox(zCalibration, best.box, videoWidth, videoHeight),
      }
    })
    return result
  }, [quantizedIndex, frames, classNames, classColorMap, videoWidth, videoHeight, zCalibration])

  if (classNames.length === 0) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
          Live Tracked Detection Readout
        </span>
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground mr-1">Update speed</span>
          {SPEED_OPTIONS.map((opt) => (
            <button
              key={opt.skip}
              onClick={() => setSkipFrames(opt.skip)}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                skipFrames === opt.skip
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted/50 text-muted-foreground hover:bg-muted'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
      {readouts.map((r) => (
        <div key={r.className} className="rounded-lg border border-border bg-neutral-900/80 p-3">
          <div className="flex items-center gap-2 mb-2.5">
            <span
              className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
              style={{ backgroundColor: r.color }}
            />
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {r.className}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 2xl:grid-cols-6 gap-2">
            <ReadoutBox label="Confidence" value={r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : null} />
            <ReadoutBox label="Track" value={r.trackId != null ? `#${r.trackId}` : null} />
            <ReadoutBox
              label="Source"
              value={r.trackSource === 'lost' ? 'extrap.' : r.trackSource === 'matched' ? 'measured' : null}
            />
            <ReadoutBox label="X (px)" value={r.x != null ? r.x.toFixed(1) : null} />
            <ReadoutBox label="Y (px)" value={r.y != null ? r.y.toFixed(1) : null} />
            <ReadoutBox label="Z (mm)" value={r.z != null ? r.z.toFixed(1) : null} />
          </div>
        </div>
      ))}
      <div className="rounded-lg border border-border bg-neutral-900/80 p-3">
        <div className="flex items-center gap-2 mb-2.5">
          <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0 bg-amber-400" />
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Spreader ↔ Container Skew
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <ReadoutBox
            label="Skew (°)"
            value={currentFrame?.skew_deg != null ? currentFrame.skew_deg.toFixed(2) : null}
          />
          <ReadoutBox
            label="Spreader (°)"
            value={currentFrame?.spreader_deg != null ? currentFrame.spreader_deg.toFixed(2) : null}
          />
          <ReadoutBox
            label="Container (°)"
            value={currentFrame?.container_deg != null ? currentFrame.container_deg.toFixed(2) : null}
          />
        </div>
      </div>
    </div>
  )
}

function ReadoutBox({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-md bg-neutral-950 border border-neutral-800 px-3 py-2.5 text-center">
      <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
        {label}
      </div>
      <div className="font-mono text-lg font-semibold tabular-nums leading-none text-sky-400 min-h-[1.25rem]">
        {value ?? <span className="text-neutral-600">--</span>}
      </div>
    </div>
  )
}
