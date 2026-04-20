import { useMemo, useRef, useState } from 'react'
import { ArrowUpDown } from 'lucide-react'
import type { InferenceResult } from '@/types'

interface ZGapReadoutProps {
  frames: InferenceResult[]
  currentTime: number
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

export default function ZGapReadout({ frames, currentTime }: ZGapReadoutProps) {
  const classNames = useMemo(() => {
    const names = new Set<string>()
    for (const f of frames) {
      for (const d of f.detections) {
        if (d.z_mm != null) names.add(d.class_name)
      }
    }
    return Array.from(names).sort()
  }, [frames])

  const [classA, setClassA] = useState('')
  const [classB, setClassB] = useState('')

  const resolvedA = classA || classNames[0] || ''
  const resolvedB = classB || classNames[1] || classNames[0] || ''

  const prevRef = useRef<{ idx: number; gap: number | null; zA: number | null; zB: number | null }>({
    idx: -1, gap: null, zA: null, zB: null,
  })

  const frameIndex = findClosestFrameIndex(frames, currentTime)

  const { gap, zA, zB } = useMemo(() => {
    if (frameIndex === prevRef.current.idx) {
      return { gap: prevRef.current.gap, zA: prevRef.current.zA, zB: prevRef.current.zB }
    }

    const frame = frameIndex >= 0 && frameIndex < frames.length ? frames[frameIndex] : null
    if (!frame || !resolvedA || !resolvedB) {
      prevRef.current = { idx: frameIndex, gap: null, zA: null, zB: null }
      return { gap: null, zA: null, zB: null }
    }

    const detA = frame.detections.find((d) => d.class_name === resolvedA && d.z_mm != null)
    const detB = frame.detections.find((d) => d.class_name === resolvedB && d.z_mm != null)

    const zA = detA?.z_mm ?? null
    const zB = detB?.z_mm ?? null
    const gap = zA != null && zB != null ? Math.abs(zA - zB) : null

    prevRef.current = { idx: frameIndex, gap, zA, zB }
    return { gap, zA, zB }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIndex, frames, resolvedA, resolvedB])

  if (classNames.length < 2) return null

  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-1.5">
        <ArrowUpDown className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
          Z-Gap Estimate
        </span>
      </div>

      <div className="rounded-lg border border-border bg-neutral-900/80 p-3 space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-muted-foreground block mb-1">Class A</label>
            <select
              value={resolvedA}
              onChange={(e) => setClassA(e.target.value)}
              className="w-full rounded border bg-background px-2 py-1 text-xs h-7"
            >
              {classNames.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-muted-foreground block mb-1">Class B</label>
            <select
              value={resolvedB}
              onChange={(e) => setClassB(e.target.value)}
              className="w-full rounded border bg-background px-2 py-1 text-xs h-7"
            >
              {classNames.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-md bg-neutral-950 border border-neutral-800 px-3 py-2.5 text-center">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
              {resolvedA || 'A'} Z
            </div>
            <div className="font-mono text-base font-semibold tabular-nums leading-none text-sky-400 min-h-[1.25rem]">
              {zA != null ? `${zA.toFixed(0)}` : <span className="text-neutral-600">--</span>}
            </div>
            <div className="text-[9px] text-muted-foreground mt-0.5">mm</div>
          </div>
          <div className="rounded-md bg-neutral-950 border border-neutral-800 px-3 py-2.5 text-center">
            <div className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5">
              {resolvedB || 'B'} Z
            </div>
            <div className="font-mono text-base font-semibold tabular-nums leading-none text-sky-400 min-h-[1.25rem]">
              {zB != null ? `${zB.toFixed(0)}` : <span className="text-neutral-600">--</span>}
            </div>
            <div className="text-[9px] text-muted-foreground mt-0.5">mm</div>
          </div>
          <div className="rounded-md bg-neutral-950 border border-amber-700/50 px-3 py-2.5 text-center">
            <div className="text-[10px] font-medium text-amber-500/80 uppercase tracking-wider mb-1.5">
              Gap
            </div>
            <div className="font-mono text-lg font-bold tabular-nums leading-none text-amber-400 min-h-[1.25rem]">
              {gap != null ? `${gap.toFixed(0)}` : <span className="text-neutral-600">--</span>}
            </div>
            <div className="text-[9px] text-muted-foreground mt-0.5">mm</div>
          </div>
        </div>
      </div>
    </div>
  )
}
