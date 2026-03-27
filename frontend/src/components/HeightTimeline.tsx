import { useMemo } from 'react'
import type { InferenceResult } from '@/types'

interface HeightTimelineProps {
  frames: InferenceResult[]
  /** Current video playback time in seconds */
  currentTime?: number
  /** Total video duration in seconds */
  duration?: number
}

interface ZPoint {
  frameNumber: number
  timestamp: number
  zMm: number
}

export default function HeightTimeline({ frames, currentTime, duration }: HeightTimelineProps) {
  const points = useMemo<ZPoint[]>(() => {
    const result: ZPoint[] = []
    for (const frame of frames) {
      const hookDet = frame.detections.find(
        (d) => d.class_name === 'crane hook' && d.z_mm != null
      )
      if (hookDet?.z_mm != null) {
        result.push({
          frameNumber: frame.frame_number,
          timestamp: frame.timestamp,
          zMm: hookDet.z_mm,
        })
      }
    }
    return result
  }, [frames])

  if (points.length < 2) return null

  const zValues = points.map((p) => p.zMm)
  const minZ = Math.min(...zValues)
  const maxZ = Math.max(...zValues)
  const zRange = maxZ - minZ || 1

  const width = 800
  const height = 56
  const padX = 40
  const padTop = 6
  const padBottom = 16
  const plotW = width - padX * 2
  const plotH = height - padTop - padBottom

  const minT = points[0].timestamp
  const maxT = points[points.length - 1].timestamp
  const tRange = maxT - minT || 1

  const toX = (t: number) => padX + ((t - minT) / tRange) * plotW
  const toY = (z: number) => padTop + plotH - ((z - minZ) / zRange) * plotH

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.timestamp).toFixed(1)},${toY(p.zMm).toFixed(1)}`)
    .join(' ')

  const showPlayhead = currentTime != null && duration != null && duration > 0
  const playheadX = showPlayhead
    ? padX + (currentTime! / duration!) * plotW
    : 0

  const yTicks = [
    { z: minZ, label: `${Math.round(minZ)}` },
    { z: maxZ, label: `${Math.round(maxZ)}` },
  ]

  return (
    <div className="w-full">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
          Z Height (mm)
        </span>
        <span className="text-[10px] text-muted-foreground">
          {Math.round(minZ)} – {Math.round(maxZ)} mm
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-14 rounded border border-border bg-muted/20"
        preserveAspectRatio="none"
      >
        {/* Grid lines */}
        <line
          x1={padX} y1={toY(minZ)} x2={padX + plotW} y2={toY(minZ)}
          stroke="currentColor" strokeOpacity={0.08} strokeWidth={1}
        />
        <line
          x1={padX} y1={toY(maxZ)} x2={padX + plotW} y2={toY(maxZ)}
          stroke="currentColor" strokeOpacity={0.08} strokeWidth={1}
        />

        {/* Y-axis labels */}
        {yTicks.map((tick) => (
          <text
            key={tick.z}
            x={padX - 4}
            y={toY(tick.z) + 3}
            textAnchor="end"
            className="fill-muted-foreground"
            fontSize={8}
          >
            {tick.label}
          </text>
        ))}

        {/* Area fill */}
        <path
          d={`${pathD} L${toX(points[points.length - 1].timestamp).toFixed(1)},${(padTop + plotH).toFixed(1)} L${toX(points[0].timestamp).toFixed(1)},${(padTop + plotH).toFixed(1)} Z`}
          className="fill-primary/10"
        />

        {/* Line */}
        <path
          d={pathD}
          fill="none"
          className="stroke-primary"
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Data points */}
        {points.map((p) => (
          <circle
            key={p.frameNumber}
            cx={toX(p.timestamp)}
            cy={toY(p.zMm)}
            r={2}
            className="fill-primary"
          />
        ))}

        {/* Playhead */}
        {showPlayhead && (
          <line
            x1={playheadX}
            y1={padTop}
            x2={playheadX}
            y2={padTop + plotH}
            stroke="currentColor"
            strokeOpacity={0.5}
            strokeWidth={1}
            strokeDasharray="2,2"
          />
        )}
      </svg>
    </div>
  )
}
