import { useMemo } from 'react'
import type { InferenceResult } from '@/types'

interface PositionTimelineProps {
  frames: InferenceResult[]
  currentTime?: number
  duration?: number
  metric: 'z' | 'x' | 'y'
  targetClass?: string
  videoWidth?: number
  videoHeight?: number
}

interface DataPoint {
  timestamp: number
  value: number
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const METRIC_CONFIG = {
  z: { title: 'Z Position vs. Time', unit: 'mm', color: '#22c55e' },
  x: { title: 'X Position vs. Time', unit: 'px', color: '#38bdf8' },
  y: { title: 'Y Position vs. Time', unit: 'px', color: '#fb923c' },
}

export default function PositionTimeline({
  frames,
  currentTime,
  duration,
  metric,
  targetClass = 'crane hook',
  videoWidth = 1,
  videoHeight = 1,
}: PositionTimelineProps) {
  const config = METRIC_CONFIG[metric]

  const points = useMemo<DataPoint[]>(() => {
    const result: DataPoint[] = []
    for (const frame of frames) {
      const det = frame.detections.find(
        (d) => d.class_name === targetClass && (metric !== 'z' || d.z_mm != null)
      )
      if (!det) continue

      let value: number | undefined
      if (metric === 'z') value = det.z_mm ?? undefined
      else if (metric === 'x') value = det.box.x * videoWidth
      else value = det.box.y * videoHeight

      if (value != null) {
        result.push({ timestamp: frame.timestamp, value })
      }
    }
    return result
  }, [frames, metric, targetClass, videoWidth, videoHeight])

  if (points.length < 2) return null

  const values = points.map((p) => p.value)
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const vRange = maxV - minV || 1
  const vPad = vRange * 0.08
  const plotMinV = minV - vPad
  const plotMaxV = maxV + vPad
  const plotVRange = plotMaxV - plotMinV

  const svgW = 800
  const svgH = 140
  const padLeft = 48
  const padRight = 8
  const padTop = 10
  const padBottom = 24
  const plotW = svgW - padLeft - padRight
  const plotH = svgH - padTop - padBottom

  const minT = 0
  const maxT = duration && duration > 0 ? duration : points[points.length - 1].timestamp
  const tRange = maxT - minT || 1

  const toX = (t: number) => padLeft + ((t - minT) / tRange) * plotW
  const toY = (v: number) => padTop + plotH - ((v - plotMinV) / plotVRange) * plotH

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.timestamp).toFixed(1)},${toY(p.value).toFixed(1)}`)
    .join(' ')

  const areaD = `${pathD} L${toX(points[points.length - 1].timestamp).toFixed(1)},${(padTop + plotH).toFixed(1)} L${toX(points[0].timestamp).toFixed(1)},${(padTop + plotH).toFixed(1)} Z`

  const showPlayhead = currentTime != null && duration != null && duration > 0
  const playheadX = showPlayhead ? toX(currentTime!) : 0

  const yTickCount = 3
  const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => {
    const v = minV + (vRange * i) / yTickCount
    return { v, label: `${Math.round(v)}` }
  })

  const xTickInterval = tRange <= 30 ? 5 : tRange <= 120 ? 15 : tRange <= 300 ? 30 : 60
  const xTicks: { t: number; label: string }[] = []
  for (let t = 0; t <= maxT; t += xTickInterval) {
    xTicks.push({ t, label: formatTime(t) })
  }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
          {config.title}
        </span>
        <span className="text-[10px] text-muted-foreground font-mono">
          {Math.round(minV)} – {Math.round(maxV)} {config.unit}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        className="w-full rounded-lg border border-border bg-neutral-900/80"
        preserveAspectRatio="none"
      >
        {/* Horizontal grid lines */}
        {yTicks.map((tick) => (
          <line
            key={tick.v}
            x1={padLeft}
            y1={toY(tick.v)}
            x2={padLeft + plotW}
            y2={toY(tick.v)}
            stroke="currentColor"
            strokeOpacity={0.07}
            strokeWidth={1}
          />
        ))}

        {/* Y-axis labels */}
        {yTicks.map((tick) => (
          <text
            key={tick.v}
            x={padLeft - 4}
            y={toY(tick.v) + 3}
            textAnchor="end"
            className="fill-muted-foreground"
            fontSize={9}
            fontFamily="monospace"
          >
            {tick.label}
          </text>
        ))}

        {/* Y-axis unit */}
        <text
          x={padLeft - 4}
          y={padTop - 1}
          textAnchor="end"
          className="fill-muted-foreground"
          fontSize={8}
        >
          {config.unit}
        </text>

        {/* X-axis labels */}
        {xTicks.map((tick) => (
          <text
            key={tick.t}
            x={toX(tick.t)}
            y={svgH - 4}
            textAnchor="middle"
            className="fill-muted-foreground"
            fontSize={9}
            fontFamily="monospace"
          >
            {tick.label}
          </text>
        ))}

        {/* Area fill */}
        <path d={areaD} fill={config.color} fillOpacity={0.08} />

        {/* Line */}
        <path
          d={pathD}
          fill="none"
          stroke={config.color}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Playhead */}
        {showPlayhead && (
          <>
            <line
              x1={playheadX}
              y1={padTop}
              x2={playheadX}
              y2={padTop + plotH}
              stroke="#ffffff"
              strokeOpacity={0.6}
              strokeWidth={1.5}
            />
            <circle
              cx={playheadX}
              cy={padTop}
              r={3}
              fill="#ffffff"
              fillOpacity={0.6}
            />
          </>
        )}
      </svg>
    </div>
  )
}
