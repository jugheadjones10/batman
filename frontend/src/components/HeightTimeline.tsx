import { useMemo } from 'react'
import { computeZForDetection } from '@/lib/zCalibration'
import type { InferenceResult, ZCalibration } from '@/types'

interface PositionTimelineProps {
  frames: InferenceResult[]
  currentTime?: number
  duration?: number
  metric: 'z' | 'x' | 'y'
  targetClass?: string
  videoWidth?: number
  videoHeight?: number
  zCalibration?: ZCalibration | null
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

const SVG_W = 800
const SVG_H = 140
const PAD_LEFT = 48
const PAD_RIGHT = 8
const PAD_TOP = 10
const PAD_BOTTOM = 24
const PLOT_W = SVG_W - PAD_LEFT - PAD_RIGHT
const PLOT_H = SVG_H - PAD_TOP - PAD_BOTTOM

export default function PositionTimeline({
  frames,
  currentTime,
  duration,
  metric,
  targetClass = 'crane hook',
  videoWidth = 1,
  videoHeight = 1,
  zCalibration,
}: PositionTimelineProps) {
  const config = METRIC_CONFIG[metric]

  const points = useMemo<DataPoint[]>(() => {
    const result: DataPoint[] = []
    for (const frame of frames) {
      const det = frame.detections.find((d) => d.class_name === targetClass)
      if (!det) continue

      let value: number | undefined
      if (metric === 'z') {
        value = det.z_mm ?? computeZForDetection(zCalibration, det, videoWidth, videoHeight) ?? undefined
      }
      else if (metric === 'x') value = det.box.x * videoWidth
      else value = det.box.y * videoHeight

      if (value != null) {
        result.push({ timestamp: frame.timestamp, value })
      }
    }
    return result
  }, [frames, metric, targetClass, videoWidth, videoHeight, zCalibration])

  const plot = useMemo(() => {
    if (points.length < 2) return null

    const values = points.map((p) => p.value)
    const minV = Math.min(...values)
    const maxV = Math.max(...values)
    const vRange = maxV - minV || 1
    const vPad = vRange * 0.08
    const plotMinV = minV - vPad
    const plotMaxV = maxV + vPad
    const plotVRange = plotMaxV - plotMinV

    const minT = 0
    const maxT = duration && duration > 0 ? duration : points[points.length - 1].timestamp
    const tRange = maxT - minT || 1

    const toX = (t: number) => PAD_LEFT + ((t - minT) / tRange) * PLOT_W
    const toY = (v: number) => PAD_TOP + PLOT_H - ((v - plotMinV) / plotVRange) * PLOT_H

    const pathD = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.timestamp).toFixed(1)},${toY(p.value).toFixed(1)}`)
      .join(' ')

    const areaD = `${pathD} L${toX(points[points.length - 1].timestamp).toFixed(1)},${(PAD_TOP + PLOT_H).toFixed(1)} L${toX(points[0].timestamp).toFixed(1)},${(PAD_TOP + PLOT_H).toFixed(1)} Z`

    const yTickCount = 3
    const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => {
      const v = minV + (vRange * i) / yTickCount
      return { v, y: toY(v), label: `${Math.round(v)}` }
    })

    const xTickInterval = tRange <= 30 ? 5 : tRange <= 120 ? 15 : tRange <= 300 ? 30 : 60
    const xTicks: { t: number; x: number; label: string }[] = []
    for (let t = 0; t <= maxT; t += xTickInterval) {
      xTicks.push({ t, x: toX(t), label: formatTime(t) })
    }

    return { minV, maxV, minT, tRange, pathD, areaD, yTicks, xTicks }
  }, [points, duration])

  if (!plot) return null

  const showPlayhead = currentTime != null && duration != null && duration > 0
  const playheadX = showPlayhead
    ? PAD_LEFT + (((currentTime ?? 0) - plot.minT) / plot.tRange) * PLOT_W
    : 0

  return (
    <div className="w-full">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
          {config.title}
        </span>
        <span className="text-[10px] text-muted-foreground font-mono">
          {Math.round(plot.minV)} – {Math.round(plot.maxV)} {config.unit}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="w-full rounded-lg border border-border bg-neutral-900/80"
        preserveAspectRatio="none"
      >
        {/* Horizontal grid lines */}
        {plot.yTicks.map((tick) => (
          <line
            key={tick.v}
            x1={PAD_LEFT}
            y1={tick.y}
            x2={PAD_LEFT + PLOT_W}
            y2={tick.y}
            stroke="currentColor"
            strokeOpacity={0.07}
            strokeWidth={1}
          />
        ))}

        {/* Y-axis labels */}
        {plot.yTicks.map((tick) => (
          <text
            key={tick.v}
            x={PAD_LEFT - 4}
            y={tick.y + 3}
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
          x={PAD_LEFT - 4}
          y={PAD_TOP - 1}
          textAnchor="end"
          className="fill-muted-foreground"
          fontSize={8}
        >
          {config.unit}
        </text>

        {/* X-axis labels */}
        {plot.xTicks.map((tick) => (
          <text
            key={tick.t}
            x={tick.x}
            y={SVG_H - 4}
            textAnchor="middle"
            className="fill-muted-foreground"
            fontSize={9}
            fontFamily="monospace"
          >
            {tick.label}
          </text>
        ))}

        {/* Area fill */}
        <path d={plot.areaD} fill={config.color} fillOpacity={0.08} />

        {/* Line */}
        <path
          d={plot.pathD}
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
              y1={PAD_TOP}
              x2={playheadX}
              y2={PAD_TOP + PLOT_H}
              stroke="#ffffff"
              strokeOpacity={0.6}
              strokeWidth={1.5}
            />
            <circle
              cx={playheadX}
              cy={PAD_TOP}
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
