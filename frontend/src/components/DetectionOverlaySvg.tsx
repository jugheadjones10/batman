import type { BoundingBox } from '@/types'

export interface OverlayBox {
  /** Stable key so React can reconcile through frame hops. */
  key: string | number
  /** Normalised bbox (centre + size) in [0, 1]. */
  box: BoundingBox
  color: string
  label?: string | null
  /** If true, stroke is dashed (used for Kalman-extrapolated tracks). */
  dashed?: boolean
}

interface DetectionOverlaySvgProps {
  boxes: OverlayBox[]
  videoWidth: number
  videoHeight: number
  /** Stroke width in viewport pixels (uses vector-effect to stay uniform). */
  strokeWidthPx?: number
  /** Label font size in viewport pixels. */
  fontSizePx?: number
}

/**
 * Absolute-positioned SVG that draws normalised bboxes over a sibling
 * `<video>`. Parent wrapper must be the same box the video occupies (the
 * component uses `inset-0` and `preserveAspectRatio="xMidYMid meet"`, so
 * bboxes align with video pixels as long as wrapper + video share an aspect
 * ratio).
 *
 * Coordinates follow the repo convention: `box.x / box.y` is the bbox centre,
 * `box.width / box.height` is the full extent — all in [0, 1] normalised
 * image coords. We multiply into the video's native pixel grid so strokes
 * and labels can use pixel sizes via `vectorEffect="non-scaling-stroke"`.
 */
export default function DetectionOverlaySvg({
  boxes,
  videoWidth,
  videoHeight,
  strokeWidthPx = 2,
  fontSizePx = 11,
}: DetectionOverlaySvgProps) {
  return (
    <svg
      viewBox={`0 0 ${videoWidth} ${videoHeight}`}
      preserveAspectRatio="xMidYMid meet"
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    >
      {boxes.map((b) => {
        const cx = b.box.x * videoWidth
        const cy = b.box.y * videoHeight
        const w = b.box.width * videoWidth
        const h = b.box.height * videoHeight
        const x = cx - w / 2
        const y = cy - h / 2
        return (
          <g key={b.key}>
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill="none"
              stroke={b.color}
              strokeWidth={strokeWidthPx}
              strokeDasharray={b.dashed ? '6 4' : undefined}
              vectorEffect="non-scaling-stroke"
            />
            {b.label ? (
              <g>
                <rect
                  x={x}
                  y={Math.max(0, y - fontSizePx - 4)}
                  width={b.label.length * fontSizePx * 0.6 + 6}
                  height={fontSizePx + 4}
                  fill={b.color}
                  fillOpacity={0.85}
                />
                <text
                  x={x + 3}
                  y={Math.max(fontSizePx, y - 3)}
                  fill="#0a0a0a"
                  fontSize={fontSizePx}
                  fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                  fontWeight={600}
                  style={{ userSelect: 'none' }}
                >
                  {b.label}
                </text>
              </g>
            ) : null}
          </g>
        )
      })}
    </svg>
  )
}
