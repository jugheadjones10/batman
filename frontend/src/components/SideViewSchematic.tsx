import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Box } from 'lucide-react'
import { api } from '@/api/client'
import { computeZForBox } from '@/lib/zCalibration'
import type { BoundingBox, InferenceResult } from '@/types'

interface SideViewSchematicProps {
  frames: InferenceResult[]
  currentTime: number
  videoWidth: number
  videoHeight: number
  projectName: string
  runName: string
  videoId: string
  inferenceId: string
}

// ISO shipping container constants (standard dry-box, not high-cube).
const ISO_CONTAINER_SHORT_SIDE_MM = 2438
const ISO_CONTAINER_HEIGHT_MM = 2591
const ISO_CONTAINER_LENGTHS_MM = [6058, 12192, 13716] as const
const ISO_CONTAINER_LABELS: Record<number, string> = {
  6058: '20 ft',
  12192: '40 ft',
  13716: '45 ft',
}

// SVG viewport geometry.
const SVG_W = 440
const SVG_H = 300
const AXIS_X = SVG_W / 2
const TOP_Y = 36
const BOTTOM_Y = SVG_H - 18
const HORIZ_FOOTPRINT_FRAC = 0.55
// Dimension bracket columns drawn to the left of the shapes.
const INNER_BRACKET_X = 70
const OUTER_BRACKET_X = 22
const MIN_Z_AXIS_MM = 8000
const Z_AXIS_HEADROOM_MM = 2500
const Z_AXIS_ROUNDING_MM = 1000

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

interface SlotData {
  z: number | null
  box: BoundingBox | null
}

function bestDetection(frame: InferenceResult | null, cls: string): SlotData {
  if (!cls || !frame) return { z: null, box: null }
  const dets = frame.detections.filter((d) => d.class_name === cls)
  if (dets.length === 0) return { z: null, box: null }
  const best = dets.reduce((a, b) => (a.confidence > b.confidence ? a : b))
  return { z: best.z_mm ?? null, box: best.box }
}

export default function SideViewSchematic({
  frames,
  currentTime,
  videoWidth,
  videoHeight,
  projectName,
  runName,
  videoId,
  inferenceId,
}: SideViewSchematicProps) {
  const { data: calResp } = useQuery({
    queryKey: ['z-calibration', projectName, runName, videoId, inferenceId],
    queryFn: () => api.inference.getZCalibration(projectName, runName, videoId, inferenceId),
  })
  const calibration = calResp?.z_calibration ?? null

  const allClasses = useMemo(() => {
    const names = new Set<string>()
    for (const f of frames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [frames])

  const zCapableClasses = useMemo(() => {
    const names = new Set<string>()
    for (const f of frames) {
      for (const d of f.detections) {
        if (d.z_mm != null) names.add(d.class_name)
      }
    }
    return Array.from(names)
  }, [frames])

  const [spreaderClass, setSpreaderClass] = useState('')
  const [containerClass, setContainerClass] = useState('')

  // Prefer a class whose name literally mentions "spreader" (but not "container"),
  // so a hypothetical `container_spreader` class doesn't claim both slots.
  const defaultSpreader =
    (calibration?.measurement_source === 'round_feature_equivalent_length' &&
    calibration.reference_class &&
    allClasses.includes(calibration.reference_class)
      ? calibration.reference_class
      : undefined) ??
    allClasses.find((c) => /spreader/i.test(c) && !/container/i.test(c)) ??
    (calibration?.reference_class && allClasses.includes(calibration.reference_class)
      ? calibration.reference_class
      : undefined) ??
    zCapableClasses[0] ?? allClasses[0] ?? ''
  const resolvedSpreader = spreaderClass || defaultSpreader
  const defaultContainer =
    allClasses.find((c) => /container/i.test(c) && c !== resolvedSpreader) ??
    allClasses.find((c) => c !== resolvedSpreader) ?? allClasses[0] ?? ''
  const resolvedContainer = containerClass || defaultContainer

  const frameIndex = findClosestFrameIndex(frames, currentTime)
  const frame = frameIndex >= 0 && frameIndex < frames.length ? frames[frameIndex] : null

  // Pick the container's ISO length ONCE for the whole run. Prefer the
  // calibration's declared `length_mm` (the user told us exactly which ISO size
  // they're picking); otherwise infer from the median container aspect ratio.
  const containerLengthMm = useMemo(() => {
    if (calibration?.length_mm != null && calibration.length_mm > 0) {
      return calibration.length_mm
    }
    const ratios: number[] = []
    for (const f of frames) {
      for (const d of f.detections) {
        if (d.class_name !== resolvedContainer) continue
        const w = d.box.width * videoWidth
        const h = d.box.height * videoHeight
        if (w <= 0 || h <= 0) continue
        ratios.push(Math.max(w, h) / Math.min(w, h))
      }
    }
    if (ratios.length === 0) return ISO_CONTAINER_LENGTHS_MM[1]
    ratios.sort((a, b) => a - b)
    const median = ratios[Math.floor(ratios.length / 2)]
    const estimated = ISO_CONTAINER_SHORT_SIDE_MM * median
    return ISO_CONTAINER_LENGTHS_MM.reduce(
      (best, cand) =>
        Math.abs(cand - estimated) < Math.abs(best - estimated) ? cand : best,
      ISO_CONTAINER_LENGTHS_MM[0] as number,
    )
  }, [calibration, frames, resolvedContainer, videoWidth, videoHeight])

  const { spreader, container } = useMemo(() => {
    const s = bestDetection(frame, resolvedSpreader)
    const c = bestDetection(frame, resolvedContainer)
    return { spreader: s, container: c }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIndex, resolvedSpreader, resolvedContainer, frames])

  // Under the simplified PDF model, the spreader and container share the same
  // real-world length ℓ, so their on-screen footprints match exactly.
  const spreaderRealLengthMm = containerLengthMm

  // Spreader-top z resolution mirrors the container below:
  //   1. `det.z_mm` persisted by the backend (calibration target).
  //   2. Client-side flat-model extrapolation from the bbox.
  // The fallback matters for ByteTrack-extrapolated frames: Kalman-predicted
  // boxes deliberately ship without z_mm (see compute_bytetrack_frames docstring),
  // and without this the spreader would silently vanish from the schematic
  // even though the tracker still has a confident box for it.
  let zSpreader: number | null = spreader.z
  let spreaderZSource: 'measured' | 'estimated' | null = spreader.z != null ? 'measured' : null
  if (zSpreader == null && spreader.box != null) {
    const computed = computeZForBox(
      calibration,
      spreader.box,
      videoWidth,
      videoHeight,
      resolvedSpreader,
    )
    if (computed != null && Number.isFinite(computed) && computed > 0) {
      zSpreader = computed
      spreaderZSource = 'estimated'
    }
  }

  // Container-top z resolution:
  //   1. `det.z_mm` persisted by the backend (target of the last calibration).
  //   2. Client-side flat-model extrapolation (same k / m,c as every class).
  let zContainerTop: number | null = container.z
  let containerZSource: 'measured' | 'estimated' | null = container.z != null ? 'measured' : null
  if (zContainerTop == null && container.box != null) {
    const computed = computeZForBox(
      calibration,
      container.box,
      videoWidth,
      videoHeight,
      resolvedContainer,
    )
    if (computed != null && Number.isFinite(computed) && computed > 0) {
      zContainerTop = computed
      containerZSource = 'estimated'
    }
  }

  const zContainerBottom = zContainerTop != null ? zContainerTop + ISO_CONTAINER_HEIGHT_MM : null
  const spreaderToContainer =
    zSpreader != null && zContainerTop != null ? zContainerTop - zSpreader : null

  // Zoom the vertical axis around the stack in the current frame. Using the
  // deepest value across the entire run can make shallow frames bunch up near
  // the camera, especially after switching to a small round-feature proxy.
  const deepestVisibleMm = Math.max(zSpreader ?? 0, zContainerBottom ?? zContainerTop ?? 0)
  const zMax = Math.max(
    Math.ceil((deepestVisibleMm + Z_AXIS_HEADROOM_MM) / Z_AXIS_ROUNDING_MM) *
      Z_AXIS_ROUNDING_MM,
    MIN_Z_AXIS_MM,
  )

  const mmToY = (mm: number) => TOP_Y + (mm / zMax) * (BOTTOM_Y - TOP_Y)

  const maxRealWidthMm = Math.max(containerLengthMm, spreaderRealLengthMm)
  const pxPerMmHoriz = (SVG_W * HORIZ_FOOTPRINT_FRAC) / maxRealWidthMm
  const spreaderHalfPx = (spreaderRealLengthMm * pxPerMmHoriz) / 2
  const containerHalfPx = (containerLengthMm * pxPerMmHoriz) / 2
  const containerPxHeight = ISO_CONTAINER_HEIGHT_MM * ((BOTTOM_Y - TOP_Y) / zMax)

  const containerTopY = zContainerTop != null ? mmToY(zContainerTop) : null
  const containerBottomY = containerTopY != null ? containerTopY + containerPxHeight : null
  const spreaderY = zSpreader != null ? mmToY(zSpreader) : null

  const ticks = useMemo(() => {
    const out: { mm: number; y: number }[] = []
    for (let mm = 0; mm <= zMax; mm += 1000) {
      out.push({ mm, y: mmToY(mm) })
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zMax])

  if (allClasses.length === 0) return null

  const containerLabel = ISO_CONTAINER_LABELS[containerLengthMm] ?? `${(containerLengthMm / 1000).toFixed(1)} m`

  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-1.5">
        <Box className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
          Side-View Schematic
        </span>
      </div>

      <div className="rounded-lg border border-border bg-neutral-900/80 p-3 space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="text-[10px] text-muted-foreground block mb-1">
              {calibration?.measurement_source === 'round_feature_equivalent_length'
                ? 'Spreader proxy class'
                : 'Spreader class'}
            </label>
            <select
              value={resolvedSpreader}
              onChange={(e) => setSpreaderClass(e.target.value)}
              className="w-full rounded border bg-background px-2 py-1 text-xs h-7"
            >
              {allClasses.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-muted-foreground block mb-1">Container class</label>
            <select
              value={resolvedContainer}
              onChange={(e) => setContainerClass(e.target.value)}
              className="w-full rounded border bg-background px-2 py-1 text-xs h-7"
            >
              {allClasses.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="rounded-md bg-neutral-950 border border-neutral-800 overflow-hidden">
          <svg
            viewBox={`0 0 ${SVG_W} ${SVG_H}`}
            className="w-full h-auto"
            role="img"
            aria-label="Side-view schematic of camera, spreader and container"
          >
            {/* z-axis tick cross marks at the axis */}
            {ticks.map((t) => (
              <line
                key={`tick-${t.mm}`}
                x1={AXIS_X - 4}
                x2={AXIS_X + 4}
                y1={t.y}
                y2={t.y}
                stroke="#3f3f46"
                strokeWidth={1}
              />
            ))}
            {/* z-axis tick labels anchored to the far-right edge so they don't
                collide with the container body */}
            {ticks.map((t) => (
              <text
                key={`tick-label-${t.mm}`}
                x={SVG_W - 4}
                y={t.y + 3}
                fontSize={8}
                fill="#71717a"
                fontFamily="monospace"
                textAnchor="end"
              >
                {(t.mm / 1000).toFixed(0)}m
              </text>
            ))}

            {/* central vertical axis (dashed) */}
            <line
              x1={AXIS_X}
              x2={AXIS_X}
              y1={TOP_Y}
              y2={BOTTOM_Y}
              stroke="#3f3f46"
              strokeDasharray="2 3"
              strokeWidth={1}
            />

            {/* camera marker */}
            <g>
              <rect
                x={AXIS_X - 26}
                y={TOP_Y - 22}
                width={52}
                height={20}
                rx={3}
                fill="#1e293b"
                stroke="#475569"
              />
              <circle cx={AXIS_X} cy={TOP_Y - 12} r={5} fill="#0f172a" stroke="#60a5fa" strokeWidth={1.5} />
              <text
                x={AXIS_X}
                y={TOP_Y - 26}
                fontSize={9}
                fill="#cbd5e1"
                fontFamily="ui-sans-serif, system-ui"
                fontWeight={600}
                textAnchor="middle"
              >
                Camera
              </text>
            </g>

            {/* spreader */}
            {spreaderY != null && (
              <g style={{ transform: `translateY(${spreaderY - TOP_Y}px)` }}>
                <SpreaderShape
                  centerX={AXIS_X}
                  anchorY={TOP_Y}
                  halfWidthPx={spreaderHalfPx}
                  estimated={spreaderZSource === 'estimated'}
                />
              </g>
            )}

            {/* container */}
            {containerTopY != null && (
              <g style={{ transform: `translateY(${containerTopY - TOP_Y}px)` }}>
                <ContainerShape
                  centerX={AXIS_X}
                  topY={TOP_Y}
                  halfWidthPx={containerHalfPx}
                  heightPx={containerPxHeight}
                  estimated={containerZSource === 'estimated'}
                />
                {/* ISO length label centered inside the container body */}
                <text
                  x={AXIS_X}
                  y={TOP_Y + containerPxHeight / 2 + 3}
                  fontSize={Math.max(8, Math.min(10, containerPxHeight * 0.45))}
                  fill="#fef3c7"
                  fontFamily="ui-sans-serif, system-ui"
                  fontWeight={600}
                  textAnchor="middle"
                  style={{ pointerEvents: 'none' }}
                >
                  {containerLabel}
                  {containerZSource === 'estimated' ? ' (est.)' : ''}
                </text>
              </g>
            )}

            {/* Dimension brackets — directly labelled distances on the diagram.
                Inner tier stacks the three stepped measurements; outer tier shows
                the total Camera → Container-bottom. */}
            {spreaderY != null && zSpreader != null && (
              <DimensionBracket
                x={INNER_BRACKET_X}
                y1={TOP_Y}
                y2={spreaderY}
                label={`${zSpreader.toFixed(0)} mm`}
                tone="cyan"
                guideToX={AXIS_X - spreaderHalfPx - 2}
                avoidLabelBeforeX={OUTER_BRACKET_X + 8}
              />
            )}
            {spreaderY != null && containerTopY != null && spreaderToContainer != null && (
              <DimensionBracket
                x={INNER_BRACKET_X}
                y1={spreaderY}
                y2={containerTopY}
                label={`${spreaderToContainer.toFixed(0)} mm`}
                tone="amber"
                guideToX={AXIS_X - containerHalfPx - 2}
                avoidLabelBeforeX={OUTER_BRACKET_X + 8}
              />
            )}
            {containerTopY != null && containerBottomY != null && (
              <DimensionBracket
                x={INNER_BRACKET_X}
                y1={containerTopY}
                y2={containerBottomY}
                label={`${ISO_CONTAINER_HEIGHT_MM} mm`}
                tone="neutral"
                guideToX={AXIS_X - containerHalfPx - 2}
                avoidLabelBeforeX={OUTER_BRACKET_X + 8}
              />
            )}
            {containerBottomY != null && zContainerBottom != null && (
              <DimensionBracket
                x={OUTER_BRACKET_X}
                y1={TOP_Y}
                y2={containerBottomY}
                label={`${zContainerBottom.toFixed(0)} mm`}
                tone="emerald"
                emphasized
                labelRotated
              />
            )}

            {/* "no detection this frame" hint */}
            {spreaderY == null && containerTopY == null && (
              <text
                x={AXIS_X}
                y={TOP_Y + (BOTTOM_Y - TOP_Y) / 2}
                fontSize={10}
                fill="#52525b"
                fontFamily="ui-sans-serif, system-ui"
                textAnchor="middle"
              >
                No detections in this frame
              </text>
            )}
          </svg>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Parametric shape helpers
// ---------------------------------------------------------------------------

interface ContainerShapeProps {
  centerX: number
  topY: number
  halfWidthPx: number
  heightPx: number
  estimated: boolean
}

/**
 * ISO dry-box container rendered parametrically so it respects the true on-screen
 * aspect ratio (very flat — 12 m × 2.6 m at 40 ft). Details: four corner castings,
 * evenly-spaced corrugation ribs, a door panel with a horizontal latch bar.
 */
function ContainerShape({ centerX, topY, halfWidthPx, heightPx, estimated }: ContainerShapeProps) {
  const left = centerX - halfWidthPx
  const right = centerX + halfWidthPx
  const bottom = topY + heightPx
  const widthPx = halfWidthPx * 2
  const cast = Math.max(4, Math.min(9, heightPx * 0.28))
  const doorInset = Math.max(6, Math.min(14, widthPx * 0.045))
  const doorX = right - doorInset
  const doorLatchY = topY + heightPx / 2

  // Corrugation ribs at ~18 px spacing; skip the door region.
  const ribSpacing = 18
  const ribStart = left + cast + 4
  const ribEnd = doorX - 4
  const ribs: number[] = []
  for (let x = ribStart + ribSpacing; x < ribEnd; x += ribSpacing) ribs.push(x)

  return (
    <g>
      {/* body */}
      <rect
        x={left}
        y={topY}
        width={widthPx}
        height={heightPx}
        rx={1}
        fill="#b45309"
        fillOpacity={estimated ? 0.35 : 0.6}
        stroke="#fbbf24"
        strokeDasharray={estimated ? '4 3' : undefined}
      />
      {ribs.map((x) => (
        <line
          key={x}
          x1={x}
          x2={x}
          y1={topY + 3}
          y2={bottom - 3}
          stroke="#78350f"
          strokeOpacity={0.55}
          strokeWidth={1}
        />
      ))}
      {/* door panel */}
      <line
        x1={doorX}
        x2={doorX}
        y1={topY + 2}
        y2={bottom - 2}
        stroke="#fbbf24"
        strokeOpacity={0.7}
        strokeWidth={1}
      />
      <line
        x1={doorX + 2}
        x2={right - cast - 1}
        y1={doorLatchY}
        y2={doorLatchY}
        stroke="#fbbf24"
        strokeOpacity={0.7}
        strokeWidth={1}
      />
      {/* corner castings */}
      <rect x={left} y={topY} width={cast} height={cast} fill="#78350f" stroke="#fbbf24" strokeWidth={0.5} />
      <rect x={right - cast} y={topY} width={cast} height={cast} fill="#78350f" stroke="#fbbf24" strokeWidth={0.5} />
      <rect x={left} y={bottom - cast} width={cast} height={cast} fill="#78350f" stroke="#fbbf24" strokeWidth={0.5} />
      <rect x={right - cast} y={bottom - cast} width={cast} height={cast} fill="#78350f" stroke="#fbbf24" strokeWidth={0.5} />
    </g>
  )
}

interface SpreaderShapeProps {
  centerX: number
  anchorY: number
  halfWidthPx: number
  estimated?: boolean
}

/**
 * Container spreader: horizontal beam with twist-lock castings at each corner
 * and a small hoist bail on top centre. Centered vertically on `anchorY`.
 *
 * When `estimated` is true, the shape is rendered dashed and at reduced opacity
 * to signal that its z came from a bbox-only flat-model extrapolation (typical
 * for ByteTrack Kalman-predicted frames that have no backend-provided z_mm).
 */
function SpreaderShape({ centerX, anchorY, halfWidthPx, estimated = false }: SpreaderShapeProps) {
  const left = centerX - halfWidthPx
  const right = centerX + halfWidthPx
  const beamHeight = 8
  const beamTop = anchorY - beamHeight / 2
  const lockWidth = 6
  const lockDrop = 4
  const bailHalfBase = Math.min(14, Math.max(8, halfWidthPx * 0.18))
  const bailHalfTop = bailHalfBase * 0.45
  const bailHeight = 7
  const fillAlpha = estimated ? 0.4 : 0.85
  const dash = estimated ? '4 3' : undefined

  return (
    <g>
      {/* beam body (between the two twist-lock castings) */}
      <rect
        x={left + lockWidth / 2}
        y={beamTop}
        width={halfWidthPx * 2 - lockWidth}
        height={beamHeight}
        rx={1}
        fill="#0e7490"
        fillOpacity={fillAlpha}
        stroke="#22d3ee"
        strokeDasharray={dash}
      />
      {/* subtle inner highlight line */}
      <line
        x1={left + lockWidth / 2 + 2}
        x2={right - lockWidth / 2 - 2}
        y1={anchorY}
        y2={anchorY}
        stroke="#22d3ee"
        strokeOpacity={0.3}
        strokeWidth={1}
      />
      {/* twist-lock castings (corners, dropped below the beam) */}
      <rect
        x={left}
        y={beamTop}
        width={lockWidth}
        height={beamHeight + lockDrop}
        fill="#155e75"
        fillOpacity={estimated ? 0.55 : 1}
        stroke="#22d3ee"
        strokeWidth={0.8}
        strokeDasharray={dash}
      />
      <rect
        x={right - lockWidth}
        y={beamTop}
        width={lockWidth}
        height={beamHeight + lockDrop}
        fill="#155e75"
        fillOpacity={estimated ? 0.55 : 1}
        stroke="#22d3ee"
        strokeWidth={0.8}
        strokeDasharray={dash}
      />
      {/* hoist bail (trapezoid) on top centre */}
      <path
        d={`M ${centerX - bailHalfBase} ${beamTop} L ${centerX - bailHalfTop} ${beamTop - bailHeight} L ${centerX + bailHalfTop} ${beamTop - bailHeight} L ${centerX + bailHalfBase} ${beamTop} Z`}
        fill="#0e7490"
        fillOpacity={fillAlpha}
        stroke="#22d3ee"
        strokeWidth={1}
        strokeDasharray={dash}
      />
    </g>
  )
}

// ---------------------------------------------------------------------------
// Dimension bracket (engineering-drawing style dimension line + label)
// ---------------------------------------------------------------------------

type BracketTone = 'cyan' | 'amber' | 'emerald' | 'neutral'

interface DimensionBracketProps {
  x: number
  y1: number
  y2: number
  label: string
  tone: BracketTone
  emphasized?: boolean
  labelRotated?: boolean
  /** If set, draw dashed horizontal guides from both caps out to this x. */
  guideToX?: number
  /** If the left-aligned label would cross this x, place it to the right instead. */
  avoidLabelBeforeX?: number
}

function DimensionBracket({
  x,
  y1,
  y2,
  label,
  tone,
  emphasized = false,
  labelRotated = false,
  guideToX,
  avoidLabelBeforeX,
}: DimensionBracketProps) {
  const toneColor: Record<BracketTone, string> = {
    cyan: '#67e8f9',
    amber: '#fcd34d',
    emerald: '#6ee7b7',
    neutral: '#a1a1aa',
  }
  const color = toneColor[tone]
  const cap = 5
  const strokeWidth = emphasized ? 1.6 : 1
  const [yTop, yBot] = y1 <= y2 ? [y1, y2] : [y2, y1]
  const mid = (yTop + yBot) / 2
  const span = Math.abs(yBot - yTop)
  // Labels are only drawn when the bracket is tall enough to host them; the
  // emphasized total bracket always labels regardless.
  const showLabel = span >= 12 || emphasized
  const fontSize = emphasized ? 9 : 8
  const leftLabelX = x - cap - 2
  const estimatedLabelWidth = label.length * fontSize * 0.58
  const useRightLabel =
    !labelRotated &&
    avoidLabelBeforeX != null &&
    leftLabelX - estimatedLabelWidth < avoidLabelBeforeX

  return (
    <g>
      {guideToX != null && (
        <>
          <line
            x1={x + cap}
            x2={guideToX}
            y1={yTop}
            y2={yTop}
            stroke={color}
            strokeOpacity={0.3}
            strokeDasharray="2 3"
            strokeWidth={1}
          />
          <line
            x1={x + cap}
            x2={guideToX}
            y1={yBot}
            y2={yBot}
            stroke={color}
            strokeOpacity={0.3}
            strokeDasharray="2 3"
            strokeWidth={1}
          />
        </>
      )}
      <line x1={x} x2={x} y1={yTop} y2={yBot} stroke={color} strokeWidth={strokeWidth} />
      <line x1={x - cap} x2={x + cap} y1={yTop} y2={yTop} stroke={color} strokeWidth={strokeWidth} />
      <line x1={x - cap} x2={x + cap} y1={yBot} y2={yBot} stroke={color} strokeWidth={strokeWidth} />
      {showLabel &&
        (labelRotated ? (
          <text
            x={x - 6}
            y={mid}
            fontSize={fontSize}
            fill={color}
            fontFamily="ui-sans-serif, system-ui"
            fontWeight={emphasized ? 600 : 500}
            textAnchor="middle"
            transform={`rotate(-90 ${x - 6} ${mid})`}
          >
            {label}
          </text>
        ) : (
          <text
            x={useRightLabel ? x + cap + 3 : leftLabelX}
            y={mid + 3}
            fontSize={fontSize}
            fill={color}
            fontFamily="ui-sans-serif, system-ui"
            fontWeight={emphasized ? 600 : 500}
            textAnchor={useRightLabel ? 'start' : 'end'}
          >
            {label}
          </text>
        ))}
    </g>
  )
}
