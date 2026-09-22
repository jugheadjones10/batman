/**
 * Offline reproduction of the inference-page stacking pipeline.
 *
 * Reads ByteTrack frames dumped by the backend (`/tmp/stacking1_tracked.json`),
 * applies the same One Euro smoothing and `analyzeStacking` call the frontend
 * makes, prints the state timeline, and dumps a per-frame diagnostic JSON for
 * visualization (`/tmp/stacking1_analysis.json`).
 *
 * Bundle & run from the frontend directory:
 *   node_modules/.bin/esbuild scripts/debugStacking.ts --bundle --platform=node \
 *     --format=esm --alias:@=./src --outfile=/tmp/debugStacking.mjs
 *   node /tmp/debugStacking.mjs <result.json path>
 */
import fs from 'node:fs'
import { smoothFramesPerTrack } from '@/lib/oneEuroFilter'
import { analyzeStacking } from '@/lib/stackingDistance'
import { resolveSpreaderClass, resolveContainerClass } from '@/lib/trackingPresentation'
import { computeZForBox } from '@/lib/zCalibration'
import type { InferenceResult, ZCalibration } from '@/types'

const resultPath = process.argv[2]
const trackedPath = process.argv[3] ?? '/tmp/stacking1_tracked.json'
const outPath = process.argv[4] ?? '/tmp/stacking1_analysis.json'

const resultData = JSON.parse(fs.readFileSync(resultPath, 'utf8'))
const calibration: ZCalibration | null = resultData.z_calibration ?? null
const tracked: InferenceResult[] = JSON.parse(fs.readFileSync(trackedPath, 'utf8')).frames

const vw = resultData.video_resolution?.width ?? 1920
const vh = resultData.video_resolution?.height ?? 1080

const smoothed = smoothFramesPerTrack(tracked, { minCutoff: 1.0, beta: 0.007 })

const names = new Set<string>()
for (const f of smoothed) for (const d of f.detections) names.add(d.class_name)
const allClasses = Array.from(names).sort()
const spreaderClass = resolveSpreaderClass(allClasses, calibration)
const containerClass = resolveContainerClass(allClasses, spreaderClass)
console.log('classes:', allClasses, '| spreader:', spreaderClass, '| container:', containerClass)

const analysis = analyzeStacking({
  frames: smoothed,
  spreaderClass,
  containerClass,
  videoWidth: vw,
  videoHeight: vh,
  calibration,
})

// --- State timeline ---------------------------------------------------------
let prevState = ''
for (let i = 0; i < analysis.frames.length; i++) {
  const info = analysis.frames[i]
  const key = `${info.state}|carried=${info.carriedTrackId}|target=${info.targetTrackId}|empty=${info.emptyTargetTrackId}`
  if (key !== prevState) {
    const t = smoothed[i].timestamp
    console.log(
      `[${t.toFixed(2).padStart(7)}s f${String(i).padStart(4)}] ${info.state.padEnd(8)} ` +
        `carried=${String(info.carriedTrackId).padEnd(4)} target=${String(info.targetTrackId).padEnd(4)} ` +
        `targetZ=${info.targetZMm?.toFixed(0) ?? '-'} empty=${String(info.emptyTargetTrackId).padEnd(4)} ` +
        `emptyZ=${info.emptyTargetZMm?.toFixed(0) ?? '-'}`,
    )
    prevState = key
  }
}

console.log('\nepisodes:')
for (const ep of analysis.episodes) {
  console.log(
    `  ${ep.kind.padEnd(6)} lock=f${ep.lockFrameIndex} (${smoothed[ep.lockFrameIndex].timestamp.toFixed(2)}s)` +
      ` end=${ep.endFrameIndex != null ? `f${ep.endFrameIndex} (${smoothed[ep.endFrameIndex].timestamp.toFixed(2)}s)` : 'never'}` +
      ` track=${ep.targetTrackId} targetZ=${ep.targetZMm?.toFixed(0)}`,
  )
}

// --- Physical invariant checks ------------------------------------------------
// These encode failures actually observed on real footage. Each one is a
// physical impossibility, not a heuristic preference.
const failures: string[] = []
const boxOf = (d: { box: { x: number; y: number; width: number; height: number } }) => d.box
const overlapSmaller = (
  a: { box: { x: number; y: number; width: number; height: number } },
  b: { box: { x: number; y: number; width: number; height: number } },
): number => {
  const A = boxOf(a)
  const B = boxOf(b)
  const ix =
    Math.min(A.x + A.width / 2, B.x + B.width / 2) - Math.max(A.x - A.width / 2, B.x - B.width / 2)
  const iy =
    Math.min(A.y + A.height / 2, B.y + B.height / 2) -
    Math.max(A.y - A.height / 2, B.y - B.height / 2)
  if (ix <= 0 || iy <= 0) return 0
  const areaA = A.width * A.height
  const areaB = B.width * B.height
  return (ix * iy) / Math.min(areaA, areaB)
}

for (const ep of analysis.episodes) {
  const label = `${ep.kind}@${smoothed[ep.lockFrameIndex].timestamp.toFixed(2)}s`

  // 1. The target may never be the spreader's own blob: measuring the load
  //    against itself pins the gap near zero for the whole descent.
  if (ep.targetTrackId != null) {
    const f = smoothed[ep.lockFrameIndex]
    // Any spreader-body class, including load-state variants.
    const body = f.detections.filter(
      (d) => /spreader/i.test(d.class_name) && !/container/i.test(d.class_name),
    )
    const target = f.detections.find((d) => d.track_id === ep.targetTrackId)
    if (target && body.length > 0 && ep.kind === 'place') {
      const merged = body.some((b) => overlapSmaller(b, target) >= 0.8)
      if (merged) failures.push(`${label}: placement target #${ep.targetTrackId} is the spreader blob`)
    }
  }

  // 2. A completed episode must have actually reached contact.
  if (ep.endReason === 'completed') {
    const end = ep.endFrameIndex ?? analysis.frames.length - 1
    let minGap = Infinity
    for (let i = ep.lockFrameIndex; i <= end; i++) {
      const g = analysis.frames[i].gapMm
      if (g != null) minGap = Math.min(minGap, g)
    }
    if (minGap > 250) {
      failures.push(`${label}: completed but gap never closed (min ${minGap.toFixed(0)} mm)`)
    }
  }

  // 3. A placement can only happen while carrying.
  if (ep.kind === 'place') {
    const before = analysis.frames[Math.max(0, ep.lockFrameIndex - 1)]
    if (before.state !== 'carrying' && before.state !== 'locked') {
      failures.push(`${label}: placement lock while state=${before.state}`)
    }
  }
}

// 4. The carried track must not flap between candidates frame-to-frame.
let flips = 0
for (let i = 1; i < analysis.frames.length; i++) {
  const a = analysis.frames[i - 1].carriedTrackId
  const b = analysis.frames[i].carriedTrackId
  if (a != null && b != null && a !== b) flips++
}
if (flips > analysis.frames.length / 200) {
  failures.push(`carried track flips ${flips} times (unstable spreader/load identity)`)
}

// 5. A clip that starts with a load cannot begin with a pickup. Only checkable
//    when the model reports the load state; motion inference alone cannot know.
if (analysis.loadStateSource === 'class') {
  const startsLoaded = analysis.frames[0]?.state === 'carrying' || analysis.frames[0]?.state === 'locked'
  const first = analysis.episodes[0]
  if (startsLoaded && first?.kind === 'pickup') {
    failures.push('clip starts loaded but its first episode is a pickup')
  }
}

console.log(`\nload state source: ${analysis.loadStateSource}`)
console.log('checks:')
if (failures.length === 0) {
  console.log('  all physical invariants hold')
} else {
  for (const f of failures) console.log(`  FAIL ${f}`)
}

// --- Per-frame diagnostic dump for visualization -----------------------------
const zOf = (d: (typeof smoothed)[number]['detections'][number]): number | null => {
  if (d.z_mm != null && Number.isFinite(d.z_mm) && d.z_mm > 0) return d.z_mm
  const z = computeZForBox(calibration, d.box, vw, vh, d.class_name)
  return z != null && Number.isFinite(z) && z > 0 ? z : null
}

const dump = smoothed.map((f, i) => ({
  i,
  t: f.timestamp,
  info: analysis.frames[i],
  dets: f.detections.map((d) => ({
    cls: d.class_name,
    tid: d.track_id ?? null,
    src: d.track_source ?? null,
    conf: Math.round(d.confidence * 100) / 100,
    box: d.box,
    z: zOf(d),
  })),
}))
fs.writeFileSync(outPath, JSON.stringify(dump))
console.log(`\nwrote ${outPath}`)
