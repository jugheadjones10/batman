import { useState, useRef, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Loader2, RefreshCw, Activity } from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import SideViewSchematic from '@/components/SideViewSchematic'
import DetectionOverlaySvg, { type OverlayBox } from '@/components/DetectionOverlaySvg'
import { useSyncedVideoMirror } from '@/hooks/useSyncedVideoMirror'
import { smoothFramesPerTrack } from '@/lib/oneEuroFilter'
import {
  buildTrackedOverlayBoxes,
  COLOR_EXTRAPOLATED,
  COLOR_MEASURED,
  DEFAULT_OEF,
  DEFAULT_TRACKER_PARAMS,
  findClosestFrameIndex,
  pickPrimaryTrackPerClassFrames,
} from '@/lib/trackingPresentation'
import type { Detection, InferenceResult } from '@/types'

// Re-query 250 ms after the last slider change so dragging doesn't fire a
// burst of requests to the backend.
const DEBOUNCE_MS = 250
const RAW_COLORS = ['#22d3ee', '#f472b6', '#a78bfa', '#facc15', '#fb923c', '#4ade80']

interface TrackerParams {
  track_activation_threshold: number
  lost_track_buffer: number
  minimum_matching_threshold: number
}

/** Raw-side overlay: the single max-confidence detection per class. */
function pickRawOverlayBoxes(
  frame: InferenceResult | null,
  colorMap: Record<string, string>,
): { boxes: OverlayBox[]; total: number } {
  if (!frame) return { boxes: [], total: 0 }
  const bestByClass = new Map<string, Detection>()
  for (const d of frame.detections) {
    const cur = bestByClass.get(d.class_name)
    if (!cur || d.confidence > cur.confidence) bestByClass.set(d.class_name, d)
  }
  const boxes: OverlayBox[] = []
  for (const [cls, d] of bestByClass) {
    boxes.push({
      key: `raw-${cls}`,
      box: d.box,
      color: colorMap[cls] ?? RAW_COLORS[0],
      label: `${cls} ${(d.confidence * 100).toFixed(0)}%`,
    })
  }
  return { boxes, total: frame.detections.length }
}

export default function TrackingComparePage() {
  const { projectName, runName, videoId, inferenceId } = useParams<{
    projectName: string
    runName: string
    videoId: string
    inferenceId: string
  }>()

  // Immediate UI state (slider thumb position) vs. the debounced "committed"
  // state that actually drives the /bytetrack-frames fetch. This keeps the
  // controls feeling live without hammering the backend on every pixel of
  // drag.
  const [uiParams, setUiParams] = useState<TrackerParams>({ ...DEFAULT_TRACKER_PARAMS })
  const [committedParams, setCommittedParams] = useState<TrackerParams>({ ...DEFAULT_TRACKER_PARAMS })
  useEffect(() => {
    const t = setTimeout(() => setCommittedParams(uiParams), DEBOUNCE_MS)
    return () => clearTimeout(t)
  }, [uiParams])

  // One Euro filter state is pure frontend post-processing; no debounce
  // needed because changing the knobs just re-runs the in-memory filter.
  const [oef, setOef] = useState<{
    enabled: boolean
    minCutoff: number
    beta: number
  }>({ ...DEFAULT_OEF })

  // Video refs + playhead. RAF keeps overlays in sync during playback; the
  // `seeked` / `timeupdate` listeners handle the paused-but-scrubbing case
  // so frame-level stepping through the timeline also updates the overlays.
  const videoRef = useRef<HTMLVideoElement>(null)
  const rightVideoRef = useRef<HTMLVideoElement>(null)
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  const rafRef = useRef<number>(0)
  useEffect(() => {
    const tick = () => {
      const v = videoRef.current
      if (v && !v.paused && !v.ended) setVideoTime(v.currentTime)
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  useSyncedVideoMirror(videoRef, rightVideoRef, true, inferenceId ?? '')

  const { data: video } = useQuery({
    queryKey: ['video', projectName, videoId],
    queryFn: () => api.videos.get(projectName!, videoId!),
    enabled: !!projectName && !!videoId,
  })

  const { data: detailResult, isLoading: detailLoading } = useQuery({
    queryKey: ['inference-result-detail', projectName, runName, videoId, inferenceId],
    queryFn: () => api.inference.getResult(projectName!, runName!, videoId!, inferenceId!),
    enabled: !!projectName && !!runName && !!videoId && !!inferenceId,
  })

  // Re-attach seek/timeupdate listeners once the video element mounts (it's
  // behind a loading guard that keeps videoRef null until detailResult
  // arrives). This keeps the overlays in sync while the video is paused and
  // the user is scrubbing or frame-stepping.
  const videoReady = !!detailResult
  useEffect(() => {
    if (!videoReady) return
    const v = videoRef.current
    if (!v) return
    const update = () => setVideoTime(v.currentTime)
    v.addEventListener('seeked', update)
    v.addEventListener('timeupdate', update)
    v.addEventListener('loadedmetadata', update)
    return () => {
      v.removeEventListener('seeked', update)
      v.removeEventListener('timeupdate', update)
      v.removeEventListener('loadedmetadata', update)
    }
  }, [videoReady])

  const { data: trackedData, isFetching: trackedFetching } = useQuery({
    queryKey: [
      'inference-bytetrack-frames',
      projectName,
      runName,
      videoId,
      inferenceId,
      committedParams.track_activation_threshold,
      committedParams.lost_track_buffer,
      committedParams.minimum_matching_threshold,
    ],
    queryFn: () =>
      api.inference.getBytetrackFrames(projectName!, runName!, videoId!, inferenceId!, {
        track_activation_threshold: committedParams.track_activation_threshold,
        lost_track_buffer: committedParams.lost_track_buffer,
        minimum_matching_threshold: committedParams.minimum_matching_threshold,
      }),
    enabled: !!projectName && !!runName && !!videoId && !!inferenceId,
    staleTime: Infinity,
  })

  const rawFrames = useMemo(() => detailResult?.frames ?? [], [detailResult?.frames])
  const trackedFrames = useMemo(() => trackedData?.frames ?? [], [trackedData?.frames])

  // One Euro post-processing. Runs per track_id across the whole ordered
  // frame list, smoothing (cx, cy, w, h) independently. Because the filter
  // is causal, running it end-to-end here produces the exact same output
  // it would emit live on each frame. When disabled, it's an identity pass.
  const smoothedTrackedFrames = useMemo<InferenceResult[]>(() => {
    if (!oef.enabled || trackedFrames.length === 0) return trackedFrames
    return smoothFramesPerTrack(trackedFrames, {
      minCutoff: oef.minCutoff,
      beta: oef.beta,
    })
  }, [trackedFrames, oef.enabled, oef.minCutoff, oef.beta])

  const primaryTrackedFrames = useMemo(
    () => pickPrimaryTrackPerClassFrames(smoothedTrackedFrames),
    [smoothedTrackedFrames],
  )

  const classColorMap = useMemo<Record<string, string>>(() => {
    const names = new Set<string>()
    for (const f of rawFrames) for (const d of f.detections) names.add(d.class_name)
    const sorted = Array.from(names).sort()
    const map: Record<string, string> = {}
    sorted.forEach((n, i) => {
      map[n] = RAW_COLORS[i % RAW_COLORS.length]
    })
    return map
  }, [rawFrames])

  const rawIdx = useMemo(() => findClosestFrameIndex(rawFrames, videoTime), [rawFrames, videoTime])
  const trackedIdx = useMemo(
    () => findClosestFrameIndex(primaryTrackedFrames, videoTime),
    [primaryTrackedFrames, videoTime],
  )
  const rawFrame = rawIdx >= 0 ? rawFrames[rawIdx] : null
  const trackedFrame = trackedIdx >= 0 ? primaryTrackedFrames[trackedIdx] : null

  const rawOverlay = useMemo(
    () => pickRawOverlayBoxes(rawFrame, classColorMap),
    [rawFrame, classColorMap],
  )
  const trackedOverlay = useMemo(
    () => buildTrackedOverlayBoxes(trackedFrame),
    [trackedFrame],
  )

  const vw = video?.width ?? 1920
  const vh = video?.height ?? 1080
  const fps = video?.fps ?? 25

  const videoStreamUrl =
    projectName && videoId ? api.videos.streamUrl(projectName, videoId, true) : ''

  const resetParams = () => setUiParams({ ...DEFAULT_TRACKER_PARAMS })
  const isModified =
    uiParams.track_activation_threshold !== DEFAULT_TRACKER_PARAMS.track_activation_threshold ||
    uiParams.lost_track_buffer !== DEFAULT_TRACKER_PARAMS.lost_track_buffer ||
    uiParams.minimum_matching_threshold !== DEFAULT_TRACKER_PARAMS.minimum_matching_threshold

  const resetOef = () => setOef({ ...DEFAULT_OEF })
  const oefModified =
    oef.enabled !== DEFAULT_OEF.enabled ||
    oef.minCutoff !== DEFAULT_OEF.minCutoff ||
    oef.beta !== DEFAULT_OEF.beta

  const createdAt = detailResult?.created_at
    ? new Date(detailResult.created_at).toLocaleString('en-SG', { timeZone: 'Asia/Singapore' })
    : null

  return (
    <div className="container max-w-[1800px] py-6 px-6 lg:px-8">
      <div className="flex items-center justify-between mb-4">
        <Link
          to={`/projects/${projectName}/inference?run=${encodeURIComponent(runName ?? '')}&video=${encodeURIComponent(videoId ?? '')}&inferenceId=${encodeURIComponent(inferenceId ?? '')}`}
        >
          <Button variant="ghost" size="sm" className="gap-1.5">
            <ArrowLeft className="h-4 w-4" />
            Back to Inference
          </Button>
        </Link>
        <div className="flex items-center gap-2">
          {trackedFetching && (
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Re-tracking
            </span>
          )}
        </div>
      </div>

      <Card className="mb-4">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Activity className="h-4 w-4 text-muted-foreground" />
            Raw vs ByteTrack — live tuning
          </CardTitle>
          <CardDescription className="text-xs">
            {runName} → {video?.filename ?? videoId}
            {createdAt ? ` · Run on ${createdAt} SGT` : ''}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {detailLoading || !detailResult ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <>
              {/* Tracker settings */}
              <div className="rounded-lg border border-border bg-muted/30 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Tracker settings</h3>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={resetParams}
                    disabled={!isModified}
                    className="gap-1.5"
                  >
                    <RefreshCw className="h-3 w-3" />
                    Reset to defaults
                  </Button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <SliderControl
                    label="track_activation_threshold"
                    tooltip="Minimum detection confidence to spawn / activate a new track."
                    value={uiParams.track_activation_threshold}
                    min={0}
                    max={1}
                    step={0.05}
                    valueSuffix=""
                    formatValue={(v) => v.toFixed(2)}
                    onChange={(v) =>
                      setUiParams((p) => ({ ...p, track_activation_threshold: v }))
                    }
                  />
                  <NumberControl
                    label="lost_track_buffer"
                    tooltip="Frames to keep extrapolating a track after its detection disappears before terminating."
                    value={uiParams.lost_track_buffer}
                    min={1}
                    max={600}
                    step={1}
                    hint={fps > 0 ? `~${(uiParams.lost_track_buffer / fps).toFixed(1)} s @ ${fps.toFixed(0)} fps` : undefined}
                    onChange={(v) => setUiParams((p) => ({ ...p, lost_track_buffer: v }))}
                  />
                  <SliderControl
                    label="minimum_matching_threshold"
                    tooltip="IoU gate for associating a detection to an existing track. Lower = more lenient matching through jitter."
                    value={uiParams.minimum_matching_threshold}
                    min={0}
                    max={1}
                    step={0.05}
                    valueSuffix=""
                    formatValue={(v) => v.toFixed(2)}
                    onChange={(v) =>
                      setUiParams((p) => ({ ...p, minimum_matching_threshold: v }))
                    }
                  />
                </div>
              </div>

              {/* One Euro post-processing — applied per track_id on top of
                  ByteTrack's Kalman output. Pure frontend; no refetch. */}
              <div
                className={`rounded-lg border border-border bg-muted/30 p-4 space-y-3 transition-opacity ${
                  oef.enabled ? '' : 'opacity-60'
                }`}
              >
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-medium">
                      Post-tracker smoothing — One Euro filter
                    </h3>
                    <p className="text-[11px] text-muted-foreground mt-0.5">
                      Layered on top of ByteTrack's Kalman posterior to scrub
                      residual detector jitter. Causal, per-track.
                    </p>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={oef.enabled}
                        onChange={(e) =>
                          setOef((p) => ({ ...p, enabled: e.target.checked }))
                        }
                      />
                      Enable
                    </label>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={resetOef}
                      disabled={!oefModified}
                      className="gap-1.5"
                    >
                      <RefreshCw className="h-3 w-3" />
                      Reset
                    </Button>
                  </div>
                </div>
                <div
                  className={`grid grid-cols-1 md:grid-cols-2 gap-4 ${
                    oef.enabled ? '' : 'pointer-events-none'
                  }`}
                >
                  <SliderControl
                    label="min_cutoff (Hz)"
                    tooltip="Cutoff frequency at rest. Lower = heavier smoothing on stationary objects (more lag on sudden motion)."
                    value={oef.minCutoff}
                    min={0.1}
                    max={5.0}
                    step={0.05}
                    valueSuffix=""
                    formatValue={(v) => v.toFixed(2)}
                    onChange={(v) => setOef((p) => ({ ...p, minCutoff: v }))}
                  />
                  <SliderControl
                    label="beta"
                    tooltip="Speed coefficient. Higher = filter unclamps faster when the object moves (less lag, but lets more real-motion jitter through)."
                    value={oef.beta}
                    min={0}
                    max={0.1}
                    step={0.001}
                    valueSuffix=""
                    formatValue={(v) => v.toFixed(3)}
                    onChange={(v) => setOef((p) => ({ ...p, beta: v }))}
                  />
                </div>
              </div>

              {/* Live counters */}
              <div className="flex items-center justify-between text-xs font-mono px-1">
                <div className="flex items-center gap-4">
                  <span className="text-muted-foreground">
                    frame <span className="text-foreground tabular-nums">{rawIdx >= 0 ? rawIdx : '--'}</span>
                  </span>
                  <span className="text-muted-foreground">
                    raw dets:{' '}
                    <span className="text-foreground tabular-nums">{rawOverlay.total}</span>{' '}
                    (shown {rawOverlay.boxes.length})
                  </span>
                  <span className="text-muted-foreground">
                    tracked primary:{' '}
                    <span className="text-foreground tabular-nums">
                      {trackedOverlay.measured + trackedOverlay.extrapolated}
                    </span>{' '}
                    (<span style={{ color: COLOR_MEASURED }}>measured {trackedOverlay.measured}</span>,{' '}
                    <span style={{ color: COLOR_EXTRAPOLATED }}>extrapolated {trackedOverlay.extrapolated}</span>)
                  </span>
                </div>
                {trackedData?.bytetrack_config && (
                  <span className="text-muted-foreground">
                    live: act={trackedData.bytetrack_config.track_activation_threshold} · buf=
                    {trackedData.bytetrack_config.lost_track_buffer} · match=
                    {trackedData.bytetrack_config.minimum_matching_threshold}
                  </span>
                )}
              </div>

              {/* Video pair */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <VideoPane
                  title="Raw (per-frame best)"
                  legend={
                    <>
                      <Swatch color="#22d3ee" /> max-conf per class
                    </>
                  }
                  videoRef={videoRef}
                  src={videoStreamUrl}
                  isMaster
                  boxes={rawOverlay.boxes}
                  videoWidth={vw}
                  videoHeight={vh}
                  onDuration={setVideoDuration}
                />
                <VideoPane
                  title="ByteTrack (best track per class)"
                  legend={
                    <>
                      <Swatch color={COLOR_MEASURED} /> measured
                      <span className="mx-2 text-muted-foreground">·</span>
                      <Swatch color={COLOR_EXTRAPOLATED} dashed /> extrapolated (Kalman)
                    </>
                  }
                  videoRef={rightVideoRef}
                  src={videoStreamUrl}
                  boxes={trackedOverlay.boxes}
                  videoWidth={vw}
                  videoHeight={vh}
                />
              </div>

              {/* Schematic pair */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div>
                  <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-1">
                    Raw (no tracker)
                  </div>
                  <SideViewSchematic
                    frames={rawFrames}
                    currentTime={videoTime}
                    videoWidth={vw}
                    videoHeight={vh}
                    projectName={projectName!}
                    runName={runName!}
                    videoId={videoId!}
                    inferenceId={inferenceId!}
                  />
                </div>
                <div>
                  <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-widest mb-1">
                    ByteTrack best track/class {oef.enabled ? '+ One Euro' : '(raw Kalman)'}
                  </div>
                  {primaryTrackedFrames.length > 0 ? (
                    <SideViewSchematic
                      frames={primaryTrackedFrames}
                      currentTime={videoTime}
                      videoWidth={vw}
                      videoHeight={vh}
                      projectName={projectName!}
                      runName={runName!}
                      videoId={videoId!}
                      inferenceId={inferenceId!}
                    />
                  ) : (
                    <div className="rounded-lg border border-dashed border-border bg-muted/30 p-6 text-center text-xs text-muted-foreground">
                      {trackedFetching ? 'Fetching tracked frames...' : 'No tracked frames available.'}
                    </div>
                  )}
                </div>
              </div>
              {/* videoDuration is captured so consumers (future timeline) can use it */}
              <span className="sr-only">duration: {videoDuration.toFixed(2)}s</span>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Small local controls — kept inline because they're only used here.
// ---------------------------------------------------------------------------

interface SliderControlProps {
  label: string
  tooltip?: string
  value: number
  min: number
  max: number
  step: number
  formatValue: (v: number) => string
  valueSuffix?: string
  onChange: (v: number) => void
}

function SliderControl({
  label,
  tooltip,
  value,
  min,
  max,
  step,
  formatValue,
  valueSuffix,
  onChange,
}: SliderControlProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs font-medium" title={tooltip}>
          {label}
        </label>
        <span className="text-xs font-mono text-sky-400 tabular-nums">
          {formatValue(value)}
          {valueSuffix}
        </span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  )
}

interface NumberControlProps {
  label: string
  tooltip?: string
  value: number
  min: number
  max: number
  step: number
  hint?: string
  onChange: (v: number) => void
}

function NumberControl({
  label,
  tooltip,
  value,
  min,
  max,
  step,
  hint,
  onChange,
}: NumberControlProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs font-medium" title={tooltip}>
          {label}
        </label>
        {hint && <span className="text-[10px] text-muted-foreground">{hint}</span>}
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      <div className="flex items-center gap-2 mt-1">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => {
            const n = Number(e.target.value)
            if (Number.isFinite(n)) onChange(Math.max(min, Math.min(max, n)))
          }}
          className="flex-1 h-7 px-2 rounded border border-border bg-background text-xs font-mono"
        />
        <span className="text-[10px] text-muted-foreground">frames</span>
      </div>
    </div>
  )
}

interface VideoPaneProps {
  title: string
  legend: React.ReactNode
  videoRef: React.RefObject<HTMLVideoElement>
  src: string
  isMaster?: boolean
  boxes: OverlayBox[]
  videoWidth: number
  videoHeight: number
  onDuration?: (d: number) => void
}

function VideoPane({
  title,
  legend,
  videoRef,
  src,
  isMaster = false,
  boxes,
  videoWidth,
  videoHeight,
  onDuration,
}: VideoPaneProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide">
          {title}
        </div>
        <div className="flex items-center text-[10px] text-muted-foreground">{legend}</div>
      </div>
      <div
        className="relative w-full bg-black rounded-lg overflow-hidden"
        style={{ aspectRatio: `${videoWidth} / ${videoHeight}` }}
      >
        <video
          ref={videoRef}
          src={src}
          controls={isMaster}
          muted
          playsInline
          preload="auto"
          className="absolute inset-0 w-full h-full"
          onLoadedMetadata={() => {
            if (onDuration && videoRef.current) onDuration(videoRef.current.duration)
          }}
        />
        <DetectionOverlaySvg
          boxes={boxes}
          videoWidth={videoWidth}
          videoHeight={videoHeight}
        />
      </div>
    </div>
  )
}

function Swatch({ color, dashed }: { color: string; dashed?: boolean }) {
  return (
    <span
      className="inline-block align-middle mr-1 rounded-sm"
      style={{
        width: 10,
        height: 10,
        border: `1.5px ${dashed ? 'dashed' : 'solid'} ${color}`,
      }}
    />
  )
}
