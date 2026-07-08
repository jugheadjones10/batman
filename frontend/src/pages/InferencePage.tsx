import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useParams, Link, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Play,
  Settings,
  Loader2,
  Zap,
  Video,
  Trash2,
  X,
  Grid3X3,
  CheckCircle2,
  Server,
  Monitor,
  ImageIcon,
  Ruler,
  Activity,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { useToast } from '@/components/ui/Toaster'
import GpuConnectionPanel from '@/components/GpuConnectionPanel'
import LogViewer from '@/components/LogViewer'
import PositionTimeline from '@/components/HeightTimeline'
import LiveDetectionReadout from '@/components/LiveDetectionReadout'
import SideViewSchematic from '@/components/SideViewSchematic'
import DetectionOverlaySvg from '@/components/DetectionOverlaySvg'
import { smoothFramesPerTrack } from '@/lib/oneEuroFilter'
import {
  buildStackingOverlayBoxes,
  buildTrackedOverlayBoxes,
  COLOR_EXTRAPOLATED,
  COLOR_MEASURED,
  DEFAULT_OEF,
  DEFAULT_TRACKER_PARAMS,
  findClosestFrameIndex,
  pickPrimaryTrackPerClassFrames,
  resolveContainerClass,
  resolveSpreaderClass,
} from '@/lib/trackingPresentation'
import { analyzeStacking } from '@/lib/stackingDistance'
import type {
  InferenceConfig,
  InferenceGPUSubmitRequest,
  InferenceProgressEvent,
  InferenceResult,
  RFDETRModelSize,
  GPUType,
  Video as ProjectVideo,
  ZCalibration,
} from '@/types'

/**
 * Live progress for a streaming local inference run. `stage` is the coarsest
 * label (what's happening right now); `current`/`total` drive the percent bar.
 * Stays non-null while inference is in-flight; cleared on terminal events.
 */
type InferenceProgress = {
  stage: 'loading_model' | 'running_inference' | 'encoding_video' | 'post_processing'
  current: number
  total: number
  avgFps: number
  etaS: number | null
}

type SelectedInferenceCell = {
  run: string
  video: string
  inferenceId: string
}

function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return ''
  const s = Math.round(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  if (m < 60) return `${m}m ${rem}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

const STAGE_LABELS: Record<InferenceProgress['stage'], string> = {
  loading_model: 'Loading model',
  running_inference: 'Running inference',
  encoding_video: 'Encoding output video',
  post_processing: 'Post-processing & saving',
}

export default function InferencePage() {
  const { projectName } = useParams<{ projectName: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const { toast } = useToast()
  const queryClient = useQueryClient()

  // Initialize from URL search params so the detail panel renders on the
  // first paint when arriving with ?run=&video=&inferenceId= (e.g., Back
  // button from the calibration page). Relying on the sync effect below
  // alone leaves selectedCell null for the initial render, which flashes
  // an empty page and reads as "selection wasn't preserved".
  const [selectedCell, setSelectedCell] = useState<SelectedInferenceCell | null>(() => {
    const run = searchParams.get('run')
    const video = searchParams.get('video')
    const inferenceId = searchParams.get('inferenceId')
    if (run && video && inferenceId) return { run, video, inferenceId }
    return null
  })
  const [showRunPanel, setShowRunPanel] = useState(false)
  // Keyed by run name (unique on disk) rather than `run.id`, which is NOT
  // guaranteed unique in historical meta.json files and caused two runs to
  // appear selected at once when they happened to share an id.
  const [runTarget, setRunTarget] = useState<{ runName: string; videoId: string } | null>(null)
  const [config, setConfig] = useState<Partial<InferenceConfig>>({
    confidence_threshold: 0.5,
    iou_threshold: 0.45,
    enable_tracking: false,
    tracking_mode: 'visible_only',
    detection_interval: 1,
  })

  // GPU inference state
  const [runMode, setRunMode] = useState<'local' | 'gpu'>('local')
  const [gpuInferConfig, setGpuInferConfig] = useState<{
    run_name: string | null
    confidence: number
    frame_interval: number
    track: boolean
    track_thresh: number
    model: RFDETRModelSize
    gpu_type: GPUType
    time_limit: string
    test_only: boolean
    no_video: boolean
  }>({
    run_name: null,
    confidence: 0.5,
    frame_interval: 1,
    track: false,
    track_thresh: 0.25,
    model: 'base',
    gpu_type: 'a100-80',
    time_limit: '04:00:00',
    test_only: false,
    no_video: false,
  })
  const [gpuJobName, setGpuJobName] = useState<string | null>(null)

  const { data: videos } = useQuery({
    queryKey: ['videos', projectName],
    queryFn: () => api.videos.list(projectName!),
    enabled: !!projectName,
  })

  const { data: runs } = useQuery({
    queryKey: ['training-runs', projectName],
    queryFn: () => api.training.listRuns(projectName!),
    enabled: !!projectName,
  })

  const { data: matrix, isLoading: matrixLoading } = useQuery({
    queryKey: ['inference-results', projectName],
    queryFn: () => api.inference.listResults(projectName!),
    enabled: !!projectName,
  })

  const { data: detailResult, isLoading: detailLoading } = useQuery({
    queryKey: [
      'inference-result-detail',
      projectName,
      selectedCell?.run,
      selectedCell?.video,
      selectedCell?.inferenceId,
    ],
    queryFn: () =>
      api.inference.getResult(
        projectName!,
        selectedCell!.run,
        selectedCell!.video,
        selectedCell!.inferenceId,
      ),
    enabled: !!projectName && !!selectedCell,
  })

  const { data: selectedVideo } = useQuery({
    queryKey: ['video', projectName, selectedCell?.video],
    queryFn: () => api.videos.get(projectName!, selectedCell!.video),
    enabled: !!projectName && !!selectedCell,
  })

  // ByteTrack-processed frames for the selected result. These are used as the
  // presentation frames for the main overlay, readouts, schematic and graphs.
  const { data: bytetrackData, isFetching: bytetrackFetching } = useQuery({
    queryKey: [
      'inference-bytetrack-frames',
      projectName,
      selectedCell?.run,
      selectedCell?.video,
      selectedCell?.inferenceId,
      DEFAULT_TRACKER_PARAMS.track_activation_threshold,
      DEFAULT_TRACKER_PARAMS.lost_track_buffer,
      DEFAULT_TRACKER_PARAMS.minimum_matching_threshold,
    ],
    queryFn: () =>
      api.inference.getBytetrackFrames(
        projectName!,
        selectedCell!.run,
        selectedCell!.video,
        selectedCell!.inferenceId,
        {
          track_activation_threshold: DEFAULT_TRACKER_PARAMS.track_activation_threshold,
          lost_track_buffer: DEFAULT_TRACKER_PARAMS.lost_track_buffer,
          minimum_matching_threshold: DEFAULT_TRACKER_PARAMS.minimum_matching_threshold,
        },
      ),
    enabled: !!projectName && !!selectedCell,
    staleTime: Infinity,
  })

  const { data: zCalibrationResp } = useQuery({
    queryKey: [
      'z-calibration',
      projectName,
      selectedCell?.run,
      selectedCell?.video,
      selectedCell?.inferenceId,
    ],
    queryFn: () =>
      api.inference.getZCalibration(
        projectName!,
        selectedCell!.run,
        selectedCell!.video,
        selectedCell!.inferenceId,
      ),
    enabled: !!projectName && !!selectedCell,
  })
  const zCalibration = zCalibrationResp?.z_calibration ?? null

  const rawFrames = useMemo(() => detailResult?.frames ?? [], [detailResult?.frames])
  const trackedFrames = useMemo(() => bytetrackData?.frames ?? [], [bytetrackData?.frames])
  const smoothedTrackedFrames = useMemo<InferenceResult[]>(() => {
    if (trackedFrames.length === 0) return trackedFrames
    return smoothFramesPerTrack(trackedFrames, {
      minCutoff: DEFAULT_OEF.minCutoff,
      beta: DEFAULT_OEF.beta,
    })
  }, [trackedFrames])
  const primaryTrackedFrames = useMemo(
    () => pickPrimaryTrackPerClassFrames(smoothedTrackedFrames),
    [smoothedTrackedFrames],
  )
  const presentationFrames = primaryTrackedFrames

  const completedRuns = runs?.filter((r) => r.status === 'completed') || []

  const loadModelMutation = useMutation({
    mutationFn: (runName: string) => api.inference.loadModel(projectName!, runName),
  })

  // Streaming inference state. `progress` is non-null while the SSE stream is
  // open, so the button and progress bar stay latched to the real backend
  // state (no more "finished early" when a timeout aborts the fetch).
  const [progress, setProgress] = useState<InferenceProgress | null>(null)
  const inferenceAbortRef = useRef<AbortController | null>(null)

  const isInferenceRunning = progress !== null || loadModelMutation.isPending

  const startInference = useCallback(
    async ({ runName, videoId }: { runName: string; videoId: string }) => {
      if (!projectName) return
      const controller = new AbortController()
      inferenceAbortRef.current = controller

      try {
        setProgress({
          stage: 'loading_model',
          current: 0,
          total: 0,
          avgFps: 0,
          etaS: null,
        })
        await loadModelMutation.mutateAsync(runName)
        // Flip to "starting inference" immediately so the bar animates from 0%
        // rather than looking frozen between model-load and the first frame.
        setProgress({
          stage: 'running_inference',
          current: 0,
          total: 0,
          avgFps: 0,
          etaS: null,
        })

        const stream = api.inference.runOnVideoWithProgress(
          projectName,
          videoId,
          {
            // Backend resolves the active model from `current_run_name` set by
            // load-model; this field is kept for backwards compatibility only.
            model_run_id: 0,
            confidence_threshold: config.confidence_threshold || 0.5,
            iou_threshold: config.iou_threshold || 0.45,
            max_detections: 100,
            enable_tracking: config.enable_tracking ?? false,
            tracking_mode: config.tracking_mode || 'visible_only',
            detection_interval: config.detection_interval ?? 1,
          },
          controller.signal,
        )

        let completed: Extract<InferenceProgressEvent, { type: 'complete' }> | null = null

        for await (const event of stream) {
          if (event.type === 'stage') {
            // `loading_model` is FE-only (before the stream opens); the BE
            // emits inference/encoding/post_processing. Preserve counts across
            // stage transitions so the bar doesn't jump backwards.
            setProgress((prev) => ({
              stage: event.stage,
              current: event.stage === 'running_inference' ? prev?.current ?? 0 : prev?.total ?? 0,
              total: event.total_frames ?? prev?.total ?? 0,
              avgFps: prev?.avgFps ?? 0,
              etaS: event.stage === 'running_inference' ? prev?.etaS ?? null : null,
            }))
          } else if (event.type === 'progress') {
            setProgress({
              stage: 'running_inference',
              current: event.current,
              total: event.total,
              avgFps: event.avg_fps,
              etaS: event.eta_s,
            })
          } else if (event.type === 'complete') {
            completed = event
          } else if (event.type === 'error') {
            throw new Error(event.message)
          }
        }

        if (!completed) {
          throw new Error('Stream closed before completion')
        }

        queryClient.invalidateQueries({ queryKey: ['inference-results', projectName] })
        setShowRunPanel(false)
        toast({
          title: 'Inference complete & saved',
          description: `${completed.total_frames} frames at ${completed.avg_fps.toFixed(1)} FPS`,
          type: 'success',
        })
      } catch (err) {
        if ((err as Error).name === 'AbortError') {
          toast({ title: 'Inference cancelled' })
        } else {
          toast({
            title: 'Inference failed',
            description: (err as Error).message,
            type: 'error',
          })
        }
      } finally {
        inferenceAbortRef.current = null
        setProgress(null)
      }
    },
    [projectName, config, loadModelMutation, queryClient, toast],
  )

  const cancelInference = useCallback(() => {
    inferenceAbortRef.current?.abort()
  }, [])

  // Warn the user if they try to leave while inference is streaming — closing
  // the tab kills the fetch and the backend loses its client, so we surface it.
  useEffect(() => {
    if (!isInferenceRunning) return
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault()
      e.returnValue = ''
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [isInferenceRunning])

  const { data: gpuStatus } = useQuery({
    queryKey: ['gpu-status'],
    queryFn: () => api.gpu.getStatus(),
    refetchInterval: 10000,
  })

  const { data: deviceInfo } = useQuery({
    queryKey: ['device-info'],
    queryFn: () => api.device.getInfo(),
    enabled: runMode === 'local',
  })

  const gpuSubmitMutation = useMutation({
    mutationFn: () => {
      const req: InferenceGPUSubmitRequest = {
        run_name: gpuInferConfig.run_name,
        video_ids: null,
        test_only: gpuInferConfig.test_only,
        model: gpuInferConfig.model,
        confidence: gpuInferConfig.confidence,
        frame_interval: gpuInferConfig.frame_interval,
        track: gpuInferConfig.track,
        track_thresh: gpuInferConfig.track_thresh,
        track_buffer: 30,
        match_thresh: 0.8,
        no_video: gpuInferConfig.no_video,
        gpu: {
          gpu_type: gpuInferConfig.gpu_type,
          num_gpus: 1,
          time_limit: gpuInferConfig.time_limit,
        },
      }
      return api.inference.submitGpu(projectName!, req)
    },
    onSuccess: (result) => {
      setGpuJobName(result.run_name)
      setShowRunPanel(false)
      toast({
        title: 'Inference submitted to GPU',
        description: `Job ${result.job_id}`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'GPU submission failed', description: error.message, type: 'error' })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: ({ run, video, inferenceId }: { run: string; video: string; inferenceId: string }) =>
      api.inference.deleteResult(projectName!, run, video, inferenceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['inference-results', projectName] })
      selectCell(null)
      toast({ title: 'Result deleted', type: 'success' })
    },
  })

  const videoFilename = useCallback(
    (videoId: string) => {
      const v = videos?.find((vid) => String(vid.id) === videoId)
      return v?.filename || videoId
    },
    [videos]
  )

  const selectCell = useCallback((cell: typeof selectedCell) => {
    setSelectedCell(cell)
    if (cell) {
      setSearchParams(
        {
          run: cell.run,
          video: cell.video,
          inferenceId: cell.inferenceId,
        },
        { replace: true },
      )
    } else {
      setSearchParams({}, { replace: true })
    }
  }, [setSearchParams])

  useEffect(() => {
    const run = searchParams.get('run')
    const video = searchParams.get('video')
    const inferenceId = searchParams.get('inferenceId')
    if (!run || !video || !inferenceId) return
    setSelectedCell((prev) => {
      if (
        prev?.run === run &&
        prev?.video === video &&
        prev?.inferenceId === inferenceId
      ) {
        return prev
      }
      return { run, video, inferenceId }
    })
  }, [searchParams])

  const hasResults = matrix && matrix.runs.length > 0 && matrix.videos.length > 0

  return (
    <div className="container max-w-[1800px] py-8 px-6 lg:px-8">
      <div className="flex items-center justify-end gap-3 mb-6">
        <div className="w-64">
          <GpuConnectionPanel />
        </div>
        <Button onClick={() => setShowRunPanel(true)} className="gap-2">
          <Play className="h-4 w-4" />
          Run Inference
        </Button>
      </div>

      {/* Results Matrix */}
      {matrixLoading ? (
        <Card>
          <CardContent className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </CardContent>
        </Card>
      ) : hasResults ? (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Runs × Videos</CardTitle>
            <CardDescription>
              Click a cell to view detailed results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    <th className="p-2 text-left border-b border-border font-medium text-muted-foreground">
                      Run
                    </th>
                    {matrix!.videos.map((vid) => (
                      <th
                        key={vid}
                        className="p-2 text-center border-b border-border font-medium text-muted-foreground max-w-[140px] truncate"
                        title={videoFilename(vid)}
                      >
                        {videoFilename(vid)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix!.runs.map((run) => (
                    <tr key={run} className="border-b border-border/50">
                      <td className="p-2 font-mono text-xs whitespace-nowrap">{run}</td>
                      {matrix!.videos.map((vid) => {
                        const results = matrix!.results[run]?.[vid]
                        const latest = results?.[0]
                        const isSelected =
                          selectedCell?.run === run && selectedCell?.video === vid
                        return (
                          <td key={vid} className="p-1 text-center">
                            {latest ? (
                              <button
                                onClick={() =>
                                  selectCell({
                                    run,
                                    video: vid,
                                    inferenceId: latest.inference_id,
                                  })
                                }
                                className={`
                                  w-full p-2 rounded-md border transition-colors text-xs
                                  ${isSelected
                                    ? 'border-primary bg-primary/10'
                                    : 'border-border hover:border-primary/50 hover:bg-muted/50'
                                  }
                                `}
                              >
                                <div className="flex items-center justify-center gap-1 text-green-600 dark:text-green-400">
                                  <CheckCircle2 className="h-3 w-3" />
                                  <span>{latest.stats.total_detections}</span>
                                </div>
                                <div className="text-muted-foreground mt-0.5">
                                  {latest.stats.avg_inference_time_ms.toFixed(0)}ms
                                </div>
                                {results!.length > 1 && (
                                  <div className="text-muted-foreground mt-0.5 text-[10px]">
                                    {results!.length} runs
                                  </div>
                                )}
                              </button>
                            ) : (
                              <div className="p-2 text-muted-foreground/30">&mdash;</div>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="mb-6">
          <CardContent className="py-12 text-center text-muted-foreground">
            <Grid3X3 className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p className="text-lg font-medium">No inference results yet</p>
            <p className="text-sm mt-1">
              Click "Run Inference" to evaluate a trained model on your videos
            </p>
          </CardContent>
        </Card>
      )}

      {/* Detail Panel */}
      {selectedCell && (
        <Card className="mb-6">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>
                {selectedCell.run} → {videoFilename(selectedCell.video)}
              </CardTitle>
              <CardDescription>
                {detailResult?.created_at
                  ? `Run on ${new Date(detailResult.created_at).toLocaleString('en-SG', { timeZone: 'Asia/Singapore' })} SGT`
                  : 'Loading...'}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  deleteMutation.mutate({
                    run: selectedCell.run,
                    video: selectedCell.video,
                    inferenceId: selectedCell.inferenceId,
                  })
                }
                disabled={deleteMutation.isPending}
                className="gap-1 text-destructive"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete
              </Button>
              <Button variant="ghost" size="sm" onClick={() => selectCell(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {/* Run selector when multiple results exist */}
            {(() => {
              const cellResults =
                matrix?.results[selectedCell.run]?.[selectedCell.video]
              if (cellResults && cellResults.length > 1) {
                return (
                  <div className="mb-4">
                    <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                      Inference Runs ({cellResults.length})
                    </label>
                    <div className="flex flex-wrap gap-1.5">
                      {cellResults.map((r) => {
                        const ts = r.created_at
                          ? new Date(r.created_at).toLocaleString('en-SG', {
                              timeZone: 'Asia/Singapore',
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit',
                            })
                          : r.inference_id
                        return (
                          <button
                            key={r.inference_id}
                            onClick={() =>
                              selectCell({
                                ...selectedCell,
                                inferenceId: r.inference_id,
                              })
                            }
                            className={`
                              px-2.5 py-1 rounded-md border text-xs transition-colors
                              ${selectedCell.inferenceId === r.inference_id
                                ? 'border-primary bg-primary/10 font-medium'
                                : 'border-border hover:border-primary/50'
                              }
                            `}
                          >
                            {ts}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                )
              }
              return null
            })()}

            {detailLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            ) : detailResult ? (
              <div className="space-y-4">
                <InferencePlaybackPanel
                  projectName={projectName!}
                  selectedCell={selectedCell}
                  videos={videos}
                  selectedVideo={selectedVideo}
                  presentationFrames={presentationFrames}
                  smoothedTrackedFrames={smoothedTrackedFrames}
                  rawFrames={rawFrames}
                  zCalibration={zCalibration}
                  bytetrackFetching={bytetrackFetching}
                />

                {/* Full-width bottom section */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3 bg-muted/50 rounded-lg">
                    <div className="text-xs text-muted-foreground">Total Frames</div>
                    <div className="text-lg font-semibold">{detailResult.stats.total_frames}</div>
                  </div>
                  <div className="p-3 bg-muted/50 rounded-lg">
                    <div className="text-xs text-muted-foreground">Keyframes</div>
                    <div className="text-lg font-semibold">{detailResult.stats.keyframes}</div>
                  </div>
                  <div className="p-3 bg-muted/50 rounded-lg">
                    <div className="text-xs text-muted-foreground">Total Detections</div>
                    <div className="text-lg font-semibold">
                      {detailResult.stats.total_detections}
                    </div>
                  </div>
                  <div className="p-3 bg-muted/50 rounded-lg">
                    <div className="text-xs text-muted-foreground">Avg Inference</div>
                    <div className="text-lg font-semibold">
                      {detailResult.stats.avg_inference_time_ms.toFixed(1)}ms
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="px-2 py-1 bg-muted rounded">
                    conf: {detailResult.config.confidence_threshold}
                  </span>
                  {detailResult.config.iou_threshold != null && (
                    <span className="px-2 py-1 bg-muted rounded">
                      IoU: {detailResult.config.iou_threshold}
                    </span>
                  )}
                  <span className="px-2 py-1 bg-muted rounded">
                    interval: {detailResult.config.frame_interval}
                  </span>
                  <span className="px-2 py-1 bg-muted rounded">
                    tracking: {detailResult.config.tracking ? detailResult.config.tracking_mode : 'off'}
                  </span>
                </div>

                {detailResult.frames && detailResult.frames.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Detection Timeline</h4>
                    <div className="h-16 flex items-end gap-px bg-muted/30 rounded p-1 overflow-hidden">
                      {detailResult.frames.map((frame, i) => {
                        const maxDet = Math.max(
                          1,
                          ...detailResult.frames.map((f: { detections: unknown[] }) => f.detections.length)
                        )
                        const height = (frame.detections.length / maxDet) * 100
                        return (
                          <div
                            key={i}
                            className="flex-1 min-w-[1px] bg-primary/60 rounded-t hover:bg-primary transition-colors"
                            style={{ height: `${Math.max(height, 2)}%` }}
                            title={`Frame ${frame.frame_number}: ${frame.detections.length} detections`}
                          />
                        )
                      })}
                    </div>
                  </div>
                )}

                {detailResult.frames && detailResult.frames.length > 0 && selectedCell && (
                  <div className="flex items-center gap-3 p-3 bg-muted/30 rounded-lg border border-border">
                    <ImageIcon className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium">Extract Frames</p>
                      <p className="text-xs text-muted-foreground">
                        Browse frames with detection overlays and download selected frames as images with bounding box data.
                      </p>
                    </div>
                    <Link
                      to={`/projects/${projectName}/inference/${encodeURIComponent(selectedCell.run)}/${encodeURIComponent(selectedCell.video)}/${encodeURIComponent(selectedCell.inferenceId)}/frames`}
                    >
                      <Button variant="outline" size="sm" className="gap-1.5 whitespace-nowrap">
                        <ImageIcon className="h-3.5 w-3.5" />
                        Open Frame Viewer
                      </Button>
                    </Link>
                  </div>
                )}

                {detailResult.frames && detailResult.frames.length > 0 && selectedCell && (
                  <div className="flex items-center gap-3 p-3 bg-muted/30 rounded-lg border border-border">
                    <Ruler className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium">Z-Axis Height Estimation</p>
                      <p className="text-xs text-muted-foreground">
                        Browse frames, select calibration points at known distances, and fit a depth model.
                      </p>
                    </div>
                    <Link
                      to={`/projects/${projectName}/inference/${encodeURIComponent(selectedCell.run)}/${encodeURIComponent(selectedCell.video)}/${encodeURIComponent(selectedCell.inferenceId)}/z-calibration`}
                    >
                      <Button variant="outline" size="sm" className="gap-1.5 whitespace-nowrap">
                        <Ruler className="h-3.5 w-3.5" />
                        Open Calibration
                      </Button>
                    </Link>
                  </div>
                )}

                {detailResult.frames && detailResult.frames.length > 0 && selectedCell && (
                  <div className="flex items-center gap-3 p-3 bg-muted/30 rounded-lg border border-border">
                    <Activity className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium">Tracking Comparison</p>
                      <p className="text-xs text-muted-foreground">
                        Side-by-side raw vs ByteTrack with live sliders for the three tracker parameters.
                      </p>
                    </div>
                    <Link
                      to={`/projects/${projectName}/inference/${encodeURIComponent(selectedCell.run)}/${encodeURIComponent(selectedCell.video)}/${encodeURIComponent(selectedCell.inferenceId)}/tracking-compare`}
                    >
                      <Button variant="outline" size="sm" className="gap-1.5 whitespace-nowrap">
                        <Activity className="h-3.5 w-3.5" />
                        Open Tracking Compare
                      </Button>
                    </Link>
                  </div>
                )}
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* GPU Log Viewer */}
      {gpuJobName && (
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              GPU Inference Logs
              <span className="ml-2 text-xs font-normal text-muted-foreground">{gpuJobName}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <LogViewer url={api.inference.gpuLogsUrl(projectName!, gpuJobName)} />
          </CardContent>
        </Card>
      )}

      {/* Run Inference Slide-over */}
      {showRunPanel && (
        <div className="fixed inset-0 z-50 bg-black/40" onClick={() => setShowRunPanel(false)}>
          <div
            className="absolute right-0 top-0 h-full w-full max-w-md bg-background border-l border-border shadow-xl overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold">Run Inference</h2>
                <Button variant="ghost" size="sm" onClick={() => setShowRunPanel(false)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>

              {/* Mode tabs */}
              <div className="flex gap-1 mb-6 p-1 bg-muted rounded-lg">
                <button
                  onClick={() => setRunMode('local')}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-sm font-medium transition-colors ${
                    runMode === 'local' ? 'bg-background shadow-sm' : 'text-muted-foreground'
                  }`}
                >
                  <Monitor className="h-3.5 w-3.5" />
                  Local
                </button>
                <button
                  onClick={() => setRunMode('gpu')}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-sm font-medium transition-colors ${
                    runMode === 'gpu' ? 'bg-background shadow-sm' : 'text-muted-foreground'
                  }`}
                >
                  <Server className="h-3.5 w-3.5" />
                  GPU Cluster
                </button>
              </div>

              {runMode === 'local' ? (
                <>
                  {/* Local Inference */}
                  {deviceInfo && (
                    <div className="mb-4 p-3 rounded-lg bg-muted/50 text-sm">
                      <span className="text-muted-foreground">Running on </span>
                      <span className="font-medium">{deviceInfo.name}</span>
                      {deviceInfo.memory_gb != null && (
                        <span className="text-muted-foreground"> ({deviceInfo.memory_gb} GB)</span>
                      )}
                    </div>
                  )}
                  <div className="mb-6">
                    <label className="text-sm font-medium mb-2 block">
                      <Zap className="h-4 w-4 inline mr-1" />
                      Model (Training Run)
                    </label>
                    {completedRuns.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No trained models available</p>
                    ) : (
                      <div className="space-y-2">
                        {completedRuns.map((run) => (
                          <button
                            key={run.name}
                            onClick={() =>
                              setRunTarget((prev) => ({
                                videoId: prev?.videoId ?? '',
                                runName: run.name,
                              }))
                            }
                            className={`w-full p-3 rounded-lg border text-left transition-colors ${
                              runTarget?.runName === run.name
                                ? 'border-primary bg-primary/10'
                                : 'border-border hover:border-primary/50'
                            }`}
                          >
                            <div className="font-medium text-sm">{run.name}</div>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                              <span>{run.model}</span>
                              {run.metrics?.mAP50 && (
                                <span>&middot; mAP50: {(run.metrics.mAP50 * 100).toFixed(1)}%</span>
                              )}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="mb-6">
                    <label className="text-sm font-medium mb-2 block">
                      <Video className="h-4 w-4 inline mr-1" />
                      Video
                    </label>
                    <select
                      value={runTarget?.videoId || ''}
                      onChange={(e) =>
                        setRunTarget((prev) => ({
                          runName: prev?.runName ?? '',
                          videoId: e.target.value,
                        }))
                      }
                      className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
                    >
                      <option value="">Select a video...</option>
                      {videos?.map((video) => (
                        <option key={video.id} value={String(video.id)}>
                          {video.filename}
                          {video.exclude_from_training ? ' [TEST]' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="mb-6 space-y-4">
                    <h3 className="text-sm font-medium flex items-center gap-1">
                      <Settings className="h-4 w-4" /> Settings
                    </h3>
                    <div>
                      <label className="text-sm mb-1 block">
                        Confidence: {((config.confidence_threshold || 0.5) * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={(config.confidence_threshold || 0.5) * 100}
                        onChange={(e) =>
                          setConfig({ ...config, confidence_threshold: Number(e.target.value) / 100 })
                        }
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="text-sm mb-1 block">
                        Detection Interval: every {config.detection_interval ?? 1} frames
                      </label>
                      <input
                        type="range"
                        min={1}
                        max={15}
                        value={config.detection_interval ?? 1}
                        onChange={(e) =>
                          setConfig({ ...config, detection_interval: Number(e.target.value) })
                        }
                        className="w-full"
                      />
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.enable_tracking}
                        onChange={(e) => setConfig({ ...config, enable_tracking: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm">Enable tracking</span>
                    </label>
                  </div>

                  {progress && (
                    <div className="mb-4 rounded-lg border border-border bg-muted/40 p-3">
                      {(() => {
                        const hasCounts = progress.total > 0
                        const pct = hasCounts
                          ? Math.min(100, Math.max(0, (progress.current / progress.total) * 100))
                          : 0
                        const showIndeterminate =
                          !hasCounts ||
                          progress.stage === 'loading_model' ||
                          progress.stage === 'encoding_video' ||
                          progress.stage === 'post_processing'
                        return (
                          <>
                            <div className="flex items-center justify-between text-xs font-medium mb-2">
                              <span className="flex items-center gap-1.5 text-foreground">
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                {STAGE_LABELS[progress.stage]}
                              </span>
                              <span className="text-muted-foreground tabular-nums">
                                {hasCounts && progress.stage === 'running_inference'
                                  ? `${progress.current}/${progress.total} (${pct.toFixed(1)}%)`
                                  : showIndeterminate
                                    ? ''
                                    : `${pct.toFixed(0)}%`}
                              </span>
                            </div>
                            <div className="h-2 w-full overflow-hidden rounded-full bg-border">
                              {showIndeterminate ? (
                                <div
                                  className="h-full w-1/3 rounded-full bg-primary/70"
                                  style={{
                                    animation: 'inference-indeterminate 1.4s ease-in-out infinite',
                                  }}
                                />
                              ) : (
                                <div
                                  className="h-full rounded-full bg-primary transition-[width] duration-150 ease-out"
                                  style={{ width: `${pct}%` }}
                                />
                              )}
                            </div>
                            {progress.stage === 'running_inference' && progress.avgFps > 0 && (
                              <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground tabular-nums">
                                <span>{progress.avgFps.toFixed(1)} FPS</span>
                                <span>
                                  {progress.etaS != null && progress.etaS > 0
                                    ? `ETA ${formatEta(progress.etaS)}`
                                    : ''}
                                </span>
                              </div>
                            )}
                          </>
                        )
                      })()}
                      <style>{`
                        @keyframes inference-indeterminate {
                          0% { transform: translateX(-100%); }
                          100% { transform: translateX(400%); }
                        }
                      `}</style>
                    </div>
                  )}

                  <Button
                    className="w-full gap-2"
                    disabled={
                      !runTarget?.runName ||
                      !runTarget?.videoId ||
                      isInferenceRunning
                    }
                    onClick={() => {
                      if (runTarget?.runName && runTarget?.videoId) {
                        startInference({
                          runName: runTarget.runName,
                          videoId: runTarget.videoId,
                        })
                      }
                    }}
                  >
                    {isInferenceRunning ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    {progress?.stage === 'loading_model' || loadModelMutation.isPending
                      ? 'Loading model...'
                      : progress?.stage === 'running_inference'
                        ? 'Running inference...'
                        : progress?.stage === 'encoding_video'
                          ? 'Encoding video...'
                          : progress?.stage === 'post_processing'
                            ? 'Saving...'
                            : 'Run & Save'}
                  </Button>

                  {isInferenceRunning && (
                    <Button
                      variant="outline"
                      className="w-full mt-2 gap-2"
                      onClick={cancelInference}
                    >
                      <X className="h-4 w-4" />
                      Cancel
                    </Button>
                  )}
                </>
              ) : (
                <>
                  {/* GPU Inference Config */}
                  <div className="mb-4">
                    <label className="text-sm font-medium mb-2 block">Training Run</label>
                    {completedRuns.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No trained models available</p>
                    ) : (
                      <select
                        value={gpuInferConfig.run_name || ''}
                        onChange={(e) =>
                          setGpuInferConfig({
                            ...gpuInferConfig,
                            run_name: e.target.value || null,
                          })
                        }
                        className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
                      >
                        <option value="">Latest run</option>
                        {completedRuns.map((run) => (
                          <option key={run.name} value={run.name}>
                            {run.name} {run.metrics?.mAP50 ? `(mAP: ${(run.metrics.mAP50 * 100).toFixed(1)}%)` : ''}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  <div className="mb-4">
                    <label className="text-sm font-medium mb-2 block">GPU Type</label>
                    <select
                      value={gpuInferConfig.gpu_type}
                      onChange={(e) =>
                        setGpuInferConfig({ ...gpuInferConfig, gpu_type: e.target.value as GPUType })
                      }
                      className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
                    >
                      <option value="h200">H200 (141 GB)</option>
                      <option value="h100-96">H100-96 (96 GB)</option>
                      <option value="h100-47">H100-47 (47 GB)</option>
                      <option value="a100-80">A100-80 (80 GB)</option>
                      <option value="a100-40">A100-40 (40 GB)</option>
                      <option value="nv">NV (misc)</option>
                    </select>
                  </div>

                  <div className="mb-4 space-y-3">
                    <h3 className="text-sm font-medium flex items-center gap-1">
                      <Settings className="h-4 w-4" /> Settings
                    </h3>
                    <div>
                      <label className="text-sm mb-1 block">
                        Confidence: {(gpuInferConfig.confidence * 100).toFixed(0)}%
                      </label>
                      <input
                        type="range"
                        min={0}
                        max={100}
                        value={gpuInferConfig.confidence * 100}
                        onChange={(e) =>
                          setGpuInferConfig({ ...gpuInferConfig, confidence: Number(e.target.value) / 100 })
                        }
                        className="w-full"
                      />
                    </div>
                    <div>
                      <label className="text-sm mb-1 block">
                        Frame Interval: every {gpuInferConfig.frame_interval} frames
                      </label>
                      <input
                        type="range"
                        min={1}
                        max={15}
                        value={gpuInferConfig.frame_interval}
                        onChange={(e) =>
                          setGpuInferConfig({ ...gpuInferConfig, frame_interval: Number(e.target.value) })
                        }
                        className="w-full"
                      />
                    </div>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={gpuInferConfig.track}
                        onChange={(e) => setGpuInferConfig({ ...gpuInferConfig, track: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm">Enable tracking</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={gpuInferConfig.test_only}
                        onChange={(e) => setGpuInferConfig({ ...gpuInferConfig, test_only: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm">Test-only videos</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={gpuInferConfig.no_video}
                        onChange={(e) => setGpuInferConfig({ ...gpuInferConfig, no_video: e.target.checked })}
                        className="rounded"
                      />
                      <span className="text-sm">No output video (JSON only)</span>
                    </label>
                  </div>

                  <Button
                    className="w-full gap-2"
                    disabled={!gpuStatus?.connected || gpuSubmitMutation.isPending}
                    onClick={() => gpuSubmitMutation.mutate()}
                  >
                    {gpuSubmitMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Server className="h-4 w-4" />
                    )}
                    {!gpuStatus?.connected ? 'Connect GPU First' : 'Submit to GPU'}
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface InferencePlaybackPanelProps {
  projectName: string
  selectedCell: SelectedInferenceCell
  videos?: ProjectVideo[]
  selectedVideo?: ProjectVideo
  presentationFrames: InferenceResult[]
  /** All smoothed ByteTrack tracks (pre primary-pick); needed for stacking analysis. */
  smoothedTrackedFrames: InferenceResult[]
  rawFrames: InferenceResult[]
  zCalibration: ZCalibration | null
  bytetrackFetching: boolean
}

function InferencePlaybackPanel({
  projectName,
  selectedCell,
  videos,
  selectedVideo,
  presentationFrames,
  smoothedTrackedFrames,
  rawFrames,
  zCalibration,
  bytetrackFetching,
}: InferencePlaybackPanelProps) {
  const [graphClass, setGraphClass] = useState<string>('crane hook')
  const videoRef = useRef<HTMLVideoElement>(null)
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const tick = () => {
      const v = videoRef.current
      if (v && !v.paused && !v.ended) {
        setVideoTime(v.currentTime)
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  useEffect(() => {
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
  }, [selectedCell.inferenceId])

  const availableClasses = useMemo(() => {
    const sourceFrames = presentationFrames.length > 0 ? presentationFrames : rawFrames
    if (sourceFrames.length === 0) return []
    const names = new Set<string>()
    for (const f of sourceFrames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [presentationFrames, rawFrames])

  useEffect(() => {
    if (availableClasses.length > 0 && !availableClasses.includes(graphClass)) {
      setGraphClass(availableClasses[0])
    }
  }, [availableClasses, graphClass])

  const currentTrackedFrameIndex = useMemo(
    () => findClosestFrameIndex(presentationFrames, videoTime),
    [presentationFrames, videoTime],
  )
  const currentTrackedFrame =
    currentTrackedFrameIndex >= 0 ? presentationFrames[currentTrackedFrameIndex] : null
  const trackedOverlay = useMemo(
    () => buildTrackedOverlayBoxes(currentTrackedFrame),
    [currentTrackedFrame],
  )

  const vid = selectedVideo ?? videos?.find((v) => String(v.id) === selectedCell.video)
  const vw = vid?.width || 1920
  const vh = vid?.height || 1080

  // Loaded-spreader stacking analysis over ALL smoothed tracks (the primary
  // pick collapses containers to the center one, which is exactly the carried
  // container we must exclude).
  const stackingAnalysis = useMemo(() => {
    if (smoothedTrackedFrames.length === 0) return null
    const names = new Set<string>()
    for (const f of smoothedTrackedFrames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    const allClasses = Array.from(names).sort()
    const spreaderClass = resolveSpreaderClass(allClasses, zCalibration)
    const containerClass = resolveContainerClass(allClasses, spreaderClass)
    if (!spreaderClass || !containerClass || spreaderClass === containerClass) return null
    return analyzeStacking({
      frames: smoothedTrackedFrames,
      spreaderClass,
      containerClass,
      videoWidth: vw,
      videoHeight: vh,
      calibration: zCalibration,
    })
  }, [smoothedTrackedFrames, zCalibration, vw, vh])

  // smoothedTrackedFrames and presentationFrames are index-aligned (the
  // primary pick is a per-frame map), so the tracked frame index is valid for
  // both the stacking analysis and the full smoothed frame.
  const currentSmoothedFrame =
    currentTrackedFrameIndex >= 0
      ? smoothedTrackedFrames[currentTrackedFrameIndex] ?? null
      : null
  const stackingBoxes = useMemo(
    () => buildStackingOverlayBoxes(currentSmoothedFrame, stackingAnalysis, currentTrackedFrameIndex),
    [currentSmoothedFrame, stackingAnalysis, currentTrackedFrameIndex],
  )
  const overlayBoxes = useMemo(
    () => [...trackedOverlay.boxes, ...stackingBoxes],
    [trackedOverlay.boxes, stackingBoxes],
  )

  const videoSrc = api.videos.streamUrl(projectName, selectedCell.video, true)
  const trackedCount = trackedOverlay.measured + trackedOverlay.extrapolated
  const hasPresentationFrames = presentationFrames.length > 0

  return (
    <>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="min-w-0">
          <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
            <h4 className="text-sm font-medium">Tracked Video</h4>
            <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
              {bytetrackFetching && (
                <span className="inline-flex items-center gap-1.5">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Tracking
                </span>
              )}
              <span>
                <span style={{ color: COLOR_MEASURED }}>measured {trackedOverlay.measured}</span>
                <span className="mx-1">/</span>
                <span style={{ color: COLOR_EXTRAPOLATED }}>
                  extrapolated {trackedOverlay.extrapolated}
                </span>
                <span className="ml-1">shown {trackedCount}</span>
              </span>
            </div>
          </div>
          <div
            className="relative w-full bg-black rounded-lg overflow-hidden"
            style={{ aspectRatio: `${vw} / ${vh}` }}
          >
            <video
              ref={videoRef}
              key={`${selectedCell.run}-${selectedCell.video}-${selectedCell.inferenceId}`}
              src={videoSrc}
              controls
              className="absolute inset-0 h-full w-full"
              preload="auto"
              playsInline
              onLoadedMetadata={() => {
                if (videoRef.current) setVideoDuration(videoRef.current.duration)
              }}
            />
            <DetectionOverlaySvg
              boxes={overlayBoxes}
              videoWidth={vw}
              videoHeight={vh}
            />
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Overlay uses ByteTrack primary tracks with One Euro smoothing; dashed boxes are
            Kalman extrapolations through missed detections.
          </p>
        </div>
        <div>
          {hasPresentationFrames ? (
            <SideViewSchematic
              frames={presentationFrames}
              currentTime={videoTime}
              videoWidth={vw}
              videoHeight={vh}
              projectName={projectName}
              runName={selectedCell.run}
              videoId={selectedCell.video}
              inferenceId={selectedCell.inferenceId}
              stacking={stackingAnalysis}
            />
          ) : (
            <div className="rounded-lg border border-dashed border-border bg-muted/30 p-6 text-center text-xs text-muted-foreground">
              {bytetrackFetching
                ? 'Preparing tracked schematic...'
                : 'No tracked frames available.'}
            </div>
          )}
        </div>
      </div>

      {hasPresentationFrames && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="space-y-2 min-w-0">
            <div className="flex items-end justify-between gap-3 flex-wrap">
              <h4 className="text-sm font-medium">Position Graphs</h4>
              {availableClasses.length > 1 && (
                <div className="w-full sm:w-56">
                  <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wide block mb-1.5">
                    Graph Class
                  </label>
                  <select
                    value={graphClass}
                    onChange={(e) => setGraphClass(e.target.value)}
                    className="w-full rounded-md border border-border bg-background px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                  >
                    {availableClasses.map((cls) => (
                      <option key={cls} value={cls}>{cls}</option>
                    ))}
                  </select>
                </div>
              )}
            </div>
            <PositionTimeline
              frames={presentationFrames}
              currentTime={videoTime}
              duration={videoDuration}
              metric="z"
              targetClass={graphClass}
              videoWidth={vw}
              videoHeight={vh}
              zCalibration={zCalibration}
            />
            <PositionTimeline
              frames={presentationFrames}
              currentTime={videoTime}
              duration={videoDuration}
              metric="x"
              targetClass={graphClass}
              videoWidth={vw}
              videoHeight={vh}
            />
            <PositionTimeline
              frames={presentationFrames}
              currentTime={videoTime}
              duration={videoDuration}
              metric="y"
              targetClass={graphClass}
              videoWidth={vw}
              videoHeight={vh}
            />
          </div>
          <LiveDetectionReadout
            frames={presentationFrames}
            currentTime={videoTime}
            videoWidth={vw}
            videoHeight={vh}
            zCalibration={zCalibration}
          />
        </div>
      )}
    </>
  )
}
