import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
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
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { useToast } from '@/components/ui/Toaster'
import GpuConnectionPanel from '@/components/GpuConnectionPanel'
import LogViewer from '@/components/LogViewer'
import PositionTimeline from '@/components/HeightTimeline'
import LiveDetectionReadout from '@/components/LiveDetectionReadout'
import type { InferenceConfig, InferenceGPUSubmitRequest, RFDETRModelSize, GPUType } from '@/types'

export default function InferencePage() {
  const { projectName } = useParams<{ projectName: string }>()
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const [selectedCell, setSelectedCell] = useState<{
    run: string
    video: string
    inferenceId: string
  } | null>(null)
  const [showRunPanel, setShowRunPanel] = useState(false)
  const [runTarget, setRunTarget] = useState<{ runId: number; videoId: string } | null>(null)
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

  const availableClasses = useMemo(() => {
    if (!detailResult?.frames) return []
    const names = new Set<string>()
    for (const f of detailResult.frames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [detailResult?.frames])

  useEffect(() => {
    if (availableClasses.length > 0 && !availableClasses.includes(graphClass)) {
      setGraphClass(availableClasses[0])
    }
  }, [availableClasses, graphClass])

  const completedRuns = runs?.filter((r) => r.status === 'completed') || []

  const loadModelMutation = useMutation({
    mutationFn: (runId: number) => api.inference.loadModel(projectName!, runId),
  })

  const runInferenceMutation = useMutation({
    mutationFn: async ({ runId, videoId }: { runId: number; videoId: string }) => {
      await loadModelMutation.mutateAsync(runId)
      return api.inference.runOnVideo(projectName!, videoId, {
        model_run_id: runId,
        confidence_threshold: config.confidence_threshold || 0.5,
        iou_threshold: config.iou_threshold || 0.45,
        max_detections: 100,
        enable_tracking: config.enable_tracking ?? false,
        tracking_mode: config.tracking_mode || 'visible_only',
        detection_interval: config.detection_interval ?? 1,
      })
    },
    onSuccess: (data: any) => {
      queryClient.invalidateQueries({ queryKey: ['inference-results', projectName] })
      setShowRunPanel(false)
      toast({
        title: 'Inference complete & saved',
        description: `${data.total_frames} frames at ${data.avg_fps.toFixed(1)} FPS`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Inference failed', description: error.message, type: 'error' })
    },
  })

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
  }, [])

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
                {/* Two-column layout: left = video + graphs, right = readout */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                  {/* Left column: Video + Position Graphs */}
                  <div className="space-y-4 min-w-0">
                    {detailResult.has_video && selectedCell && (() => {
                      const vid = videos?.find((v) => String(v.id) === selectedCell.video)
                      return (
                        <div>
                          <h4 className="text-sm font-medium mb-2">Detected Video</h4>
                          <video
                            ref={videoRef}
                            key={`${selectedCell.run}-${selectedCell.video}-${selectedCell.inferenceId}`}
                            src={api.inference.videoUrl(
                              projectName!,
                              selectedCell.run,
                              selectedCell.video,
                              selectedCell.inferenceId
                            )}
                            controls
                            className="w-full rounded-lg bg-black"
                            preload="auto"
                            playsInline
                            onLoadedMetadata={() => {
                              if (videoRef.current) setVideoDuration(videoRef.current.duration)
                            }}
                          />
                          {detailResult.frames && detailResult.frames.length > 0 && (
                            <div className="mt-2 space-y-2">
                              <PositionTimeline
                                frames={detailResult.frames}
                                currentTime={videoTime}
                                duration={videoDuration}
                                metric="z"
                                targetClass={graphClass}
                              />
                              <PositionTimeline
                                frames={detailResult.frames}
                                currentTime={videoTime}
                                duration={videoDuration}
                                metric="x"
                                targetClass={graphClass}
                                videoWidth={vid?.width || 1920}
                                videoHeight={vid?.height || 1080}
                              />
                              <PositionTimeline
                                frames={detailResult.frames}
                                currentTime={videoTime}
                                duration={videoDuration}
                                metric="y"
                                targetClass={graphClass}
                                videoWidth={vid?.width || 1920}
                                videoHeight={vid?.height || 1080}
                              />
                            </div>
                          )}
                          <p className="text-xs text-muted-foreground mt-2">
                            If the video does not load above,{' '}
                            <a
                              href={api.inference.videoUrl(
                                projectName!,
                                selectedCell.run,
                                selectedCell.video,
                                selectedCell.inferenceId
                              )}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-primary hover:underline"
                            >
                              open it in a new tab
                            </a>
                            .
                          </p>
                        </div>
                      )
                    })()}
                  </div>

                  {/* Right column: Graph class selector + Live Readout */}
                  <div className="space-y-4">
                    {availableClasses.length > 1 && (
                      <div>
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

                    {detailResult.frames && detailResult.frames.length > 0 && (() => {
                      const vid = videos?.find((v) => String(v.id) === selectedCell?.video)
                      const vw = vid?.width || 1920
                      const vh = vid?.height || 1080
                      return (
                        <div className="xl:max-h-[calc(100vh-200px)] xl:overflow-y-auto">
                          <LiveDetectionReadout
                            frames={detailResult.frames}
                            currentTime={videoTime}
                            videoWidth={vw}
                            videoHeight={vh}
                          />
                        </div>
                      )
                    })()}
                  </div>
                </div>

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
                            key={run.id}
                            onClick={() =>
                              setRunTarget((prev) => ({ ...prev!, runId: run.id }))
                            }
                            className={`w-full p-3 rounded-lg border text-left transition-colors ${
                              runTarget?.runId === run.id
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
                        setRunTarget((prev) => ({ runId: prev?.runId || 0, videoId: e.target.value }))
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

                  <Button
                    className="w-full gap-2"
                    disabled={
                      !runTarget?.runId ||
                      !runTarget?.videoId ||
                      runInferenceMutation.isPending ||
                      loadModelMutation.isPending
                    }
                    onClick={() => {
                      if (runTarget?.runId && runTarget?.videoId) {
                        runInferenceMutation.mutate({
                          runId: runTarget.runId,
                          videoId: runTarget.videoId,
                        })
                      }
                    }}
                  >
                    {runInferenceMutation.isPending || loadModelMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                    {loadModelMutation.isPending
                      ? 'Loading model...'
                      : runInferenceMutation.isPending
                        ? 'Running inference...'
                        : 'Run & Save'}
                  </Button>
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
                          <option key={run.id} value={run.name}>
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
