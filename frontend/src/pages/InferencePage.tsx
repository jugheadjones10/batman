import { useState, useRef } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Play,
  Pause,
  Download,
  Settings,
  Loader2,
  Zap,
  BarChart3,
  Video,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { useToast } from '@/components/ui/Toaster'
import { formatDuration } from '@/lib/utils'
import type { InferenceConfig, InferenceResult, Video as VideoType } from '@/types'

export default function InferencePage() {
  const { projectName } = useParams<{ projectName: string }>()
  const { toast } = useToast()
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const [selectedVideoId, setSelectedVideoId] = useState<number | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [results, setResults] = useState<InferenceResult[]>([])
  const [config, setConfig] = useState<Partial<InferenceConfig>>({
    confidence_threshold: 0.5,
    iou_threshold: 0.45,
    enable_tracking: true,
    tracking_mode: 'visible_only',
    detection_interval: 5, // Detect every 5 frames (5x faster)
  })

  const { data: project } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

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

  const completedRuns = runs?.filter((r) => r.status === 'completed') || []

  const loadModelMutation = useMutation({
    mutationFn: (runId: number) => api.inference.loadModel(projectName!, runId),
    onSuccess: () => {
      toast({ title: 'Model loaded', type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Failed to load model', description: error.message, type: 'error' })
    },
  })

  const runInferenceMutation = useMutation({
    mutationFn: () =>
      api.inference.runOnVideo(projectName!, selectedVideoId!, {
        model_run_id: selectedRunId!,
        confidence_threshold: config.confidence_threshold || 0.5,
        iou_threshold: config.iou_threshold || 0.45,
        max_detections: 100,
        enable_tracking: config.enable_tracking ?? true,
        tracking_mode: config.tracking_mode || 'visible_only',
        detection_interval: config.detection_interval || 5, // Detect every 5 frames for speed
      }),
    onSuccess: (data) => {
      setResults(data.results)
      toast({
        title: 'Inference complete',
        description: `${data.processed_frames} frames at ${data.avg_fps.toFixed(1)} FPS`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Inference failed', description: error.message, type: 'error' })
    },
  })

  const exportMutation = useMutation({
    mutationFn: () =>
      api.inference.exportVideo(projectName!, selectedVideoId!, {
        model_run_id: selectedRunId!,
        confidence_threshold: config.confidence_threshold || 0.5,
        iou_threshold: config.iou_threshold || 0.45,
        max_detections: 100,
        enable_tracking: config.enable_tracking ?? true,
        tracking_mode: config.tracking_mode || 'visible_only',
        detection_interval: config.detection_interval || 5,
      }),
    onSuccess: (data) => {
      toast({
        title: 'Video exported',
        description: `Saved to ${data.output_path}`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Export failed', description: error.message, type: 'error' })
    },
  })

  const handleLoadModel = async (runId: number) => {
    setSelectedRunId(runId)
    await loadModelMutation.mutateAsync(runId)
  }

  const handleRunInference = async () => {
    if (!selectedVideoId || !selectedRunId) {
      toast({ title: 'Select video and model first', type: 'error' })
      return
    }

    setIsRunning(true)
    await runInferenceMutation.mutateAsync()
    setIsRunning(false)
  }

  const selectedVideo = videos?.find((v) => v.id === selectedVideoId)

  // Draw detections on canvas
  const drawDetections = (frame: InferenceResult) => {
    const video = videoRef.current
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!video || !canvas || !ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    frame.detections.forEach((det) => {
      const color = colors[det.class_id % colors.length]

      // Convert normalized coords to pixel coords
      const x = (det.box.x - det.box.width / 2) * canvas.width
      const y = (det.box.y - det.box.height / 2) * canvas.height
      const w = det.box.width * canvas.width
      const h = det.box.height * canvas.height

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)

      // Label
      const label = det.track_id
        ? `[${det.track_id}] ${det.class_name} ${(det.confidence * 100).toFixed(0)}%`
        : `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`

      ctx.font = '12px JetBrains Mono'
      const metrics = ctx.measureText(label)
      ctx.fillStyle = color
      ctx.fillRect(x, y - 18, metrics.width + 8, 18)
      ctx.fillStyle = '#000'
      ctx.fillText(label, x + 4, y - 5)
    })
  }

  return (
    <div className="container max-w-6xl py-8 px-6 lg:px-8">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-3xl font-bold flex items-center gap-3">
            <Play className="h-8 w-8 text-primary" />
            Inference
          </h1>
          <p className="text-muted-foreground mt-1">
            Run your trained model on videos
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Video preview */}
        <div className="lg:col-span-2">
          <Card>
            <CardContent className="p-0">
              <div className="relative aspect-video bg-black rounded-t-lg overflow-hidden">
                {selectedVideo ? (
                  <>
                    <video
                      ref={videoRef}
                      src={api.videos.streamUrl(projectName!, selectedVideo.id)}
                      className="w-full h-full object-contain"
                      controls
                    />
                    <canvas
                      ref={canvasRef}
                      width={selectedVideo.width}
                      height={selectedVideo.height}
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ objectFit: 'contain' }}
                    />
                  </>
                ) : (
                  <div className="flex items-center justify-center h-full text-muted-foreground">
                    <p>Select a video to preview</p>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="p-4 border-t border-border">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Button
                      onClick={handleRunInference}
                      disabled={!selectedVideoId || !selectedRunId || isRunning}
                      className="gap-2"
                    >
                      {isRunning ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      Run Inference
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => exportMutation.mutate()}
                      disabled={!selectedVideoId || !selectedRunId || exportMutation.isPending}
                      className="gap-2"
                    >
                      {exportMutation.isPending ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="h-4 w-4" />
                      )}
                      Export Video
                    </Button>
                  </div>

                  {results.length > 0 && (
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <BarChart3 className="h-4 w-4" />
                        {results.length} frames
                      </span>
                      <span className="flex items-center gap-1">
                        <Zap className="h-4 w-4" />
                        {(results.reduce((acc, r) => acc + r.inference_time_ms, 0) / results.length).toFixed(1)}ms avg
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Results */}
          {results.length > 0 && (
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Detection Results</CardTitle>
                <CardDescription>
                  {results.reduce((acc, r) => acc + r.detections.length, 0)} total detections
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-32 overflow-y-auto">
                  <div className="flex gap-1">
                    {results.map((frame) => (
                      <div
                        key={frame.frame_number}
                        className="w-1 bg-primary/30 hover:bg-primary transition-colors cursor-pointer"
                        style={{ height: `${Math.min(frame.detections.length * 10, 100)}%` }}
                        onClick={() => {
                          if (videoRef.current) {
                            videoRef.current.currentTime = frame.timestamp
                          }
                          drawDetections(frame)
                        }}
                        title={`Frame ${frame.frame_number}: ${frame.detections.length} detections`}
                      />
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Video selector */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Video className="h-5 w-5" />
                Select Video
              </CardTitle>
            </CardHeader>
            <CardContent>
              <select
                value={selectedVideoId || ''}
                onChange={(e) => setSelectedVideoId(Number(e.target.value) || null)}
                className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
              >
                <option value="">Select a video...</option>
                {videos?.map((video) => (
                  <option key={video.id} value={video.id}>
                    {video.filename}
                  </option>
                ))}
              </select>
            </CardContent>
          </Card>

          {/* Model selector */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Select Model
              </CardTitle>
            </CardHeader>
            <CardContent>
              {completedRuns.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No trained models available
                </p>
              ) : (
                <div className="space-y-2">
                  {completedRuns.map((run) => (
                    <button
                      key={run.id}
                      onClick={() => handleLoadModel(run.id)}
                      className={`
                        w-full p-3 rounded-lg border text-left transition-colors
                        ${selectedRunId === run.id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                        }
                      `}
                    >
                      <div className="font-medium text-sm">{run.name}</div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                        <span>{run.base_model}</span>
                        {run.metrics?.mAP50 && (
                          <span>• mAP50: {(run.metrics.mAP50 * 100).toFixed(1)}%</span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Settings */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">
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
                <label className="text-sm font-medium mb-2 block">
                  IoU Threshold: {((config.iou_threshold || 0.45) * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={(config.iou_threshold || 0.45) * 100}
                  onChange={(e) =>
                    setConfig({ ...config, iou_threshold: Number(e.target.value) / 100 })
                  }
                  className="w-full"
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">
                  Detection Interval: every {config.detection_interval || 5} frames
                </label>
                <input
                  type="range"
                  min={1}
                  max={15}
                  value={config.detection_interval || 5}
                  onChange={(e) =>
                    setConfig({ ...config, detection_interval: Number(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  {config.detection_interval === 1 
                    ? 'Max accuracy (slowest)'
                    : `${config.detection_interval}x faster, uses tracking between detections`
                  }
                </p>
              </div>

              <div>
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

              {config.enable_tracking && (
                <div>
                  <label className="text-sm font-medium mb-2 block">Tracking Mode</label>
                  <select
                    value={config.tracking_mode}
                    onChange={(e) =>
                      setConfig({ ...config, tracking_mode: e.target.value as any })
                    }
                    className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
                  >
                    <option value="visible_only">Visible Only</option>
                    <option value="occlusion_tolerant">Occlusion Tolerant</option>
                  </select>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

