import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  ArrowLeft,
  Loader2,
  Plus,
  Trash2,
  Ruler,
  Check,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'
import { cn } from '@/lib/utils'
import type { Detection, InferenceResult, ZCalibrationLabel } from '@/types'

const DETECTION_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
const FRAME_INTERVALS = [1, 2, 5, 10, 15, 30, 60] as const

function pickDefaultInterval(totalFrames: number): number {
  if (totalFrames <= 60) return 1
  if (totalFrames <= 150) return 5
  if (totalFrames <= 500) return 10
  if (totalFrames <= 1500) return 30
  return 60
}

interface CalibrationPoint {
  frameIndex: number
  frame_number: number
  z_mm: string
}

export default function ZCalibrationPage() {
  const { projectName, runName, videoId, inferenceId } = useParams<{
    projectName: string
    runName: string
    videoId: string
    inferenceId: string
  }>()
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [calibrationPoints, setCalibrationPoints] = useState<CalibrationPoint[]>([])
  const thumbnailStripRef = useRef<HTMLDivElement>(null)
  const [frameInterval, setFrameInterval] = useState<number | null>(null)

  const { data: video } = useQuery({
    queryKey: ['video', projectName, videoId],
    queryFn: () => api.videos.get(projectName!, videoId!),
    enabled: !!projectName && !!videoId,
  })

  const { data: detailResult, isLoading } = useQuery({
    queryKey: ['inference-result-detail', projectName, runName, videoId, inferenceId],
    queryFn: () => api.inference.getResult(projectName!, runName!, videoId!, inferenceId!),
    enabled: !!projectName && !!runName && !!videoId && !!inferenceId,
  })

  const { data: existingCal } = useQuery({
    queryKey: ['z-calibration', projectName, runName, videoId, inferenceId],
    queryFn: () => api.inference.getZCalibration(projectName!, runName!, videoId!, inferenceId!),
    enabled: !!projectName && !!runName && !!videoId && !!inferenceId,
  })

  const allFrames: InferenceResult[] = useMemo(
    () => detailResult?.frames ?? [],
    [detailResult],
  )

  useEffect(() => {
    if (frameInterval === null && allFrames.length > 0) {
      setFrameInterval(pickDefaultInterval(allFrames.length))
    }
  }, [allFrames.length, frameInterval])

  const effectiveInterval = frameInterval ?? 1

  const filteredFrames = useMemo(
    () => allFrames.filter((_, i) => i % effectiveInterval === 0),
    [allFrames, effectiveInterval],
  )

  useEffect(() => {
    setCurrentFrameIndex(0)
    setCalibrationPoints([])
  }, [effectiveInterval])

  // Load existing calibration points when data arrives
  useEffect(() => {
    if (!existingCal?.z_calibration?.labels?.length || allFrames.length === 0 || calibrationPoints.length > 0) return
    const points: CalibrationPoint[] = []
    for (const label of existingCal.z_calibration.labels) {
      const idx = filteredFrames.findIndex((f) => f.frame_number === label.frame_number)
      if (idx !== -1) {
        points.push({ frameIndex: idx, frame_number: label.frame_number, z_mm: String(label.z_mm) })
      }
    }
    if (points.length > 0) setCalibrationPoints(points)
  }, [existingCal, allFrames.length, filteredFrames, calibrationPoints.length])

  const currentFrame = filteredFrames[currentFrameIndex]
  const hasExistingZ = existingCal?.z_calibration?.model != null

  const classNames = useMemo(() => {
    const names = new Set<string>()
    for (const f of allFrames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [allFrames])

  const classColorMap = useMemo(() => {
    const map: Record<string, string> = {}
    classNames.forEach((name, i) => {
      map[name] = DETECTION_COLORS[i % DETECTION_COLORS.length]
    })
    return map
  }, [classNames])

  const selectedFrameNumbers = useMemo(
    () => new Set(calibrationPoints.map((p) => p.frame_number)),
    [calibrationPoints],
  )

  const goToFrame = useCallback(
    (index: number) => {
      if (filteredFrames.length > 0 && index >= 0 && index < filteredFrames.length) {
        setCurrentFrameIndex(index)
      }
    },
    [filteredFrames.length],
  )

  const goToFrameRef = useRef(goToFrame)
  goToFrameRef.current = goToFrame

  const addCurrentFrame = useCallback(() => {
    if (!currentFrame) return
    if (selectedFrameNumbers.has(currentFrame.frame_number)) {
      toast({ title: 'Already added', description: `Frame ${currentFrame.frame_number} is already a calibration point`, type: 'error' })
      return
    }
    if (currentFrame.detections.length === 0) {
      toast({ title: 'No detections', description: 'This frame has no detections to calibrate against', type: 'error' })
      return
    }
    setCalibrationPoints((prev) => [
      ...prev,
      { frameIndex: currentFrameIndex, frame_number: currentFrame.frame_number, z_mm: '' },
    ])
  }, [currentFrame, currentFrameIndex, selectedFrameNumbers, toast])

  const removePoint = useCallback((frameNumber: number) => {
    setCalibrationPoints((prev) => prev.filter((p) => p.frame_number !== frameNumber))
  }, [])

  const updateDistance = useCallback((frameNumber: number, value: string) => {
    setCalibrationPoints((prev) =>
      prev.map((p) => (p.frame_number === frameNumber ? { ...p, z_mm: value } : p)),
    )
  }, [])

  const navigateToPoint = useCallback(
    (frameIndex: number) => {
      goToFrame(frameIndex)
    },
    [goToFrame],
  )

  const validLabels = useMemo((): ZCalibrationLabel[] => {
    return calibrationPoints
      .filter((p) => {
        const z = parseFloat(p.z_mm)
        return !isNaN(z) && z > 0
      })
      .map((p) => ({
        frame_number: p.frame_number,
        z_mm: parseFloat(p.z_mm),
        detection_index: 0,
      }))
  }, [calibrationPoints])

  const calibrateMutation = useMutation({
    mutationFn: async () => {
      await api.inference.saveZCalibration(projectName!, runName!, videoId!, inferenceId!, validLabels)
      return api.inference.applyZEstimation(projectName!, runName!, videoId!, inferenceId!)
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['z-calibration', projectName, runName, videoId, inferenceId] })
      queryClient.invalidateQueries({ queryKey: ['inference-result-detail', projectName, runName, videoId, inferenceId] })
      toast({ title: 'Z estimation applied', description: `Model: ${result.model.type}`, type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Z calibration failed', description: error.message, type: 'error' })
    },
  })


  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goToFrameRef.current(currentFrameIndex - 1)
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        goToFrameRef.current(currentFrameIndex + 1)
      }
      if (e.key === ' ') {
        e.preventDefault()
        addCurrentFrame()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentFrameIndex, addCurrentFrame])

  useEffect(() => {
    const strip = thumbnailStripRef.current
    const thumb = strip?.querySelector(`[data-index="${currentFrameIndex}"]`)
    if (strip && thumb) {
      thumb.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' })
    }
  }, [currentFrameIndex])

  if (!projectName || !videoId || !runName || !inferenceId) return null

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (allFrames.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <p className="text-muted-foreground">No frames in this inference result.</p>
        <Link to={`/projects/${projectName}/inference`}>
          <Button variant="ghost">Back to inference</Button>
        </Link>
      </div>
    )
  }

  const imageUrl = currentFrame
    ? api.videos.frameUrl(projectName, videoId, currentFrame.frame_number)
    : ''

  const isCurrentFrameAdded = currentFrame ? selectedFrameNumbers.has(currentFrame.frame_number) : false
  const hasIncompletePoints = calibrationPoints.some((p) => !p.z_mm || parseFloat(p.z_mm) <= 0 || isNaN(parseFloat(p.z_mm)))

  return (
    <div className="h-[calc(100vh-4rem)] flex overflow-hidden">
      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 bg-neutral-900">
        {/* Top bar */}
        <div className="flex-shrink-0 px-4 py-2 border-b border-border flex items-center gap-3 flex-wrap">
          <Link to={`/projects/${projectName}/inference`}>
            <Button variant="ghost" size="sm" className="gap-1 h-8">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          </Link>
          <div className="flex items-center gap-1.5">
            <Ruler className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-sm font-medium">Z-Axis Calibration</span>
          </div>
          <span className="text-sm text-muted-foreground truncate">
            {runName} / {video?.filename ?? videoId}
          </span>
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-muted-foreground">Every</span>
            <select
              value={effectiveInterval}
              onChange={(e) => setFrameInterval(Number(e.target.value))}
              className="rounded border bg-background px-2 py-0.5 text-xs h-7"
            >
              {FRAME_INTERVALS.map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
            <span className="text-[11px] text-muted-foreground">frames</span>
            <span className="text-[11px] text-muted-foreground ml-1">
              ({filteredFrames.length} of {allFrames.length})
            </span>
          </div>
          <div className="flex-1" />
          {hasExistingZ && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
              Calibrated
            </span>
          )}
          <span className="text-xs text-muted-foreground">
            {calibrationPoints.length} point{calibrationPoints.length !== 1 ? 's' : ''}
          </span>
          <Button
            variant={isCurrentFrameAdded ? 'default' : 'outline'}
            size="sm"
            className="gap-1.5 h-8"
            onClick={addCurrentFrame}
            disabled={isCurrentFrameAdded || !currentFrame?.detections.length}
          >
            {isCurrentFrameAdded ? (
              <Check className="h-3.5 w-3.5" />
            ) : (
              <Plus className="h-3.5 w-3.5" />
            )}
            {isCurrentFrameAdded ? 'Added' : 'Add Frame'}
          </Button>
        </div>

        {/* Frame viewer */}
        <div className="flex-1 flex items-center justify-center min-h-0 p-2 relative">
          {currentFrame && (
            <div className="relative max-w-full max-h-full" style={{ aspectRatio: video ? `${video.width}/${video.height}` : undefined }}>
              <img
                key={currentFrame.frame_number}
                src={imageUrl}
                alt={`Frame ${currentFrame.frame_number}`}
                className="max-w-full max-h-[calc(100vh-14rem)] object-contain rounded"
                draggable={false}
              />
              {currentFrame.detections.map((det: Detection, i: number) => {
                const left = (det.box.x - det.box.width / 2) * 100
                const top = (det.box.y - det.box.height / 2) * 100
                const width = det.box.width * 100
                const height = det.box.height * 100
                const color = classColorMap[det.class_name] || '#FF6B6B'
                return (
                  <div
                    key={i}
                    className="absolute pointer-events-none"
                    style={{
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${width}%`,
                      height: `${height}%`,
                      border: `2px solid ${color}`,
                      borderRadius: 2,
                    }}
                  >
                    <span
                      className="absolute -top-5 left-0 text-[10px] font-medium px-1 rounded-sm whitespace-nowrap"
                      style={{ backgroundColor: color, color: '#000' }}
                    >
                      {det.class_name} {(det.confidence * 100).toFixed(0)}%
                      {det.z_mm != null && ` Z:${det.z_mm.toFixed(0)}mm`}
                    </span>
                  </div>
                )
              })}
              {isCurrentFrameAdded && (
                <div className="absolute top-2 right-2 bg-primary text-primary-foreground px-2 py-1 rounded text-xs font-medium flex items-center gap-1">
                  <Ruler className="h-3 w-3" />
                  Calibration Point
                </div>
              )}
            </div>
          )}
        </div>

        {/* Bottom filmstrip */}
        <div className="relative flex-shrink-0 bg-secondary border-t border-border px-4 py-2">
          <div className="flex items-center gap-2 mb-2">
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(0)}>
              <SkipBack className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(currentFrameIndex - 1)}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="text-xs font-mono min-w-[80px] text-center text-muted-foreground">
              {filteredFrames.length > 0 ? currentFrameIndex + 1 : 0} / {filteredFrames.length}
            </span>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(currentFrameIndex + 1)}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(filteredFrames.length - 1)}>
              <SkipForward className="h-3.5 w-3.5" />
            </Button>
            {currentFrame && (
              <span className="text-[11px] text-muted-foreground ml-2">
                Frame {currentFrame.frame_number} &middot; {currentFrame.timestamp.toFixed(2)}s &middot; {currentFrame.detections.length} detection{currentFrame.detections.length !== 1 ? 's' : ''}
              </span>
            )}
            <div className="flex-1" />
            <span className="text-[10px] text-muted-foreground hidden sm:block">
              &larr; &rarr; navigate &middot; Space add frame
            </span>
          </div>
          <div
            ref={thumbnailStripRef}
            className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin"
            style={{ maxHeight: 44 }}
          >
            {filteredFrames.map((frame, i) => {
              const isCalPoint = selectedFrameNumbers.has(frame.frame_number)
              const isCurrent = i === currentFrameIndex
              const hasDetections = frame.detections.length > 0
              return (
                <button
                  key={frame.frame_number}
                  data-index={i}
                  onClick={() => goToFrame(i)}
                  className={cn(
                    'relative flex-shrink-0 w-10 h-10 rounded border-2 transition-colors flex flex-col items-center justify-center',
                    isCalPoint
                      ? 'border-amber-500 bg-amber-500/20'
                      : isCurrent
                      ? 'border-primary bg-primary/10'
                      : hasDetections
                      ? 'border-green-500/50 bg-green-500/5 hover:border-green-500/80'
                      : 'border-border bg-muted/30 hover:border-primary/50',
                  )}
                  title={`Frame ${frame.frame_number} · ${frame.timestamp.toFixed(1)}s · ${frame.detections.length} detection${frame.detections.length !== 1 ? 's' : ''}`}
                >
                  {isCalPoint ? (
                    <Ruler className="h-3.5 w-3.5 text-amber-500" />
                  ) : (
                    <>
                      <span className="text-[9px] font-mono text-muted-foreground leading-none">
                        {frame.frame_number}
                      </span>
                      {hasDetections && (
                        <div className="flex items-center gap-0.5 mt-0.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                          <span className="text-[8px] text-green-500 leading-none">
                            {frame.detections.length}
                          </span>
                        </div>
                      )}
                    </>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* Right sidebar - calibration */}
      <div className="w-72 flex-shrink-0 bg-secondary border-l border-border flex flex-col overflow-hidden">
        {/* Existing calibration status */}
        {hasExistingZ && existingCal?.z_calibration && (
          <div className="flex-shrink-0 p-3 border-b border-border">
            <label className="text-[10px] font-medium text-muted-foreground mb-1.5 block uppercase tracking-wide">
              Current Model
            </label>
            <div className="text-xs space-y-1 p-2 bg-muted/50 rounded">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Formula</span>
                <span className="font-mono">
                  {existingCal.z_calibration.model?.type === 'k_over_s'
                    ? `Z = ${existingCal.z_calibration.model.k?.toFixed(0)} / s`
                    : `Z = ${existingCal.z_calibration.model?.a?.toFixed(0)} / s + ${existingCal.z_calibration.model?.b?.toFixed(0)}`}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Points</span>
                <span>{existingCal.z_calibration.labels.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Class</span>
                <span>{existingCal.z_calibration.class_name}</span>
              </div>
            </div>
          </div>
        )}

        {/* Calibration points list */}
        <div className="flex-1 overflow-y-auto p-3 min-h-0">
          <div className="flex items-center justify-between mb-2">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
              Calibration Points ({calibrationPoints.length})
            </label>
          </div>

          {calibrationPoints.length === 0 ? (
            <div className="text-xs text-muted-foreground py-6 text-center space-y-2">
              <Ruler className="h-8 w-8 mx-auto text-muted-foreground/40" />
              <p>No calibration points yet.</p>
              <p>Browse to a frame with detections and press <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Space</kbd> or click <strong>Add Frame</strong> to add it.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {calibrationPoints.map((point) => (
                <div
                  key={point.frame_number}
                  className={cn(
                    'p-2 rounded border text-xs transition-colors cursor-pointer',
                    currentFrame?.frame_number === point.frame_number
                      ? 'border-amber-500/60 bg-amber-500/10'
                      : 'border-border bg-muted/30 hover:border-border/80',
                  )}
                  onClick={() => navigateToPoint(point.frameIndex)}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-medium">Frame {point.frame_number}</span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        removePoint(point.frame_number)
                      }}
                      className="text-muted-foreground hover:text-destructive transition-colors p-0.5"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-muted-foreground whitespace-nowrap">Distance:</span>
                    <Input
                      type="number"
                      placeholder="mm"
                      value={point.z_mm}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => updateDistance(point.frame_number, e.target.value)}
                      className="h-6 text-xs flex-1 min-w-0"
                    />
                    <span className="text-muted-foreground">mm</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Info text */}
        <div className="flex-shrink-0 px-3 py-2 border-t border-border">
          <p className="text-[10px] text-muted-foreground leading-relaxed">
            {calibrationPoints.length === 0
              ? 'Select frames at known distances from the camera to build a depth model.'
              : calibrationPoints.length === 1 && validLabels.length === 1
              ? '1 point → Z = k/s model. Add more points for linear regression.'
              : `${validLabels.length} valid point${validLabels.length !== 1 ? 's' : ''} → linear regression model.`}
          </p>
        </div>

        {/* Actions */}
        <div className="flex-shrink-0 p-3 border-t border-border space-y-2">
          <Button
            size="sm"
            className="w-full gap-1.5"
            disabled={validLabels.length === 0 || hasIncompletePoints || calibrateMutation.isPending}
            onClick={() => calibrateMutation.mutate()}
          >
            {calibrateMutation.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Ruler className="h-3.5 w-3.5" />
            )}
            {calibrateMutation.isPending ? 'Calibrating...' : 'Calibrate & Estimate'}
          </Button>

        </div>
      </div>
    </div>
  )
}
