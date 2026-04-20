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
  Info,
  X,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'
import { cn } from '@/lib/utils'
import type { Detection, InferenceResult, ZCalibrationLabel, ZCalibrationTarget } from '@/types'

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

  const [sizeMetric, setSizeMetric] = useState<'h_px' | 'w_px'>('h_px')
  const [referenceClassName, setReferenceClassName] = useState('')
  const [referenceRealWidth, setReferenceRealWidth] = useState('')
  const [targets, setTargets] = useState<{ class_name: string; real_width_mm: string }[]>([])
  const [showInfo, setShowInfo] = useState(false)

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

  // Load existing calibration settings when data arrives
  const [didLoadExisting, setDidLoadExisting] = useState(false)
  useEffect(() => {
    if (!existingCal?.z_calibration || allFrames.length === 0 || didLoadExisting) return
    const cal = existingCal.z_calibration

    if (cal.labels?.length) {
      const points: CalibrationPoint[] = []
      for (const label of cal.labels) {
        const idx = filteredFrames.findIndex((f) => f.frame_number === label.frame_number)
        if (idx !== -1) {
          points.push({ frameIndex: idx, frame_number: label.frame_number, z_mm: String(label.z_mm) })
        }
      }
      if (points.length > 0) setCalibrationPoints(points)
    }

    if (cal.size_metric === 'w_px') setSizeMetric('w_px')
    if (cal.class_name) setReferenceClassName(cal.class_name)
    if (cal.reference_real_width_mm) setReferenceRealWidth(String(cal.reference_real_width_mm))
    if (cal.targets?.length) {
      setTargets(cal.targets.map((t) => ({ class_name: t.class_name, real_width_mm: String(t.real_width_mm) })))
    }
    setDidLoadExisting(true)
  }, [existingCal, allFrames.length, filteredFrames, didLoadExisting])

  const currentFrame = filteredFrames[currentFrameIndex]
  const hasExistingZ = existingCal?.z_calibration?.model != null

  const classNames = useMemo(() => {
    const names = new Set<string>()
    for (const f of allFrames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [allFrames])

  // Auto-select reference class when classes are available and none is set
  useEffect(() => {
    if (!referenceClassName && classNames.length > 0) {
      setReferenceClassName(classNames[0])
    }
  }, [classNames, referenceClassName])

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

  const refWidth = parseFloat(referenceRealWidth)

  // Auto-seed the reference class as a target so it always gets distance estimates
  useEffect(() => {
    if (referenceClassName && !isNaN(refWidth) && refWidth > 0 && targets.length === 0 && !didLoadExisting) {
      setTargets([{ class_name: referenceClassName, real_width_mm: String(refWidth) }])
    }
  }, [referenceClassName, refWidth, targets.length, didLoadExisting])

  const hasMultiTarget = !isNaN(refWidth) && refWidth > 0 && targets.length > 0

  const calibrateMutation = useMutation({
    mutationFn: async () => {
      const opts: {
        sizeMetric?: string
        referenceRealWidthMm?: number | null
        targets?: ZCalibrationTarget[] | null
      } = { sizeMetric }

      if (hasMultiTarget) {
        opts.referenceRealWidthMm = refWidth
        opts.targets = targets
          .filter((t) => t.class_name.trim() && !isNaN(parseFloat(t.real_width_mm)) && parseFloat(t.real_width_mm) > 0)
          .map((t) => ({ class_name: t.class_name.trim(), real_width_mm: parseFloat(t.real_width_mm) }))
      }

      await api.inference.saveZCalibration(
        projectName!, runName!, videoId!, inferenceId!, validLabels,
        referenceClassName || classNames[0] || 'crane hook', opts,
      )
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
      <div className="w-80 flex-shrink-0 bg-secondary border-l border-border flex flex-col overflow-hidden">
        {/* Header with info button */}
        <div className="flex-shrink-0 p-3 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Ruler className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-xs font-medium">Distance Calibration</span>
          </div>
          <div className="flex items-center gap-1.5">
            {hasExistingZ && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
                Active
              </span>
            )}
            <button
              onClick={() => setShowInfo(true)}
              className="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded"
              title="How does this work?"
            >
              <Info className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Existing model summary (collapsed) */}
        {hasExistingZ && existingCal?.z_calibration && (
          <div className="flex-shrink-0 px-3 py-2 border-b border-border">
            <div className="text-[11px] p-2 bg-muted/50 rounded space-y-0.5">
              {existingCal.z_calibration.model?.type === 'multi_target' ? (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model</span>
                  <span className="font-mono">
                    f={existingCal.z_calibration.model.focal_length_px?.toFixed(0)}px
                    &middot; {existingCal.z_calibration.model.targets?.length ?? 0} target{(existingCal.z_calibration.model.targets?.length ?? 0) !== 1 ? 's' : ''}
                  </span>
                </div>
              ) : (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Model</span>
                  <span className="font-mono">
                    {existingCal.z_calibration.model?.type === 'k_over_s'
                      ? `Z = ${existingCal.z_calibration.model.k?.toFixed(0)}/s`
                      : `Z = ${existingCal.z_calibration.model?.a?.toFixed(0)}/s + ${existingCal.z_calibration.model?.b?.toFixed(0)}`}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Scrollable config area */}
        <div className="flex-1 overflow-y-auto min-h-0">

          {/* Section 1: Reference Object */}
          <div className="p-3 border-b border-border space-y-2.5">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide block">
              1. Reference Object
            </label>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              The object you will calibrate with. You must know its real-world size.
            </p>

            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-muted-foreground w-12 flex-shrink-0">Class</span>
                <select
                  value={referenceClassName}
                  onChange={(e) => setReferenceClassName(e.target.value)}
                  className="flex-1 rounded border bg-background px-2 py-1 text-xs h-7 min-w-0"
                >
                  {classNames.map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-muted-foreground w-12 flex-shrink-0">Width</span>
                <div className="flex-1 flex items-center gap-1.5">
                  <Input
                    type="number"
                    placeholder="e.g. 2500"
                    value={referenceRealWidth}
                    onChange={(e) => setReferenceRealWidth(e.target.value)}
                    className="h-7 text-xs flex-1 min-w-0"
                  />
                  <span className="text-[11px] text-muted-foreground">mm</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] text-muted-foreground w-12 flex-shrink-0">Metric</span>
                <div className="flex gap-1 flex-1">
                  {(['h_px', 'w_px'] as const).map((m) => (
                    <button
                      key={m}
                      onClick={() => setSizeMetric(m)}
                      className={cn(
                        'flex-1 text-[11px] py-1 rounded border transition-colors',
                        sizeMetric === m
                          ? 'border-primary bg-primary/10 text-foreground font-medium'
                          : 'border-border bg-muted/30 text-muted-foreground hover:border-primary/50',
                      )}
                    >
                      {m === 'h_px' ? 'Height' : 'Width'}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Section 2: Estimation Targets */}
          <div className="p-3 border-b border-border space-y-2.5">
            <div className="flex items-center justify-between">
              <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                2. Estimation Targets
              </label>
              <button
                onClick={() => setTargets((prev) => [...prev, { class_name: '', real_width_mm: '2438' }])}
                className="text-[10px] text-primary hover:underline flex items-center gap-0.5"
              >
                <Plus className="h-2.5 w-2.5" /> Add
              </button>
            </div>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Classes to estimate distance for. Each needs a known real-world width.
            </p>

            {targets.length === 0 ? (
              <div className="text-[11px] text-muted-foreground py-3 text-center bg-muted/30 rounded border border-dashed border-border">
                No targets yet. Add the classes you want to measure.
              </div>
            ) : (
              <div className="space-y-1.5">
                <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground px-0.5">
                  <span className="flex-1">Class name</span>
                  <span className="w-[72px] text-right">Width (mm)</span>
                  <span className="w-5" />
                </div>
                {targets.map((tgt, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    {classNames.length > 0 ? (
                      <select
                        value={tgt.class_name}
                        onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? { ...t, class_name: e.target.value } : t))}
                        className="flex-1 rounded border bg-background px-2 py-1 text-xs h-7 min-w-0"
                      >
                        <option value="">Select class...</option>
                        {classNames.map((name) => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                    ) : (
                      <Input
                        type="text"
                        placeholder="class name"
                        value={tgt.class_name}
                        onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? { ...t, class_name: e.target.value } : t))}
                        className="h-7 text-xs flex-1 min-w-0"
                      />
                    )}
                    <Input
                      type="number"
                      placeholder="mm"
                      value={tgt.real_width_mm}
                      onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? { ...t, real_width_mm: e.target.value } : t))}
                      className="h-7 text-xs w-[72px]"
                    />
                    <button
                      onClick={() => setTargets((prev) => prev.filter((_, j) => j !== i))}
                      className="text-muted-foreground hover:text-destructive transition-colors p-0.5 flex-shrink-0"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <p className="text-[10px] text-muted-foreground italic">
              ISO container width: 2438mm (same for 20ft / 40ft / 45ft).
            </p>
          </div>

          {/* Section 3: Calibration Points */}
          <div className="p-3 space-y-2.5">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide block">
              3. Calibration Points ({calibrationPoints.length})
            </label>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              {referenceClassName
                ? <>Frames where <strong>{referenceClassName}</strong> is at a known distance from the camera.</>
                : 'Frames where the reference object is at a known distance from the camera.'}
            </p>

            {calibrationPoints.length === 0 ? (
              <div className="text-xs text-muted-foreground py-5 text-center space-y-2">
                <Ruler className="h-7 w-7 mx-auto text-muted-foreground/30" />
                <p>No calibration points yet.</p>
                <p className="text-[10px]">
                  Browse to a frame and press <kbd className="px-1 py-0.5 bg-muted rounded text-[10px]">Space</kbd> or click <strong>Add Frame</strong>.
                </p>
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
        </div>

        {/* Status text */}
        <div className="flex-shrink-0 px-3 py-2 border-t border-border">
          <p className="text-[10px] text-muted-foreground leading-relaxed">
            {!hasMultiTarget
              ? calibrationPoints.length === 0
                ? 'Enter a reference width and add targets to enable multi-target mode, or add calibration points for single-class mode.'
                : calibrationPoints.length === 1 && validLabels.length === 1
                ? '1 point \u2192 Z = k/s model. Add more points for higher accuracy.'
                : `${validLabels.length} valid point${validLabels.length !== 1 ? 's' : ''} \u2192 linear regression model.`
              : `Multi-target: calibrate on ${referenceClassName || 'reference'}, estimate distance to ${targets.filter((t) => t.class_name.trim()).map((t) => t.class_name).join(', ') || '...'}.`}
          </p>
        </div>

        {/* Actions */}
        <div className="flex-shrink-0 p-3 border-t border-border">
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

      {/* Info popup overlay */}
      {showInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowInfo(false)}>
          <div
            className="bg-secondary border border-border rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-sm font-semibold">How Distance Calibration Works</h3>
              <button onClick={() => setShowInfo(false)} className="text-muted-foreground hover:text-foreground transition-colors">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto text-xs text-muted-foreground space-y-4 leading-relaxed">
              <div>
                <h4 className="text-foreground font-medium mb-1">The idea</h4>
                <p>
                  Objects farther from the camera appear smaller. If you know an object's
                  real-world size and can measure its apparent size in pixels, the pinhole
                  camera model gives you the distance: <code className="px-1 py-0.5 bg-muted rounded text-[11px]">D = k / s</code> where
                  <code className="px-1 py-0.5 bg-muted rounded text-[11px]">s</code> is the bounding box size in pixels.
                </p>
              </div>

              <div>
                <h4 className="text-foreground font-medium mb-1">Single-class mode</h4>
                <p>
                  Pick frames where the object is at a known distance. The system fits
                  the constant <code className="px-1 py-0.5 bg-muted rounded text-[11px]">k</code> (1 point) or a linear
                  model <code className="px-1 py-0.5 bg-muted rounded text-[11px]">D = a/s + b</code> (2+ points) directly.
                  No reference width or targets needed.
                </p>
              </div>

              <div>
                <h4 className="text-foreground font-medium mb-1">Multi-target mode</h4>
                <p>
                  When you provide a <strong>reference object width</strong> and <strong>estimation targets</strong>,
                  the system works differently:
                </p>
                <ol className="list-decimal list-inside space-y-1 mt-2 ml-1">
                  <li>You calibrate on the <strong>reference object</strong> (e.g., a spreader) at known distance(s).</li>
                  <li>The system derives the camera's effective focal length: <code className="px-1 py-0.5 bg-muted rounded text-[11px]">f = D &times; s / W<sub>ref</sub></code></li>
                  <li>For each target class with known real width <code className="px-1 py-0.5 bg-muted rounded text-[11px]">W<sub>target</sub></code>, it
                    computes <code className="px-1 py-0.5 bg-muted rounded text-[11px]">k = f &times; W<sub>target</sub></code></li>
                  <li>Every frame, every detection of every target class gets a distance estimate.</li>
                </ol>
                <p className="mt-2">
                  This lets you calibrate on <em>one</em> object and measure distance to <em>multiple</em> different objects.
                </p>
              </div>

              <div>
                <h4 className="text-foreground font-medium mb-1">Size metric</h4>
                <p>
                  <strong>Height (h_px)</strong> uses the bounding box height — best for side-view cameras
                  (e.g., crane hook swinging vertically).
                  <strong> Width (w_px)</strong> uses the bounding box width — best for top-down cameras
                  (e.g., viewing containers from above, where width is the constant 2.438m ISO dimension).
                </p>
              </div>

              <div>
                <h4 className="text-foreground font-medium mb-1">Tips</h4>
                <ul className="list-disc list-inside space-y-1 ml-1">
                  <li>Use 2+ calibration points spanning the full operating range for best accuracy.</li>
                  <li>ISO shipping containers are always <strong>2438mm wide</strong>, regardless of length (20ft / 40ft / 45ft).</li>
                  <li>The reference object should be included as a target too if you want its distance.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
