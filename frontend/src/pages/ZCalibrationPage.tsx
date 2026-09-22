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
import type {
  Detection,
  InferenceResult,
  ZCalibrationLabel,
} from '@/types'

const DETECTION_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
const FRAME_INTERVALS = [1, 2, 5, 10, 15, 30, 60] as const

// ISO-standard dry-box container lengths. The spreader telescopes to match the
// container it's picking, so both share the same ℓ for a given calibration.
const ISO_LENGTHS = [
  { mm: 6058, label: '20 ft (6058 mm)' },
  { mm: 12192, label: '40 ft (12192 mm)' },
  { mm: 13716, label: '45 ft (13716 mm)' },
] as const

function pickDefaultInterval(totalFrames: number): number {
  if (totalFrames <= 60) return 1
  if (totalFrames <= 150) return 5
  if (totalFrames <= 500) return 10
  if (totalFrames <= 1500) return 30
  return 60
}

interface CalibrationPoint {
  detection_index?: number
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

  const [referenceClass, setReferenceClass] = useState('')
  const [lengthMm, setLengthMm] = useState<number | null>(null)
  const [customTargets, setTargets] = useState<string[] | null>(null)
  const [roundFeatureDiameterMm, setRoundFeatureDiameterMm] = useState('')
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = useState(true)
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

  // Load existing calibration settings when data arrives
  const [didLoadExisting, setDidLoadExisting] = useState(false)
  useEffect(() => {
    if (!existingCal?.z_calibration || allFrames.length === 0 || didLoadExisting) return
    const cal = existingCal.z_calibration
    // Whole-object labels measure a different reference. Keep the saved model
    // active, but require fresh round-feature points before replacing it.
    if (cal.measurement_source !== 'round_feature_equivalent_length') {
      if (cal.length_mm != null) setLengthMm(cal.length_mm)
      setDidLoadExisting(true)
      return
    }

    if (cal.labels?.length) {
      const points: CalibrationPoint[] = []
      for (const label of cal.labels) {
        const idx = allFrames.findIndex((f) => f.frame_number === label.frame_number)
        if (idx !== -1) {
          points.push({ frame_number: label.frame_number, z_mm: String(label.z_mm), detection_index: label.detection_index })
        }
      }
      if (points.length > 0) setCalibrationPoints(points)
    }

    if (cal.reference_class) setReferenceClass(cal.reference_class)
    if (cal.length_mm != null) setLengthMm(cal.length_mm)
    setTargets([...(cal.targets ?? [cal.reference_class])])
    if (cal.round_feature_diameter_mm != null) {
      setRoundFeatureDiameterMm(String(cal.round_feature_diameter_mm))
    }
    setAdvancedSettingsOpen(!(
      Number.isFinite(cal.length_mm) && (cal.length_mm ?? 0) > 0 &&
      Number.isFinite(cal.round_feature_diameter_mm) && (cal.round_feature_diameter_mm ?? 0) > 0
    ))
    setDidLoadExisting(true)
  }, [existingCal, allFrames, didLoadExisting])

  const currentFrame = filteredFrames[currentFrameIndex]
  const hasExistingZ = existingCal?.z_calibration?.model != null

  const classNames = useMemo(() => {
    const names = new Set<string>()
    for (const f of allFrames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [allFrames])

  // Saved settings take precedence even when both queries finish together.
  useEffect(() => {
    const savedRoundReference = existingCal?.z_calibration?.measurement_source === 'round_feature_equivalent_length'
      && existingCal.z_calibration.reference_class
    if (referenceClass || savedRoundReference || classNames.length === 0) return
    const roundClass = classNames.find((name) => /^round$/i.test(name))
      ?? classNames.find((name) => /round/i.test(name))
    if (roundClass) setReferenceClass(roundClass)
  }, [classNames, referenceClass, existingCal])

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
    if (!currentFrame.detections.some((det) => det.class_name === referenceClass)) {
      toast({ title: 'Reference not visible', description: `Choose a frame with a visible ${referenceClass}.`, type: 'error' })
      return
    }
    setCalibrationPoints((prev) => [
      ...prev,
      { frame_number: currentFrame.frame_number, z_mm: '' },
    ])
  }, [currentFrame, referenceClass, selectedFrameNumbers, toast])

  const removePoint = useCallback((frameNumber: number) => {
    setCalibrationPoints((prev) => prev.filter((p) => p.frame_number !== frameNumber))
  }, [])

  const updateDistance = useCallback((frameNumber: number, value: string) => {
    setCalibrationPoints((prev) =>
      prev.map((p) => (p.frame_number === frameNumber ? { ...p, z_mm: value } : p)),
    )
  }, [])

  const navigateToPoint = useCallback(
    (frameNumber: number) => {
      const visibleIndex = filteredFrames.findIndex((frame) => frame.frame_number === frameNumber)
      if (visibleIndex >= 0) {
        goToFrame(visibleIndex)
      } else {
        setFrameInterval(1)
        setCurrentFrameIndex(allFrames.findIndex((frame) => frame.frame_number === frameNumber))
      }
    },
    [allFrames, filteredFrames, goToFrame],
  )

  const validLabels = useMemo((): ZCalibrationLabel[] => {
    return calibrationPoints.flatMap((point) => {
      const z = Number(point.z_mm)
      const frame = allFrames.find((item) => item.frame_number === point.frame_number)
      if (!Number.isFinite(z) || z <= 0 || !frame) return []
      const savedIndex = point.detection_index
      const index = savedIndex != null && frame.detections[savedIndex]?.class_name === referenceClass
        ? savedIndex
        : frame.detections.findIndex((det) => det.class_name === referenceClass)
      return index < 0 ? [] : [{ frame_number: point.frame_number, z_mm: z, detection_index: index }]
    })
  }, [allFrames, calibrationPoints, referenceClass])

  // Scale transfer relates the round feature to whole spreader/container boxes.
  const targets = customTargets ?? classNames.filter(
    (name) => name === referenceClass || /spreader|container/i.test(name),
  )

  const roundFeatureDiameter = Number(roundFeatureDiameterMm)
  const equivalentSizeRatio =
    lengthMm != null &&
    lengthMm > 0 &&
    Number.isFinite(roundFeatureDiameter) &&
    roundFeatureDiameter > 0
      ? lengthMm / roundFeatureDiameter
      : null

  const addTarget = useCallback(() => {
    const available = classNames.find((c) => !targets.includes(c))
    if (available) setTargets([...targets, available])
  }, [classNames, targets])

  const calibrateMutation = useMutation({
    mutationFn: async () => {
      const refClass = referenceClass
      if (!refClass) throw new Error('No reference class selected')
      const cleaned = targets.filter((t) => t.trim().length > 0)
      const allTargets = Array.from(new Set([refClass, ...cleaned]))
      if (lengthMm == null || lengthMm <= 0) {
        throw new Error('Round feature calibration requires a container/spreader length')
      }
      if (!Number.isFinite(roundFeatureDiameter) || roundFeatureDiameter <= 0) {
        throw new Error('Round feature calibration requires a positive round feature diameter')
      }

      await api.inference.saveZCalibration(
        projectName!, runName!, videoId!, inferenceId!, validLabels, refClass,
        {
          lengthMm: lengthMm ?? null,
          targetClasses: allTargets,
          measurementSource: 'round_feature_equivalent_length',
          featureToSpreaderZOffsetMm: existingCal?.z_calibration?.measurement_source === 'round_feature_equivalent_length'
            ? existingCal.z_calibration.feature_to_spreader_z_offset_mm ?? 0
            : 0,
          roundFeatureDiameterMm: roundFeatureDiameter,
        },
      )
      return api.inference.applyZEstimation(projectName!, runName!, videoId!, inferenceId!)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['z-calibration', projectName, runName, videoId, inferenceId] })
      queryClient.invalidateQueries({ queryKey: ['inference-result-detail', projectName, runName, videoId, inferenceId] })
      toast({ title: 'Distance calibration applied', description: 'Return to inference to view the spreader-to-target gap.', type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Distance calibration failed', description: error.message, type: 'error' })
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
  const inferenceBackUrl = `/projects/${encodeURIComponent(projectName)}/inference?run=${encodeURIComponent(runName)}&video=${encodeURIComponent(videoId)}&inferenceId=${encodeURIComponent(inferenceId)}`

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
        <Link to={inferenceBackUrl}>
          <Button variant="ghost">Back to inference</Button>
        </Link>
      </div>
    )
  }

  const imageUrl = currentFrame
    ? api.videos.frameUrl(projectName, videoId, currentFrame.frame_number)
    : ''

  const isCurrentFrameAdded = currentFrame ? selectedFrameNumbers.has(currentFrame.frame_number) : false
  const hasIncompletePoints = validLabels.length !== calibrationPoints.length
  const needsRoundDimensions = equivalentSizeRatio == null
  return (
    <div className="h-[calc(100vh-4rem)] flex overflow-hidden">
      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0 bg-neutral-900">
        {/* Top bar */}
        <div className="flex-shrink-0 px-4 py-2 border-b border-border flex items-center gap-3 flex-wrap">
          <Link to={inferenceBackUrl}>
            <Button variant="ghost" size="sm" className="gap-1 h-8">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          </Link>
          <div className="flex items-center gap-1.5">
            <Ruler className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="text-sm font-medium">Distance Calibration</span>
          </div>
          <span className="text-sm text-muted-foreground truncate">
            {runName} / {video?.filename ?? videoId}
          </span>
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-muted-foreground">Every</span>
            <select
              value={effectiveInterval}
              onChange={(e) => { setFrameInterval(Number(e.target.value)); setCurrentFrameIndex(0) }}
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
            disabled={isCurrentFrameAdded || !currentFrame?.detections.some((det) => det.class_name === referenceClass)}
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
                const isRoundReference = det.class_name === referenceClass
                return (
                  <div
                    key={i}
                    className="absolute pointer-events-none"
                    style={{
                      left: `${left}%`,
                      top: `${top}%`,
                      width: `${width}%`,
                      height: `${height}%`,
                      border: `${isRoundReference ? 3 : 2}px solid ${color}`,
                      borderRadius: 2,
                      opacity: isRoundReference ? 1 : 0.35,
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
        {hasExistingZ && existingCal?.z_calibration?.model && (
          <details className="flex-shrink-0 px-3 py-2 border-b border-border">
            <summary className="text-xs cursor-pointer">Saved calibration details</summary>
            <div className="text-[11px] p-2 bg-muted/50 rounded space-y-0.5">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model</span>
                <span className="font-mono">
                  {existingCal.z_calibration.model.type === 'k_over_s'
                    ? `Z = ${existingCal.z_calibration.model.k?.toFixed(0)}/s`
                    : `Z = ${existingCal.z_calibration.model.m?.toFixed(0)}/s + ${existingCal.z_calibration.model.c?.toFixed(0)}`}
                </span>
              </div>
              {existingCal.z_calibration.length_mm != null && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">ℓ</span>
                  <span className="font-mono">{existingCal.z_calibration.length_mm} mm</span>
                </div>
              )}
              {existingCal.z_calibration.measurement_source === 'round_feature_equivalent_length' && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Round Ø</span>
                  <span className="font-mono">
                    {existingCal.z_calibration.round_feature_diameter_mm ?? '?'} mm
                  </span>
                </div>
              )}
              {existingCal.z_calibration.targets?.length ? (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Targets</span>
                  <span className="font-mono truncate max-w-[180px]" title={existingCal.z_calibration.targets.join(', ')}>
                    {existingCal.z_calibration.targets.length}
                  </span>
                </div>
              ) : null}
            </div>
          </details>
        )}

        {/* Scrollable config area */}
        <div className="flex-1 overflow-y-auto min-h-0">

          <p className="p-3 text-xs text-muted-foreground border-b border-border">
            Calibrate once to estimate the vertical gap from the empty spreader or the bottom of its load to the target container.
          </p>
          {existingCal?.z_calibration && existingCal.z_calibration.measurement_source !== 'round_feature_equivalent_length' && (
            <p className="p-3 text-xs text-muted-foreground border-b border-border">
              The saved calibration uses the whole object. Add new round-feature distance points to replace it.
              The saved calibration stays active until you apply the new one.
            </p>
          )}
          {/* Reference object */}
          <div className="p-3 border-b border-border space-y-2.5">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide block">
              1. Round feature class
            </label>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              The round feature provides spreader distance. Matching boxes are emphasized in the viewer.
            </p>
            <select
              aria-label="Round feature class"
              value={referenceClass}
              onChange={(e) => setReferenceClass(e.target.value)}
              className="w-full rounded border bg-background px-2 py-1 text-xs h-7"
            >
              <option value="">Select the detected round feature class</option>
              {classNames.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            {!referenceClass && (
              <p className="text-[10px] text-muted-foreground">
                No round class was selected automatically. Choose its class if it has a different name;
                otherwise run inference with a model trained to detect the round feature.
              </p>
            )}
          </div>

          {/* Known camera distances */}
          <div className="p-3 space-y-2.5">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide block">
              2. Known distances ({calibrationPoints.length})
            </label>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              {referenceClass
                ? <>Frames where <strong>{referenceClass}</strong> is visible at a known spreader distance from the camera.</>
                : 'Frames where the round feature is visible at a known distance from the camera.'}
            </p>

            <p className="text-[10px] text-muted-foreground leading-relaxed">
              One known distance is the minimum. Add a second at a different height to account for bias.
              Enter camera-to-object distance, not the gap to the target container.
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
                    onClick={() => navigateToPoint(point.frame_number)}
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
                      <span className="text-muted-foreground whitespace-nowrap">Camera distance:</span>
                      <Input
                        type="number"
                        min="0.1"
                        step="any"
                        aria-label={`Camera distance for frame ${point.frame_number} in millimetres`}
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
          <details
            className="border-t border-border"
            open={advancedSettingsOpen}
            onToggle={(e) => setAdvancedSettingsOpen(e.currentTarget.open)}
          >
            <summary className="p-3 text-xs font-medium cursor-pointer">
              Advanced settings{needsRoundDimensions ? ' — dimensions required' : ''}
            </summary>
            <p className="px-3 pb-2 text-[10px] text-muted-foreground">
              Set the physical dimensions used to relate the round feature to container size.
              Saved settings are kept when you recalibrate.
            </p>
          {/* Section 2: Container length (ℓ) */}
          <div className="p-3 border-b border-border space-y-2.5">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide block">
              Container/spreader length
            </label>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Required with the feature diameter to relate round-feature size to container size.
            </p>
            <select
              aria-label="Container/spreader length"
              value={lengthMm ?? ''}
              onChange={(e) => setLengthMm(e.target.value ? Number(e.target.value) : null)}
              className="w-full rounded border bg-background px-2 py-1 text-xs h-7"
            >
              <option value="">Select container length</option>
              {ISO_LENGTHS.map((l) => (
                <option key={l.mm} value={l.mm}>{l.label}</option>
              ))}
            </select>
          </div>

          {/* Round feature dimensions */}
            <div className="p-3 border-b border-border space-y-2.5">
              <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide block">
                Round Feature Diameter
              </label>
              <Input
                type="number"
                min="0"
                step="0.1"
                aria-label="Round feature diameter in millimetres"
                placeholder="Diameter (mm)"
                value={roundFeatureDiameterMm}
                onChange={(e) => setRoundFeatureDiameterMm(e.target.value)}
                className="h-7 text-xs"
              />
              <p className="text-[10px] text-muted-foreground leading-relaxed">
                {equivalentSizeRatio != null
                  ? `Equivalent spreader-size ratio: ${equivalentSizeRatio.toFixed(3)}x`
                  : 'Enter diameter and length to preview the equivalent size ratio.'}
              </p>
            </div>

          {/* Section 4: Estimation Targets */}
          <div className="p-3 border-b border-border space-y-2.5">
            <div className="flex items-center justify-between">
              <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
                Classes sharing this calibration
              </label>
              <button
                onClick={addTarget}
                disabled={targets.length >= classNames.length}
                className="text-[10px] text-primary hover:underline flex items-center gap-0.5 disabled:opacity-40 disabled:no-underline"
              >
                <Plus className="h-2.5 w-2.5" /> Add
              </button>
            </div>
            <p className="text-[10px] text-muted-foreground leading-relaxed">
              Classes to estimate distance for. All inherit the same fit — one model, broadcast by class name.
            </p>

            {targets.length === 0 ? (
              <div className="text-[11px] text-muted-foreground py-3 text-center bg-muted/30 rounded border border-dashed border-border">
                The reference object is always included.
              </div>
            ) : (
              <div className="space-y-1.5">
                {targets.map((tgt, i) => (
                  <div key={i} className="flex items-center gap-1.5">
                    <select
                      value={tgt}
                      onChange={(e) => setTargets(targets.map((t, j) => j === i ? e.target.value : t))}
                      className="flex-1 rounded border bg-background px-2 py-1 text-xs h-7 min-w-0"
                    >
                      <option value="">Select class...</option>
                      {classNames.map((name) => (
                        <option key={name} value={name}>{name}</option>
                      ))}
                    </select>
                    <button
                      onClick={() => setTargets(targets.filter((_, j) => j !== i))}
                      className="text-muted-foreground hover:text-destructive transition-colors p-0.5 flex-shrink-0"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <p className="text-[10px] text-muted-foreground italic">
              The round feature is converted to equivalent spreader length; container targets still use their whole bbox length.
            </p>
          </div>

          </details>
        </div>

        {/* Status text */}
        <div className="flex-shrink-0 px-3 py-2 border-t border-border">
          <p className="text-[10px] text-muted-foreground leading-relaxed">
            {needsRoundDimensions
              ? 'Enter the length and round feature diameter in Advanced settings.'
              : hasIncompletePoints
              ? 'Each point needs a positive distance and a visible reference object.'
              : validLabels.length === 0
              ? 'Add a frame at a known camera distance to get started.'
              : `${validLabels.length} distance point${validLabels.length === 1 ? '' : 's'} ready. Applies to ${Array.from(new Set([referenceClass, ...targets])).join(', ')}.`}

          </p>
        </div>

        {/* Actions */}
        <div className="flex-shrink-0 p-3 border-t border-border">
          <Button
            size="sm"
            className="w-full gap-1.5"
            disabled={validLabels.length === 0 || hasIncompletePoints || needsRoundDimensions || calibrateMutation.isPending}
            onClick={() => calibrateMutation.mutate()}
          >
            {calibrateMutation.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Ruler className="h-3.5 w-3.5" />
            )}
            {calibrateMutation.isPending ? 'Calibrating...' : 'Apply calibration'}
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
                <h4 className="text-foreground font-medium mb-1">Minimum setup</h4>
                <p>Choose the detected round feature, add a frame, and enter its known distance
                  from the camera in millimetres. The detected box supplies its pixel size automatically.
                  A second frame at a different height allows the fit to account for a constant distance offset.</p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">When dimensions are needed</h4>
                <p>Enter the container length and round feature diameter in Advanced. These relate the small
                  feature to whole-container boxes, so both use the same distance scale.</p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">From camera distance to clearance</h4>
                <p>Tracking identifies the load state and target. Empty clearance is target-top distance
                  minus spreader distance. Loaded clearance also subtracts the container height,
                  currently assumed to be 2,591 mm. This is a vertical estimate for the overhead camera view.</p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">Visibility matters</h4>
                <p>Use fully visible reference boxes and keep camera zoom and object size consistent.
                  Calibration cannot recover a hidden target by itself; tracking may infer its plane from
                  touchdown. Those targets are marked as inferred in the schematic.</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
