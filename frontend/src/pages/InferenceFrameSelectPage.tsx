import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  ArrowLeft,
  Loader2,
  Download,
  Check,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { useToast } from '@/components/ui/Toaster'
import { cn } from '@/lib/utils'
import type { Detection, InferenceResult } from '@/types'

const DETECTION_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
const FRAME_INTERVALS = [1, 2, 5, 10, 15, 30, 60] as const

function pickDefaultInterval(totalFrames: number): number {
  if (totalFrames <= 60) return 1
  if (totalFrames <= 150) return 5
  if (totalFrames <= 500) return 10
  if (totalFrames <= 1500) return 30
  return 60
}

export default function InferenceFrameSelectPage() {
  const { projectName, runName, videoId, inferenceId } = useParams<{
    projectName: string
    runName: string
    videoId: string
    inferenceId: string
  }>()
  const { toast } = useToast()

  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [selectedFrameIndices, setSelectedFrameIndices] = useState<Set<number>>(new Set())
  const lastClickedIndexRef = useRef<number | null>(null)
  const thumbnailStripRef = useRef<HTMLDivElement>(null)
  const [extracting, setExtracting] = useState(false)
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

  const allFrames: InferenceResult[] = useMemo(
    () => detailResult?.frames ?? [],
    [detailResult],
  )

  // Set default interval once we know the frame count
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

  // Reset navigation when interval changes
  useEffect(() => {
    setCurrentFrameIndex(0)
    setSelectedFrameIndices(new Set())
    lastClickedIndexRef.current = null
  }, [effectiveInterval])

  const currentFrame = filteredFrames[currentFrameIndex]

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

  const handleFilmstripClick = useCallback(
    (index: number, e: React.MouseEvent) => {
      const isMetaKey = e.metaKey || e.ctrlKey
      const isShift = e.shiftKey

      if (isShift && lastClickedIndexRef.current !== null) {
        const start = Math.min(lastClickedIndexRef.current, index)
        const end = Math.max(lastClickedIndexRef.current, index)
        setSelectedFrameIndices((prev) => {
          const next = new Set(prev)
          for (let i = start; i <= end; i++) next.add(i)
          return next
        })
      } else if (isMetaKey) {
        setSelectedFrameIndices((prev) => {
          const next = new Set(prev)
          if (next.has(index)) next.delete(index)
          else next.add(index)
          return next
        })
      }

      lastClickedIndexRef.current = index
      goToFrame(index)
    },
    [goToFrame],
  )

  const toggleCurrentFrame = useCallback(() => {
    setSelectedFrameIndices((prev) => {
      const next = new Set(prev)
      if (next.has(currentFrameIndex)) next.delete(currentFrameIndex)
      else next.add(currentFrameIndex)
      return next
    })
  }, [currentFrameIndex])

  const handleDownload = useCallback(async () => {
    if (!projectName || !runName || !videoId || !inferenceId || selectedFrameIndices.size === 0) return
    setExtracting(true)
    try {
      const frameNumbers = Array.from(selectedFrameIndices).map((i) => filteredFrames[i].frame_number)
      const blob = await api.inference.extractFrames(projectName, runName, videoId, inferenceId, frameNumbers)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${projectName}_${runName}_${inferenceId}_frames.zip`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      toast({
        title: 'Frames downloaded',
        description: `${selectedFrameIndices.size} frames exported as ZIP`,
        type: 'success',
      })
    } catch (error: any) {
      toast({ title: 'Download failed', description: error.message, type: 'error' })
    } finally {
      setExtracting(false)
    }
  }, [projectName, runName, videoId, inferenceId, selectedFrameIndices, filteredFrames, toast])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'Escape' && selectedFrameIndices.size > 0) {
        e.preventDefault()
        setSelectedFrameIndices(new Set())
        return
      }
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
        toggleCurrentFrame()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentFrameIndex, selectedFrameIndices.size, toggleCurrentFrame])

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

  const selectedCount = selectedFrameIndices.size

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
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            <span className="text-[11px] text-muted-foreground">frames</span>
            <span className="text-[11px] text-muted-foreground ml-1">
              ({filteredFrames.length} of {allFrames.length})
            </span>
          </div>
          <div className="flex-1" />
          <span className="text-xs text-muted-foreground">
            {selectedCount} selected
          </span>
          <Button
            variant={selectedFrameIndices.has(currentFrameIndex) ? 'default' : 'outline'}
            size="sm"
            className="gap-1.5 h-8"
            onClick={toggleCurrentFrame}
          >
            <Check className="h-3.5 w-3.5" />
            {selectedFrameIndices.has(currentFrameIndex) ? 'Selected' : 'Select Frame'}
          </Button>
          <Button
            size="sm"
            className="gap-1.5 h-8"
            disabled={selectedCount === 0 || extracting}
            onClick={handleDownload}
          >
            {extracting ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Download className="h-3.5 w-3.5" />
            )}
            Download ZIP
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
              {/* Detection bounding box overlays */}
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
                    </span>
                  </div>
                )
              })}
              {/* Selection badge */}
              {selectedFrameIndices.has(currentFrameIndex) && (
                <div className="absolute top-2 right-2 bg-primary text-primary-foreground px-2 py-1 rounded text-xs font-medium flex items-center gap-1">
                  <Check className="h-3 w-3" />
                  Selected
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
              &larr; &rarr; navigate &middot; Space toggle select &middot; Cmd-click multi-select &middot; Shift-click range
            </span>
          </div>
          {/* Multi-select floating bar */}
          {selectedFrameIndices.size > 0 && (
            <div className="absolute left-4 right-4 bottom-full mb-1 z-10 flex items-center gap-2 px-2 py-1.5 rounded-md bg-neutral-900/90 border border-blue-500/40 backdrop-blur-md shadow-lg">
              <span className="text-xs font-medium text-blue-400">
                {selectedFrameIndices.size} frame{selectedFrameIndices.size !== 1 ? 's' : ''} selected
              </span>
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-[11px] px-2 text-muted-foreground hover:text-foreground"
                onClick={() => {
                  const allIndices = new Set<number>()
                  filteredFrames.forEach((_, i) => allIndices.add(i))
                  setSelectedFrameIndices(allIndices)
                }}
              >
                Select all
              </Button>
              <Button
                size="sm"
                className="h-6 text-[11px] px-2 gap-1"
                disabled={extracting}
                onClick={handleDownload}
              >
                {extracting ? (
                  <Loader2 className="h-3 w-3 animate-spin" />
                ) : (
                  <Download className="h-3 w-3" />
                )}
                Download {selectedFrameIndices.size} frames
              </Button>
              <button
                onClick={() => setSelectedFrameIndices(new Set())}
                className="text-[11px] text-muted-foreground hover:text-foreground ml-1"
                title="Deselect all (Esc)"
              >
                ✕
              </button>
            </div>
          )}
          <div
            ref={thumbnailStripRef}
            className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin"
            style={{ maxHeight: 44 }}
          >
            {filteredFrames.map((frame, i) => {
              const isMultiSelected = selectedFrameIndices.has(i)
              const isCurrent = i === currentFrameIndex
              const hasDetections = frame.detections.length > 0
              return (
                <button
                  key={frame.frame_number}
                  data-index={i}
                  onClick={(e) => handleFilmstripClick(i, e)}
                  className={cn(
                    'relative flex-shrink-0 w-10 h-10 rounded border-2 transition-colors flex flex-col items-center justify-center',
                    isMultiSelected
                      ? 'border-blue-500 bg-blue-500/20'
                      : isCurrent
                      ? 'border-amber-400 bg-amber-400/10'
                      : hasDetections
                      ? 'border-green-500/50 bg-green-500/5 hover:border-green-500/80'
                      : 'border-border bg-muted/30 hover:border-primary/50',
                  )}
                  title={`Frame ${frame.frame_number} · ${frame.timestamp.toFixed(1)}s · ${frame.detections.length} detection${frame.detections.length !== 1 ? 's' : ''}`}
                >
                  {isMultiSelected ? (
                    <div className="w-4 h-4 rounded-full bg-blue-500 flex items-center justify-center">
                      <svg viewBox="0 0 12 12" className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="2,6 5,9 10,3" />
                      </svg>
                    </div>
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

      {/* Right sidebar - detection details */}
      <div className="w-64 flex-shrink-0 bg-secondary border-l border-border flex flex-col overflow-hidden">
        <div className="flex-shrink-0 p-3 border-b border-border">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Classes ({classNames.length})
          </label>
          {classNames.length > 0 ? (
            <div className="space-y-0.5 max-h-32 overflow-y-auto">
              {classNames.map((cls) => (
                <div key={cls} className="flex items-center gap-2 px-2 py-1 text-xs">
                  <span
                    className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: classColorMap[cls] }}
                  />
                  <span className="truncate">{cls}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">No detections</p>
          )}
        </div>
        <div className="flex-1 overflow-y-auto p-3 min-h-0">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Detections ({currentFrame?.detections.length ?? 0})
          </label>
          {!currentFrame?.detections.length ? (
            <p className="text-xs text-muted-foreground py-4 text-center">No detections in this frame</p>
          ) : (
            <div className="space-y-0.5">
              {currentFrame.detections.map((det: Detection, i: number) => (
                <div
                  key={i}
                  className="flex items-center gap-2 px-2 py-1.5 rounded text-xs bg-muted/30"
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: classColorMap[det.class_name] }}
                  />
                  <div className="flex-1 min-w-0">
                    <span className="truncate block">{det.class_name}</span>
                    <span className="text-[10px] text-muted-foreground">
                      {(det.confidence * 100).toFixed(1)}%
                      {det.track_id != null && ` · track ${det.track_id}`}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="flex-shrink-0 p-3 border-t border-border space-y-2">
          <div className="text-[10px] text-muted-foreground uppercase tracking-wide font-medium">
            Frame Info
          </div>
          {currentFrame && (
            <div className="text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Frame</span>
                <span className="font-mono">{currentFrame.frame_number}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Time</span>
                <span className="font-mono">{currentFrame.timestamp.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Detections</span>
                <span>{currentFrame.detections.length}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
