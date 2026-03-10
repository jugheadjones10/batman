import { useState, useRef, useEffect, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  Trash2,
  ArrowLeft,
  Loader2,
  Wand2,
  XCircle,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { useToast } from '@/components/ui/Toaster'
import { useStore } from '@/store/useStore'
import { cn } from '@/lib/utils'
import type { Annotation, BoundingBox, Frame } from '@/types'
import { AnnotationCanvas, type AnnotationCanvasHandle } from '@/components/AnnotationCanvas'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { Input } from '@/components/ui/Input'

const FRAME_INTERVALS = [1, 2, 5, 10, 15, 30] as const
type FilmstripMode = 'interval' | 'annotated'

export default function VideoAnnotatePage() {
  const { projectName, videoId } = useParams<{ projectName: string; videoId: string }>()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const selectedAnnotationId = useStore((s) => s.selectedAnnotationId)
  const setSelectedAnnotation = useStore((s) => s.setSelectedAnnotation)
  const selectedClassId = useStore((s) => s.selectedClassId)
  const setSelectedClassId = useStore((s) => s.setSelectedClassId)

  const containerRef = useRef<HTMLDivElement>(null)
  const thumbnailStripRef = useRef<HTMLDivElement>(null)
  const annotationCanvasRef = useRef<AnnotationCanvasHandle>(null)

  const [filmstripMode, setFilmstripMode] = useState<FilmstripMode>('interval')
  const [frameInterval, setFrameInterval] = useState(5)
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0)
  const [selectedFrameIndices, setSelectedFrameIndices] = useState<Set<number>>(new Set())
  const lastClickedIndexRef = useRef<number | null>(null)
  const [samModalOpen, setSamModalOpen] = useState(false)
  const [samClassDescriptions, setSamClassDescriptions] = useState<Record<string, string>>({})
  const [samConfidence, setSamConfidence] = useState(0.25)
  const [samSkipLabeled, setSamSkipLabeled] = useState(true)
  const [samFrameScope, setSamFrameScope] = useState<'all_visible' | 'selected' | 'unlabeled'>('all_visible')
  const [labelingJobId, setLabelingJobId] = useState<string | null>(null)
  const [isLabelingRunning, setIsLabelingRunning] = useState(false)
  const [labelingProgress, setLabelingProgress] = useState<{ progress: number; message: string } | null>(null)

  const { data: project } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  const { data: video } = useQuery({
    queryKey: ['video', projectName, videoId],
    queryFn: () => api.videos.get(projectName!, videoId!),
    enabled: !!projectName && !!videoId,
  })

  const { data: frames = [], isLoading: framesLoading } = useQuery({
    queryKey: ['video-frames', projectName, videoId],
    queryFn: () => api.videos.getFrames(projectName!, videoId!),
    enabled: !!projectName && !!videoId,
  })

  const extractFramesMutation = useMutation({
    mutationFn: () =>
      api.videos.extractFrames(projectName!, videoId!, {
        mode: 'frames',
        interval: frameInterval,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
      queryClient.invalidateQueries({ queryKey: ['video', projectName, videoId] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({ title: 'Frames extracted', type: 'success' })
    },
    onError: (e: Error) => {
      toast({ title: 'Extract failed', description: e.message, type: 'error' })
    },
  })

  const filteredFrames: Frame[] =
    filmstripMode === 'annotated'
      ? frames.filter((f) => (f.annotation_count ?? 0) > 0)
      : frames.filter((_, i) => i % frameInterval === 0)

  const currentFrame = filteredFrames[currentFrameIndex]
  const currentFrameId = currentFrame ? String(currentFrame.id) : null

  const { data: annotations } = useQuery({
    queryKey: ['annotations', projectName, currentFrameId],
    queryFn: () => api.annotations.listForFrame(projectName!, currentFrameId!),
    enabled: !!projectName && !!currentFrameId,
  })

  const createAnnotationMutation = useMutation({
    mutationFn: (data: { frame_id: number | string; class_label_id: number; box: BoundingBox }) =>
      api.annotations.create(projectName!, data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrameId] })
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
      annotationCanvasRef.current?.onAnnotationCreated(data)
    },
  })

  const updateAnnotationMutation = useMutation({
    mutationFn: ({ id, box }: { id: number; box: BoundingBox }) =>
      api.annotations.update(projectName!, id, { box }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrameId] })
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
    },
  })

  const deleteAnnotationMutation = useMutation({
    mutationFn: (id: number) => api.annotations.delete(projectName!, id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrameId] })
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
      setSelectedAnnotation(null)
    },
  })

  const clearFrameMutation = useMutation({
    mutationFn: (frameId: string) => api.annotations.clearFrame(projectName!, frameId),
    onSuccess: (data, frameId) => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, frameId] })
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
      queryClient.invalidateQueries({ queryKey: ['video', projectName, videoId] })
      setSelectedAnnotation(null)
      toast({ title: 'Frame cleared', description: `Removed ${data.deleted} annotation(s)`, type: 'success' })
    },
    onError: (e: Error) => {
      toast({ title: 'Clear failed', description: e.message, type: 'error' })
    },
  })

  const clearSelectedFramesMutation = useMutation({
    mutationFn: (frameIds: string[]) => api.annotations.clearFrames(projectName!, frameIds),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName] })
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
      queryClient.invalidateQueries({ queryKey: ['video', projectName, videoId] })
      setSelectedAnnotation(null)
      setSelectedFrameIndices(new Set())
      toast({
        title: 'Frames cleared',
        description: `Removed ${data.deleted} annotation(s) from ${data.frames_cleared} frame(s)`,
        type: 'success',
      })
    },
    onError: (e: Error) => {
      toast({ title: 'Clear failed', description: e.message, type: 'error' })
    },
  })

  const updateAnnotationClassMutation = useMutation({
    mutationFn: ({ id, class_label_id }: { id: number; class_label_id: number }) =>
      api.annotations.update(projectName!, id, { class_label_id }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrameId] })
    },
  })

  const updateClassMutateRef = useRef(updateAnnotationClassMutation.mutate)
  updateClassMutateRef.current = updateAnnotationClassMutation.mutate

  const handleSelectClass = useCallback((classIndex: number) => {
    setSelectedClassId(classIndex)
    const selId = useStore.getState().selectedAnnotationId
    if (selId !== null) {
      updateClassMutateRef.current({ id: selId, class_label_id: classIndex })
    }
  }, [setSelectedClassId])

  useEffect(() => {
    if (filteredFrames.length > 0 && currentFrameIndex >= filteredFrames.length) {
      setCurrentFrameIndex(Math.max(0, filteredFrames.length - 1))
    }
  }, [filteredFrames.length, currentFrameIndex])

  useEffect(() => {
    setSelectedFrameIndices(new Set())
    lastClickedIndexRef.current = null
  }, [filmstripMode, frameInterval])

  const handleCreateAnnotation = useCallback(
    (box: BoundingBox, classId: number) => {
      if (!currentFrame) return
      createAnnotationMutation.mutate({ frame_id: currentFrame.id, class_label_id: classId, box })
    },
    [currentFrame, createAnnotationMutation]
  )

  const handleUpdateAnnotation = useCallback(
    (id: number, box: BoundingBox) => updateAnnotationMutation.mutate({ id, box }),
    [updateAnnotationMutation]
  )

  const handleDeleteAnnotation = useCallback(
    (id: number) => deleteAnnotationMutation.mutate(id),
    [deleteAnnotationMutation]
  )

  const handleRestoreAnnotation = useCallback(
    async (annotation: Annotation): Promise<Annotation | void> => {
      if (!currentFrameId || !projectName) return
      const created = await api.annotations.create(projectName, {
        frame_id: currentFrameId,
        class_label_id: annotation.class_label_id,
        box: annotation.box,
      })
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrameId] })
      queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
      return created
    },
    [currentFrameId, projectName, queryClient, videoId]
  )

  const goToFrame = useCallback(
    (index: number) => {
      if (filteredFrames.length > 0 && index >= 0 && index < filteredFrames.length) {
        setCurrentFrameIndex(index)
        setSelectedAnnotation(null)
      }
    },
    [filteredFrames.length, setSelectedAnnotation]
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
        lastClickedIndexRef.current = index
      } else {
        setSelectedFrameIndices(new Set())
        lastClickedIndexRef.current = index
        goToFrame(index)
        return
      }

      lastClickedIndexRef.current = index
    },
    [goToFrame]
  )

  useEffect(() => {
    if (samModalOpen && project) {
      const desc: Record<string, string> = {}
      project.classes.forEach((c) => {
        desc[c] = project.class_descriptions?.[c] ?? c
      })
      setSamClassDescriptions(desc)
    }
  }, [samModalOpen, project])

  useEffect(() => {
    if (!labelingJobId || !projectName) return
    const t = setInterval(async () => {
      try {
        const status = await api.labeling.getLabelingStatus(projectName, labelingJobId)
        setLabelingProgress({ progress: status.progress, message: status.message })
        if (status.status === 'completed') {
          setLabelingJobId(null)
          setIsLabelingRunning(false)
          setLabelingProgress(null)
          setSamModalOpen(false)
          queryClient.invalidateQueries({ queryKey: ['annotations', projectName] })
          queryClient.invalidateQueries({ queryKey: ['video-frames', projectName, videoId] })
          queryClient.invalidateQueries({ queryKey: ['video', projectName, videoId] })
          toast({ title: 'Auto-label complete', description: status.message, type: 'success' })
        } else if (status.status === 'failed') {
          setLabelingJobId(null)
          setIsLabelingRunning(false)
          setLabelingProgress(null)
          toast({ title: 'Auto-label failed', description: status.message, type: 'error' })
        }
      } catch {
        // ignore
      }
    }, 1000)
    return () => clearInterval(t)
  }, [labelingJobId, projectName, queryClient, toast, videoId])

  const runSamAutoLabel = useCallback(async () => {
    if (!projectName || !videoId || !project) return
    await api.projects.updateClassDescriptions(projectName, samClassDescriptions)
    queryClient.invalidateQueries({ queryKey: ['project', projectName] })
    let frameIds: (number | string)[]
    if (samFrameScope === 'all_visible') {
      frameIds = filteredFrames.map((f) => f.id)
    } else if (samFrameScope === 'selected' && currentFrame) {
      frameIds = [currentFrame.id]
    } else {
      frameIds = filteredFrames.filter((f) => (f.annotation_count ?? 0) === 0).map((f) => f.id)
    }
    if (frameIds.length === 0) {
      toast({ title: 'No frames', description: 'No frames to label for the selected scope.', type: 'error' })
      return
    }
    const { job_id } = await api.labeling.autoLabel(projectName, {
      video_ids: [videoId],
      frame_ids: frameIds,
      class_descriptions: samClassDescriptions,
      confidence: samConfidence,
      skip_labeled_frames: samSkipLabeled,
    })
    setLabelingJobId(job_id)
    setIsLabelingRunning(true)
    setLabelingProgress({ progress: 0, message: 'Starting...' })
  }, [
    projectName,
    videoId,
    project,
    filteredFrames,
    currentFrame,
    samFrameScope,
    samClassDescriptions,
    samConfidence,
    samSkipLabeled,
    queryClient,
    toast,
  ])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'Escape') {
        if (selectedFrameIndices.size > 0) {
          e.preventDefault()
          setSelectedFrameIndices(new Set())
          return
        }
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goToFrameRef.current(currentFrameIndex - 1)
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        goToFrameRef.current(currentFrameIndex + 1)
      }
      const num = parseInt(e.key)
      const classCount = project?.classes?.length ?? 0
      if (!isNaN(num) && num >= 1 && num <= classCount) {
        e.preventDefault()
        handleSelectClass(num - 1)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentFrameIndex, project?.classes?.length, handleSelectClass, selectedFrameIndices.size])

  useEffect(() => {
    const strip = thumbnailStripRef.current
    const thumb = strip?.querySelector(`[data-index="${currentFrameIndex}"]`)
    if (strip && thumb) {
      thumb.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' })
    }
  }, [currentFrameIndex])

  const annotatedCount = frames.filter((f) => (f.annotation_count ?? 0) > 0).length
  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

  if (!projectName || !videoId) return null

  if (framesLoading || !video) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (frames.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <p className="text-muted-foreground">No frames extracted yet.</p>
        <p className="text-sm text-muted-foreground">
          Extract one frame every <strong>{frameInterval}</strong> frames to annotate.
        </p>
        <div className="flex items-center gap-2">
          <span className="text-sm">Interval:</span>
          <select
            value={frameInterval}
            onChange={(e) => setFrameInterval(Number(e.target.value))}
            className="rounded border bg-background px-2 py-1 text-sm"
          >
            {FRAME_INTERVALS.map((n) => (
              <option key={n} value={n}>
                Every {n} frame{n > 1 ? 's' : ''}
              </option>
            ))}
          </select>
        </div>
        <Button
          onClick={() => extractFramesMutation.mutate()}
          disabled={extractFramesMutation.isPending}
        >
          {extractFramesMutation.isPending ? 'Extracting…' : 'Extract frames'}
        </Button>
        <Link to={`/projects/${projectName}`}>
          <Button variant="ghost">Back to project</Button>
        </Link>
      </div>
    )
  }

  const imageUrl = currentFrame
    ? api.videos.frameImageUrl(projectName, videoId, String(currentFrame.id))
    : ''

  return (
    <div className="h-[calc(100vh-4rem)] flex overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0 bg-neutral-900">
        <div className="flex-shrink-0 px-4 py-2 border-b border-border flex items-center gap-3 flex-wrap">
          <Link to={`/projects/${projectName}`}>
            <Button variant="ghost" size="sm" className="gap-1 h-8">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          </Link>
          <span className="text-sm text-muted-foreground truncate">
            {video.filename}
          </span>
          <div className="flex items-center gap-1">
            <span className="text-[11px] text-muted-foreground">View:</span>
            <button
              onClick={() => setFilmstripMode('interval')}
              className={cn(
                'px-2 py-0.5 rounded text-[11px] font-medium',
                filmstripMode === 'interval' ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'
              )}
            >
              Interval
            </button>
            <button
              onClick={() => setFilmstripMode('annotated')}
              className={cn(
                'px-2 py-0.5 rounded text-[11px] font-medium',
                filmstripMode === 'annotated' ? 'bg-primary text-primary-foreground' : 'bg-muted hover:bg-muted/80'
              )}
            >
              Annotated
            </button>
          </div>
          {filmstripMode === 'interval' && (
            <div className="flex items-center gap-1">
              <span className="text-[11px] text-muted-foreground">Every</span>
              <select
                value={frameInterval}
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
            </div>
          )}
          <div className="flex-1" />
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5 h-8"
            onClick={() => setSamModalOpen(true)}
            disabled={!project?.classes?.length}
          >
            <Wand2 className="h-3.5 w-3.5" />
            Auto-label with SAM3
          </Button>
        </div>

        <div ref={containerRef} className="flex-1 flex flex-col min-h-0 p-2">
          {currentFrame ? (
            <AnnotationCanvas
              ref={annotationCanvasRef}
              imageUrl={imageUrl}
              imageWidth={video.width}
              imageHeight={video.height}
              annotations={annotations ?? []}
              selectedAnnotationId={selectedAnnotationId}
              selectedClassId={selectedClassId}
              classes={project?.classes ?? []}
              onCreateAnnotation={handleCreateAnnotation}
              onUpdateAnnotation={handleUpdateAnnotation}
              onDeleteAnnotation={handleDeleteAnnotation}
              onSelectAnnotation={setSelectedAnnotation}
              onRestoreAnnotation={handleRestoreAnnotation}
            />
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              {filmstripMode === 'annotated' ? (
                <p>No annotated frames yet. Switch to Interval to label frames.</p>
              ) : (
                <p>No frames in this range.</p>
              )}
            </div>
          )}
        </div>

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
            <span className="text-[11px] text-muted-foreground ml-2">
              {annotatedCount} / {frames.length} frames annotated
            </span>
            <div className="flex-1" />
            <span className="text-[10px] text-muted-foreground hidden sm:block">
              ← → navigate • 1-9 class • Del delete • ⌘Z undo • ⌘Y / ⌘⇧Z redo • ⌘-click / Shift-click multi-select
            </span>
          </div>
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
                variant="ghost"
                size="sm"
                className="h-6 text-[11px] px-2 text-destructive hover:text-destructive"
                disabled={clearSelectedFramesMutation.isPending}
                onClick={() => {
                  const frameIds = Array.from(selectedFrameIndices)
                    .filter((i) => filteredFrames[i])
                    .map((i) => String(filteredFrames[i].id))
                  const annotatedCount = Array.from(selectedFrameIndices).filter(
                    (i) => filteredFrames[i] && (filteredFrames[i].annotation_count ?? 0) > 0
                  ).length
                  if (frameIds.length === 0) return
                  const msg = annotatedCount > 0
                    ? `Clear annotations from ${frameIds.length} frame(s)? (${annotatedCount} have annotations)`
                    : `${frameIds.length} frame(s) selected have no annotations.`
                  if (annotatedCount === 0) {
                    toast({ title: 'Nothing to clear', description: 'Selected frames have no annotations.', type: 'error' })
                    return
                  }
                  if (confirm(msg)) {
                    clearSelectedFramesMutation.mutate(frameIds)
                  }
                }}
              >
                {clearSelectedFramesMutation.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <XCircle className="h-3 w-3 mr-1" />
                )}
                Clear annotations
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
            style={{ maxHeight: 56 }}
          >
            {filteredFrames.map((frame, i) => {
              const isSelected = selectedFrameIndices.has(i)
              return (
                <button
                  key={frame.id}
                  data-index={i}
                  onClick={(e) => handleFilmstripClick(i, e)}
                  className={cn(
                    'relative flex-shrink-0 w-12 h-12 rounded overflow-hidden border-2 transition-colors',
                    isSelected
                      ? 'border-blue-500 ring-1 ring-blue-500'
                      : i === currentFrameIndex
                      ? 'border-amber-400 ring-1 ring-amber-400'
                      : (frame.annotation_count ?? 0) > 0
                      ? 'border-green-500/50 hover:border-green-500/80'
                      : 'border-border hover:border-primary/50'
                  )}
                >
                  <img
                    src={api.videos.frameImageUrl(projectName, videoId, String(frame.id))}
                    alt={`Frame ${frame.frame_number}`}
                    className={cn('w-full h-full object-cover', isSelected && 'brightness-75')}
                  />
                  {isSelected && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-4 h-4 rounded-full bg-blue-500 flex items-center justify-center">
                        <svg viewBox="0 0 12 12" className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="2,6 5,9 10,3" />
                        </svg>
                      </div>
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="w-64 flex-shrink-0 bg-secondary border-l border-border flex flex-col overflow-hidden">
        <div className="flex-shrink-0 p-3 border-b border-border">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Label ({project?.classes.length ?? 0})
          </label>
          {project?.classes && project.classes.length > 0 ? (
            <div className="space-y-0.5 max-h-32 overflow-y-auto">
              {project.classes.map((cls, i) => (
                <button
                  key={cls}
                  onClick={() => handleSelectClass(i)}
                  className={cn(
                    'w-full flex items-center gap-2 px-2 py-1 rounded text-sm text-left',
                    selectedClassId === i ? 'bg-primary/20 text-primary' : 'hover:bg-muted'
                  )}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: colors[i % colors.length] }}
                  />
                  <span className="truncate text-xs">{cls}</span>
                  <span className="text-[10px] text-muted-foreground">{i + 1}</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Add classes in the project page.</p>
          )}
        </div>
        <div className="flex-1 overflow-y-auto p-3 min-h-0">
          <div className="flex items-center justify-between mb-1">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
              Regions ({annotations?.length ?? 0})
            </label>
            {annotations && annotations.length > 0 && currentFrameId && (
              <button
                onClick={() => {
                  if (confirm(`Remove all ${annotations.length} annotation(s) from this frame?`)) {
                    clearFrameMutation.mutate(currentFrameId)
                  }
                }}
                disabled={clearFrameMutation.isPending}
                className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-destructive transition-colors"
                title="Clear all annotations on this frame"
              >
                <XCircle className="h-3 w-3" />
                Clear frame
              </button>
            )}
          </div>
          {!annotations?.length ? (
            <p className="text-xs text-muted-foreground py-4 text-center">Click and drag to draw</p>
          ) : (
            <div className="space-y-0.5">
              {annotations.map((ann) => (
                <div
                  key={ann.id}
                  onClick={() => setSelectedAnnotation(ann.id)}
                  className={cn(
                    'flex items-center gap-2 px-2 py-1 rounded text-xs cursor-pointer group',
                    selectedAnnotationId === ann.id ? 'bg-primary/20' : 'hover:bg-muted'
                  )}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: ann.class_color }}
                  />
                  <span className="flex-1 truncate">{ann.class_name}</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteAnnotationMutation.mutate(ann.id)
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-destructive"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* SAM3 Auto-label modal */}
      {samModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => !isLabelingRunning && setSamModalOpen(false)}
        >
          <Card
            className="w-full max-w-lg max-h-[90vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <CardHeader className="py-4">
              <CardTitle className="text-lg flex items-center gap-2">
                <Wand2 className="h-5 w-5" />
                Auto-label with SAM3
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Run SAM3 on selected frames. Choose scope, edit class descriptions, then Run.
              </p>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto py-0 space-y-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">Frames to label</label>
                <div className="flex flex-col gap-1">
                  {(['all_visible', 'selected', 'unlabeled'] as const).map((scope) => (
                    <label key={scope} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="samFrameScope"
                        checked={samFrameScope === scope}
                        onChange={() => setSamFrameScope(scope)}
                        disabled={isLabelingRunning}
                      />
                      <span className="text-sm">
                        {scope === 'all_visible' &&
                          `All visible frames (${filteredFrames.length})`}
                        {scope === 'selected' &&
                          `Current frame only`}
                        {scope === 'unlabeled' &&
                          `Unlabeled frames only (${filteredFrames.filter((f) => (f.annotation_count ?? 0) === 0).length})`}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">Class descriptions (SAM prompts)</label>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {project?.classes.map((cls) => (
                    <div key={cls}>
                      <span className="text-xs text-muted-foreground block mb-0.5">{cls}</span>
                      <Input
                        value={samClassDescriptions[cls] ?? cls}
                        onChange={(e) =>
                          setSamClassDescriptions((prev) => ({ ...prev, [cls]: e.target.value }))
                        }
                        placeholder={cls}
                        className="h-8 text-sm"
                        disabled={isLabelingRunning}
                      />
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1 block">
                  Confidence threshold: {samConfidence.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={samConfidence}
                  onChange={(e) => setSamConfidence(Number(e.target.value))}
                  disabled={isLabelingRunning}
                  className="w-full"
                />
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={samSkipLabeled}
                  onChange={(e) => setSamSkipLabeled(e.target.checked)}
                  disabled={isLabelingRunning}
                />
                <span className="text-sm">Skip already-labeled frames</span>
              </label>
              {labelingProgress && (
                <div className="space-y-1">
                  <Progress value={labelingProgress.progress * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground">{labelingProgress.message}</p>
                </div>
              )}
            </CardContent>
            <CardFooter className="py-4 gap-2">
              <Button
                variant="outline"
                onClick={() => !isLabelingRunning && setSamModalOpen(false)}
                disabled={isLabelingRunning}
              >
                Cancel
              </Button>
              <Button onClick={runSamAutoLabel} disabled={isLabelingRunning}>
                {isLabelingRunning ? 'Running...' : 'Run'}
              </Button>
            </CardFooter>
          </Card>
        </div>
      )}
    </div>
  )
}
