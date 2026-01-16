import { useState, useRef, useEffect, useCallback } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  Wand2,
  Loader2,
  Trash2,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { useToast } from '@/components/ui/Toaster'
import { useStore } from '@/store/useStore'
import { cn, formatDuration } from '@/lib/utils'
import type { Annotation, BoundingBox } from '@/types'

type DragMode = 'none' | 'draw' | 'move' | 'resize'
type ResizeHandle = 'nw' | 'ne' | 'sw' | 'se' | null

export default function AnnotatePage() {
  const { projectName } = useParams<{ projectName: string }>()
  const [searchParams] = useSearchParams()
  const videoIdParam = searchParams.get('video')
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const {
    currentVideo,
    setCurrentVideo,
    currentFrameIndex,
    setCurrentFrameIndex,
    selectedAnnotationId,
    setSelectedAnnotation,
    selectedClassId,
    setSelectedClassId,
  } = useStore()

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })

  // Drag state
  const [dragMode, setDragMode] = useState<DragMode>('none')
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [drawingBox, setDrawingBox] = useState<BoundingBox | null>(null)
  const [resizeHandle, setResizeHandle] = useState<ResizeHandle>(null)
  const [originalBox, setOriginalBox] = useState<BoundingBox | null>(null)
  const [cursor, setCursor] = useState('crosshair')

  // Fetch project data
  const { data: project } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  // Fetch videos
  const { data: videos } = useQuery({
    queryKey: ['videos', projectName],
    queryFn: () => api.videos.list(projectName!),
    enabled: !!projectName,
  })

  // Fetch frames for current video
  const { data: frames } = useQuery({
    queryKey: ['frames', projectName, currentVideo?.id],
    queryFn: () => api.videos.getFrames(projectName!, currentVideo!.id),
    enabled: !!projectName && !!currentVideo,
  })

  const currentFrame = frames?.[currentFrameIndex]

  // Fetch annotations for current frame
  const { data: annotations } = useQuery({
    queryKey: ['annotations', projectName, currentFrame?.id],
    queryFn: () => api.annotations.listForFrame(projectName!, currentFrame!.id),
    enabled: !!projectName && !!currentFrame,
  })

  // Create annotation mutation
  const createAnnotationMutation = useMutation({
    mutationFn: (data: { frame_id: number; class_label_id: number; box: BoundingBox }) =>
      api.annotations.create(projectName!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrame?.id] })
    },
  })

  // Update annotation mutation
  const updateAnnotationMutation = useMutation({
    mutationFn: ({ id, box }: { id: number; box: BoundingBox }) =>
      api.annotations.update(projectName!, id, { box }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrame?.id] })
    },
  })

  // Delete annotation mutation
  const deleteAnnotationMutation = useMutation({
    mutationFn: (id: number) => api.annotations.delete(projectName!, id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentFrame?.id] })
      setSelectedAnnotation(null)
    },
  })

  // Auto-label mutation
  const autoLabelMutation = useMutation({
    mutationFn: () =>
      api.labeling.autoLabel(projectName!, {
        video_ids: currentVideo ? [currentVideo.id] : undefined,
      }),
    onSuccess: (data) => {
      toast({ title: 'Auto-labeling started', description: `Job ID: ${data.job_id}` })
    },
    onError: (error: Error) => {
      toast({ title: 'Auto-labeling failed', description: error.message, type: 'error' })
    },
  })

  // Set video from URL param or first video as current if none selected
  useEffect(() => {
    if (videos?.length && !currentVideo) {
      if (videoIdParam) {
        const video = videos.find((v) => v.id === Number(videoIdParam))
        if (video) {
          setCurrentVideo(video)
          return
        }
      }
      setCurrentVideo(videos[0])
    }
  }, [videos, currentVideo, setCurrentVideo, videoIdParam])

  // Update canvas size
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current && currentVideo) {
        const container = containerRef.current
        const aspectRatio = currentVideo.width / currentVideo.height
        const maxWidth = container.clientWidth - 16
        const maxHeight = container.clientHeight - 16

        let width = maxWidth
        let height = width / aspectRatio

        if (height > maxHeight) {
          height = maxHeight
          width = height * aspectRatio
        }

        setCanvasSize({ width: Math.floor(width), height: Math.floor(height) })
      }
    }

    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [currentVideo])

  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

  // Draw annotations
  const drawAnnotations = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || !currentVideo) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw existing annotations
    annotations?.forEach((ann) => {
      const isSelected = ann.id === selectedAnnotationId
      const color = ann.class_color || colors[0]
      drawBox(ctx, ann.box, color, isSelected, ann.class_name)
    })

    // Draw current drawing box
    if (drawingBox) {
      const color = colors[selectedClassId % colors.length]
      drawBox(ctx, drawingBox, color, true)
    }
  }, [annotations, selectedAnnotationId, drawingBox, currentVideo, selectedClassId, canvasSize])

  useEffect(() => {
    drawAnnotations()
  }, [drawAnnotations])

  const drawBox = (
    ctx: CanvasRenderingContext2D,
    box: BoundingBox,
    color: string,
    isSelected: boolean,
    label?: string
  ) => {
    const { width, height } = canvasSize
    if (width === 0 || height === 0) return

    // Convert normalized coords to pixel coords
    const x = (box.x - box.width / 2) * width
    const y = (box.y - box.height / 2) * height
    const w = box.width * width
    const h = box.height * height

    // Draw fill
    ctx.fillStyle = color + (isSelected ? '30' : '15')
    ctx.fillRect(x, y, w, h)

    // Draw border
    ctx.strokeStyle = color
    ctx.lineWidth = isSelected ? 2 : 1.5
    ctx.strokeRect(x, y, w, h)

    // Draw label
    if (label) {
      ctx.font = '11px system-ui, sans-serif'
      const metrics = ctx.measureText(label)
      const padding = 4
      const labelHeight = 16

      ctx.fillStyle = color
      ctx.fillRect(x, y - labelHeight, metrics.width + padding * 2, labelHeight)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, x + padding, y - 4)
    }

    // Draw corner handles for selected
    if (isSelected) {
      const handleSize = 8
      ctx.fillStyle = '#fff'
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ;[
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h],
      ].forEach(([hx, hy]) => {
        ctx.fillRect(hx - handleSize / 2, hy - handleSize / 2, handleSize, handleSize)
        ctx.strokeRect(hx - handleSize / 2, hy - handleSize / 2, handleSize, handleSize)
      })
    }
  }

  // Get normalized coordinates from mouse event
  const getNormalizedCoords = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect()
    if (!rect) return null
    return {
      x: (e.clientX - rect.left) / canvasSize.width,
      y: (e.clientY - rect.top) / canvasSize.height,
    }
  }

  // Check if point is on a resize handle
  const getResizeHandle = (x: number, y: number, box: BoundingBox): ResizeHandle => {
    const handleSize = 12 / canvasSize.width // Handle size in normalized coords

    const left = box.x - box.width / 2
    const right = box.x + box.width / 2
    const top = box.y - box.height / 2
    const bottom = box.y + box.height / 2

    if (Math.abs(x - left) < handleSize && Math.abs(y - top) < handleSize) return 'nw'
    if (Math.abs(x - right) < handleSize && Math.abs(y - top) < handleSize) return 'ne'
    if (Math.abs(x - left) < handleSize && Math.abs(y - bottom) < handleSize) return 'sw'
    if (Math.abs(x - right) < handleSize && Math.abs(y - bottom) < handleSize) return 'se'

    return null
  }

  // Check if point is inside a box
  const isInsideBox = (x: number, y: number, box: BoundingBox): boolean => {
    const left = box.x - box.width / 2
    const right = box.x + box.width / 2
    const top = box.y - box.height / 2
    const bottom = box.y + box.height / 2
    return x >= left && x <= right && y >= top && y <= bottom
  }

  // Get annotation at point
  const getAnnotationAtPoint = (x: number, y: number): Annotation | null => {
    if (!annotations) return null
    for (let i = annotations.length - 1; i >= 0; i--) {
      if (isInsideBox(x, y, annotations[i].box)) {
        return annotations[i]
      }
    }
    return null
  }

  // Update cursor based on mouse position
  const updateCursor = (x: number, y: number) => {
    // Check selected annotation first for resize handles
    if (selectedAnnotationId) {
      const selectedAnn = annotations?.find((a) => a.id === selectedAnnotationId)
      if (selectedAnn) {
        const handle = getResizeHandle(x, y, selectedAnn.box)
        if (handle === 'nw' || handle === 'se') {
          setCursor('nwse-resize')
          return
        }
        if (handle === 'ne' || handle === 'sw') {
          setCursor('nesw-resize')
          return
        }
        if (isInsideBox(x, y, selectedAnn.box)) {
          setCursor('move')
          return
        }
      }
    }

    // Check if hovering over any annotation
    const hovered = getAnnotationAtPoint(x, y)
    if (hovered) {
      setCursor('pointer')
      return
    }

    setCursor('crosshair')
  }

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    const coords = getNormalizedCoords(e)
    if (!coords) return

    const { x, y } = coords

    // Check if clicking on resize handle of selected annotation
    if (selectedAnnotationId) {
      const selectedAnn = annotations?.find((a) => a.id === selectedAnnotationId)
      if (selectedAnn) {
        const handle = getResizeHandle(x, y, selectedAnn.box)
        if (handle) {
          setDragMode('resize')
          setResizeHandle(handle)
          setDragStart({ x, y })
          setOriginalBox({ ...selectedAnn.box })
          return
        }

        // Check if clicking inside selected annotation to move it
        if (isInsideBox(x, y, selectedAnn.box)) {
          setDragMode('move')
          setDragStart({ x, y })
          setOriginalBox({ ...selectedAnn.box })
          return
        }
      }
    }

    // Check if clicking on any annotation
    const clickedAnn = getAnnotationAtPoint(x, y)
    if (clickedAnn) {
      setSelectedAnnotation(clickedAnn.id)
      // Start move immediately
      setDragMode('move')
      setDragStart({ x, y })
      setOriginalBox({ ...clickedAnn.box })
      return
    }

    // Start drawing new box
    setSelectedAnnotation(null)
    setDragMode('draw')
    setDragStart({ x, y })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    const coords = getNormalizedCoords(e)
    if (!coords) return

    const { x, y } = coords

    // Update cursor when not dragging
    if (dragMode === 'none') {
      updateCursor(x, y)
      return
    }

    if (!dragStart) return

    if (dragMode === 'draw') {
      const box: BoundingBox = {
        x: (dragStart.x + x) / 2,
        y: (dragStart.y + y) / 2,
        width: Math.abs(x - dragStart.x),
        height: Math.abs(y - dragStart.y),
      }
      setDrawingBox(box)
    }

    if (dragMode === 'move' && originalBox && selectedAnnotationId) {
      const dx = x - dragStart.x
      const dy = y - dragStart.y
      const newBox: BoundingBox = {
        x: Math.max(originalBox.width / 2, Math.min(1 - originalBox.width / 2, originalBox.x + dx)),
        y: Math.max(originalBox.height / 2, Math.min(1 - originalBox.height / 2, originalBox.y + dy)),
        width: originalBox.width,
        height: originalBox.height,
      }
      // Update local state for immediate feedback
      setDrawingBox(null)
      // Update the annotation in the cache for live preview
      queryClient.setQueryData(
        ['annotations', projectName, currentFrame?.id],
        (old: Annotation[] | undefined) =>
          old?.map((a) => (a.id === selectedAnnotationId ? { ...a, box: newBox } : a))
      )
    }

    if (dragMode === 'resize' && originalBox && selectedAnnotationId && resizeHandle) {
      let left = originalBox.x - originalBox.width / 2
      let right = originalBox.x + originalBox.width / 2
      let top = originalBox.y - originalBox.height / 2
      let bottom = originalBox.y + originalBox.height / 2

      // Adjust based on handle
      if (resizeHandle.includes('w')) left = Math.min(x, right - 0.01)
      if (resizeHandle.includes('e')) right = Math.max(x, left + 0.01)
      if (resizeHandle.includes('n')) top = Math.min(y, bottom - 0.01)
      if (resizeHandle.includes('s')) bottom = Math.max(y, top + 0.01)

      // Clamp to canvas
      left = Math.max(0, left)
      right = Math.min(1, right)
      top = Math.max(0, top)
      bottom = Math.min(1, bottom)

      const newBox: BoundingBox = {
        x: (left + right) / 2,
        y: (top + bottom) / 2,
        width: right - left,
        height: bottom - top,
      }

      queryClient.setQueryData(
        ['annotations', projectName, currentFrame?.id],
        (old: Annotation[] | undefined) =>
          old?.map((a) => (a.id === selectedAnnotationId ? { ...a, box: newBox } : a))
      )
    }
  }

  const handleMouseUp = () => {
    if (dragMode === 'draw' && drawingBox && currentFrame) {
      if (drawingBox.width > 0.01 && drawingBox.height > 0.01) {
        createAnnotationMutation.mutate({
          frame_id: currentFrame.id,
          class_label_id: selectedClassId,
          box: drawingBox,
        })
      }
      setDrawingBox(null)
    }

    if ((dragMode === 'move' || dragMode === 'resize') && selectedAnnotationId) {
      // Get the current box from cache and save it
      const currentAnnotations = queryClient.getQueryData<Annotation[]>([
        'annotations',
        projectName,
        currentFrame?.id,
      ])
      const ann = currentAnnotations?.find((a) => a.id === selectedAnnotationId)
      if (ann && originalBox) {
        // Only update if box actually changed
        if (
          ann.box.x !== originalBox.x ||
          ann.box.y !== originalBox.y ||
          ann.box.width !== originalBox.width ||
          ann.box.height !== originalBox.height
        ) {
          updateAnnotationMutation.mutate({ id: selectedAnnotationId, box: ann.box })
        }
      }
    }

    setDragMode('none')
    setDragStart(null)
    setOriginalBox(null)
    setResizeHandle(null)
  }

  // Navigation
  const goToFrame = (index: number) => {
    if (frames && index >= 0 && index < frames.length) {
      setCurrentFrameIndex(index)
      setSelectedAnnotation(null)
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goToFrame(currentFrameIndex - 1)
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        goToFrame(currentFrameIndex + 1)
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotationId) {
        e.preventDefault()
        deleteAnnotationMutation.mutate(selectedAnnotationId)
      }
      if (e.key === 'Escape') {
        setSelectedAnnotation(null)
      }
      const num = parseInt(e.key)
      if (!isNaN(num) && num >= 1 && num <= (project?.classes.length || 0)) {
        setSelectedClassId(num - 1)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentFrameIndex, selectedAnnotationId, project?.classes.length])

  return (
    <div className="h-[calc(100vh-4rem)] flex overflow-hidden">
      {/* Main annotation area */}
      <div className="flex-1 flex flex-col min-w-0 bg-neutral-900">
        {/* Canvas area */}
        <div
          ref={containerRef}
          className="flex-1 flex items-center justify-center p-2 overflow-hidden min-h-0"
        >
          {currentFrame ? (
            <div
              className="relative flex-shrink-0"
              style={{ width: canvasSize.width, height: canvasSize.height }}
            >
              <img
                src={api.videos.frameUrl(projectName!, currentVideo!.id, currentFrame.frame_number)}
                alt={`Frame ${currentFrame.frame_number}`}
                className="absolute inset-0 w-full h-full object-contain"
                draggable={false}
              />
              <canvas
                ref={canvasRef}
                width={canvasSize.width}
                height={canvasSize.height}
                className="absolute inset-0"
                style={{ cursor }}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
            </div>
          ) : (
            <div className="text-center text-muted-foreground">
              <p>No frames to annotate</p>
              <p className="text-sm mt-1">Extract frames from a video first</p>
            </div>
          )}
        </div>

        {/* Timeline */}
        <div className="flex-shrink-0 bg-secondary border-t border-border px-4 py-2">
          <div className="flex items-center gap-2 mb-1.5">
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(0)}>
              <SkipBack className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(currentFrameIndex - 1)}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="text-xs font-mono min-w-[70px] text-center text-muted-foreground">
              {currentFrameIndex + 1} / {frames?.length || 0}
            </span>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame(currentFrameIndex + 1)}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToFrame((frames?.length || 1) - 1)}>
              <SkipForward className="h-3.5 w-3.5" />
            </Button>

            {currentFrame && (
              <span className="text-xs text-muted-foreground ml-1">
                {formatDuration(currentFrame.timestamp)}
              </span>
            )}

            <div className="flex-1" />

            <span className="text-[10px] text-muted-foreground hidden sm:block">
              ← → navigate • Del delete • 1-9 class
            </span>
          </div>

          <div className="relative h-5 bg-muted rounded overflow-hidden">
            {frames && frames.length > 0 && frames.map((frame, i) => (
              <div
                key={frame.id}
                className={cn(
                  'absolute h-full w-1 -ml-0.5 cursor-pointer transition-colors',
                  i === currentFrameIndex
                    ? 'bg-amber-400'
                    : 'bg-amber-400/30 hover:bg-amber-400/60'
                )}
                style={{
                  left: `${(i / Math.max(frames.length - 1, 1)) * 100}%`,
                }}
                onClick={() => goToFrame(i)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 bg-secondary border-l border-border flex flex-col overflow-hidden">
        {/* Video selector */}
        <div className="flex-shrink-0 p-3 border-b border-border">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Video
          </label>
          <select
            value={currentVideo?.id || ''}
            onChange={(e) => {
              const video = videos?.find((v) => v.id === Number(e.target.value))
              if (video) {
                setCurrentVideo(video)
                setCurrentFrameIndex(0)
              }
            }}
            className="w-full h-8 px-2 rounded border border-border bg-background text-sm"
          >
            {videos?.map((video) => (
              <option key={video.id} value={video.id}>
                {video.filename}
              </option>
            ))}
          </select>
        </div>

        {/* Classes */}
        <div className="flex-shrink-0 p-3 border-b border-border">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Label ({project?.classes.length || 0})
          </label>
          {project?.classes.length === 0 ? (
            <p className="text-xs text-muted-foreground py-2">Add classes in project settings</p>
          ) : (
            <div className="space-y-0.5 max-h-32 overflow-y-auto">
              {project?.classes.map((cls, i) => (
                <button
                  key={cls}
                  onClick={() => setSelectedClassId(i)}
                  className={cn(
                    'w-full flex items-center gap-2 px-2 py-1 rounded text-sm text-left transition-colors',
                    selectedClassId === i ? 'bg-primary/20 text-primary' : 'hover:bg-muted text-foreground'
                  )}
                >
                  <span
                    className="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                    style={{ backgroundColor: colors[i % colors.length] }}
                  />
                  <span className="truncate flex-1 text-xs">{cls}</span>
                  <span className="text-[10px] text-muted-foreground">{i + 1}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Annotations list */}
        <div className="flex-1 overflow-y-auto p-3 min-h-0">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Regions ({annotations?.length || 0})
          </label>

          {annotations?.length === 0 ? (
            <p className="text-xs text-muted-foreground py-4 text-center">Click and drag to draw</p>
          ) : (
            <div className="space-y-0.5">
              {annotations?.map((ann) => (
                <div
                  key={ann.id}
                  onClick={() => setSelectedAnnotation(ann.id)}
                  className={cn(
                    'flex items-center gap-2 px-2 py-1 rounded text-xs cursor-pointer transition-colors group',
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
                    className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-destructive transition-opacity"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Auto-label button */}
        <div className="flex-shrink-0 p-3 border-t border-border">
          <Button
            variant="outline"
            size="sm"
            className="w-full gap-2 h-8 text-xs"
            onClick={() => autoLabelMutation.mutate()}
            disabled={autoLabelMutation.isPending || !project?.classes.length}
          >
            {autoLabelMutation.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Wand2 className="h-3.5 w-3.5" />
            )}
            Auto-Label
          </Button>
        </div>
      </div>
    </div>
  )
}
