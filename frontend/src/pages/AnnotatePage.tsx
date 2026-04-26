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
  FolderOpen,
  Plus,
  Wand2,
  XCircle,
  Loader2,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'
import { useStore } from '@/store/useStore'
import { cn } from '@/lib/utils'
import type { Annotation, BoundingBox } from '@/types'
import { AnnotationCanvas, type AnnotationCanvasHandle } from '@/components/AnnotationCanvas'
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'

export default function AnnotatePage() {
  const { projectName } = useParams<{ projectName: string }>()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const currentImageIndex = useStore((s) => s.currentImageIndex)
  const setCurrentImageIndex = useStore((s) => s.setCurrentImageIndex)
  const selectedAnnotationId = useStore((s) => s.selectedAnnotationId)
  const setSelectedAnnotation = useStore((s) => s.setSelectedAnnotation)
  const selectedClassId = useStore((s) => s.selectedClassId)
  const setSelectedClassId = useStore((s) => s.setSelectedClassId)

  const containerRef = useRef<HTMLDivElement>(null)
  const thumbnailStripRef = useRef<HTMLDivElement>(null)
  const annotationCanvasRef = useRef<AnnotationCanvasHandle>(null)

  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [newClassName, setNewClassName] = useState('')
  const [selectedImageIndices, setSelectedImageIndices] = useState<Set<number>>(new Set())
  const lastClickedIndexRef = useRef<number | null>(null)
  const [samModalOpen, setSamModalOpen] = useState(false)
  const [samClassDescriptions, setSamClassDescriptions] = useState<Record<string, string>>({})
  const [samConfidence, setSamConfidence] = useState(0.25)
  const [samSkipLabeled, setSamSkipLabeled] = useState(true)
  const [labelingJobId, setLabelingJobId] = useState<string | null>(null)
  const [isLabelingRunning, setIsLabelingRunning] = useState(false)
  const [labelingProgress, setLabelingProgress] = useState<{ progress: number; message: string } | null>(null)

  // Fetch project data
  const { data: project } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  const { data: manualDatasets } = useQuery({
    queryKey: ['manual-data-datasets', projectName],
    queryFn: () => api.manualData.listDatasets(projectName!),
    enabled: !!projectName,
  })

  // Fetch manual data images
  const { data: manualData } = useQuery({
    queryKey: ['manual-data-images', projectName, selectedDataset],
    queryFn: () => api.manualData.listImages(projectName!, 0, 500, selectedDataset ?? undefined),
    enabled: !!projectName,
  })

  const images = manualData?.images ?? []
  const currentImage = images[currentImageIndex]

  // Fetch annotations for current image (frame_id)
  const { data: annotations } = useQuery({
    queryKey: ['annotations', projectName, currentImage?.frame_id],
    queryFn: () => api.annotations.listForFrame(projectName!, currentImage!.frame_id),
    enabled: !!projectName && !!currentImage,
  })

  // Create annotation mutation
  const createAnnotationMutation = useMutation({
    mutationFn: (data: { frame_id: number | string; class_label_id: number; box: BoundingBox; polygon?: number[][] }) =>
      api.annotations.create(projectName!, data),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentImage?.frame_id] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      annotationCanvasRef.current?.onAnnotationCreated(data)
    },
  })

  // Update annotation mutation
  const updateAnnotationMutation = useMutation({
    mutationFn: ({ id, box, class_label_id, polygon }: { id: number; box?: BoundingBox; class_label_id?: number; polygon?: number[][] }) => {
      return api.annotations.update(projectName!, id, {
        ...(box && { box }),
        ...(class_label_id !== undefined && { class_label_id }),
        ...(polygon !== undefined && { polygon }),
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentImage?.frame_id] })
    },
  })

  const addClassMutation = useMutation({
    mutationFn: (name: string) => {
      const next = [...(project?.classes ?? []), name.trim()]
      return api.projects.updateClasses(projectName!, next)
    },
    onSuccess: (_, addedName) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      const nextIndex = (project?.classes?.length ?? 0)
      setSelectedClassId(nextIndex)
      setNewClassName('')
      toast({ title: 'Class added', description: `"${addedName.trim()}" is now selected.`, type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Add class failed', description: error.message, type: 'error' })
    },
  })

  // Update annotation class mutation
  const updateAnnotationClassMutation = useMutation({
    mutationFn: ({ id, class_label_id }: { id: number; class_label_id: number }) =>
      api.annotations.update(projectName!, id, { class_label_id }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentImage?.frame_id] })
    },
  })

  // Delete annotation mutation
  const deleteAnnotationMutation = useMutation({
    mutationFn: (id: number) => api.annotations.delete(projectName!, id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentImage?.frame_id] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      setSelectedAnnotation(null)
    },
  })

  const clearFrameMutation = useMutation({
    mutationFn: (frameId: string | number) => api.annotations.clearFrame(projectName!, frameId),
    onSuccess: (data, frameId) => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, frameId] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      setSelectedAnnotation(null)
      toast({ title: 'Image cleared', description: `Removed ${data.deleted} annotation(s)`, type: 'success' })
    },
    onError: (e: Error) => {
      toast({ title: 'Clear failed', description: e.message, type: 'error' })
    },
  })

  const clearSelectedImagesMutation = useMutation({
    mutationFn: (frameIds: string[]) => api.annotations.clearFrames(projectName!, frameIds),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      setSelectedAnnotation(null)
      setSelectedImageIndices(new Set())
      toast({
        title: 'Images cleared',
        description: `Removed ${data.deleted} annotation(s) from ${data.frames_cleared} image(s)`,
        type: 'success',
      })
    },
    onError: (e: Error) => {
      toast({ title: 'Clear failed', description: e.message, type: 'error' })
    },
  })

  const updateClassMutateRef = useRef(updateAnnotationClassMutation.mutate)
  updateClassMutateRef.current = updateAnnotationClassMutation.mutate

  // Select a class: always update the active class, AND re-label the selected annotation if one exists
  const handleSelectClass = useCallback((classIndex: number) => {
    setSelectedClassId(classIndex)
    const selId = useStore.getState().selectedAnnotationId
    if (selId !== null) {
      updateClassMutateRef.current({ id: selId, class_label_id: classIndex })
    }
  }, [setSelectedClassId])

  // Clamp currentImageIndex when images change
  useEffect(() => {
    if (images.length > 0 && currentImageIndex >= images.length) {
      setCurrentImageIndex(Math.max(0, images.length - 1))
    }
  }, [images.length, currentImageIndex, setCurrentImageIndex])

  const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

  const handleCreateAnnotation = useCallback(
    (box: BoundingBox, classId: number, polygon?: number[][]) => {
      if (!currentImage) return
      createAnnotationMutation.mutate({ frame_id: currentImage.frame_id, class_label_id: classId, box, polygon })
    },
    [currentImage, createAnnotationMutation]
  )

  const handleUpdateAnnotation = useCallback(
    (id: number, box: BoundingBox, polygon?: number[][] | null) => {
      const polyArg = polygon === null ? undefined : polygon
      updateAnnotationMutation.mutate({ id, box, polygon: polyArg })
    },
    [updateAnnotationMutation]
  )

  const handleDeleteAnnotation = useCallback(
    (id: number) => deleteAnnotationMutation.mutate(id),
    [deleteAnnotationMutation]
  )

  const handleRestoreAnnotation = useCallback(
    async (annotation: Annotation): Promise<Annotation | void> => {
      if (!currentImage || !projectName) return
      const created = await api.annotations.create(projectName, {
        frame_id: currentImage.frame_id,
        class_label_id: annotation.class_label_id,
        box: annotation.box,
        ...(annotation.polygon && { polygon: annotation.polygon }),
      })
      queryClient.invalidateQueries({ queryKey: ['annotations', projectName, currentImage.frame_id] })
      return created
    },
    [currentImage, projectName, queryClient]
  )

  const handleDatasetChange = useCallback((ds: string | null) => {
    setSelectedDataset(ds)
    setCurrentImageIndex(0)
    setSelectedAnnotation(null)
    setSelectedImageIndices(new Set())
    lastClickedIndexRef.current = null
  }, [setCurrentImageIndex, setSelectedAnnotation])

  // Init SAM modal class descriptions from project when opening
  useEffect(() => {
    if (samModalOpen && project) {
      const desc: Record<string, string> = {}
      project.classes.forEach((c) => {
        desc[c] = project.class_descriptions?.[c] ?? c
      })
      setSamClassDescriptions(desc)
    }
  }, [samModalOpen, project])

  // Poll labeling job status
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
          queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
          queryClient.invalidateQueries({ queryKey: ['manual-data-datasets', projectName] })
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
  }, [labelingJobId, projectName, queryClient, toast])

  const runSamAutoLabel = useCallback(async () => {
    if (!projectName || !project || !manualDatasets) return
    // Save class descriptions to project
    await api.projects.updateClassDescriptions(projectName, samClassDescriptions)
    queryClient.invalidateQueries({ queryKey: ['project', projectName] })
    const sourceKeys =
      selectedDataset === null
        ? manualDatasets.datasets.map((d) => d.source_key)
        : [manualDatasets.datasets.find((d) => d.name === selectedDataset)?.source_key].filter(Boolean) as string[]
    if (sourceKeys.length === 0) {
      toast({ title: 'No dataset', description: 'Select a dataset first.', type: 'error' })
      return
    }
    const { job_id } = await api.labeling.autoLabel(projectName, {
      source_keys: sourceKeys,
      class_descriptions: samClassDescriptions,
      confidence: samConfidence,
      skip_labeled_frames: samSkipLabeled,
    })
    setLabelingJobId(job_id)
    setIsLabelingRunning(true)
    setLabelingProgress({ progress: 0, message: 'Starting...' })
  }, [
    projectName,
    project,
    manualDatasets,
    selectedDataset,
    samClassDescriptions,
    samConfidence,
    samSkipLabeled,
    queryClient,
    toast,
  ])

  // Navigation
  const goToImage = useCallback((index: number) => {
    if (images.length > 0 && index >= 0 && index < images.length) {
      setCurrentImageIndex(index)
      setSelectedAnnotation(null)
    }
  }, [images.length, setCurrentImageIndex, setSelectedAnnotation])

  const handleFilmstripClick = useCallback(
    (index: number, e: React.MouseEvent) => {
      const isMetaKey = e.metaKey || e.ctrlKey
      const isShift = e.shiftKey

      if (isShift && lastClickedIndexRef.current !== null) {
        const start = Math.min(lastClickedIndexRef.current, index)
        const end = Math.max(lastClickedIndexRef.current, index)
        setSelectedImageIndices((prev) => {
          const next = new Set(prev)
          for (let i = start; i <= end; i++) next.add(i)
          return next
        })
      } else if (isMetaKey) {
        setSelectedImageIndices((prev) => {
          const next = new Set(prev)
          if (next.has(index)) next.delete(index)
          else next.add(index)
          return next
        })
        lastClickedIndexRef.current = index
      } else {
        setSelectedImageIndices(new Set())
        lastClickedIndexRef.current = index
        goToImage(index)
        return
      }

      lastClickedIndexRef.current = index
    },
    [goToImage]
  )

  // Keep mutable refs
  const stateRef = useRef({ currentImageIndex, classCount: project?.classes.length || 0 })
  stateRef.current = { currentImageIndex, classCount: project?.classes.length || 0 }

  const goToImageRef = useRef(goToImage)
  goToImageRef.current = goToImage

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      if (e.key === 'Escape') {
        if (selectedImageIndices.size > 0) {
          e.preventDefault()
          setSelectedImageIndices(new Set())
          return
        }
      }

      const { currentImageIndex: idx, classCount } = stateRef.current

      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goToImageRef.current(idx - 1)
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault()
        goToImageRef.current(idx + 1)
      }
      const num = parseInt(e.key)
      if (!isNaN(num) && num >= 1 && num <= classCount) {
        e.preventDefault()
        handleSelectClass(num - 1)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleSelectClass, selectedImageIndices.size])

  // Scroll thumbnail into view when current image changes
  useEffect(() => {
    const strip = thumbnailStripRef.current
    const thumb = strip?.querySelector(`[data-index="${currentImageIndex}"]`)
    if (strip && thumb) {
      thumb.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' })
    }
  }, [currentImageIndex])

  if (!projectName) return null

  return (
    <div className="h-[calc(100vh-4rem)] flex overflow-hidden">
      {/* Main annotation area */}
      <div className="flex-1 flex flex-col min-w-0 bg-neutral-900">
        {/* Back link + dataset filter */}
        <div className="flex-shrink-0 px-4 py-2 border-b border-border flex items-center gap-2">
          <Link to={`/projects/${projectName}`}>
            <Button variant="ghost" size="sm" className="gap-1 h-8">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back
            </Button>
          </Link>

          {manualDatasets && manualDatasets.datasets.length > 1 && (
            <>
              <div className="w-px h-5 bg-border" />
              <FolderOpen className="h-3.5 w-3.5 text-muted-foreground" />
              <div className="flex items-center gap-1 flex-wrap">
                <button
                  onClick={() => handleDatasetChange(null)}
                  className={cn(
                    'px-2 py-0.5 rounded text-[11px] font-medium transition-colors',
                    selectedDataset === null
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground hover:text-foreground'
                  )}
                >
                  All
                </button>
                {manualDatasets.datasets.map((ds) => (
                  <button
                    key={ds.source_key}
                    onClick={() => handleDatasetChange(ds.name)}
                    className={cn(
                      'px-2 py-0.5 rounded text-[11px] font-medium transition-colors',
                      selectedDataset === ds.name
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:text-foreground'
                    )}
                  >
                    {ds.name === '(root)' ? 'Root' : ds.name} ({ds.image_count})
                  </button>
                ))}
              </div>
            </>
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

        {/* Canvas area */}
        <div
          ref={containerRef}
          className="flex-1 flex items-center justify-center p-2 overflow-hidden min-h-0"
        >
          {currentImage ? (
            <AnnotationCanvas
              ref={annotationCanvasRef}
              imageUrl={currentImage.url}
              imageWidth={currentImage.width}
              imageHeight={currentImage.height}
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
            <div className="text-center text-muted-foreground">
              <p>No images to annotate</p>
              <p className="text-sm mt-1">
                Place images in <code className="bg-muted px-1 rounded">manual_data/</code> and sync
              </p>
              <Link to={`/projects/${projectName}`}>
                <Button variant="outline" size="sm" className="mt-4">
                  Go to project
                </Button>
              </Link>
            </div>
          )}
        </div>

        {/* Navigation bar with thumbnail strip */}
        <div className="relative flex-shrink-0 bg-secondary border-t border-border px-4 py-2">
          <div className="flex items-center gap-2 mb-2">
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToImage(0)}>
              <SkipBack className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToImage(currentImageIndex - 1)}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="text-xs font-mono min-w-[70px] text-center text-muted-foreground">
              {currentImageIndex + 1} / {images.length || 0}
            </span>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToImage(currentImageIndex + 1)}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => goToImage(images.length - 1)}>
              <SkipForward className="h-3.5 w-3.5" />
            </Button>

            <div className="flex-1" />

            <span className="text-[10px] text-muted-foreground hidden sm:block">
              ← → navigate • 1-9 class • Del delete • ⌘Z undo • ⌘⇧Z redo • ⌘-click / Shift-click multi-select
            </span>
          </div>
          {selectedImageIndices.size > 0 && (
            <div className="absolute left-4 right-4 bottom-full mb-1 z-10 flex items-center gap-2 px-2 py-1.5 rounded-md bg-neutral-900/90 border border-blue-500/40 backdrop-blur-md shadow-lg">
              <span className="text-xs font-medium text-blue-400">
                {selectedImageIndices.size} image{selectedImageIndices.size !== 1 ? 's' : ''} selected
              </span>
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-[11px] px-2 text-muted-foreground hover:text-foreground"
                onClick={() => {
                  const allIndices = new Set<number>()
                  images.forEach((_, i) => allIndices.add(i))
                  setSelectedImageIndices(allIndices)
                }}
              >
                Select all
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-[11px] px-2 text-destructive hover:text-destructive"
                disabled={clearSelectedImagesMutation.isPending}
                onClick={() => {
                  const frameIds = Array.from(selectedImageIndices)
                    .filter((i) => images[i])
                    .map((i) => String(images[i].frame_id))
                  const annotatedCount = Array.from(selectedImageIndices).filter(
                    (i) => images[i] && images[i].annotation_count > 0
                  ).length
                  if (frameIds.length === 0) return
                  if (annotatedCount === 0) {
                    toast({ title: 'Nothing to clear', description: 'Selected images have no annotations.', type: 'error' })
                    return
                  }
                  if (confirm(`Clear annotations from ${frameIds.length} image(s)? (${annotatedCount} have annotations)`)) {
                    clearSelectedImagesMutation.mutate(frameIds)
                  }
                }}
              >
                {clearSelectedImagesMutation.isPending ? (
                  <Loader2 className="h-3 w-3 animate-spin mr-1" />
                ) : (
                  <XCircle className="h-3 w-3 mr-1" />
                )}
                Clear annotations
              </Button>
              <button
                onClick={() => setSelectedImageIndices(new Set())}
                className="text-[11px] text-muted-foreground hover:text-foreground ml-1"
                title="Deselect all (Esc)"
              >
                ✕
              </button>
            </div>
          )}

          {/* Thumbnail strip */}
          <div
            ref={thumbnailStripRef}
            className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin"
            style={{ maxHeight: 56 }}
          >
            {images.map((img, i) => {
              const isSelected = selectedImageIndices.has(i)
              return (
                <button
                  key={img.frame_id}
                  data-index={i}
                  onClick={(e) => handleFilmstripClick(i, e)}
                  className={cn(
                    'relative flex-shrink-0 w-12 h-12 rounded overflow-hidden border-2 transition-colors',
                    isSelected
                      ? 'border-blue-500 ring-1 ring-blue-500'
                      : i === currentImageIndex
                      ? 'border-amber-400 ring-1 ring-amber-400'
                      : img.annotation_count > 0
                      ? 'border-green-500/50 hover:border-green-500/80'
                      : 'border-border hover:border-primary/50'
                  )}
                >
                  <img
                    src={img.url}
                    alt={img.filename}
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

      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 bg-secondary border-l border-border flex flex-col overflow-hidden">
        {/* Classes */}
        <div className="flex-shrink-0 p-3 border-b border-border">
          <label className="text-[10px] font-medium text-muted-foreground mb-1 block uppercase tracking-wide">
            Label ({project?.classes.length || 0})
          </label>
          {project?.classes.length === 0 ? (
            <p className="text-xs text-muted-foreground py-2 mb-2">Add your first class below.</p>
          ) : (
            <div className="space-y-0.5 max-h-32 overflow-y-auto mb-2">
              {project?.classes.map((cls, i) => (
                <button
                  key={cls}
                  onClick={() => handleSelectClass(i)}
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
          <div className="flex gap-1.5">
            <Input
              placeholder="New class name"
              value={newClassName}
              onChange={(e) => setNewClassName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const name = newClassName.trim()
                  if (name && !addClassMutation.isPending) addClassMutation.mutate(name)
                }
              }}
              className="h-8 text-xs flex-1 min-w-0"
            />
            <Button
              size="sm"
              variant="secondary"
              className="h-8 px-2 shrink-0"
              disabled={
                !newClassName.trim() ||
                addClassMutation.isPending ||
                (project?.classes ?? []).includes(newClassName.trim())
              }
              onClick={() => addClassMutation.mutate(newClassName)}
            >
              <Plus className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* Annotations list */}
        <div className="flex-1 overflow-y-auto p-3 min-h-0">
          <div className="flex items-center justify-between mb-1">
            <label className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
              Regions ({annotations?.length || 0})
            </label>
            {annotations && annotations.length > 0 && currentImage && (
              <button
                onClick={() => {
                  if (confirm(`Remove all ${annotations.length} annotation(s) from this image?`)) {
                    clearFrameMutation.mutate(String(currentImage.frame_id))
                  }
                }}
                disabled={clearFrameMutation.isPending}
                className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-destructive transition-colors"
                title="Clear all annotations on this image"
              >
                <XCircle className="h-3 w-3" />
                Clear image
              </button>
            )}
          </div>

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
                Run SAM3 on all images in the current dataset. Edit class descriptions to improve detection.
              </p>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto py-0 space-y-4">
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
                <span className="text-sm">Skip already-labeled images</span>
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
