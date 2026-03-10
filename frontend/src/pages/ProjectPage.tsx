import { useState, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import {
  PenTool,
  Plus,
  Image,
  Tag,
  Loader2,
  RefreshCw,
  FolderOpen,
  Database,
  Trash2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  SkipBack,
  SkipForward,
  Upload,
  FolderUp,
  Pencil,
  Video,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { cn } from '@/lib/utils'
import { useToast } from '@/components/ui/Toaster'

export default function ProjectPage() {
  const { projectName } = useParams<{ projectName: string }>()
  const [newClass, setNewClass] = useState('')
  const [renamingClass, setRenamingClass] = useState<string | null>(null)
  const [newClassName, setNewClassName] = useState('')
  const [mergingClasses, setMergingClasses] = useState<string[]>([])
  const [expandedDatasets, setExpandedDatasets] = useState<Set<string>>(new Set())
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [renamingDataset, setRenamingDataset] = useState<string | null>(null)
  const [datasetRenameValue, setDatasetRenameValue] = useState('')
  const [uploadSubdir, setUploadSubdir] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const folderInputRef = useRef<HTMLInputElement>(null)
  const videoFileInputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isDraggingVideo, setIsDraggingVideo] = useState(false)
  const [videoExcludeFromTraining, setVideoExcludeFromTraining] = useState(false)
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const { t } = useTranslation()

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

  const { data: manualDatasets } = useQuery({
    queryKey: ['manual-data-datasets', projectName],
    queryFn: () => api.manualData.listDatasets(projectName!),
    enabled: !!projectName,
  })

  const { data: manualData, isLoading: manualDataLoading } = useQuery({
    queryKey: ['manual-data-images', projectName, selectedDataset],
    queryFn: () => api.manualData.listImages(projectName!, 0, 500, selectedDataset ?? undefined),
    enabled: !!projectName,
  })

  const { data: importedDatasets } = useQuery({
    queryKey: ['imported-datasets', projectName],
    queryFn: () => api.import.listDatasets(projectName!),
    enabled: !!projectName,
  })

  const { data: classDetails } = useQuery({
    queryKey: ['class-details', projectName],
    queryFn: () => api.classes.getDetails(projectName!),
    enabled: !!projectName,
  })

  const syncMutation = useMutation({
    mutationFn: () => api.manualData.sync(projectName!),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({
        title: t('project.synced'),
        description: t('project.syncedDesc', { added: data.images_added, removed: data.images_removed }),
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: t('project.syncFailed'), description: error.message, type: 'error' })
    },
  })

  const uploadMutation = useMutation({
    mutationFn: ({ files, dataset }: { files: File[]; dataset?: string }) =>
      api.manualData.upload(projectName!, files, dataset),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      setUploadSubdir('')
      if (fileInputRef.current) fileInputRef.current.value = ''
      toast({
        title: t('project.uploaded'),
        description: t('project.uploadedDesc', { count: data.uploaded, dataset: data.dataset }),
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: t('project.uploadFailed'), description: error.message, type: 'error' })
    },
  })

  const videoUploadMutation = useMutation({
    mutationFn: async ({ files, excludeFromTraining }: { files: File[]; excludeFromTraining?: boolean }) => {
      const results = []
      for (const file of files) {
        const result = await api.videos.upload(projectName!, file, {
          exclude_from_training: excludeFromTraining ?? false,
        })
        results.push(result)
      }
      return results
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      if (videoFileInputRef.current) videoFileInputRef.current.value = ''
      toast({
        title: t('project.videosAdded'),
        description: t('project.videosAddedDesc', { count: data.length }),
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: t('project.videoUploadFailed'), description: error.message, type: 'error' })
    },
  })

  const isImageFile = (f: File) => /\.(jpe?g|png|webp|bmp)$/i.test(f.name)
  const isVideoFile = (f: File) => /\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(f.name)

  /** Recursively collect all files from a FileSystemDirectoryEntry. */
  async function readDirEntry(entry: FileSystemDirectoryEntry): Promise<File[]> {
    const results: File[] = []
    const reader = entry.createReader()
    // readEntries returns up to 100 at a time; must loop until empty
    let batch: FileSystemEntry[]
    do {
      batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej))
      for (const child of batch) {
        if (child.isFile) {
          const file = await new Promise<File>((res, rej) =>
            (child as FileSystemFileEntry).file(res, rej),
          )
          results.push(file)
        } else if (child.isDirectory) {
          results.push(...(await readDirEntry(child as FileSystemDirectoryEntry)))
        }
      }
    } while (batch.length > 0)
    return results
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const items = Array.from(e.dataTransfer.items)
    const allFiles: File[] = []
    let detectedDataset: string | undefined = uploadSubdir.trim() || undefined

    for (const item of items) {
      const entry = item.webkitGetAsEntry?.()
      if (!entry) continue
      if (entry.isDirectory) {
        if (!detectedDataset) detectedDataset = entry.name
        const inner = await readDirEntry(entry as FileSystemDirectoryEntry)
        allFiles.push(...inner.filter(isImageFile))
      } else if (entry.isFile) {
        const file = await new Promise<File>((res, rej) =>
          (entry as FileSystemFileEntry).file(res, rej),
        )
        if (isImageFile(file)) allFiles.push(file)
      }
    }

    if (allFiles.length) {
      uploadMutation.mutate({ files: allFiles, dataset: detectedDataset })
    }
  }

  const handleVideoDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDraggingVideo(false)
    const files = e.dataTransfer.files ? Array.from(e.dataTransfer.files).filter(isVideoFile) : []
    if (files.length) {
      videoUploadMutation.mutate({ files, excludeFromTraining: videoExcludeFromTraining })
    }
  }

  const updateClassesMutation = useMutation({
    mutationFn: (classes: string[]) => api.projects.updateClasses(projectName!, classes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      toast({ title: t('project.classesUpdated'), type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('project.classesUpdateFailed'), description: error.message, type: 'error' })
    },
  })

  const renameClassMutation = useMutation({
    mutationFn: ({ oldName, newName }: { oldName: string; newName: string }) =>
      api.classes.rename(projectName!, oldName, newName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      toast({ title: t('project.classRenamed'), type: 'success' })
      setRenamingClass(null)
      setNewClassName('')
    },
    onError: (error: Error) => {
      toast({ title: t('project.renameFailed'), description: error.message, type: 'error' })
    },
  })

  const mergeClassesMutation = useMutation({
    mutationFn: ({ sources, target }: { sources: string[]; target: string }) =>
      api.classes.merge(projectName!, sources, target),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      toast({ title: t('project.classesMerged'), description: data.message, type: 'success' })
      setMergingClasses([])
    },
    onError: (error: Error) => {
      toast({ title: t('project.mergeFailed'), description: error.message, type: 'error' })
    },
  })

  const deleteClassMutation = useMutation({
    mutationFn: (className: string) => api.classes.delete(projectName!, className, true),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      toast({
        title: t('project.classDeleted'),
        description: t('project.classDeletedDesc', { count: data.annotations_deleted }),
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: t('project.deleteFailed'), description: error.message, type: 'error' })
    },
  })

  const deleteDatasetMutation = useMutation({
    mutationFn: (videoId: number | string) => api.import.deleteDataset(projectName!, videoId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['imported-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({
        title: t('project.datasetDeleted'),
        description: t('project.datasetDeletedDesc', { images: data.images_deleted, annotations: data.annotations_deleted }),
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: t('project.deleteFailed'), description: error.message, type: 'error' })
    },
  })

  const deleteVideoMutation = useMutation({
    mutationFn: (videoId: number | string) => api.videos.delete(projectName!, videoId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({ title: t('project.videoRemoved'), type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('project.deleteFailed'), description: error.message, type: 'error' })
    },
  })

  const deleteImageMutation = useMutation({
    mutationFn: (frameId: string) => api.manualData.deleteImage(projectName!, frameId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      const desc = data.annotations_deleted > 0
        ? t('project.imageDeletedAnnotations', { count: data.annotations_deleted })
        : undefined
      toast({ title: t('project.imageDeleted'), description: desc, type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('project.deleteFailed'), description: error.message, type: 'error' })
    },
  })

  const renameDatasetMutation = useMutation({
    mutationFn: ({ oldName, newName }: { oldName: string; newName: string }) =>
      api.manualData.renameDataset(projectName!, oldName, newName),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['manual-data-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      if (selectedDataset === data.old_name) setSelectedDataset(data.new_name)
      setRenamingDataset(null)
      toast({ title: t('project.datasetRenamed'), type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('project.renameFailed'), description: error.message, type: 'error' })
    },
  })

  const handleAddClass = () => {
    if (!newClass.trim() || !project) return
    const classes = [...project.classes, newClass.trim()]
    updateClassesMutation.mutate(classes)
    setNewClass('')
  }

  const handleRemoveClass = (className: string, annotationCount: number) => {
    if (!project) return
    if (annotationCount > 0) {
      if (!confirm(t('project.deleteClassConfirm', { name: className, count: annotationCount }))) {
        return
      }
    }
    deleteClassMutation.mutate(className)
  }

  const images = manualData?.images ?? []
  const totalImages = manualData?.total ?? 0
  const annotatedCount = images.filter((img) => img.annotation_count > 0).length

  return (
    <div className="container max-w-6xl mx-auto py-8 px-6 lg:px-8">
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content - Videos then Images */}
        <div className="lg:col-span-2 space-y-6">
          {/* Videos */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Video className="h-5 w-5" />
                  {t('project.videos')}
                </CardTitle>
                <CardDescription>
                  {t('project.videosDesc')}{' '}
                  <code className="text-xs bg-muted px-1 rounded">python -m cli.videos add --project ... &lt;file.mp4&gt;</code>
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  'flex flex-wrap items-end gap-3 p-4 rounded-lg border mb-4 transition-colors',
                  isDraggingVideo ? 'border-primary bg-primary/10 border-2' : 'bg-muted/50 border-border'
                )}
                onDragEnter={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (e.dataTransfer.types.includes('Files')) setIsDraggingVideo(true)
                }}
                onDragOver={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  e.dataTransfer.dropEffect = 'copy'
                }}
                onDragLeave={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDraggingVideo(false)
                }}
                onDrop={handleVideoDrop}
              >
                <input
                  ref={videoFileInputRef}
                  type="file"
                  accept=".mp4,.avi,.mov,.mkv,.webm,.m4v"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const files = e.target.files ? Array.from(e.target.files) : []
                    if (files.length) {
                      videoUploadMutation.mutate({ files, excludeFromTraining: videoExcludeFromTraining })
                    }
                    e.target.value = ''
                  }}
                />
                <Button
                  variant="secondary"
                  size="sm"
                  className="gap-1.5"
                  disabled={videoUploadMutation.isPending}
                  onClick={() => videoFileInputRef.current?.click()}
                >
                  {videoUploadMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="h-4 w-4" />
                  )}
                  {t('project.chooseVideos')}
                </Button>
                <label className="flex items-center gap-2 cursor-pointer text-sm">
                  <input
                    type="checkbox"
                    checked={videoExcludeFromTraining}
                    onChange={(e) => setVideoExcludeFromTraining(e.target.checked)}
                    className="rounded"
                  />
                  <span>{t('project.testOnly')}</span>
                </label>
                <p className="text-xs text-muted-foreground w-full mt-0.5">
                  {isDraggingVideo ? t('project.dropVideos') : t('project.dropVideosHint')}
                </p>
              </div>
              {videos && videos.length > 0 ? (
                <ul className="space-y-2">
                  {videos.map((v) => (
                    <li
                      key={String(v.id)}
                      className="flex items-center gap-3 p-2 rounded-lg border border-border hover:border-primary/30 transition-colors group"
                    >
                      <Link
                        to={`/projects/${projectName}/annotate/video/${v.id}`}
                        className="shrink-0 w-[88px] h-[56px] rounded-md overflow-hidden bg-muted border border-border block"
                      >
                        <img
                          src={api.videos.thumbnailUrl(projectName!, v.id)}
                          alt={v.filename}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      </Link>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium truncate">{v.filename}</p>
                        <p className="text-xs text-muted-foreground">
                          {v.width}×{v.height} · {v.fps?.toFixed(1) ?? 0}fps · {(v.duration ?? 0).toFixed(1)}s
                          {v.frame_count != null && v.frame_count > 0 && ` · ${t('project.framesAnnotated', { annotated: v.annotated_frame_count ?? v.annotation_count ?? 0, total: v.frame_count })}`}
                          {v.exclude_from_training && ` · ${t('project.testOnlyBadge')}`}
                        </p>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <Link to={`/projects/${projectName}/annotate/video/${v.id}`}>
                          <Button variant="outline" size="sm" className="gap-1">
                            <PenTool className="h-3.5 w-3.5" />
                            {t('common.annotate')}
                          </Button>
                        </Link>
                        <button
                          onClick={() => {
                            if (confirm(`Remove "${v.filename}" from project?`)) {
                              deleteVideoMutation.mutate(v.id)
                            }
                          }}
                          className="p-1.5 text-muted-foreground hover:text-destructive transition-colors"
                          title={t('project.removeVideo')}
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground text-center py-4">
                  {t('project.noVideos')}
                </p>
              )}
            </CardContent>
          </Card>

          {/* Images */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Image className="h-5 w-5" />
                  {t('project.images')}
                </CardTitle>
                <CardDescription>
                  {totalImages > 0
                    ? t('project.imagesAnnotated', { annotated: annotatedCount, total: totalImages })
                    : t('project.imagesPlaceholder')}
                </CardDescription>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => syncMutation.mutate()}
                  disabled={syncMutation.isPending}
                  className="gap-1"
                >
                  {syncMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  {t('common.refresh')}
                </Button>
                {totalImages > 0 && (
                  <Link to={`/projects/${projectName}/annotate`}>
                    <Button size="sm" className="gap-1">
                      <PenTool className="h-4 w-4" />
                      {t('common.annotate')}
                    </Button>
                  </Link>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {/* Upload images */}
              <div
                className={cn(
                  'flex flex-wrap items-end gap-2 p-3 rounded-lg border mb-4 transition-colors',
                  isDragging ? 'border-primary bg-primary/10 border-2' : 'bg-muted/50 border-border'
                )}
                onDragEnter={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (e.dataTransfer.types.includes('Files')) setIsDragging(true)
                }}
                onDragOver={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  e.dataTransfer.dropEffect = 'copy'
                }}
                onDragLeave={(e) => {
                  e.preventDefault()
                  e.stopPropagation()
                  if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false)
                }}
                onDrop={handleDrop}
              >
                <div className="flex-1 min-w-[140px]">
                  <label className="text-xs text-muted-foreground block mb-1">{t('project.subfolderLabel')}</label>
                  <Input
                    placeholder={t('project.subfolderPlaceholder')}
                    value={uploadSubdir}
                    onChange={(e) => setUploadSubdir(e.target.value)}
                    className="h-9"
                  />
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,.webp,.bmp"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const files = e.target.files ? Array.from(e.target.files) : []
                    if (files.length) {
                      uploadMutation.mutate({
                        files,
                        dataset: uploadSubdir.trim() || undefined,
                      })
                    }
                    e.target.value = ''
                  }}
                />
                <input
                  ref={folderInputRef}
                  type="file"
                  // @ts-expect-error – non-standard but widely supported
                  webkitdirectory=""
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const allFiles = e.target.files ? Array.from(e.target.files) : []
                    const images = allFiles.filter((f) => isImageFile(f))
                    if (images.length) {
                      // Auto-detect folder name from webkitRelativePath if no manual subdir
                      let dataset = uploadSubdir.trim() || undefined
                      if (!dataset) {
                        const rel = (images[0] as File & { webkitRelativePath?: string }).webkitRelativePath || ''
                        const folderName = rel.split('/')[0]
                        if (folderName) dataset = folderName
                      }
                      uploadMutation.mutate({ files: images, dataset })
                    }
                    e.target.value = ''
                  }}
                />
                <Button
                  variant="secondary"
                  size="sm"
                  className="gap-1.5"
                  disabled={uploadMutation.isPending}
                  onClick={() => fileInputRef.current?.click()}
                >
                  {uploadMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="h-4 w-4" />
                  )}
                  {t('common.uploadImages')}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="gap-1.5"
                  disabled={uploadMutation.isPending}
                  onClick={() => folderInputRef.current?.click()}
                >
                  {uploadMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FolderUp className="h-4 w-4" />
                  )}
                  {t('common.uploadFolder')}
                </Button>
                <p className="text-xs text-muted-foreground w-full mt-0.5">
                  {isDragging ? t('project.dropImages') : t('project.dropImagesHint')}
                </p>
              </div>

              {manualDataLoading ? (
                <div className="grid grid-cols-4 gap-2">
                  {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                    <div key={i} className="aspect-square bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : totalImages === 0 ? (
                <div className="text-center py-12 border-2 border-dashed border-border rounded-lg">
                  <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground mb-2 font-medium">
                    {t('project.noImages')}
                  </p>
                  <p className="text-sm text-muted-foreground mb-4 max-w-md mx-auto">
                    {t('project.noImagesDesc')}
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => syncMutation.mutate()}
                    disabled={syncMutation.isPending}
                    className="gap-2"
                  >
                    {syncMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                    {t('common.scanForImages')}
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {manualDatasets && manualDatasets.datasets.length > 1 && (
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <button
                        onClick={() => setSelectedDataset(null)}
                        className={cn(
                          'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                          selectedDataset === null
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted text-muted-foreground hover:text-foreground'
                        )}
                      >
                        {t('project.all')} ({manualDatasets.datasets.reduce((s, d) => s + d.image_count, 0)})
                      </button>
                      {manualDatasets.datasets.map((ds) => (
                        <div key={ds.source_key} className="relative group/pill flex items-center">
                          {renamingDataset === ds.name ? (
                            <form
                              className="flex items-center gap-1"
                              onSubmit={(e) => {
                                e.preventDefault()
                                const val = datasetRenameValue.trim()
                                if (val && val !== ds.name) {
                                  renameDatasetMutation.mutate({ oldName: ds.name, newName: val })
                                } else {
                                  setRenamingDataset(null)
                                }
                              }}
                            >
                              <input
                                autoFocus
                                className="px-2 py-0.5 rounded-md text-xs font-medium border border-primary bg-background w-32 focus:outline-none"
                                value={datasetRenameValue}
                                onChange={(e) => setDatasetRenameValue(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Escape') setRenamingDataset(null) }}
                                onBlur={() => setRenamingDataset(null)}
                              />
                              {renameDatasetMutation.isPending && (
                                <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                              )}
                            </form>
                          ) : (
                            <>
                              <button
                                onClick={() => setSelectedDataset(ds.name)}
                                className={cn(
                                  'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                                  selectedDataset === ds.name
                                    ? 'bg-primary text-primary-foreground'
                                    : 'bg-muted text-muted-foreground hover:text-foreground'
                                )}
                              >
                                {ds.name === '(root)' ? t('project.root') : ds.name} ({ds.image_count})
                              </button>
                              {ds.name !== '(root)' && (
                                <button
                                  className="ml-0.5 p-0.5 rounded opacity-0 group-hover/pill:opacity-100 transition-opacity text-muted-foreground hover:text-foreground"
                                  title={t('project.renameDataset')}
                                  onClick={() => {
                                    setRenamingDataset(ds.name)
                                    setDatasetRenameValue(ds.name)
                                  }}
                                >
                                  <Pencil className="h-3 w-3" />
                                </button>
                              )}
                            </>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                <div className="grid grid-cols-4 sm:grid-cols-5 gap-2">
                  {images.slice(0, 20).map((img) => (
                    <div
                      key={img.frame_id}
                      className="relative aspect-square rounded-lg overflow-hidden border border-border hover:border-primary/50 transition-colors group"
                    >
                      <img
                        src={img.url}
                        alt={img.filename}
                        className="w-full h-full object-cover"
                      />
                      {img.annotation_count > 0 && (
                        <div
                          className="absolute top-1 right-1 w-2 h-2 rounded-full bg-green-500"
                          title={`${img.annotation_count} annotations`}
                        />
                      )}
                      <button
                        className="absolute bottom-1 right-1 p-1 rounded bg-black/60 text-white opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
                        title={t('common.delete')}
                        onClick={() => {
                          const msg = img.annotation_count > 0
                            ? t('project.deleteImageConfirm', { count: img.annotation_count })
                            : t('project.deleteImageConfirmSimple')
                          if (confirm(msg)) deleteImageMutation.mutate(String(img.frame_id))
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Imported Datasets */}
          {importedDatasets && importedDatasets.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  {t('project.importedDatasets')}
                </CardTitle>
                <CardDescription>
                  {t('project.importedDatasetsDesc', { count: importedDatasets.length })}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {importedDatasets.map((ds) => (
                  <div key={ds.video_id} className="group border border-border rounded-lg overflow-hidden">
                    <div
                      onClick={() => {
                        setExpandedDatasets((prev) => {
                          const next = new Set(prev)
                          const key = String(ds.video_id)
                          if (next.has(key)) next.delete(key)
                          else next.add(key)
                          return next
                        })
                      }}
                      className="flex items-center gap-3 p-3 hover:bg-muted/50 transition-colors cursor-pointer"
                    >
                      {expandedDatasets.has(String(ds.video_id)) ? (
                        <ChevronDown className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      ) : (
                        <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{ds.source}</p>
                        <p className="text-xs text-muted-foreground">
                          {ds.image_count} images · {ds.annotation_count} annotations
                        </p>
                        {ds.classes && ds.classes.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {ds.classes.map((cls) => (
                              <span
                                key={cls}
                                className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
                              >
                                {cls}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          if (confirm(`Delete this imported dataset? (${ds.image_count} images, ${ds.annotation_count} annotations)`)) {
                            deleteDatasetMutation.mutate(ds.video_id)
                          }
                        }}
                        className="p-1 opacity-0 group-hover:opacity-100 hover:text-destructive transition-opacity"
                        title={t('project.deleteDataset')}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>

                    {expandedDatasets.has(String(ds.video_id)) && (
                      <div className="border-t border-border p-3">
                        <ImportedDatasetPreview projectName={projectName!} videoId={ds.video_id} totalImages={ds.image_count} />
                      </div>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Classes */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Tag className="h-5 w-5" />
                {t('project.classes')}
              </CardTitle>
              <CardDescription>
                {t('project.classesDesc')}
                {mergingClasses.length > 0 && (
                  <span className="ml-2 text-primary">
                    {t('project.classesMerging', { count: mergingClasses.length })}
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2 mb-4">
                <Input
                  placeholder={t('project.addClass')}
                  value={newClass}
                  onChange={(e) => setNewClass(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
                />
                <Button onClick={handleAddClass} size="icon">
                  <Plus className="h-4 w-4" />
                </Button>
              </div>

              {mergingClasses.length > 0 && (
                <div className="mb-3 p-2 bg-muted rounded-lg text-sm">
                  <p className="text-muted-foreground mb-2">{t('project.clickToMerge')}</p>
                  <Button size="sm" variant="outline" onClick={() => setMergingClasses([])}>
                    {t('project.cancelMerge')}
                  </Button>
                </div>
              )}

              {(!classDetails || classDetails.length === 0) ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  {t('project.noClasses')}
                </p>
              ) : (
                <div className="space-y-2">
                  {classDetails.map((cls, i) => (
                    <div
                      key={cls.name}
                      className={`group rounded-lg border transition-colors ${
                        mergingClasses.includes(cls.name)
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <div className="flex items-center gap-2 p-2">
                        <span
                          className="w-3 h-3 rounded-full flex-shrink-0"
                          style={{
                            backgroundColor: [
                              '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'
                            ][i % 7]
                          }}
                        />
                        {renamingClass === cls.name ? (
                          <div className="flex-1 flex gap-2">
                            <Input
                              value={newClassName}
                              onChange={(e) => setNewClassName(e.target.value)}
                              className="h-7 text-sm"
                              autoFocus
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  renameClassMutation.mutate({ oldName: cls.name, newName: newClassName })
                                } else if (e.key === 'Escape') {
                                  setRenamingClass(null)
                                }
                              }}
                            />
                            <Button
                              size="sm"
                              className="h-7"
                              onClick={() => renameClassMutation.mutate({ oldName: cls.name, newName: newClassName })}
                            >
                              {t('common.save')}
                            </Button>
                          </div>
                        ) : (
                          <>
                            <span className="flex-1 font-medium text-sm">{cls.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {t('project.annotations', { count: cls.annotation_count })}
                            </span>
                          </>
                        )}
                        {renamingClass !== cls.name && (
                          <div className="opacity-0 group-hover:opacity-100 flex gap-1 transition-opacity">
                            {mergingClasses.length > 0 && !mergingClasses.includes(cls.name) ? (
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-6 text-xs"
                                onClick={() => {
                                  mergeClassesMutation.mutate({ sources: mergingClasses, target: cls.name })
                                }}
                              >
                                {t('project.mergeHere')}
                              </Button>
                            ) : (
                              <>
                                <button
                                  onClick={() => {
                                    if (mergingClasses.includes(cls.name)) {
                                      setMergingClasses(mergingClasses.filter(c => c !== cls.name))
                                    } else {
                                      setMergingClasses([...mergingClasses, cls.name])
                                    }
                                  }}
                                  className="text-xs text-muted-foreground hover:text-primary"
                                  title="Select for merge"
                                >
                                  {mergingClasses.includes(cls.name) ? '✓' : '○'}
                                </button>
                                <button
                                  onClick={() => {
                                    setRenamingClass(cls.name)
                                    setNewClassName(cls.name)
                                  }}
                                  className="text-xs text-muted-foreground hover:text-primary"
                                  title="Rename"
                                >
                                  ✎
                                </button>
                                <button
                                  onClick={() => handleRemoveClass(cls.name, cls.annotation_count)}
                                  className="text-xs text-muted-foreground hover:text-destructive"
                                  title="Delete"
                                >
                                  ×
                                </button>
                              </>
                            )}
                          </div>
                        )}
                      </div>
                      {cls.annotation_sources && Object.keys(cls.annotation_sources).length > 0 && (
                        <div className="px-2 pb-2 pt-0 flex items-center gap-2 flex-wrap">
                          {Object.entries(cls.annotation_sources).map(([source, count]) => (
                            <span
                              key={source}
                              className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
                                source === 'manual_data' || source.startsWith('manual_data__')
                                  ? 'bg-green-500/20 text-green-400'
                                  : 'bg-gray-500/20 text-gray-400'
                              }`}
                            >
                              {source}: {count}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
    </div>
    </div>
  )
}

const PAGE_SIZE = 20

function ImportedDatasetPreview({ projectName, videoId, totalImages }: { projectName: string; videoId: number | string; totalImages: number }) {
  const [page, setPage] = useState(0)
  const offset = page * PAGE_SIZE
  const totalPages = Math.max(1, Math.ceil(totalImages / PAGE_SIZE))

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ['imported-dataset-images', projectName, videoId, offset],
    queryFn: () => api.import.listImages(projectName, videoId, offset, PAGE_SIZE),
    placeholderData: (prev) => prev,
  })

  if (isLoading) {
    return (
      <div className="grid grid-cols-4 sm:grid-cols-5 gap-2">
        {Array.from({ length: Math.min(PAGE_SIZE, totalImages) }, (_, i) => (
          <div key={i} className="aspect-square bg-muted animate-pulse rounded" />
        ))}
      </div>
    )
  }

  if (!data || data.images.length === 0) {
    return <p className="text-xs text-muted-foreground text-center py-2">No images</p>
  }

  return (
    <div>
      <div className={cn('grid grid-cols-4 sm:grid-cols-5 gap-2 transition-opacity', isFetching && 'opacity-50')}>
        {data.images.map((img) => (
          <div
            key={img.frame_id}
            className="relative aspect-square rounded overflow-hidden border border-border"
          >
            <img
              src={img.url}
              alt={img.original_filename}
              className="w-full h-full object-cover"
            />
            {img.split && (
              <span className="absolute bottom-0 left-0 right-0 text-[9px] text-center bg-black/60 text-white py-0.5">
                {img.split}
              </span>
            )}
          </div>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-3">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={page === 0}
            onClick={() => setPage(0)}
          >
            <SkipBack className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            <ChevronLeft className="h-3.5 w-3.5" />
          </Button>
          <span className="text-xs font-mono text-muted-foreground min-w-[60px] text-center">
            {page + 1} / {totalPages}
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 w-7 p-0"
            disabled={page >= totalPages - 1}
            onClick={() => setPage(totalPages - 1)}
          >
            <SkipForward className="h-3.5 w-3.5" />
          </Button>
        </div>
      )}
    </div>
  )
}
