import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Play,
  PenTool,
  Cpu,
  Plus,
  Image,
  Tag,
  Loader2,
  RefreshCw,
  Grid3X3,
  FolderOpen,
  Database,
  Trash2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  SkipBack,
  SkipForward,
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
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
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

  const { data: inferenceMatrix } = useQuery({
    queryKey: ['inference-results', projectName],
    queryFn: () => api.inference.listResults(projectName!),
    enabled: !!projectName,
  })

  const syncMutation = useMutation({
    mutationFn: () => api.manualData.sync(projectName!),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({
        title: 'Images synced',
        description: `${data.images_added} added, ${data.images_removed} removed`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Sync failed', description: error.message, type: 'error' })
    },
  })

  const updateClassesMutation = useMutation({
    mutationFn: (classes: string[]) => api.projects.updateClasses(projectName!, classes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({ title: 'Classes updated', type: 'success' })
    },
  })

  const renameClassMutation = useMutation({
    mutationFn: ({ oldName, newName }: { oldName: string; newName: string }) =>
      api.classes.rename(projectName!, oldName, newName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      toast({ title: 'Class renamed', type: 'success' })
      setRenamingClass(null)
      setNewClassName('')
    },
    onError: (error: Error) => {
      toast({ title: 'Rename failed', description: error.message, type: 'error' })
    },
  })

  const mergeClassesMutation = useMutation({
    mutationFn: ({ sources, target }: { sources: string[]; target: string }) =>
      api.classes.merge(projectName!, sources, target),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      toast({ title: 'Classes merged', description: data.message, type: 'success' })
      setMergingClasses([])
    },
    onError: (error: Error) => {
      toast({ title: 'Merge failed', description: error.message, type: 'error' })
    },
  })

  const deleteClassMutation = useMutation({
    mutationFn: (className: string) => api.classes.delete(projectName!, className, true),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      queryClient.invalidateQueries({ queryKey: ['manual-data-images', projectName] })
      toast({
        title: 'Class deleted',
        description: `${data.annotations_deleted} annotations removed`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Delete failed', description: error.message, type: 'error' })
    },
  })

  const deleteDatasetMutation = useMutation({
    mutationFn: (videoId: number | string) => api.import.deleteDataset(projectName!, videoId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['imported-datasets', projectName] })
      queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({
        title: 'Dataset deleted',
        description: `${data.images_deleted} images, ${data.annotations_deleted} annotations removed`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Delete failed', description: error.message, type: 'error' })
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
      if (!confirm(`Delete class "${className}" and its ${annotationCount} annotations?`)) {
        return
      }
    }
    deleteClassMutation.mutate(className)
  }

  const images = manualData?.images ?? []
  const totalImages = manualData?.total ?? 0
  const annotatedCount = images.filter((img) => img.annotation_count > 0).length

  if (projectLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (!project) {
    return (
      <div className="container max-w-4xl mx-auto py-8 px-6">
        <Card>
          <CardContent className="py-16 text-center">
            <h2 className="text-xl font-semibold mb-2">Project not found</h2>
            <Link to="/projects" className="text-primary hover:underline">
              Back to projects
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="container max-w-6xl mx-auto py-8 px-6 lg:px-8">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-3xl font-bold">{project.name}</h1>
          {project.description && (
            <p className="text-muted-foreground mt-1">{project.description}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Link to={`/projects/${projectName}/train`}>
            <Button variant="outline" className="gap-2">
              <Cpu className="h-4 w-4" />
              Train
            </Button>
          </Link>
          <Link to={`/projects/${projectName}/inference`}>
            <Button className="gap-2">
              <Play className="h-4 w-4" />
              Run Inference
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content - Images */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Image className="h-5 w-5" />
                  Images
                </CardTitle>
                <CardDescription>
                  {totalImages > 0
                    ? `${annotatedCount}/${totalImages} annotated`
                    : 'Place images in project_root/manual_data/ folder'}
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
                  Refresh
                </Button>
                {totalImages > 0 && (
                  <Link to={`/projects/${projectName}/annotate`}>
                    <Button size="sm" className="gap-1">
                      <PenTool className="h-4 w-4" />
                      Annotate
                    </Button>
                  </Link>
                )}
              </div>
            </CardHeader>
            <CardContent>
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
                    No images yet
                  </p>
                  <p className="text-sm text-muted-foreground mb-4 max-w-md mx-auto">
                    Place your images (jpg, png, webp, bmp) in the project folder at{' '}
                    <code className="bg-muted px-1 rounded">manual_data/</code>, then click
                    Refresh to load them.
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
                    Scan for images
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
                        All ({manualDatasets.datasets.reduce((s, d) => s + d.image_count, 0)})
                      </button>
                      {manualDatasets.datasets.map((ds) => (
                        <button
                          key={ds.source_key}
                          onClick={() => setSelectedDataset(ds.name)}
                          className={cn(
                            'px-2.5 py-1 rounded-md text-xs font-medium transition-colors',
                            selectedDataset === ds.name
                              ? 'bg-primary text-primary-foreground'
                              : 'bg-muted text-muted-foreground hover:text-foreground'
                          )}
                        >
                          {ds.name === '(root)' ? 'Root' : ds.name} ({ds.image_count})
                        </button>
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
                  Imported Datasets
                </CardTitle>
                <CardDescription>
                  {importedDatasets.length} dataset{importedDatasets.length !== 1 ? 's' : ''} from external sources
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
                        title="Delete dataset"
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
          {/* Inference Results Summary */}
          {inferenceMatrix && inferenceMatrix.runs.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Grid3X3 className="h-5 w-5" />
                  Inference
                </CardTitle>
                <CardDescription>
                  {inferenceMatrix.runs.length} run{inferenceMatrix.runs.length !== 1 ? 's' : ''} available
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Link to={`/projects/${projectName}/inference`}>
                  <Button variant="outline" className="w-full gap-2">
                    <Play className="h-4 w-4" />
                    View Results
                  </Button>
                </Link>
              </CardContent>
            </Card>
          )}

          {/* Classes */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Tag className="h-5 w-5" />
                Classes
              </CardTitle>
              <CardDescription>
                Object classes to detect
                {mergingClasses.length > 0 && (
                  <span className="ml-2 text-primary">
                    (Select target class to merge {mergingClasses.length} selected)
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2 mb-4">
                <Input
                  placeholder="Add class..."
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
                  <p className="text-muted-foreground mb-2">Click a class to merge into it, or:</p>
                  <Button size="sm" variant="outline" onClick={() => setMergingClasses([])}>
                    Cancel merge
                  </Button>
                </div>
              )}

              {(!classDetails || classDetails.length === 0) ? (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No classes defined yet
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
                              Save
                            </Button>
                          </div>
                        ) : (
                          <>
                            <span className="flex-1 font-medium text-sm">{cls.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {cls.annotation_count} ann.
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
                                Merge here
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
