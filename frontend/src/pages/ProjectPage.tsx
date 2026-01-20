import { useState, useRef } from 'react'
import { createPortal } from 'react-dom'
import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Upload,
  Video,
  Play,
  PenTool,
  Cpu,
  Plus,
  Trash2,
  Image,
  Tag,
  Loader2,
  BarChart3,
  ChevronRight,
  Download,
  ExternalLink,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { useToast } from '@/components/ui/Toaster'
import { formatDuration, formatNumber } from '@/lib/utils'
import type { Video as VideoType } from '@/types'

export default function ProjectPage() {
  const { projectName } = useParams<{ projectName: string }>()
  const [uploading, setUploading] = useState(false)
  const [newClass, setNewClass] = useState('')
  const [showImport, setShowImport] = useState(false)
  const [importConfig, setImportConfig] = useState({
    api_key: '',
    workspace: '',
    project: '',
    version: 1,
  })
  const [showImageGallery, setShowImageGallery] = useState<number | null>(null)
  const [galleryPage, setGalleryPage] = useState(0)
  const GALLERY_PAGE_SIZE = 100
  const [renamingClass, setRenamingClass] = useState<string | null>(null)
  const [newClassName, setNewClassName] = useState('')
  const [mergingClasses, setMergingClasses] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const { data: project, isLoading: projectLoading } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  const { data: videos, isLoading: videosLoading } = useQuery({
    queryKey: ['videos', projectName],
    queryFn: () => api.videos.list(projectName!),
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

  const { data: galleryImages, isLoading: galleryLoading } = useQuery({
    queryKey: ['gallery-images', projectName, showImageGallery, galleryPage],
    queryFn: () => api.import.listImages(projectName!, showImageGallery!, galleryPage * GALLERY_PAGE_SIZE, GALLERY_PAGE_SIZE),
    enabled: !!projectName && showImageGallery !== null,
  })

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.videos.upload(projectName!, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({ title: 'Video uploaded', type: 'success' })
      setUploading(false)
    },
    onError: (error: Error) => {
      toast({ title: 'Upload failed', description: error.message, type: 'error' })
      setUploading(false)
    },
  })

  const updateClassesMutation = useMutation({
    mutationFn: (classes: string[]) => api.projects.updateClasses(projectName!, classes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({ title: 'Classes updated', type: 'success' })
    },
  })

  const deleteVideoMutation = useMutation({
    mutationFn: (videoId: number) => api.videos.delete(projectName!, videoId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      toast({ title: 'Video deleted', type: 'success' })
    },
  })

  const [importProgress, setImportProgress] = useState<{
    active: boolean
    progress: number
    message: string
  }>({ active: false, progress: 0, message: '' })

  const handleImport = async () => {
    if (!projectName || !importConfig.api_key || !importConfig.workspace || !importConfig.project) {
      return
    }

    setImportProgress({ active: true, progress: 0, message: 'Starting import...' })

    try {
      let finalResult: any = null
      
      for await (const update of api.import.fromRoboflowWithProgress(projectName, importConfig)) {
        setImportProgress({
          active: true,
          progress: update.progress,
          message: update.message,
        })

        if (update.status === 'complete') {
          finalResult = update
        } else if (update.status === 'error') {
          throw new Error(update.message)
        }
      }

      if (finalResult) {
        queryClient.invalidateQueries({ queryKey: ['project', projectName] })
        queryClient.invalidateQueries({ queryKey: ['imported-datasets', projectName] })
        queryClient.invalidateQueries({ queryKey: ['class-details', projectName] })
        toast({
          title: 'Import successful',
          description: `${finalResult.images_imported} images, ${finalResult.annotations_imported} annotations imported`,
          type: 'success',
        })
        setShowImport(false)
        setImportConfig({ api_key: '', workspace: '', project: '', version: 1 })
      }
    } catch (error: any) {
      toast({ title: 'Import failed', description: error.message, type: 'error' })
    } finally {
      setImportProgress({ active: false, progress: 0, message: '' })
    }
  }

  const deleteDatasetMutation = useMutation({
    mutationFn: (videoId: number) => api.import.deleteDataset(projectName!, videoId),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['project', projectName] })
      queryClient.invalidateQueries({ queryKey: ['imported-datasets', projectName] })
      toast({ title: 'Dataset deleted', description: data.message, type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Delete failed', description: error.message, type: 'error' })
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
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })  // Refresh video annotation counts
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
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })  // Refresh video annotation counts
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

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    uploadMutation.mutate(file)
  }

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

  if (projectLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (!project) {
    return (
      <div className="container max-w-4xl py-8 px-6">
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
    <div className="container max-w-6xl py-8 px-6 lg:px-8">
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
        {/* Main content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Videos section */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Video className="h-5 w-5" />
                  Videos
                </CardTitle>
                <CardDescription>
                  {videos?.length || 0} video{(videos?.length || 0) !== 1 ? 's' : ''} in this project
                </CardDescription>
              </div>
              <div>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept="video/*"
                  className="hidden"
                />
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                  className="gap-2"
                >
                  {uploading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="h-4 w-4" />
                  )}
                  Upload Video
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {videosLoading ? (
                <div className="space-y-4">
                  {[1, 2].map((i) => (
                    <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : videos?.length === 0 ? (
                <div className="text-center py-12 border-2 border-dashed border-border rounded-lg">
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground mb-4">
                    No videos yet. Upload your first video to get started.
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    className="gap-2"
                  >
                    <Upload className="h-4 w-4" />
                    Upload Video
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  {videos?.map((video) => (
                    <VideoCard
                      key={video.id}
                      video={video}
                      projectName={projectName!}
                      onDelete={() => deleteVideoMutation.mutate(video.id)}
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
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
                      {/* Main row */}
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
                      
                      {/* Data sources row */}
                      {cls.annotation_sources && Object.keys(cls.annotation_sources).length > 0 && (
                        <div className="px-2 pb-2 pt-0 flex items-center gap-2 flex-wrap">
                          {Object.entries(cls.annotation_sources).map(([source, count]) => (
                            <span
                              key={source}
                              className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded ${
                                source === 'roboflow'
                                  ? 'bg-purple-500/20 text-purple-400'
                                  : source === 'local_coco'
                                  ? 'bg-blue-500/20 text-blue-400'
                                  : source === 'video'
                                  ? 'bg-green-500/20 text-green-400'
                                  : source === 'auto'
                                  ? 'bg-yellow-500/20 text-yellow-400'
                                  : 'bg-gray-500/20 text-gray-400'
                              }`}
                            >
                              {source === 'roboflow' && (
                                <svg className="w-2.5 h-2.5" viewBox="0 0 24 24" fill="currentColor">
                                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                                </svg>
                              )}
                              {source === 'video' && (
                                <Video className="w-2.5 h-2.5" />
                              )}
                              {source === 'auto' && (
                                <Cpu className="w-2.5 h-2.5" />
                              )}
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

          {/* Imported Datasets */}
          {importedDatasets && importedDatasets.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Image className="h-5 w-5" />
                  Imported Datasets
                </CardTitle>
                <CardDescription>
                  External images added to this project
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {importedDatasets.map((dataset) => (
                  <div key={dataset.video_id} className="border border-border rounded-lg p-3 overflow-hidden">
                    <div className="flex items-start gap-2 mb-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`text-xs px-2 py-0.5 rounded flex-shrink-0 ${
                            dataset.source === 'roboflow' 
                              ? 'bg-purple-500/20 text-purple-400' 
                              : 'bg-blue-500/20 text-blue-400'
                          }`}>
                            {dataset.source}
                          </span>
                          <span className="text-sm font-medium">
                            {dataset.image_count} images
                          </span>
                          <span className="text-xs text-muted-foreground">
                            ({dataset.annotation_count} annotations)
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-1 flex-shrink-0">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-xs"
                          onClick={() => {
                            setGalleryPage(0)
                            setShowImageGallery(dataset.video_id)
                          }}
                        >
                          View
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 p-0 text-destructive hover:text-destructive"
                          onClick={() => {
                            if (confirm(`Delete ${dataset.image_count} imported images?`)) {
                              deleteDatasetMutation.mutate(dataset.video_id)
                            }
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                    {/* Sample images */}
                    <div className="grid grid-cols-3 gap-1">
                      {dataset.sample_images.slice(0, 6).map((url, i) => (
                        <img
                          key={i}
                          src={`${url.startsWith('/') ? 'http://localhost:8000' : ''}${url}`}
                          alt={`Sample ${i + 1}`}
                          className="w-full h-16 object-cover rounded cursor-pointer hover:opacity-80"
                          onClick={() => {
                            setGalleryPage(0)
                            setShowImageGallery(dataset.video_id)
                          }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Image Gallery Modal - rendered via portal to escape stacking context */}
          {showImageGallery !== null && createPortal(
            <div className="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center p-8">
              <div className="bg-background rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                <div className="flex items-center justify-between p-4 border-b border-border flex-shrink-0">
                  <h3 className="font-semibold">
                    Imported Images {galleryImages && `(${galleryImages.total} total)`}
                  </h3>
                  <Button variant="ghost" size="sm" onClick={() => {
                    setShowImageGallery(null)
                    setGalleryPage(0)
                  }}>
                    ✕
                  </Button>
                </div>
                <div className="p-4 overflow-y-auto overflow-x-hidden flex-1 min-h-0">
                  {galleryLoading ? (
                    <div className="flex items-center justify-center py-12">
                      <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    </div>
                  ) : galleryImages ? (
                    <div className="grid grid-cols-4 gap-2 w-full">
                      {galleryImages.images.map((img) => (
                        <div key={img.frame_id} className="relative group">
                          <img
                            src={`http://localhost:8000${img.url}`}
                            alt={img.original_filename}
                            className="w-full h-24 object-cover rounded"
                          />
                          <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity rounded flex items-center justify-center">
                            <span className="text-xs text-white text-center px-1">
                              {img.original_filename || `Frame ${img.frame_id}`}
                            </span>
                          </div>
                          {img.split && (
                            <span className="absolute top-1 right-1 text-[10px] bg-black/50 text-white px-1 rounded">
                              {img.split}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
                {/* Pagination controls */}
                {galleryImages && galleryImages.total > GALLERY_PAGE_SIZE && (
                  <div className="flex items-center justify-between p-4 border-t border-border flex-shrink-0">
                    <span className="text-sm text-muted-foreground">
                      Showing {galleryPage * GALLERY_PAGE_SIZE + 1}-{Math.min((galleryPage + 1) * GALLERY_PAGE_SIZE, galleryImages.total)} of {galleryImages.total}
                    </span>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={galleryPage === 0 || galleryLoading}
                        onClick={() => setGalleryPage(p => p - 1)}
                      >
                        Previous
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={(galleryPage + 1) * GALLERY_PAGE_SIZE >= galleryImages.total || galleryLoading}
                        onClick={() => setGalleryPage(p => p + 1)}
                      >
                        Next
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </div>,
            document.body
          )}

          {/* Import Dataset */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="h-5 w-5" />
                Import Dataset
              </CardTitle>
              <CardDescription>
                Add images from Roboflow
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!showImport ? (
                <Button
                  variant="outline"
                  className="w-full gap-2"
                  onClick={() => setShowImport(true)}
                >
                  <ExternalLink className="h-4 w-4" />
                  Import from Roboflow
                </Button>
              ) : (
                <div className="space-y-3">
                  <Input
                    placeholder="API Key"
                    type="password"
                    value={importConfig.api_key}
                    onChange={(e) => setImportConfig({ ...importConfig, api_key: e.target.value })}
                  />
                  <Input
                    placeholder="Workspace (e.g. my-workspace)"
                    value={importConfig.workspace}
                    onChange={(e) => setImportConfig({ ...importConfig, workspace: e.target.value })}
                  />
                  <Input
                    placeholder="Project (e.g. my-project)"
                    value={importConfig.project}
                    onChange={(e) => setImportConfig({ ...importConfig, project: e.target.value })}
                  />
                  <Input
                    placeholder="Version"
                    type="number"
                    min={1}
                    value={importConfig.version}
                    onChange={(e) => setImportConfig({ ...importConfig, version: parseInt(e.target.value) || 1 })}
                  />
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      className="flex-1"
                      onClick={() => setShowImport(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      className="flex-1 gap-2 relative overflow-hidden"
                      onClick={handleImport}
                      disabled={importProgress.active || !importConfig.api_key || !importConfig.workspace || !importConfig.project}
                    >
                      {importProgress.active && (
                        <div 
                          className="absolute inset-0 bg-primary/30 transition-all duration-300"
                          style={{ width: `${importProgress.progress}%` }}
                        />
                      )}
                      <span className="relative flex items-center gap-2">
                        {importProgress.active ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            {importProgress.progress}%
                          </>
                        ) : (
                          <>
                            <Download className="h-4 w-4" />
                            Import
                          </>
                        )}
                      </span>
                    </Button>
                  </div>
                  {importProgress.active && (
                    <p className="text-xs text-muted-foreground animate-pulse">
                      {importProgress.message}
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground">
                    Get your API key at{' '}
                    <a
                      href="https://app.roboflow.com/settings/api"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline"
                    >
                      roboflow.com/settings/api
                    </a>
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function VideoCard({
  video,
  projectName,
  onDelete,
}: {
  video: VideoType
  projectName: string
  onDelete: () => void
}) {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const extractMutation = useMutation({
    mutationFn: () => api.videos.extractFrames(projectName, video.id),
    onSuccess: (frames) => {
      queryClient.invalidateQueries({ queryKey: ['videos', projectName] })
      toast({
        title: 'Frames extracted',
        description: `${frames.length} frames extracted`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Extraction failed', description: error.message, type: 'error' })
    },
  })

  return (
    <div className="group rounded-lg border border-border hover:border-primary/30 transition-colors overflow-hidden">
      {/* Video preview row */}
      <div className="flex items-center gap-4 p-4">
        <div className="relative w-36 h-24 bg-muted rounded overflow-hidden flex-shrink-0">
          <video
            src={api.videos.streamUrl(projectName, video.id)}
            className="w-full h-full object-cover"
            muted
            preload="metadata"
          />
          <div className="absolute bottom-1 right-1 px-1.5 py-0.5 bg-black/70 rounded text-xs">
            {formatDuration(video.duration)}
          </div>
        </div>

        <div className="flex-1 min-w-0">
          <h4 className="font-medium truncate text-lg">{video.filename}</h4>
          <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
            <span>{video.width}×{video.height}</span>
            <span>{video.fps.toFixed(1)} fps</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {video.frame_count === 0 ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => extractMutation.mutate()}
              disabled={extractMutation.isPending}
              className="gap-1"
            >
              {extractMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Image className="h-4 w-4" />
              )}
              Extract Frames
            </Button>
          ) : (
            <Link to={`/projects/${projectName}/annotate?video=${video.id}`}>
              <Button size="sm" className="gap-1">
                <PenTool className="h-4 w-4" />
                Annotate
                <ChevronRight className="h-4 w-4" />
              </Button>
            </Link>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={onDelete}
            className="text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-6 px-4 py-3 bg-muted/30 border-t border-border text-sm">
        <div className="flex items-center gap-2">
          <Image className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Frames:</span>
          <span className="font-medium">{formatNumber(video.frame_count)}</span>
        </div>
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground">Annotations:</span>
          <span className="font-medium">{formatNumber(video.annotation_count)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">Total frames:</span>
          <span className="font-medium">{formatNumber(video.total_frames)}</span>
        </div>
      </div>
    </div>
  )
}
