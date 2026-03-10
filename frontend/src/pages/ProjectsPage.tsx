import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Plus, Folder, Video, Tag, Trash2, ArrowRight, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { useToast } from '@/components/ui/Toaster'
import { formatDate } from '@/lib/utils'
import type { Project } from '@/types'

export default function ProjectsPage() {
  const [showCreate, setShowCreate] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null)
  const [deleteConfirmName, setDeleteConfirmName] = useState('')
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const { t } = useTranslation()

  const { data: projects, isLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: api.projects.list,
  })

  const createMutation = useMutation({
    mutationFn: api.projects.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setShowCreate(false)
      setNewProjectName('')
      setNewProjectDesc('')
      toast({ title: t('projects.created'), type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('projects.failedCreate'), description: error.message, type: 'error' })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: api.projects.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setProjectToDelete(null)
      setDeleteConfirmName('')
      toast({ title: t('projects.movedToTrash'), type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('projects.failedDelete'), description: error.message, type: 'error' })
    },
  })

  const handleCreate = () => {
    if (!newProjectName.trim()) return
    createMutation.mutate({
      name: newProjectName,
      description: newProjectDesc,
    })
  }

  return (
    <div className="container max-w-6xl py-8 px-6 lg:px-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="font-display text-3xl font-bold">{t('projects.title')}</h1>
          <p className="text-muted-foreground mt-1">
            {t('projects.subtitle')}
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)} className="gap-2">
          <Plus className="h-4 w-4" />
          {t('projects.newProject')}
        </Button>
      </div>

      {/* Create project dialog */}
      {showCreate && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Card className="border-primary">
            <CardHeader>
              <CardTitle>{t('projects.createTitle')}</CardTitle>
              <CardDescription>
                {t('projects.createSubtitle')}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">{t('projects.nameLabel')}</label>
                <Input
                  placeholder={t('projects.namePlaceholder')}
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">{t('projects.descLabel')}</label>
                <Input
                  placeholder={t('projects.descPlaceholder')}
                  value={newProjectDesc}
                  onChange={(e) => setNewProjectDesc(e.target.value)}
                />
              </div>
              <div className="flex gap-2 pt-2">
                <Button onClick={handleCreate} disabled={!newProjectName.trim()}>
                  {t('projects.createBtn')}
                </Button>
                <Button variant="ghost" onClick={() => setShowCreate(false)}>
                  {t('common.cancel')}
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Projects grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardContent className="p-6 h-48" />
            </Card>
          ))}
        </div>
      ) : projects?.length === 0 ? (
        <Card className="border-dashed">
          <CardContent className="flex flex-col items-center justify-center py-16">
            <Folder className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="font-semibold text-lg mb-2">{t('projects.noProjects')}</h3>
            <p className="text-muted-foreground text-center mb-4">
              {t('projects.noProjectsDesc')}
            </p>
            <Button onClick={() => setShowCreate(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              {t('projects.createBtn')}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects?.map((project, i) => (
            <motion.div
              key={project.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <ProjectCard
                project={project}
                onDelete={() => setProjectToDelete(project)}
              />
            </motion.div>
          ))}
        </div>
      )}

      {/* Delete project confirmation */}
      {projectToDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => {
            setProjectToDelete(null)
            setDeleteConfirmName('')
          }}
        >
          <Card
            className="w-full max-w-md shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-destructive">{t('projects.deleteTitle')}</CardTitle>
                <CardDescription>
                  {t('projects.deleteDesc', { name: projectToDelete.name })}
                </CardDescription>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setProjectToDelete(null)
                  setDeleteConfirmName('')
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {t('projects.deleteConfirmLabel', { name: projectToDelete.name })}
              </p>
              <Input
                placeholder={projectToDelete.name}
                value={deleteConfirmName}
                onChange={(e) => setDeleteConfirmName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && deleteConfirmName === projectToDelete.name) {
                    deleteMutation.mutate(projectToDelete.name)
                  }
                  if (e.key === 'Escape') {
                    setProjectToDelete(null)
                    setDeleteConfirmName('')
                  }
                }}
                className="font-mono"
                autoFocus
              />
              <div className="flex gap-2 pt-2">
                <Button
                  variant="destructive"
                  disabled={deleteConfirmName !== projectToDelete.name || deleteMutation.isPending}
                  onClick={() => deleteMutation.mutate(projectToDelete.name)}
                  className="gap-2"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  {t('projects.moveToTrash')}
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setProjectToDelete(null)
                    setDeleteConfirmName('')
                  }}
                >
                  {t('common.cancel')}
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function ProjectCard({ project, onDelete }: { project: Project; onDelete: () => void }) {
  const { t } = useTranslation()

  return (
    <Card className="group hover:border-primary/50 transition-colors">
      <CardContent className="p-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Folder className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-semibold">{project.name}</h3>
              <p className="text-sm text-muted-foreground">
                {formatDate(project.created_at)}
              </p>
            </div>
          </div>
          <button
            onClick={(e) => {
              e.preventDefault()
              onDelete()
            }}
            className="opacity-0 group-hover:opacity-100 p-2 hover:bg-destructive/10 hover:text-destructive rounded transition-all"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>

        {project.description && (
          <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
            {project.description}
          </p>
        )}

        <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
          <span className="flex items-center gap-1.5">
            <Video className="h-4 w-4" />
            {t('projects.videoCount', { count: project.video_count })}
          </span>
          <span className="flex items-center gap-1.5">
            <Tag className="h-4 w-4" />
            {t('projects.labelCount', { count: project.annotation_count })}
          </span>
        </div>

        {project.classes.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-4">
            {project.classes.slice(0, 3).map((cls) => (
              <span
                key={cls}
                className="px-2 py-0.5 text-xs rounded-full bg-primary/10 text-primary"
              >
                {cls}
              </span>
            ))}
            {project.classes.length > 3 && (
              <span className="px-2 py-0.5 text-xs rounded-full bg-muted text-muted-foreground">
                {t('common.more', { count: project.classes.length - 3 })}
              </span>
            )}
          </div>
        )}

        <Link
          to={`/projects/${project.name}`}
          className="flex items-center gap-2 text-sm font-medium text-primary hover:underline"
        >
          {t('common.openProject')}
          <ArrowRight className="h-4 w-4" />
        </Link>
      </CardContent>
    </Card>
  )
}
