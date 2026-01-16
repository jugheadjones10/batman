import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Plus, Folder, Video, Tag, Trash2, ArrowRight } from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { useToast } from '@/components/ui/Toaster'
import { formatDate, formatNumber } from '@/lib/utils'
import type { Project } from '@/types'

export default function ProjectsPage() {
  const [showCreate, setShowCreate] = useState(false)
  const [newProjectName, setNewProjectName] = useState('')
  const [newProjectDesc, setNewProjectDesc] = useState('')
  const queryClient = useQueryClient()
  const { toast } = useToast()

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
      toast({ title: 'Project created', type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Failed to create project', description: error.message, type: 'error' })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: api.projects.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      toast({ title: 'Project deleted', type: 'success' })
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
          <h1 className="font-display text-3xl font-bold">Projects</h1>
          <p className="text-muted-foreground mt-1">
            Manage your video labeling projects
          </p>
        </div>
        <Button onClick={() => setShowCreate(true)} className="gap-2">
          <Plus className="h-4 w-4" />
          New Project
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
              <CardTitle>Create New Project</CardTitle>
              <CardDescription>
                Set up a new project for video labeling
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Project Name</label>
                <Input
                  placeholder="My Detection Project"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                />
              </div>
              <div>
                <label className="text-sm font-medium mb-2 block">Description (optional)</label>
                <Input
                  placeholder="Brief description of your project"
                  value={newProjectDesc}
                  onChange={(e) => setNewProjectDesc(e.target.value)}
                />
              </div>
              <div className="flex gap-2 pt-2">
                <Button onClick={handleCreate} disabled={!newProjectName.trim()}>
                  Create Project
                </Button>
                <Button variant="ghost" onClick={() => setShowCreate(false)}>
                  Cancel
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
            <h3 className="font-semibold text-lg mb-2">No projects yet</h3>
            <p className="text-muted-foreground text-center mb-4">
              Create your first project to start labeling videos
            </p>
            <Button onClick={() => setShowCreate(true)} className="gap-2">
              <Plus className="h-4 w-4" />
              Create Project
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
                onDelete={() => deleteMutation.mutate(project.name)}
              />
            </motion.div>
          ))}
        </div>
      )}
    </div>
  )
}

function ProjectCard({ project, onDelete }: { project: Project; onDelete: () => void }) {
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
            {formatNumber(project.video_count)} videos
          </span>
          <span className="flex items-center gap-1.5">
            <Tag className="h-4 w-4" />
            {formatNumber(project.annotation_count)} labels
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
                +{project.classes.length - 3} more
              </span>
            )}
          </div>
        )}

        <Link
          to={`/projects/${project.name}`}
          className="flex items-center gap-2 text-sm font-medium text-primary hover:underline"
        >
          Open Project
          <ArrowRight className="h-4 w-4" />
        </Link>
      </CardContent>
    </Card>
  )
}

