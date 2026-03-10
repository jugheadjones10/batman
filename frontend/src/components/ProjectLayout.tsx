import { Outlet, Link, useLocation, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useTranslation } from 'react-i18next'
import { Database, Cpu, Play, Loader2 } from 'lucide-react'
import { api } from '@/api/client'
import { Card, CardContent } from '@/components/ui/Card'
import { cn } from '@/lib/utils'

export default function ProjectLayout() {
  const { projectName } = useParams<{ projectName: string }>()
  const location = useLocation()
  const { t } = useTranslation()

  const { data: project, isLoading } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  const tabs = [
    {
      label: t('nav.tabDataCollection'),
      icon: <Database className="h-4 w-4" />,
      to: `/projects/${projectName}`,
      exact: true,
    },
    {
      label: t('nav.tabTraining'),
      icon: <Cpu className="h-4 w-4" />,
      to: `/projects/${projectName}/train`,
      exact: false,
    },
    {
      label: t('nav.tabInference'),
      icon: <Play className="h-4 w-4" />,
      to: `/projects/${projectName}/inference`,
      exact: false,
    },
  ]

  const decodedPath = decodeURIComponent(location.pathname)

  const isTabActive = (tab: (typeof tabs)[number]) => {
    if (tab.exact) {
      return decodedPath === tab.to || decodedPath === tab.to + '/'
    }
    return decodedPath.startsWith(tab.to)
  }

  if (isLoading) {
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
            <h2 className="text-xl font-semibold mb-2">{t('project.notFound')}</h2>
            <Link to="/projects" className="text-primary hover:underline">
              {t('project.backToProjects')}
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div>
      {/* Project header + tab bar */}
      <div className="container max-w-6xl mx-auto pt-8 pb-4 px-6 lg:px-8">
        <div className="mb-5">
          <h1 className="font-display text-3xl font-bold">{project.name}</h1>
          {project.description && (
            <p className="text-muted-foreground mt-1">{project.description}</p>
          )}
        </div>

        <div className="flex gap-1 p-1 bg-muted rounded-lg w-fit">
          {tabs.map((tab) => {
            const active = isTabActive(tab)
            return (
              <Link
                key={tab.to}
                to={tab.to}
                className={cn(
                  'flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
                  active
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                )}
              >
                {tab.icon}
                {tab.label}
              </Link>
            )
          })}
        </div>
      </div>

      <div className="border-b border-border" />

      {/* Tab content — each page owns its own container/padding */}
      <Outlet />
    </div>
  )
}
