import { Outlet, Link, useLocation } from 'react-router-dom'
import { Folder, ChevronRight } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { setLanguage, type Language } from '@/i18n/i18n'

function LanguageToggle() {
  const { i18n } = useTranslation()
  const current = i18n.language as Language

  const toggle = () => setLanguage(current === 'en' ? 'ko' : 'en')

  return (
    <button
      onClick={toggle}
      className="flex items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:border-primary/50 hover:text-foreground"
      title={current === 'en' ? '한국어로 전환' : 'Switch to English'}
    >
      <span className={current === 'en' ? 'text-foreground font-semibold' : 'opacity-50'}>EN</span>
      <span className="text-border mx-0.5">/</span>
      <span className={current === 'ko' ? 'text-foreground font-semibold' : 'opacity-50'}>한국어</span>
    </button>
  )
}

type CrumbItem = { label: string; to?: string }

function Breadcrumbs() {
  const { pathname } = useLocation()
  const parts = pathname.split('/').filter(Boolean)

  const isVideoAnnotate =
    parts[0] === 'projects' && parts[2] === 'annotate' && parts[3] === 'video' && !!parts[4]

  const projectNameEncoded = parts[1]
  const projectName = projectNameEncoded ? decodeURIComponent(projectNameEncoded) : undefined
  const videoId = parts[4] ? decodeURIComponent(parts[4]) : undefined

  const { data: video } = useQuery({
    queryKey: ['video', projectName, videoId],
    queryFn: () => api.videos.get(projectName!, videoId!),
    enabled: isVideoAnnotate && !!projectName && !!videoId,
  })

  if (parts[0] !== 'projects' || !parts[1]) return null

  const crumbs: CrumbItem[] = [{ label: 'projects', to: '/projects' }]
  const projectPath = `/projects/${projectNameEncoded}`
  const section = parts[2]

  if (!section) {
    crumbs.push({ label: projectName! })
  } else {
    crumbs.push({ label: projectName!, to: projectPath })

    if (section === 'train') {
      crumbs.push({ label: 'Training' })
    } else if (section === 'inference') {
      crumbs.push({ label: 'Inference' })
    } else if (section === 'annotate') {
      if (isVideoAnnotate) {
        crumbs.push({ label: video?.filename ?? videoId! })
      } else {
        crumbs.push({ label: 'Annotate' })
      }
    }
  }

  return (
    <nav className="flex items-center gap-1 text-sm text-muted-foreground min-w-0">
      <ChevronRight className="h-4 w-4 flex-shrink-0" />
      {crumbs.map((crumb, i) => {
        const isLast = i === crumbs.length - 1
        return (
          <span key={i} className="flex items-center gap-1 min-w-0">
            {i > 0 && <ChevronRight className="h-4 w-4 flex-shrink-0" />}
            {crumb.to ? (
              <Link
                to={crumb.to}
                className="hover:text-foreground transition-colors truncate"
                title={crumb.label}
              >
                {crumb.label}
              </Link>
            ) : (
              <span
                className={`truncate ${isLast ? 'text-foreground font-medium' : ''}`}
                title={crumb.label}
              >
                {crumb.label}
              </span>
            )}
          </span>
        )
      })}
    </nav>
  )
}

export default function Layout() {
  const { t } = useTranslation()

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="flex h-16 items-center justify-between px-6">
          <div className="flex items-center gap-4 min-w-0">
            <Link to="/" className="flex items-center gap-3 group flex-shrink-0">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground transition-transform group-hover:scale-110">
                <svg viewBox="0 0 100 100" className="h-6 w-6">
                  <path 
                    d="M 50 25 L 20 55 L 35 55 L 35 75 L 50 60 L 65 75 L 65 55 L 80 55 Z" 
                    fill="currentColor"
                  />
                </svg>
              </div>
              <span className="font-display text-xl font-bold tracking-tight">
                Batman
              </span>
            </Link>

            <Breadcrumbs />
          </div>

          <nav className="flex items-center gap-3 flex-shrink-0">
            <LanguageToggle />
            <NavLink to="/projects" icon={<Folder className="h-4 w-4" />}>
              {t('nav.projects')}
            </NavLink>
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1">
        <Outlet />
      </main>
    </div>
  )
}

function NavLink({ 
  to, 
  icon, 
  children 
}: { 
  to: string
  icon: React.ReactNode
  children: React.ReactNode 
}) {
  const location = useLocation()
  const isActive = location.pathname.startsWith(to)

  return (
    <Link
      to={to}
      className={`
        flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors
        ${isActive 
          ? 'bg-primary text-primary-foreground' 
          : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
        }
      `}
    >
      {icon}
      {children}
    </Link>
  )
}
