import { Outlet, Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Folder, 
  Play, 
  PenTool, 
  Cpu, 
  Settings,
  ChevronRight
} from 'lucide-react'

export default function Layout() {
  const location = useLocation()
  const pathParts = location.pathname.split('/').filter(Boolean)

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur-lg">
        <div className="flex h-16 items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-3 group">
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

            {/* Breadcrumb */}
            {pathParts.length > 1 && (
              <nav className="flex items-center gap-1 text-sm text-muted-foreground">
                <ChevronRight className="h-4 w-4" />
                {pathParts.map((part, i) => (
                  <span key={i} className="flex items-center gap-1">
                    {i > 0 && <ChevronRight className="h-4 w-4" />}
                    <Link 
                      to={`/${pathParts.slice(0, i + 1).join('/')}`}
                      className="hover:text-foreground transition-colors"
                    >
                      {part}
                    </Link>
                  </span>
                ))}
              </nav>
            )}
          </div>

          <nav className="flex items-center gap-2">
            <NavLink to="/projects" icon={<Folder className="h-4 w-4" />}>
              Projects
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

