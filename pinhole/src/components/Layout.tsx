import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

export function Layout({ children }: { children: ReactNode }) {
  const location = useLocation();
  const isHome = location.pathname === '/';

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-cyan-900 selection:text-cyan-50">
      <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800/50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link
            to="/"
            className="flex items-center gap-3 group"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-500 flex items-center justify-center text-white font-bold text-sm">
              P
            </div>
            <span className="font-bold text-white tracking-tight group-hover:text-cyan-400 transition-colors">
              Pinhole Explainers
            </span>
          </Link>
          {!isHome && (
            <motion.div
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <Link
                to="/"
                className="text-sm text-slate-400 hover:text-cyan-400 transition-colors flex items-center gap-2"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
                All Topics
              </Link>
            </motion.div>
          )}
        </div>
      </nav>

      <div className="pt-16">
        {children}
      </div>

      <footer className="py-12 text-center text-slate-500 border-t border-slate-800/50 mt-24">
        <p className="font-mono text-sm">Pinhole Camera Explainers</p>
      </footer>
    </div>
  );
}
