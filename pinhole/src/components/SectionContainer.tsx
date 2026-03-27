import type { ReactNode } from 'react';
import { motion } from 'framer-motion';

interface SectionContainerProps {
  id: string;
  title: string;
  children: ReactNode;
  subtitle?: string;
}

export function SectionContainer({ id, title, subtitle, children }: SectionContainerProps) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-20%' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className="min-h-screen py-32 px-6 md:px-12 flex flex-col justify-center items-center border-b border-slate-800/50 relative overflow-hidden"
    >
      <div className="max-w-6xl w-full flex flex-col lg:flex-row gap-16 items-center">
        <div className="flex flex-col gap-6 flex-1 w-full lg:w-1/2">
          <header className="mb-4">
            {subtitle && (
              <span className="text-sm font-bold tracking-widest uppercase text-cyan-400 mb-2 block">
                {subtitle}
              </span>
            )}
            <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-6">
              {title}
            </h2>
          </header>
          {Array.isArray(children) ? children[0] : children}
        </div>
        <div className="flex-1 w-full lg:w-1/2">
          {Array.isArray(children) ? children[1] : null}
        </div>
      </div>
    </motion.section>
  );
}

export function ContentBlock({ children }: { children: ReactNode }) {
  return <div className="text-slate-300 text-lg md:text-xl leading-relaxed font-light">{children}</div>;
}

export function DiagramBlock({ children }: { children: ReactNode }) {
  return (
    <div className="w-full h-[500px] md:h-[600px] bg-slate-800/50 rounded-3xl border border-slate-700/50 shadow-2xl overflow-hidden flex flex-col items-center justify-center p-8 relative ring-1 ring-white/10 backdrop-blur-sm">
      {children}
    </div>
  );
}
