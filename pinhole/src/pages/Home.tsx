import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const topics = [
  {
    to: '/pinhole-model',
    color: 'cyan',
    badge: 'FUNDAMENTALS',
    title: 'The Pinhole Camera Model',
    description:
      'Understand how a pinhole camera works: how light rays project through a single point to form an image, the geometry of similar triangles, and the foundational equations that govern image formation.',
    tags: ['Similar Triangles', 'Projection', 'Focal Length', 'Image Formation'],
    icon: (
      <svg viewBox="0 0 64 64" className="w-full h-full">
        <rect x="4" y="16" width="24" height="32" rx="2" fill="none" stroke="currentColor" strokeWidth="2" />
        <circle cx="16" cy="32" r="6" fill="none" stroke="currentColor" strokeWidth="2" />
        <circle cx="16" cy="32" r="2" fill="currentColor" />
        <line x1="28" y1="22" x2="56" y2="42" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3" />
        <line x1="28" y1="42" x2="56" y2="22" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3" />
        <line x1="56" y1="18" x2="56" y2="46" stroke="currentColor" strokeWidth="2" />
      </svg>
    ),
  },
  {
    to: '/depth-model',
    color: 'purple',
    badge: 'APPLICATION',
    title: 'Z-Axis Depth Estimation',
    description:
      'Derive the formula Z = k/s step-by-step. Learn how to calibrate a single constant k using one known measurement, then estimate the distance of any object from its pixel size alone.',
    tags: ['Z = k/s', 'Calibration', 'Inverse Curve', 'Error Analysis'],
    icon: (
      <svg viewBox="0 0 64 64" className="w-full h-full">
        <line x1="8" y1="56" x2="56" y2="56" stroke="currentColor" strokeWidth="2" />
        <line x1="8" y1="56" x2="8" y2="8" stroke="currentColor" strokeWidth="2" />
        <path d="M12 12 Q20 14 28 30 Q36 46 56 52" fill="none" stroke="currentColor" strokeWidth="2.5" />
        <circle cx="20" cy="18" r="3" fill="currentColor" />
        <circle cx="36" cy="40" r="3" fill="currentColor" />
        <circle cx="52" cy="50" r="3" fill="currentColor" />
      </svg>
    ),
  },
];

const colorMap: Record<string, { border: string; bg: string; text: string; badge: string; tagBg: string; tagText: string; hoverBorder: string; iconText: string }> = {
  cyan: {
    border: 'border-cyan-800/50',
    bg: 'bg-cyan-950/20',
    text: 'text-cyan-400',
    badge: 'bg-cyan-950/50 border-cyan-800/50 text-cyan-400',
    tagBg: 'bg-cyan-900/30',
    tagText: 'text-cyan-300',
    hoverBorder: 'hover:border-cyan-600/70',
    iconText: 'text-cyan-500',
  },
  purple: {
    border: 'border-purple-800/50',
    bg: 'bg-purple-950/20',
    text: 'text-purple-400',
    badge: 'bg-purple-950/50 border-purple-800/50 text-purple-400',
    tagBg: 'bg-purple-900/30',
    tagText: 'text-purple-300',
    hoverBorder: 'hover:border-purple-600/70',
    iconText: 'text-purple-500',
  },
};

export function Home() {
  return (
    <>
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="pt-24 pb-16 px-6 text-center max-w-4xl mx-auto"
      >
        <div className="inline-block px-4 py-1 rounded-full bg-cyan-950/50 border border-cyan-800/50 text-cyan-400 text-sm font-bold tracking-widest mb-6">
          INTERACTIVE TUTORIALS
        </div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8">
          Pinhole Camera{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
            Explainers
          </span>
        </h1>
        <p className="text-xl text-slate-400 font-light leading-relaxed max-w-2xl mx-auto">
          Interactive guides covering the theory and application of pinhole camera geometry
          for our crane tracking system.
        </p>
      </motion.header>

      <main className="max-w-5xl mx-auto px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {topics.map((topic, i) => {
            const c = colorMap[topic.color];
            return (
              <motion.div
                key={topic.to}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: i * 0.15 }}
              >
                <Link
                  to={topic.to}
                  className={`block group rounded-3xl border ${c.border} ${c.hoverBorder} ${c.bg} p-8 transition-all duration-300 hover:shadow-2xl hover:-translate-y-1 h-full`}
                >
                  <div className={`w-16 h-16 mb-6 ${c.iconText} opacity-70 group-hover:opacity-100 transition-opacity`}>
                    {topic.icon}
                  </div>

                  <span className={`inline-block px-3 py-0.5 rounded-full border text-xs font-bold tracking-widest mb-4 ${c.badge}`}>
                    {topic.badge}
                  </span>

                  <h2 className="text-2xl font-bold text-white mb-3 group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-cyan-400 group-hover:to-purple-400 transition-all">
                    {topic.title}
                  </h2>

                  <p className="text-slate-400 leading-relaxed mb-6 text-sm">
                    {topic.description}
                  </p>

                  <div className="flex flex-wrap gap-2">
                    {topic.tags.map((tag) => (
                      <span key={tag} className={`px-2.5 py-1 rounded-md text-xs font-mono ${c.tagBg} ${c.tagText}`}>
                        {tag}
                      </span>
                    ))}
                  </div>

                  <div className={`mt-6 flex items-center gap-2 text-sm font-semibold ${c.text} group-hover:gap-3 transition-all`}>
                    Explore
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                  </div>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </main>
    </>
  );
}
