import type { ReactNode } from 'react';
import { motion } from 'framer-motion';

interface SectionProps {
  id: string;
  title: string;
  children: ReactNode;
}

export const SectionWrapper = ({ id, title, children }: SectionProps) => {
  return (
    <motion.section
      id={id}
      className="min-h-screen w-full flex flex-col justify-center items-center py-24 px-4 sm:px-8 max-w-5xl mx-auto"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <div className="w-full">
        <h2 className="text-3xl md:text-5xl font-bold mb-12 text-slate-100 tracking-tight border-b border-slate-700 pb-4">
          {title}
        </h2>
        <div className="flex flex-col lg:flex-row gap-12 items-start justify-between">
          {children}
        </div>
      </div>
    </motion.section>
  );
};
