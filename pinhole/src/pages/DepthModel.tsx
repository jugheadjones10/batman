import { Section1 } from '../sections/Section1';
import { Section2 } from '../sections/Section2';
import { Section3 } from '../sections/Section3';
import { Section4 } from '../sections/Section4';
import { Section5 } from '../sections/Section5';
import { Section6 } from '../sections/Section6';
import { Section7 } from '../sections/Section7';
import { motion } from 'framer-motion';

export function DepthModel() {
  return (
    <>
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="pt-24 pb-16 px-6 text-center max-w-4xl mx-auto"
      >
        <div className="inline-block px-4 py-1 rounded-full bg-purple-950/50 border border-purple-800/50 text-purple-400 text-sm font-bold tracking-widest mb-6">
          APPLICATION
        </div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8">
          Z-Axis{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
            Depth Estimation
          </span>
        </h1>
        <p className="text-xl text-slate-400 font-light leading-relaxed">
          How to estimate the distance of an object using nothing but its apparent pixel size.
          Scroll down to explore the mathematics behind our top-down crane tracking system.
        </p>
      </motion.header>

      <main className="flex flex-col">
        <Section1 />
        <Section2 />
        <Section3 />
        <Section4 />
        <Section5 />
        <Section6 />
        <Section7 />
      </main>
    </>
  );
}
