import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { MathBlock } from '../components/MathBlock';
import { motion, AnimatePresence } from 'framer-motion';

export function Section3() {
  const [step, setStep] = useState(1);
  
  const steps = [
    { math: "s = \\frac{f \\times S}{Z}", desc: "Start with the pinhole model." },
    { math: "Z \\times s = f \\times S", desc: "Multiply both sides by Z." },
    { math: "Z = \\frac{f \\times S}{s}", desc: "Divide by s. Now we're solving for distance." },
    { math: "k = f \\times S", desc: "Notice that focal length (f) and object size (S) never change. We can merge them." },
    { math: "Z = \\frac{k}{s}", desc: "Our final, simplified depth equation." }
  ];

  return (
    <SectionContainer id="section3" title="Rearranging for Z" subtitle="Phase 3">
      <ContentBlock>
        <p className="mb-6">
          We don't want to find the apparent pixel size <strong className="text-cyan-400">s</strong>; we can measure that directly from our image. We want to find the physical distance <strong className="text-purple-400">Z</strong>.
        </p>

        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 min-h-[120px] flex items-center justify-center relative overflow-hidden shadow-inner my-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="text-center"
            >
              <MathBlock math={steps[step - 1].math} className="text-emerald-400 text-3xl" />
              <p className="text-sm text-slate-400 mt-2">{steps[step - 1].desc}</p>
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="flex gap-2 justify-between mt-4">
          <button 
            disabled={step === 1}
            onClick={() => setStep(s => Math.max(1, s - 1))}
            className="px-6 py-3 bg-slate-800 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-700 transition-colors font-semibold"
          >
            Back
          </button>
          <div className="flex gap-2 items-center">
            {[1, 2, 3, 4, 5].map(i => (
              <div 
                key={i} 
                className={`w-3 h-3 rounded-full transition-all duration-300 ${i === step ? 'bg-emerald-400 scale-125' : 'bg-slate-700'}`}
              />
            ))}
          </div>
          <button 
            disabled={step === 5}
            onClick={() => setStep(s => Math.min(5, s + 1))}
            className="px-6 py-3 bg-emerald-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-emerald-500 transition-colors font-semibold"
          >
            Next
          </button>
        </div>
      </ContentBlock>

      <DiagramBlock>
        <div className="relative w-full h-full flex flex-col items-center justify-center">
          <motion.div 
            animate={{ 
              scale: step >= 4 ? 0 : 1,
              opacity: step >= 4 ? 0 : 1 
            }}
            transition={{ duration: 0.5 }}
            className="flex gap-16 absolute top-1/4"
          >
            <div className="w-24 h-24 rounded-2xl bg-slate-800 flex items-center justify-center border-2 border-slate-600 shadow-lg">
              <span className="text-4xl font-mono text-slate-300">f</span>
            </div>
            <div className="w-24 h-24 rounded-2xl bg-slate-800 flex items-center justify-center border-2 border-slate-600 shadow-lg">
              <span className="text-4xl font-mono text-slate-300">S</span>
            </div>
          </motion.div>
          
          <motion.div 
            animate={{ 
              scale: step >= 4 ? 1 : 0.5,
              opacity: step >= 4 ? 1 : 0,
              y: step >= 4 ? 0 : 50
            }}
            transition={{ duration: 0.5, type: 'spring' }}
            className="w-48 h-48 rounded-3xl bg-emerald-900/50 flex items-center justify-center border-4 border-emerald-500 shadow-2xl absolute"
          >
            <span className="text-6xl font-mono text-emerald-400 font-bold">k</span>
          </motion.div>
          
          <div className="absolute bottom-8 text-center w-full">
            <AnimatePresence mode="wait">
              {step >= 4 ? (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-emerald-400 font-mono text-xl"
                >
                  k is the "camera constant"
                </motion.p>
              ) : (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-slate-400 font-mono text-xl"
                >
                  f and S are separate variables
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        </div>
      </DiagramBlock>
    </SectionContainer>
  );
}
