import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { MathBlock } from '../components/MathBlock';
import { Slider } from '../components/Slider';

export function Section5() {
  const [z1, setZ1] = useState(500);
  const [s1, setS1] = useState(400);
  const [newS, setNewS] = useState(50);

  const k = z1 * s1;
  const predictedZ = newS > 0 ? k / newS : 0;

  return (
    <SectionContainer id="section5" title="Calibration: Finding k" subtitle="Phase 5">
      <ContentBlock>
        <p className="mb-6">
          We don't need to know the camera's exact focal length or the hook's true size in millimeters. 
          We only need <strong className="text-cyan-400">one known measurement</strong> to find <em>k</em>.
        </p>

        <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700/50 my-6 shadow-inner">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm text-slate-400 mb-2">Known Distance (mm)</label>
              <input 
                type="number" 
                value={z1} 
                onChange={(e) => setZ1(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white font-mono focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-2">Measured BBox (px)</label>
              <input 
                type="number" 
                value={s1} 
                onChange={(e) => setS1(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white font-mono focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
              />
            </div>
          </div>

          <div className="mt-6 flex items-center justify-center bg-cyan-900/20 p-4 rounded-xl border border-cyan-800/50">
            <MathBlock math={`k = ${z1} \\times ${s1} = ${k.toLocaleString()}`} className="text-cyan-400 m-0" block={false} />
          </div>
        </div>

        <p className="mb-6">
          Now that we have calibrated <em>k</em>, any new bounding box size can instantly be converted to distance:
        </p>

        <div className="bg-purple-900/20 p-6 rounded-2xl border border-purple-800/50">
          <label className="block text-sm text-slate-400 mb-4">New BBox Size (px)</label>
          <Slider
            label=""
            min={10}
            max={500}
            step={1}
            value={newS}
            onChange={setNewS}
            unit="px"
            color="purple"
          />
          <div className="mt-8 text-center">
            <div className="text-sm text-slate-400 uppercase tracking-widest mb-2">Predicted Distance</div>
            <div className="text-5xl font-bold font-mono text-purple-400">
              {predictedZ.toFixed(0)} <span className="text-2xl text-purple-600">mm</span>
            </div>
          </div>
        </div>
      </ContentBlock>

      <DiagramBlock>
        <div className="w-full h-full flex flex-col items-center justify-center p-8 gap-8">
          <div className="w-full bg-slate-800 rounded-2xl p-6 border-2 border-slate-700 flex flex-col items-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-50" />
            <h3 className="text-slate-400 text-sm font-bold tracking-widest uppercase mb-4">Calibration Step</h3>
            <div className="flex items-center gap-4 w-full">
              <div className="flex-1 bg-slate-900 h-24 rounded-xl flex flex-col items-center justify-center border border-slate-700">
                <span className="text-xs text-slate-500 uppercase">Input Z</span>
                <span className="font-mono text-xl text-white">{z1}</span>
              </div>
              <div className="text-2xl text-slate-600">×</div>
              <div className="flex-1 bg-slate-900 h-24 rounded-xl flex flex-col items-center justify-center border border-slate-700">
                <span className="text-xs text-slate-500 uppercase">Input s</span>
                <span className="font-mono text-xl text-white">{s1}</span>
              </div>
              <div className="text-2xl text-slate-600">=</div>
              <div className="flex-1 bg-cyan-900/30 h-24 rounded-xl flex flex-col items-center justify-center border border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
                <span className="text-xs text-cyan-500 uppercase font-bold">Constant k</span>
                <span className="font-mono text-xl text-cyan-400 font-bold">{k.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="w-8 h-8 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-slate-500">
              <path d="M12 5v14M5 12l7 7 7-7"/>
            </svg>
          </div>

          <div className="w-full bg-slate-800 rounded-2xl p-6 border-2 border-slate-700 flex flex-col items-center relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-purple-500 to-transparent opacity-50" />
            <h3 className="text-slate-400 text-sm font-bold tracking-widest uppercase mb-4">Inference Step</h3>
            <div className="flex items-center gap-4 w-full">
              <div className="flex-1 bg-cyan-900/10 h-24 rounded-xl flex flex-col items-center justify-center border border-cyan-900 border-dashed">
                <span className="text-xs text-slate-500 uppercase">Stored k</span>
                <span className="font-mono text-xl text-cyan-400/50">{k.toLocaleString()}</span>
              </div>
              <div className="text-2xl text-slate-600">÷</div>
              <div className="flex-1 bg-slate-900 h-24 rounded-xl flex flex-col items-center justify-center border border-slate-700">
                <span className="text-xs text-slate-500 uppercase">Live s</span>
                <span className="font-mono text-xl text-white">{newS}</span>
              </div>
              <div className="text-2xl text-slate-600">=</div>
              <div className="flex-1 bg-purple-900/30 h-24 rounded-xl flex flex-col items-center justify-center border border-purple-500/50 shadow-[0_0_15px_rgba(168,85,247,0.15)]">
                <span className="text-xs text-purple-500 uppercase font-bold">Result Z</span>
                <span className="font-mono text-xl text-purple-400 font-bold">{predictedZ.toFixed(0)}</span>
              </div>
            </div>
          </div>
        </div>
      </DiagramBlock>
    </SectionContainer>
  );
}
