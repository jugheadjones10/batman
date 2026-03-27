import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { MathBlock } from '../components/MathBlock';

export function Section6() {
  const [z1, setZ1] = useState(500);
  const [s1, setS1] = useState(400);
  
  const [z2, setZ2] = useState(4000);
  const [s2, setS2] = useState(50);

  const k1 = z1 * s1;
  const k2 = z2 * s2;
  
  const kAvg = (k1 + k2) / 2;
  const diff = Math.abs(k1 - k2);
  const consistency = kAvg === 0 ? 0 : (diff / kAvg) * 100;
  
  let statusColor = "bg-emerald-500";
  let statusText = "text-emerald-400";
  if (consistency > 10) {
    statusColor = "bg-red-500";
    statusText = "text-red-400";
  } else if (consistency > 5) {
    statusColor = "bg-amber-500";
    statusText = "text-amber-400";
  }

  return (
    <SectionContainer id="section6" title="Two-Point Cross-Check" subtitle="Phase 6">
      <ContentBlock>
        <p className="mb-6">
          Mathematically, one calibration point is enough to find <span className="text-purple-400 font-mono">k</span>.
          But in the real world, measurements have noise. 
        </p>
        <p className="mb-6">
          By taking two reference points at different heights, we can compute <span className="text-purple-400 font-mono">k</span> independently for both.
          If the model holds true, <span className="text-purple-400 font-mono">k₁</span> and <span className="text-purple-400 font-mono">k₂</span> should match closely. 
          If they differ by more than 5-10%, the camera might not be perfectly top-down, or the measurements are flawed.
        </p>
        
        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 text-center mb-8 shadow-inner">
          <MathBlock math="k_1 = Z_1 \times s_1 \quad \text{and} \quad k_2 = Z_2 \times s_2" className="text-slate-300 text-lg mb-2" />
          <MathBlock math="\text{Error} = \frac{|k_1 - k_2|}{\bar{k}}" className="text-slate-300 text-lg" />
        </div>
      </ContentBlock>

      <DiagramBlock>
        <div className="w-full flex flex-col gap-6">
          
          <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
            <h3 className="text-emerald-400 font-bold mb-4 uppercase tracking-wider text-sm">Reference Point 1</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Z₁ Distance (mm)</label>
                <input type="number" value={z1} onChange={(e) => setZ1(Number(e.target.value))} className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-200 font-mono" />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">s₁ Size (px)</label>
                <input type="number" value={s1} onChange={(e) => setS1(Number(e.target.value))} className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-200 font-mono" />
              </div>
            </div>
            <div className="mt-3 text-right font-mono text-emerald-400 font-bold">
              k₁ = {k1.toLocaleString()}
            </div>
          </div>

          <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
            <h3 className="text-cyan-400 font-bold mb-4 uppercase tracking-wider text-sm">Reference Point 2</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-slate-400 mb-1">Z₂ Distance (mm)</label>
                <input type="number" value={z2} onChange={(e) => setZ2(Number(e.target.value))} className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-200 font-mono" />
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">s₂ Size (px)</label>
                <input type="number" value={s2} onChange={(e) => setS2(Number(e.target.value))} className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-slate-200 font-mono" />
              </div>
            </div>
            <div className="mt-3 text-right font-mono text-cyan-400 font-bold">
              k₂ = {k2.toLocaleString()}
            </div>
          </div>

          <div className="bg-slate-800 p-5 rounded-xl border border-slate-600">
            <div className="flex justify-between items-center mb-2">
              <span className="text-slate-300 font-bold">Consistency Difference:</span>
              <span className={`font-mono font-bold text-xl ${statusText}`}>
                {consistency.toFixed(2)}%
              </span>
            </div>
            
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden mb-4 border border-slate-700">
              <div 
                className={`h-full ${statusColor} transition-all duration-500`} 
                style={{ width: `${Math.min(consistency, 100)}%` }} 
              />
            </div>
            
            <div className="text-center">
              <span className="text-xs text-slate-400 uppercase tracking-widest">Averaged Constant (k)</span>
              <div className="text-2xl text-purple-400 font-mono font-bold mt-1">
                {kAvg.toLocaleString()}
              </div>
            </div>
          </div>

        </div>
      </DiagramBlock>
    </SectionContainer>
  );
}
