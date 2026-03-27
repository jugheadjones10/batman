import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { MathBlock } from '../components/MathBlock';
import { Slider } from '../components/Slider';

export function Section7() {
  const [noise, setNoise] = useState(1);
  
  const k = 200000;
  
  const errorAt500 = (Math.pow(500, 2) / k) * noise;
  const errorAt4000 = (Math.pow(4000, 2) / k) * noise;

  const points = [];
  const minZ = 500;
  const maxZ = 4000;
  
  for (let z = minZ; z <= maxZ; z += 100) {
    const s = k / z;
    const x = ((z - minZ) / (maxZ - minZ)) * 400;
    const y = 350 - ((s - 50) / 350) * 300;
    points.push({ z, s, x, y });
  }
  
  const curvePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  
  const errorPointsUpper = [];
  const errorPointsLower = [];
  
  for (let z = minZ; z <= maxZ; z += 100) {
    const s = k / z;
    const sUpper = s - noise; 
    const sLower = s + noise;
    
    const zUpper = sUpper > 0 ? k / sUpper : maxZ * 1.5;
    const zLower = k / sLower;
    
    const xUpper = ((zUpper - minZ) / (maxZ - minZ)) * 400;
    const xLower = ((zLower - minZ) / (maxZ - minZ)) * 400;
    
    const y = 350 - ((s - 50) / 350) * 300;
    
    errorPointsUpper.push({ x: xUpper, y });
    errorPointsLower.unshift({ x: xLower, y });
  }
  
  const errorPath = [
    ...errorPointsUpper.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`),
    ...errorPointsLower.map(p => `L ${p.x} ${p.y}`),
    'Z'
  ].join(' ');

  return (
    <SectionContainer id="section7" title="Accuracy & Limitations" subtitle="Phase 7">
      <ContentBlock>
        <p className="mb-6">
          The inverse relationship has a built-in weakness: as the object gets farther away, its pixel size shrinks to tiny fractions. 
          A single pixel of noise has a dramatically different effect depending on the distance.
        </p>
        <p className="mb-6">
          Calculated via calculus (taking the derivative of our formula), the depth error <span className="text-red-400 font-mono">ΔZ</span> grows with the <strong className="text-white">square of the distance</strong>.
        </p>
        
        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 text-center mb-8 shadow-inner">
          <MathBlock math="\Delta Z \approx \frac{Z^2}{k} \cdot \Delta s" className="text-red-400" />
        </div>

        <div className="grid grid-cols-2 gap-4 mb-8">
          <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
            <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">At Z = 500mm</div>
            <div className="text-xl text-white font-mono">± {errorAt500.toFixed(2)} mm</div>
            <div className="text-xs text-slate-500 mt-1">Negligible error</div>
          </div>
          <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
            <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">At Z = 4000mm</div>
            <div className="text-xl text-red-400 font-mono">± {errorAt4000.toFixed(2)} mm</div>
            <div className="text-xs text-slate-500 mt-1">Significant error</div>
          </div>
        </div>

        <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
          <Slider
            label="Pixel Noise (Δs)"
            min={0.5}
            max={5}
            step={0.5}
            value={noise}
            onChange={setNoise}
            unit="px"
            color="red"
          />
        </div>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="-50 -50 450 450" className="w-full h-full text-slate-300 font-mono text-xs overflow-visible">
          <line x1="0" y1="350" x2="400" y2="350" stroke="#475569" strokeWidth="2" />
          <text x="410" y="355" fill="#475569" fontWeight="bold">Z (Distance)</text>
          
          <line x1="0" y1="350" x2="0" y2="0" stroke="#475569" strokeWidth="2" />
          <text x="-20" y="-10" fill="#475569" fontWeight="bold">s (Size)</text>

          <g className="opacity-50">
            {[500, 1000, 2000, 3000, 4000].map(z => {
              const x = ((z - minZ) / (maxZ - minZ)) * 400;
              return (
                <g key={z}>
                  <line x1={x} y1="350" x2={x} y2="355" stroke="#475569" />
                  <text x={x} y="370" fill="#64748b" textAnchor="middle">{z}</text>
                </g>
              );
            })}
          </g>

          <g className="opacity-50">
            {[50, 100, 200, 400].map(s => {
              const y = 350 - ((s - 50) / 350) * 300;
              return (
                <g key={s}>
                  <line x1="0" y1={y} x2="-5" y2={y} stroke="#475569" />
                  <text x="-10" y={y + 4} fill="#64748b" textAnchor="end">{s}</text>
                </g>
              );
            })}
          </g>

          <path d={errorPath} fill="rgba(239, 68, 68, 0.2)" />
          
          <path d={curvePath} fill="none" stroke="#ef4444" strokeWidth="3" />
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}
