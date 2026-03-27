import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { Slider } from '../components/Slider';
import { MathBlock } from '../components/MathBlock';

export function Section2() {
  const [z, setZ] = useState(1500);
  const f = 80;
  const S = 200;
  const s = (f * S) / (z / 10);

  return (
    <SectionContainer id="section2" title="The Core Relationship" subtitle="Phase 2">
      <ContentBlock>
        <p className="mb-6">
          How do we translate distance into pixel size? We rely on the <strong className="text-cyan-400">pinhole camera model</strong>.
        </p>
        <p className="mb-6">
          The model gives us a simple geometric relationship based on similar triangles:
        </p>
        
        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 text-center mb-8 shadow-inner">
          <MathBlock math="s = \frac{f \times S}{Z}" className="text-cyan-400" />
        </div>

        <ul className="space-y-4 text-slate-400 mb-8 list-none">
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-cyan-900/50 flex items-center justify-center text-cyan-400 font-mono font-bold border border-cyan-800">s</span>
            <span>Apparent size on the sensor (pixels)</span>
          </li>
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-slate-300 font-mono font-bold border border-slate-700">f</span>
            <span>Focal length (constant)</span>
          </li>
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-slate-300 font-mono font-bold border border-slate-700">S</span>
            <span>Real object size (constant)</span>
          </li>
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-purple-900/50 flex items-center justify-center text-purple-400 font-mono font-bold border border-purple-800">Z</span>
            <span>Distance from camera to object</span>
          </li>
        </ul>

        <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
          <Slider
            label="Distance (Z)"
            min={500}
            max={4000}
            step={50}
            value={z}
            onChange={setZ}
            unit="mm"
            color="purple"
          />
        </div>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="0 0 400 600" className="w-full h-full text-slate-300 font-mono text-xs" preserveAspectRatio="xMidYMid meet">
          <line x1="200" y1="50" x2="200" y2="550" stroke="#475569" strokeDasharray="4" strokeWidth="1" />
          
          <path d="M120 150 L280 150" stroke="#06b6d4" strokeWidth="2" />
          <circle cx="200" cy="150" r="4" fill="#06b6d4" />
          <text x="220" y="145" fill="#06b6d4">Lens Pinhole</text>
          
          <path d="M150 50 L250 50" stroke="#a855f7" strokeWidth="4" />
          <text x="260" y="55" fill="#a855f7">Sensor</text>
          
          <line x1="180" y1="150" x2="200" y2="50" stroke="#475569" strokeWidth="1" />
          
          <g transform={`translate(0, ${150 + (z / 4000) * 350})`}>
            <rect x="150" y="-10" width="100" height="20" fill="#ef4444" rx="4" />
            <text x="120" y="5" fill="#ef4444" fontWeight="bold">S</text>
            <line x1="150" y1="20" x2="250" y2="20" stroke="#ef4444" />
            
            <line x1="200" y1="-10" x2={200 - s} y2={-150 - (z / 4000) * 350 + 50} stroke="#ef4444" strokeWidth="1" strokeDasharray="2" opacity="0.5" />
            <line x1="250" y1="-10" x2={200 - (s/2)} y2={-150 - (z / 4000) * 350 + 50} stroke="#ef4444" strokeWidth="1" strokeDasharray="2" opacity="0.5" />
            <line x1="150" y1="-10" x2={200 + (s/2)} y2={-150 - (z / 4000) * 350 + 50} stroke="#ef4444" strokeWidth="1" strokeDasharray="2" opacity="0.5" />
          </g>

          <g transform="translate(0, 50)">
            <rect x={200 - (s/2)} y="-2" width={s} height="4" fill="#06b6d4" />
            <text x={200} y="-15" fill="#06b6d4" textAnchor="middle" fontWeight="bold">s = {s.toFixed(1)}px</text>
          </g>
          
          <line x1="320" y1="150" x2="320" y2={150 + (z / 4000) * 350} stroke="#a855f7" strokeWidth="2" />
          <text x="330" y={150 + ((z / 4000) * 350) / 2} fill="#a855f7" fontWeight="bold">Z = {z}mm</text>
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}
