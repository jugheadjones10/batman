import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { Slider } from '../components/Slider';

export function Section4() {
  const [z, setZ] = useState(1500);
  const k = 200000;
  
  const minZ = 500;
  const maxZ = 4000;
  const s = k / z;

  const points = [];
  for (let zVal = minZ; zVal <= maxZ; zVal += 50) {
    points.push({ z: zVal, s: k / zVal });
  }

  const mapX = (val: number) => ((val - minZ) / (maxZ - minZ)) * 300 + 40;
  const mapY = (val: number) => 300 - ((val - 50) / (400 - 50)) * 260;

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${mapX(p.z)} ${mapY(p.s)}`).join(' ');

  const showPoints = [500, 1000, 2000, 4000];

  return (
    <SectionContainer id="section4" title="The Inverse Curve" subtitle="Phase 4">
      <ContentBlock>
        <p className="mb-6">
          Because distance is in the denominator <strong className="text-cyan-400 font-mono">(Z = k/s)</strong>, the relationship isn't linear. It forms a hyperbola.
        </p>
        <p className="mb-6">
          This non-linear reality means that at close ranges, small distance changes cause huge pixel size changes. At far ranges, huge distance changes cause tiny pixel size changes.
        </p>

        <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
          <Slider
            label="Hook Distance (Z)"
            min={minZ}
            max={maxZ}
            step={50}
            value={z}
            onChange={setZ}
            unit="mm"
            color="amber"
          />
        </div>

        <ul className="mt-8 space-y-2 font-mono text-sm">
          {showPoints.map(pz => (
            <li 
              key={pz} 
              className={`p-3 rounded-lg border flex justify-between items-center transition-colors ${Math.abs(z - pz) < 25 ? 'bg-amber-900/50 border-amber-500/50 text-amber-300 shadow-[0_0_15px_rgba(245,158,11,0.2)]' : 'bg-slate-800/50 border-slate-700 text-slate-400'}`}
            >
              <span>Z = {pz}mm</span>
              <span className="opacity-50 text-xl font-sans">→</span>
              <span>s = {k / pz}px</span>
            </li>
          ))}
        </ul>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="0 0 360 340" className="w-full h-full text-slate-300 font-mono text-[10px]" preserveAspectRatio="xMidYMid meet">
          
          {[50, 100, 200, 300, 400].map(tick => (
            <g key={`y-${tick}`}>
              <line x1="35" y1={mapY(tick)} x2="340" y2={mapY(tick)} stroke="#334155" strokeWidth="1" strokeDasharray="2" />
              <text x="30" y={mapY(tick) + 3} textAnchor="end" fill="#64748b">{tick}</text>
            </g>
          ))}

          {[500, 1000, 2000, 3000, 4000].map(tick => (
            <g key={`x-${tick}`}>
              <line x1={mapX(tick)} y1="40" x2={mapX(tick)} y2="305" stroke="#334155" strokeWidth="1" strokeDasharray="2" />
              <text x={mapX(tick)} y="320" textAnchor="middle" fill="#64748b">{tick}</text>
            </g>
          ))}
          
          <text x="190" y="335" textAnchor="middle" fill="#94a3b8" fontWeight="bold">Distance Z (mm)</text>
          
          <g transform="translate(10, 170) rotate(-90)">
            <text x="0" y="0" textAnchor="middle" fill="#94a3b8" fontWeight="bold">Apparent Size s (px)</text>
          </g>

          <line x1="40" y1="40" x2="340" y2="300" stroke="#ef4444" strokeWidth="2" strokeDasharray="4" opacity="0.3" />
          <text x="240" y="160" fill="#ef4444" opacity="0.5" transform="rotate(38 240 160)">Linear (Wrong)</text>

          <path d={pathD} fill="none" stroke="#f59e0b" strokeWidth="4" className="drop-shadow-lg" />
          
          {showPoints.map(pz => (
            <circle 
              key={`pt-${pz}`} 
              cx={mapX(pz)} 
              cy={mapY(k / pz)} 
              r="4" 
              fill="#f59e0b" 
              className={Math.abs(z - pz) < 25 ? "animate-pulse" : ""}
            />
          ))}

          <g transform={`translate(${mapX(z)}, ${mapY(s)})`}>
            <circle cx="0" cy="0" r="8" fill="#fef3c7" stroke="#b45309" strokeWidth="2" className="shadow-2xl" />
            <rect x="10" y="-30" width="100" height="24" rx="4" fill="#1e293b" stroke="#f59e0b" strokeWidth="1" />
            <text x="16" y="-14" fill="#fcd34d">({z}, {s.toFixed(1)})</text>
          </g>

          <line x1="40" y1="40" x2="40" y2="300" stroke="#cbd5e1" strokeWidth="2" />
          <line x1="40" y1="300" x2="340" y2="300" stroke="#cbd5e1" strokeWidth="2" />
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}
