import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { Slider } from '../components/Slider';

export function Section1() {
  const [z, setZ] = useState(1500);

  const maxZ = 4000;
  const minZ = 500;
  
  // Map Z (500-4000) to SVG Y coordinate (100-500)
  const mapZToY = (zVal: number) => {
    return 100 + ((zVal - minZ) / (maxZ - minZ)) * 400;
  };
  
  // Apparent size inversely proportional to Z
  const baseSize = 300000;
  const apparentSize = baseSize / z;

  return (
    <SectionContainer id="section1" title="The Setup" subtitle="Phase 1">
      <ContentBlock>
        <p className="mb-6">
          Imagine a top-down camera mounted securely to the ceiling of an industrial warehouse.
          Directly below it hangs a massive steel crane hook.
        </p>
        <p className="mb-6">
          As the hook descends toward the floor, moving further away from the lens, its apparent size on the camera sensor shrinks. 
          This relationship—distance versus size—is the core principle of our depth estimation.
        </p>
        
        <div className="mt-12 bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
          <Slider
            label="Hook Distance (Z)"
            min={500}
            max={4000}
            step={50}
            value={z}
            onChange={setZ}
            unit="mm"
            color="cyan"
          />
        </div>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="0 0 400 600" className="w-full h-full text-slate-300 font-mono text-sm" preserveAspectRatio="xMidYMid meet">
          {/* Ceiling */}
          <rect x="0" y="0" width="400" height="20" fill="#334155" />
          
          {/* Camera */}
          <path d="M170 20 L230 20 L210 50 L190 50 Z" fill="#06b6d4" />
          <circle cx="200" cy="50" r="10" fill="#1e293b" stroke="#06b6d4" strokeWidth="2" />
          
          {/* Cable */}
          <line x1="200" y1="60" x2="200" y2={mapZToY(z)} stroke="#94a3b8" strokeWidth="4" />
          
          {/* Hook (abstract box for now) */}
          <rect x="175" y={mapZToY(z)} width="50" height="60" rx="10" fill="#ef4444" />
          
          {/* Z distance arrow */}
          <line x1="250" y1="60" x2="250" y2={mapZToY(z)} stroke="#06b6d4" strokeWidth="2" strokeDasharray="4" />
          <path d={`M245 ${mapZToY(z) - 10} L250 ${mapZToY(z)} L255 ${mapZToY(z) - 10} Z`} fill="#06b6d4" />
          <path d="M245 70 L250 60 L255 70 Z" fill="#06b6d4" />
          <text x="265" y={60 + (mapZToY(z) - 60) / 2} fill="#06b6d4" alignmentBaseline="middle">Z = {z}mm</text>

          {/* Camera View Inset */}
          <g transform="translate(20, 480)">
            <rect x="0" y="0" width="100" height="100" fill="#0f172a" stroke="#475569" strokeWidth="2" rx="8" />
            <text x="50" y="-10" fill="#94a3b8" textAnchor="middle" fontSize="12">Sensor View</text>
            <rect 
              x={50 - apparentSize / 2} 
              y={50 - apparentSize / 2} 
              width={apparentSize} 
              height={apparentSize * 1.2} 
              fill="#ef4444" 
              rx="4" 
            />
          </g>
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}
