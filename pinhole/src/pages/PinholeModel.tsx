import { useState } from 'react';
import { SectionContainer, ContentBlock, DiagramBlock } from '../components/SectionContainer';
import { MathBlock } from '../components/MathBlock';
import { Slider } from '../components/Slider';
import { motion, AnimatePresence } from 'framer-motion';

function PinholeIntro() {
  return (
    <SectionContainer id="pinhole-intro" title="What Is a Pinhole Camera?" subtitle="Concept">
      <ContentBlock>
        <p className="mb-6">
          A <strong className="text-cyan-400">pinhole camera</strong> is the simplest possible imaging device.
          It consists of a light-tight box with a tiny hole on one side and a flat surface (the sensor or film) on the other.
        </p>
        <p className="mb-6">
          Light rays from the scene pass through the pinhole and project an <strong className="text-white">inverted image</strong> onto the back wall.
          Because the hole is so small, only one ray from each point in the scene reaches each point on the sensor, producing a sharp (if dim) image.
        </p>
        <p className="mb-6">
          This elegant simplicity makes it the foundation of all camera mathematics. Even complex multi-element lenses are modelled as
          an "equivalent pinhole" for the purpose of geometric calculations.
        </p>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="0 0 500 400" className="w-full h-full text-slate-300 font-mono text-xs" preserveAspectRatio="xMidYMid meet">
          <rect x="180" y="60" width="6" height="280" fill="#334155" rx="1" />
          <rect x="181" y="170" width="4" height="60" fill="#0f172a" />

          <rect x="340" y="80" width="4" height="240" fill="#a855f7" opacity="0.6" />
          <text x="355" y="200" fill="#a855f7" fontSize="12" fontWeight="bold">Sensor</text>

          <circle cx="60" cy="140" r="8" fill="#06b6d4" />
          <text x="60" y="125" fill="#06b6d4" textAnchor="middle" fontSize="11">A</text>

          <circle cx="60" cy="260" r="8" fill="#ef4444" />
          <text x="60" y="280" fill="#ef4444" textAnchor="middle" fontSize="11">B</text>

          <line x1="68" y1="140" x2="183" y2="200" stroke="#06b6d4" strokeWidth="1.5" />
          <line x1="183" y1="200" x2="340" y2="260" stroke="#06b6d4" strokeWidth="1.5" strokeDasharray="4" />
          <circle cx="340" cy="260" r="4" fill="#06b6d4" />
          <text x="352" y="268" fill="#06b6d4" fontSize="10">A'</text>

          <line x1="68" y1="260" x2="183" y2="200" stroke="#ef4444" strokeWidth="1.5" />
          <line x1="183" y1="200" x2="340" y2="140" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4" />
          <circle cx="340" cy="140" r="4" fill="#ef4444" />
          <text x="352" y="145" fill="#ef4444" fontSize="10">B'</text>

          <text x="183" y="348" fill="#94a3b8" textAnchor="middle" fontSize="11" fontWeight="bold">Pinhole</text>

          <text x="100" y="380" fill="#64748b" fontSize="11" textAnchor="middle">Scene</text>
          <text x="340" y="380" fill="#64748b" fontSize="11" textAnchor="middle">Image (inverted)</text>
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}

function SimilarTriangles() {
  const [objectDist, setObjectDist] = useState(300);
  const focalLength = 80;
  const objectHeight = 120;

  const imageHeight = (focalLength * objectHeight) / objectDist;

  const pinholeX = 200;
  const sensorX = pinholeX + focalLength;
  const objectX = pinholeX - objectDist / 2;

  const centerY = 200;
  const objectTop = centerY - objectHeight / 2;
  const objectBot = centerY + objectHeight / 2;

  const imgTop = centerY + imageHeight / 2;
  const imgBot = centerY - imageHeight / 2;

  return (
    <SectionContainer id="similar-triangles" title="Similar Triangles" subtitle="Geometry">
      <ContentBlock>
        <p className="mb-6">
          The pinhole model works because of <strong className="text-cyan-400">similar triangles</strong>.
          A light ray from the top of an object passes through the pinhole and hits the sensor below the optical axis.
          A ray from the bottom hits above it.
        </p>
        <p className="mb-6">
          The triangle formed by the object and the pinhole is <em>similar</em> to the triangle formed by the image and the pinhole.
          This gives us the fundamental proportion:
        </p>

        <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 text-center mb-8 shadow-inner">
          <MathBlock math="\frac{h'}{f} = \frac{H}{Z}" className="text-cyan-400" />
        </div>

        <ul className="space-y-4 text-slate-400 mb-8 list-none">
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-cyan-900/50 flex items-center justify-center text-cyan-400 font-mono font-bold border border-cyan-800">h'</span>
            <span>Image height on the sensor</span>
          </li>
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-slate-300 font-mono font-bold border border-slate-700">f</span>
            <span>Focal length (pinhole to sensor distance)</span>
          </li>
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-slate-800 flex items-center justify-center text-slate-300 font-mono font-bold border border-slate-700">H</span>
            <span>Real object height</span>
          </li>
          <li className="flex gap-4 items-center">
            <span className="w-8 h-8 rounded bg-purple-900/50 flex items-center justify-center text-purple-400 font-mono font-bold border border-purple-800">Z</span>
            <span>Distance from pinhole to object</span>
          </li>
        </ul>

        <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
          <Slider
            label="Object Distance (Z)"
            min={100}
            max={600}
            step={10}
            value={objectDist}
            onChange={setObjectDist}
            unit=""
            color="purple"
          />
        </div>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="0 0 400 400" className="w-full h-full text-slate-300 font-mono text-xs" preserveAspectRatio="xMidYMid meet">
          <line x1={pinholeX} y1="40" x2={pinholeX} y2="360" stroke="#334155" strokeWidth="2" />
          <rect x={pinholeX - 1} y={centerY - 4} width="2" height="8" fill="#fbbf24" />

          <line x1="20" y1={centerY} x2="380" y2={centerY} stroke="#475569" strokeWidth="1" strokeDasharray="4" />
          <text x="385" y={centerY + 4} fill="#64748b" fontSize="9">Optical Axis</text>

          <line x1={objectX} y1={objectTop} x2={objectX} y2={objectBot} stroke="#06b6d4" strokeWidth="3" />
          <circle cx={objectX} cy={objectTop} r="3" fill="#06b6d4" />
          <text x={objectX} y={objectTop - 10} fill="#06b6d4" textAnchor="middle" fontSize="10" fontWeight="bold">H = {objectHeight}</text>

          <rect x={sensorX} y="60" width="3" height="280" fill="#a855f7" opacity="0.4" rx="1" />

          <line x1={sensorX} y1={imgBot} x2={sensorX} y2={imgTop} stroke="#a855f7" strokeWidth="3" />
          <circle cx={sensorX} cy={imgTop} r="3" fill="#a855f7" />
          <text x={sensorX + 10} y={imgTop + 4} fill="#a855f7" fontSize="10" fontWeight="bold">h' = {imageHeight.toFixed(1)}</text>

          <line x1={objectX} y1={objectTop} x2={pinholeX} y2={centerY} stroke="#06b6d4" strokeWidth="1" opacity="0.6" />
          <line x1={pinholeX} y1={centerY} x2={sensorX} y2={imgTop} stroke="#06b6d4" strokeWidth="1" strokeDasharray="3" opacity="0.6" />

          <line x1={objectX} y1={objectBot} x2={pinholeX} y2={centerY} stroke="#ef4444" strokeWidth="1" opacity="0.6" />
          <line x1={pinholeX} y1={centerY} x2={sensorX} y2={imgBot} stroke="#ef4444" strokeWidth="1" strokeDasharray="3" opacity="0.6" />

          <g>
            <line x1={objectX} y1="340" x2={pinholeX} y2="340" stroke="#06b6d4" strokeWidth="1" />
            <text x={(objectX + pinholeX) / 2} y="355" fill="#06b6d4" textAnchor="middle" fontSize="10">Z = {objectDist}</text>
          </g>
          <g>
            <line x1={pinholeX} y1="340" x2={sensorX} y2="340" stroke="#a855f7" strokeWidth="1" />
            <text x={(pinholeX + sensorX) / 2} y="355" fill="#a855f7" textAnchor="middle" fontSize="10">f = {focalLength}</text>
          </g>

          <polygon points={`${objectX - 8},345 ${objectX},340 ${objectX - 8},335`} fill="#06b6d4" />
          <polygon points={`${pinholeX + 8},345 ${pinholeX},340 ${pinholeX + 8},335`} fill="#06b6d4" />
          <polygon points={`${pinholeX - 8},345 ${pinholeX},340 ${pinholeX - 8},335`} fill="#a855f7" />
          <polygon points={`${sensorX + 8},345 ${sensorX},340 ${sensorX + 8},335`} fill="#a855f7" />
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}

function ProjectionEquation() {
  const [step, setStep] = useState(1);

  const steps = [
    { math: "\\frac{h'}{f} = \\frac{H}{Z}", desc: "Start from the similar triangles proportion." },
    { math: "h' = \\frac{f \\times H}{Z}", desc: "Multiply both sides by f to isolate the image size." },
    { math: "s = \\frac{f \\times S}{Z}", desc: "Rename: h' becomes s (sensor size), H becomes S (real size)." },
  ];

  return (
    <SectionContainer id="projection-eq" title="The Projection Equation" subtitle="Derivation">
      <ContentBlock>
        <p className="mb-6">
          From similar triangles, we derive the <strong className="text-cyan-400">projection equation</strong> that
          relates an object's real-world size to its appearance on the sensor.
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
            {steps.map((_, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-full transition-all duration-300 ${i + 1 === step ? 'bg-emerald-400 scale-125' : 'bg-slate-700'}`}
              />
            ))}
          </div>
          <button
            disabled={step === steps.length}
            onClick={() => setStep(s => Math.min(steps.length, s + 1))}
            className="px-6 py-3 bg-emerald-600 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-emerald-500 transition-colors font-semibold"
          >
            Next
          </button>
        </div>

        <div className="mt-10 p-5 rounded-2xl border border-emerald-800/50 bg-emerald-950/20">
          <p className="text-emerald-300 text-sm leading-relaxed">
            This is the <strong>core equation</strong> of the pinhole model. It tells us that an object's image size is
            directly proportional to the focal length and real object size, and inversely proportional to its distance.
            Every depth estimation technique we use builds on this single formula.
          </p>
        </div>
      </ContentBlock>

      <DiagramBlock>
        <div className="relative w-full h-full flex flex-col items-center justify-center gap-8 p-8">
          <AnimatePresence mode="wait">
            {step === 1 && (
              <motion.div
                key="proportion"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center gap-6"
              >
                <div className="flex flex-col items-center gap-2">
                  <div className="w-20 h-20 rounded-2xl bg-cyan-900/30 border-2 border-cyan-600/50 flex items-center justify-center">
                    <span className="text-3xl font-mono text-cyan-400">h'</span>
                  </div>
                  <div className="w-0.5 h-6 bg-slate-600" />
                  <div className="w-20 h-20 rounded-2xl bg-slate-800 border-2 border-slate-600 flex items-center justify-center">
                    <span className="text-3xl font-mono text-slate-300">f</span>
                  </div>
                </div>
                <span className="text-4xl text-slate-500">=</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="w-20 h-20 rounded-2xl bg-slate-800 border-2 border-slate-600 flex items-center justify-center">
                    <span className="text-3xl font-mono text-slate-300">H</span>
                  </div>
                  <div className="w-0.5 h-6 bg-slate-600" />
                  <div className="w-20 h-20 rounded-2xl bg-purple-900/30 border-2 border-purple-600/50 flex items-center justify-center">
                    <span className="text-3xl font-mono text-purple-400">Z</span>
                  </div>
                </div>
              </motion.div>
            )}
            {step === 2 && (
              <motion.div
                key="isolated"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center gap-6"
              >
                <div className="w-24 h-24 rounded-2xl bg-cyan-900/30 border-2 border-cyan-600/50 flex items-center justify-center shadow-lg">
                  <span className="text-4xl font-mono text-cyan-400">h'</span>
                </div>
                <span className="text-4xl text-slate-500">=</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-600 flex items-center justify-center">
                      <span className="text-2xl font-mono text-slate-300">f</span>
                    </div>
                    <span className="text-2xl text-slate-500">x</span>
                    <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-600 flex items-center justify-center">
                      <span className="text-2xl font-mono text-slate-300">H</span>
                    </div>
                  </div>
                  <div className="w-40 h-0.5 bg-slate-500" />
                  <div className="w-16 h-16 rounded-xl bg-purple-900/30 border border-purple-600/50 flex items-center justify-center">
                    <span className="text-2xl font-mono text-purple-400">Z</span>
                  </div>
                </div>
              </motion.div>
            )}
            {step === 3 && (
              <motion.div
                key="final"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex items-center gap-6"
              >
                <div className="w-24 h-24 rounded-2xl bg-emerald-900/40 border-2 border-emerald-500/50 flex items-center justify-center shadow-lg shadow-emerald-500/10">
                  <span className="text-4xl font-mono text-emerald-400 font-bold">s</span>
                </div>
                <span className="text-4xl text-slate-500">=</span>
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-600 flex items-center justify-center">
                      <span className="text-2xl font-mono text-slate-300">f</span>
                    </div>
                    <span className="text-2xl text-slate-500">x</span>
                    <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-600 flex items-center justify-center">
                      <span className="text-2xl font-mono text-slate-300">S</span>
                    </div>
                  </div>
                  <div className="w-40 h-0.5 bg-slate-500" />
                  <div className="w-16 h-16 rounded-xl bg-purple-900/30 border border-purple-600/50 flex items-center justify-center">
                    <span className="text-2xl font-mono text-purple-400">Z</span>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <p className="absolute bottom-6 text-slate-500 text-sm text-center">
            {step === 3 ? 'This is the standard pinhole projection equation' : 'Step through the derivation'}
          </p>
        </div>
      </DiagramBlock>
    </SectionContainer>
  );
}

function RealVsPinhole() {
  const [aperture, setAperture] = useState(3);

  const rayCount = Math.max(1, 2 * aperture + 1);
  const spread = aperture * 2;

  return (
    <SectionContainer id="real-vs-pinhole" title="Real Lenses vs. Pinhole" subtitle="Context">
      <ContentBlock>
        <p className="mb-6">
          Real cameras use glass lenses with a finite aperture, not a single point. A larger aperture gathers
          more light but can introduce <strong className="text-cyan-400">blur</strong> for out-of-focus objects.
        </p>
        <p className="mb-6">
          Despite this, the pinhole model remains the standard geometric model for cameras.
          The <strong className="text-white">optical center</strong> of a real lens behaves identically
          to a pinhole for the purpose of computing projection geometry.
        </p>
        <p className="mb-6">
          The key insight: for calibrated cameras, all the complex optics collapse into the same
          simple equation. The focal length <span className="text-cyan-400 font-mono">f</span> you measure
          during calibration already accounts for the lens system.
        </p>

        <div className="bg-slate-800/80 p-6 rounded-2xl border border-slate-700">
          <Slider
            label="Aperture Size"
            min={1}
            max={15}
            step={1}
            value={aperture}
            onChange={setAperture}
            unit=""
            color="cyan"
          />
          <p className="text-xs text-slate-500 mt-3">
            {aperture <= 3 ? 'Small aperture: nearly pinhole-like, sharp but dim.' :
             aperture <= 8 ? 'Medium aperture: more light, slight blur for off-focus.' :
             'Large aperture: lots of light, noticeable blur for off-focus objects.'}
          </p>
        </div>
      </ContentBlock>

      <DiagramBlock>
        <svg viewBox="0 0 400 400" className="w-full h-full text-slate-300 font-mono text-xs" preserveAspectRatio="xMidYMid meet">
          <text x="200" y="30" fill="#94a3b8" textAnchor="middle" fontSize="12" fontWeight="bold">
            {aperture <= 3 ? 'Near-Pinhole (Sharp)' : aperture <= 8 ? 'Medium Aperture' : 'Wide Aperture (Blurry)'}
          </text>

          <circle cx="60" cy="200" r="6" fill="#ef4444" />
          <text x="60" y="225" fill="#ef4444" textAnchor="middle" fontSize="10">Object</text>

          <rect x="197" y={200 - spread - 10} width="6" height={(spread + 10) * 2} fill="#334155" rx="2" />
          <rect x="198" y={200 - spread} width="4" height={spread * 2} fill="#0f172a" />

          <rect x="340" y="120" width="4" height="160" fill="#a855f7" opacity="0.4" rx="1" />

          {Array.from({ length: rayCount }).map((_, i) => {
            const t = rayCount === 1 ? 0 : (i / (rayCount - 1)) * 2 - 1;
            const passY = 200 + t * spread;
            const sensorY = 200 - t * spread * 0.5 + t * aperture * 1.5;
            return (
              <g key={i}>
                <line x1="66" y1="200" x2="200" y2={passY} stroke="#ef4444" strokeWidth="1" opacity="0.5" />
                <line x1="200" y1={passY} x2="340" y2={sensorY} stroke="#ef4444" strokeWidth="1" opacity="0.5" strokeDasharray="3" />
                <circle cx="340" cy={sensorY} r="2" fill="#ef4444" opacity="0.7" />
              </g>
            );
          })}

          <text x="200" y="365" fill="#94a3b8" textAnchor="middle" fontSize="11">
            Aperture
          </text>
          <text x="340" y="310" fill="#a855f7" fontSize="10" textAnchor="middle">Sensor</text>

          {aperture <= 3 && (
            <text x="340" y="250" fill="#22c55e" fontSize="10" textAnchor="middle" fontWeight="bold">Sharp focus</text>
          )}
          {aperture > 3 && aperture <= 8 && (
            <text x="340" y="250" fill="#eab308" fontSize="10" textAnchor="middle" fontWeight="bold">Slight spread</text>
          )}
          {aperture > 8 && (
            <text x="340" y="250" fill="#ef4444" fontSize="10" textAnchor="middle" fontWeight="bold">Circle of confusion</text>
          )}
        </svg>
      </DiagramBlock>
    </SectionContainer>
  );
}

function KeyProperties() {
  return (
    <SectionContainer id="key-properties" title="Key Properties" subtitle="Summary">
      <ContentBlock>
        <p className="mb-6">
          The pinhole camera model has several properties that make it invaluable for computer vision:
        </p>

        <div className="space-y-4">
          <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
            <h3 className="text-cyan-400 font-bold mb-2">Straight lines stay straight</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              Unlike real wide-angle lenses that can cause barrel distortion, the ideal pinhole model projects all straight
              lines in the world as straight lines on the sensor. This simplifies geometry enormously.
            </p>
          </div>

          <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
            <h3 className="text-cyan-400 font-bold mb-2">Size is inversely proportional to distance</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              An object twice as far away appears half the size. This linear inverse relationship
              is the basis of depth-from-size estimation techniques like our <span className="font-mono text-purple-400">Z = k/s</span> formula.
            </p>
          </div>

          <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
            <h3 className="text-cyan-400 font-bold mb-2">Single parameter: focal length</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              The entire projection is governed by one number, the focal length <span className="font-mono text-cyan-400">f</span>.
              In practice, we also account for the sensor's pixel density and principal point offset, but for top-down setups
              where the object is near the center, <span className="font-mono text-cyan-400">f</span> alone suffices.
            </p>
          </div>

          <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700/50">
            <h3 className="text-cyan-400 font-bold mb-2">Everything is in focus</h3>
            <p className="text-slate-400 text-sm leading-relaxed">
              An ideal pinhole has infinite depth of field. Every distance is equally in focus.
              Real cameras approximate this at small apertures (high f-stop numbers).
            </p>
          </div>
        </div>
      </ContentBlock>

      <DiagramBlock>
        <div className="w-full h-full flex flex-col items-center justify-center p-8 gap-6">
          <div className="text-center mb-4">
            <span className="text-xs text-slate-500 uppercase tracking-widest">The Complete Pinhole Model</span>
          </div>

          <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700 w-full max-w-sm">
            <div className="text-center">
              <MathBlock math="s = \frac{f \times S}{Z}" className="text-cyan-400 text-2xl" />
            </div>
            <div className="mt-6 space-y-3 text-sm">
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded bg-cyan-900/50 flex items-center justify-center text-cyan-400 font-mono font-bold border border-cyan-800 text-xs">s</span>
                <span className="text-slate-400">Image size (what we measure)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded bg-slate-700 flex items-center justify-center text-slate-300 font-mono font-bold border border-slate-600 text-xs">f</span>
                <span className="text-slate-400">Focal length (camera property)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded bg-slate-700 flex items-center justify-center text-slate-300 font-mono font-bold border border-slate-600 text-xs">S</span>
                <span className="text-slate-400">Real object size (fixed)</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-6 h-6 rounded bg-purple-900/50 flex items-center justify-center text-purple-400 font-mono font-bold border border-purple-800 text-xs">Z</span>
                <span className="text-slate-400">Distance (what we want)</span>
              </div>
            </div>
          </div>

          <svg viewBox="0 0 40 40" className="w-8 h-8 text-slate-600">
            <path d="M20 8 v16 M12 20 l8 8 8-8" stroke="currentColor" strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          </svg>

          <div className="bg-purple-900/20 rounded-2xl p-6 border border-purple-700/50 w-full max-w-sm text-center">
            <span className="text-xs text-purple-400 uppercase tracking-widest font-bold">Applied in</span>
            <p className="text-white font-semibold mt-2">Z-Axis Depth Estimation</p>
            <p className="text-slate-400 text-sm mt-1">
              Where we merge f and S into a single constant k
            </p>
            <MathBlock math="Z = \frac{k}{s}" className="text-purple-400 mt-3" />
          </div>
        </div>
      </DiagramBlock>
    </SectionContainer>
  );
}

export function PinholeModel() {
  return (
    <>
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="pt-24 pb-16 px-6 text-center max-w-4xl mx-auto"
      >
        <div className="inline-block px-4 py-1 rounded-full bg-cyan-950/50 border border-cyan-800/50 text-cyan-400 text-sm font-bold tracking-widest mb-6">
          FUNDAMENTALS
        </div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-8">
          The Pinhole{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500">
            Camera Model
          </span>
        </h1>
        <p className="text-xl text-slate-400 font-light leading-relaxed">
          Understanding how a tiny hole in a box captures the geometry of the world,
          and why this century-old model still powers modern computer vision.
        </p>
      </motion.header>

      <main className="flex flex-col">
        <PinholeIntro />
        <SimilarTriangles />
        <ProjectionEquation />
        <RealVsPinhole />
        <KeyProperties />
      </main>
    </>
  );
}
