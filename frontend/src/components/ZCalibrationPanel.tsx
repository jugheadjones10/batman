import { useState, useCallback, useEffect, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Plus, Trash2, Ruler, Video, Info, X } from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'
import type { ZCalibrationLabel, InferenceResult } from '@/types'

interface ZCalibrationPanelProps {
  projectName: string
  runName: string
  videoId: string
  inferenceId: string
  frames: InferenceResult[]
  onZEstimated?: () => void
}

// ISO-standard dry-box container lengths. The spreader telescopes to match the
// container it's picking, so both share the same ℓ for a given calibration run.
const ISO_LENGTHS = [
  { mm: 6058, label: '20 ft (6058 mm)' },
  { mm: 12192, label: '40 ft (12192 mm)' },
  { mm: 13716, label: '45 ft (13716 mm)' },
] as const

export default function ZCalibrationPanel({
  projectName,
  runName,
  videoId,
  inferenceId,
  frames,
  onZEstimated,
}: ZCalibrationPanelProps) {
  const { toast } = useToast()
  const queryClient = useQueryClient()

  const [labels, setLabels] = useState<ZCalibrationLabel[]>([])
  const [newFrameNum, setNewFrameNum] = useState('')
  const [newZMm, setNewZMm] = useState('')
  const [referenceClass, setReferenceClass] = useState('')
  const [lengthMm, setLengthMm] = useState<number | null>(null)
  const [targets, setTargets] = useState<string[]>([])
  const [showInfo, setShowInfo] = useState(false)

  const { data: existingCal, isLoading: calLoading } = useQuery({
    queryKey: ['z-calibration', projectName, runName, videoId, inferenceId],
    queryFn: () => api.inference.getZCalibration(projectName, runName, videoId, inferenceId),
  })

  const hasExistingZ = existingCal?.z_calibration?.model != null

  const classNames = useMemo(() => {
    const names = new Set<string>()
    for (const f of frames) {
      for (const d of f.detections) names.add(d.class_name)
    }
    return Array.from(names).sort()
  }, [frames])

  const [didLoadExisting, setDidLoadExisting] = useState(false)
  useEffect(() => {
    if (!existingCal?.z_calibration || didLoadExisting) return
    const cal = existingCal.z_calibration
    if (cal.reference_class) setReferenceClass(cal.reference_class)
    if (cal.length_mm != null) setLengthMm(cal.length_mm)
    if (cal.targets?.length) setTargets([...cal.targets])
    setDidLoadExisting(true)
  }, [existingCal, didLoadExisting])

  useEffect(() => {
    if (referenceClass || classNames.length === 0) return
    // Prefer a class whose name mentions "spreader" — the PDF canonical flow.
    const spreaderLike = classNames.find((c) => /spreader/i.test(c))
    setReferenceClass(spreaderLike ?? classNames[0])
  }, [classNames, referenceClass])

  // Auto-seed the reference class as a target so its own detections are
  // estimated too. Mirrors the legacy behaviour the user expects.
  useEffect(() => {
    if (referenceClass && lengthMm != null && lengthMm > 0 && targets.length === 0 && !didLoadExisting) {
      setTargets([referenceClass])
    }
  }, [referenceClass, lengthMm, targets.length, didLoadExisting])

  const addLabel = useCallback(() => {
    const fn = parseInt(newFrameNum, 10)
    const z = parseFloat(newZMm)
    if (isNaN(fn) || isNaN(z) || z <= 0) {
      toast({ title: 'Invalid input', description: 'Enter valid frame number and distance (mm)', type: 'error' })
      return
    }
    const frameExists = frames.some((f) => f.frame_number === fn)
    if (!frameExists) {
      toast({ title: 'Frame not found', description: `Frame ${fn} is not in the inference results`, type: 'error' })
      return
    }
    if (labels.some((l) => l.frame_number === fn)) {
      toast({ title: 'Duplicate', description: `Frame ${fn} already labeled`, type: 'error' })
      return
    }
    setLabels((prev) => [...prev, { frame_number: fn, z_mm: z, detection_index: 0 }])
    setNewFrameNum('')
    setNewZMm('')
  }, [newFrameNum, newZMm, frames, labels, toast])

  const removeLabel = useCallback((fn: number) => {
    setLabels((prev) => prev.filter((l) => l.frame_number !== fn))
  }, [])

  const addTarget = useCallback(() => {
    const available = classNames.find((c) => !targets.includes(c))
    if (available) setTargets((prev) => [...prev, available])
  }, [classNames, targets])

  const calibrateMutation = useMutation({
    mutationFn: async () => {
      const cleaned = targets.filter((t) => t.trim().length > 0)
      const refClass = referenceClass || classNames[0] || ''
      if (!refClass) {
        throw new Error('No reference class selected')
      }
      const allTargets = cleaned.includes(refClass) ? cleaned : [refClass, ...cleaned]

      await api.inference.saveZCalibration(
        projectName, runName, videoId, inferenceId, labels, refClass,
        { lengthMm: lengthMm ?? null, targetClasses: allTargets },
      )
      const result = await api.inference.applyZEstimation(projectName, runName, videoId, inferenceId)
      return result
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['z-calibration', projectName, runName, videoId, inferenceId] })
      queryClient.invalidateQueries({ queryKey: ['inference-result-detail', projectName, runName, videoId, inferenceId] })
      toast({
        title: 'Z estimation applied',
        description: `Model: ${result.model.type}`,
        type: 'success',
      })
      onZEstimated?.()
    },
    onError: (error: Error) => {
      toast({ title: 'Z calibration failed', description: error.message, type: 'error' })
    },
  })

  const exportVideoMutation = useMutation({
    mutationFn: () => api.inference.exportZVideo(projectName, runName, videoId, inferenceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['inference-result-detail', projectName, runName, videoId, inferenceId] })
      toast({ title: 'Video re-exported with Z overlays', type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Video export failed', description: error.message, type: 'error' })
    },
  })

  if (calLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading Z calibration...
      </div>
    )
  }

  const model = existingCal?.z_calibration?.model
  const existingLabelsCount = existingCal?.z_calibration?.labels?.length ?? 0

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Ruler className="h-4 w-4 text-muted-foreground" />
        <h4 className="text-sm font-medium">Distance Calibration</h4>
        {hasExistingZ && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
            Active
          </span>
        )}
        <button
          onClick={() => setShowInfo(true)}
          className="ml-auto text-muted-foreground hover:text-foreground transition-colors p-0.5"
          title="How does this work?"
        >
          <Info className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Existing model summary */}
      {hasExistingZ && model && (
        <div className="p-2 bg-muted/50 rounded text-xs">
          <span className="font-mono">
            {model.type === 'k_over_s'
              ? `Z = ${model.k?.toFixed(0)}/s`
              : `Z = ${model.m?.toFixed(0)}/s + ${model.c?.toFixed(0)}`}
            {existingCal?.z_calibration?.length_mm != null
              ? ` · ℓ=${existingCal.z_calibration.length_mm}mm`
              : ''}
            {' · '}
            {existingLabelsCount} pts
          </span>
        </div>
      )}

      {/* 1. Container length (ℓ) */}
      <div className="space-y-2">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
          1. Container length (ℓ)
        </span>
        <select
          value={lengthMm ?? ''}
          onChange={(e) => setLengthMm(e.target.value ? Number(e.target.value) : null)}
          className="w-full rounded border bg-background px-2 py-0.5 text-xs h-7"
        >
          <option value="">Single-class mode (no targets)</option>
          {ISO_LENGTHS.map((l) => (
            <option key={l.mm} value={l.mm}>{l.label}</option>
          ))}
        </select>
        <p className="text-[10px] text-muted-foreground/80 leading-relaxed">
          Shared by the spreader (which telescopes to match) and every container class.
        </p>
      </div>

      {/* 2. Reference class */}
      <div className="space-y-2">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
          2. Reference Class
        </span>
        <select
          value={referenceClass}
          onChange={(e) => setReferenceClass(e.target.value)}
          className="w-full rounded border bg-background px-2 py-0.5 text-xs h-7"
        >
          {classNames.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        <p className="text-[10px] text-muted-foreground/80 leading-relaxed">
          The class you can measure distance to (typically the spreader — hoist PLC readout).
        </p>
      </div>

      {/* 3. Targets */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
            3. Estimation Targets
          </span>
          <button
            onClick={addTarget}
            disabled={targets.length >= classNames.length}
            className="text-[10px] text-primary hover:underline flex items-center gap-0.5 disabled:opacity-40 disabled:no-underline"
          >
            <Plus className="h-2.5 w-2.5" /> Add
          </button>
        </div>
        {targets.length === 0 ? (
          <p className="text-[10px] text-muted-foreground">No targets. Add classes to measure distance for.</p>
        ) : (
          <div className="space-y-1.5">
            {targets.map((tgt, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <select
                  value={tgt}
                  onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? e.target.value : t))}
                  className="flex-1 rounded border bg-background px-2 py-0.5 text-xs h-7 min-w-0"
                >
                  <option value="">Select...</option>
                  {classNames.map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
                <button
                  onClick={() => setTargets((prev) => prev.filter((_, j) => j !== i))}
                  className="text-muted-foreground hover:text-destructive transition-colors p-0.5"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 4. Calibration Points */}
      <div className="space-y-2">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
          4. Calibration Points ({labels.length})
        </span>
        <p className="text-[10px] text-muted-foreground">
          {referenceClass
            ? <>Frames where <strong>{referenceClass}</strong> is at a known distance.</>
            : 'Enter frame number and known distance from camera (mm).'}
        </p>

        {labels.map((label) => (
          <div key={label.frame_number} className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground w-20">Frame {label.frame_number}</span>
            <span className="font-mono">{label.z_mm}mm</span>
            <button
              onClick={() => removeLabel(label.frame_number)}
              className="ml-auto text-muted-foreground hover:text-destructive transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}

        <div className="flex items-center gap-2">
          <Input
            type="number"
            placeholder="Frame #"
            value={newFrameNum}
            onChange={(e) => setNewFrameNum(e.target.value)}
            className="w-24 h-8 text-xs"
          />
          <Input
            type="number"
            placeholder="Distance (mm)"
            value={newZMm}
            onChange={(e) => setNewZMm(e.target.value)}
            className="w-32 h-8 text-xs"
          />
          <Button variant="outline" size="sm" onClick={addLabel} className="h-8 gap-1">
            <Plus className="h-3 w-3" />
            Add
          </Button>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <Button
          size="sm"
          className="gap-1.5"
          disabled={labels.length === 0 || calibrateMutation.isPending}
          onClick={() => calibrateMutation.mutate()}
        >
          {calibrateMutation.isPending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Ruler className="h-3.5 w-3.5" />
          )}
          {calibrateMutation.isPending ? 'Calibrating...' : 'Calibrate & Estimate'}
        </Button>

        {hasExistingZ && (
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5"
            disabled={exportVideoMutation.isPending}
            onClick={() => exportVideoMutation.mutate()}
          >
            {exportVideoMutation.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Video className="h-3.5 w-3.5" />
            )}
            {exportVideoMutation.isPending ? 'Exporting...' : 'Re-export Video with Z'}
          </Button>
        )}
      </div>

      {/* Info popup overlay */}
      {showInfo && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setShowInfo(false)}>
          <div
            className="bg-secondary border border-border rounded-lg shadow-xl max-w-lg w-full mx-4 max-h-[80vh] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-sm font-semibold">How Distance Calibration Works</h3>
              <button onClick={() => setShowInfo(false)} className="text-muted-foreground hover:text-foreground transition-colors">
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto text-xs text-muted-foreground space-y-4 leading-relaxed">
              <div>
                <h4 className="text-foreground font-medium mb-1">The idea</h4>
                <p>
                  Objects farther from the camera appear smaller. If you know an object's
                  real-world size and can measure its apparent size in pixels, the distance
                  is <code className="px-1 py-0.5 bg-muted rounded text-[11px]">Z = k / s</code>
                  , where <code className="px-1 py-0.5 bg-muted rounded text-[11px]">s</code> is
                  the <strong>longer side</strong> of the bounding box in pixels. Batman always
                  uses the longer side — it's the cleaner signal and removes the axis-picking step.
                </p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">1 calibration label</h4>
                <p>
                  Fits <code className="px-1 py-0.5 bg-muted rounded text-[11px]">k</code> exactly:
                  <code className="px-1 py-0.5 bg-muted rounded text-[11px]"> k = z · s</code>.
                </p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">2+ calibration labels</h4>
                <p>
                  Fits a line in <code className="px-1 py-0.5 bg-muted rounded text-[11px]">1/s</code>:
                  <code className="px-1 py-0.5 bg-muted rounded text-[11px]"> Z = m/s + c</code>.
                  The intercept <code className="px-1 py-0.5 bg-muted rounded text-[11px]">c</code> absorbs
                  systematic bias (detector bbox clip, tape-measure offset, optical centre shift) — usually
                  2–3× more accurate than the 1-point fit away from the calibration distance.
                </p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">Container length (ℓ)</h4>
                <p>
                  Picking an ISO length (20/40/45 ft) sets the shared real-world size for every target
                  class. The spreader telescopes to match the container, so the same fit applies to
                  both without any per-target rescaling. If you leave this blank, the calibration runs
                  in single-class mode with no targets.
                </p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">Tips</h4>
                <ul className="list-disc list-inside space-y-1 ml-1">
                  <li>Use 2+ calibration points spanning the operating range for best accuracy.</li>
                  <li>Re-calibrate if the camera moves or zoom changes.</li>
                  <li>See <code className="px-1 py-0.5 bg-muted rounded text-[11px]">docs/guides/z-axis-height-estimation.md</code> for the full derivation.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
