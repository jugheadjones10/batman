import { useState, useCallback, useEffect, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Plus, Trash2, Ruler, Video, Info, X } from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'
import type { ZCalibrationLabel, ZCalibrationTarget, InferenceResult } from '@/types'

interface ZCalibrationPanelProps {
  projectName: string
  runName: string
  videoId: string
  inferenceId: string
  frames: InferenceResult[]
  onZEstimated?: () => void
}

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
  const [sizeMetric, setSizeMetric] = useState<'h_px' | 'w_px'>('h_px')
  const [referenceClassName, setReferenceClassName] = useState('')
  const [referenceRealWidth, setReferenceRealWidth] = useState('')
  const [targets, setTargets] = useState<{ class_name: string; real_width_mm: string }[]>([])
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
    if (cal.size_metric === 'w_px') setSizeMetric('w_px')
    if (cal.class_name) setReferenceClassName(cal.class_name)
    if (cal.reference_real_width_mm) setReferenceRealWidth(String(cal.reference_real_width_mm))
    if (cal.targets?.length) {
      setTargets(cal.targets.map((t) => ({ class_name: t.class_name, real_width_mm: String(t.real_width_mm) })))
    }
    setDidLoadExisting(true)
  }, [existingCal, didLoadExisting])

  useEffect(() => {
    if (!referenceClassName && classNames.length > 0) {
      setReferenceClassName(classNames[0])
    }
  }, [classNames, referenceClassName])

  const refWidth = parseFloat(referenceRealWidth)

  useEffect(() => {
    if (referenceClassName && !isNaN(refWidth) && refWidth > 0 && targets.length === 0 && !didLoadExisting) {
      setTargets([{ class_name: referenceClassName, real_width_mm: String(refWidth) }])
    }
  }, [referenceClassName, refWidth, targets.length, didLoadExisting])

  const hasMultiTarget = !isNaN(refWidth) && refWidth > 0 && targets.length > 0

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

  const calibrateMutation = useMutation({
    mutationFn: async () => {
      const opts: {
        sizeMetric?: string
        referenceRealWidthMm?: number | null
        targets?: ZCalibrationTarget[] | null
      } = { sizeMetric }

      if (hasMultiTarget) {
        opts.referenceRealWidthMm = refWidth
        opts.targets = targets
          .filter((t) => t.class_name.trim() && !isNaN(parseFloat(t.real_width_mm)) && parseFloat(t.real_width_mm) > 0)
          .map((t) => ({ class_name: t.class_name.trim(), real_width_mm: parseFloat(t.real_width_mm) }))
      }

      await api.inference.saveZCalibration(
        projectName, runName, videoId, inferenceId, labels,
        referenceClassName || classNames[0] || 'crane hook', opts,
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
      {hasExistingZ && existingCal?.z_calibration && (
        <div className="p-2 bg-muted/50 rounded text-xs">
          {existingCal.z_calibration.model?.type === 'multi_target' ? (
            <span className="font-mono">
              f={existingCal.z_calibration.model.focal_length_px?.toFixed(0)}px
              &middot; {existingCal.z_calibration.model.targets?.length ?? 0} target{(existingCal.z_calibration.model.targets?.length ?? 0) !== 1 ? 's' : ''}
              &middot; {existingCal.z_calibration.labels.length} pts
            </span>
          ) : (
            <span className="font-mono">
              {existingCal.z_calibration.model?.type === 'k_over_s'
                ? `Z = ${existingCal.z_calibration.model.k?.toFixed(0)}/s`
                : `Z = ${existingCal.z_calibration.model?.a?.toFixed(0)}/s + ${existingCal.z_calibration.model?.b?.toFixed(0)}`}
              &middot; {existingCal.z_calibration.labels.length} pts
            </span>
          )}
        </div>
      )}

      {/* 1. Reference Object */}
      <div className="space-y-2">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">1. Reference Object</span>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground w-11 flex-shrink-0">Class</span>
          <select
            value={referenceClassName}
            onChange={(e) => setReferenceClassName(e.target.value)}
            className="flex-1 rounded border bg-background px-2 py-0.5 text-xs h-7 min-w-0"
          >
            {classNames.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground w-11 flex-shrink-0">Width</span>
          <Input
            type="number"
            placeholder="e.g. 2500"
            value={referenceRealWidth}
            onChange={(e) => setReferenceRealWidth(e.target.value)}
            className="h-7 text-xs flex-1 min-w-0"
          />
          <span className="text-muted-foreground">mm</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-muted-foreground w-11 flex-shrink-0">Metric</span>
          <select
            value={sizeMetric}
            onChange={(e) => setSizeMetric(e.target.value as 'h_px' | 'w_px')}
            className="flex-1 rounded border bg-background px-2 py-0.5 text-xs h-7"
          >
            <option value="h_px">Height (h_px)</option>
            <option value="w_px">Width (w_px)</option>
          </select>
        </div>
      </div>

      {/* 2. Targets */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">2. Estimation Targets</span>
          <button
            onClick={() => setTargets((prev) => [...prev, { class_name: '', real_width_mm: '2438' }])}
            className="text-[10px] text-primary hover:underline flex items-center gap-0.5"
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
                {classNames.length > 0 ? (
                  <select
                    value={tgt.class_name}
                    onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? { ...t, class_name: e.target.value } : t))}
                    className="flex-1 rounded border bg-background px-2 py-0.5 text-xs h-7 min-w-0"
                  >
                    <option value="">Select...</option>
                    {classNames.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                ) : (
                  <Input
                    type="text"
                    placeholder="class"
                    value={tgt.class_name}
                    onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? { ...t, class_name: e.target.value } : t))}
                    className="h-7 text-xs flex-1 min-w-0"
                  />
                )}
                <Input
                  type="number"
                  placeholder="mm"
                  value={tgt.real_width_mm}
                  onChange={(e) => setTargets((prev) => prev.map((t, j) => j === i ? { ...t, real_width_mm: e.target.value } : t))}
                  className="h-7 text-xs w-20"
                />
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

      {/* 3. Calibration Points */}
      <div className="space-y-2">
        <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">
          3. Calibration Points ({labels.length})
        </span>
        <p className="text-[10px] text-muted-foreground">
          {referenceClassName
            ? <>Frames where <strong>{referenceClassName}</strong> is at a known distance.</>
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
                  is: <code className="px-1 py-0.5 bg-muted rounded text-[11px]">D = k / s</code> where
                  <code className="px-1 py-0.5 bg-muted rounded text-[11px]"> s</code> is the bounding box size in pixels.
                </p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">Single-class mode</h4>
                <p>
                  Pick frames where the object is at a known distance. The system fits
                  <code className="px-1 py-0.5 bg-muted rounded text-[11px]"> k</code> (1 point) or a linear
                  model <code className="px-1 py-0.5 bg-muted rounded text-[11px]">D = a/s + b</code> (2+ points).
                  No reference width or targets needed.
                </p>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">Multi-target mode</h4>
                <p>
                  Provide a <strong>reference object width</strong> and <strong>targets</strong>:
                </p>
                <ol className="list-decimal list-inside space-y-1 mt-2 ml-1">
                  <li>Calibrate on the reference (e.g., spreader) at known distance(s).</li>
                  <li>System derives focal length: <code className="px-1 py-0.5 bg-muted rounded text-[11px]">f = D &times; s / W<sub>ref</sub></code></li>
                  <li>For each target, it computes <code className="px-1 py-0.5 bg-muted rounded text-[11px]">k = f &times; W<sub>target</sub></code></li>
                  <li>Every detection of every target gets a distance estimate.</li>
                </ol>
              </div>
              <div>
                <h4 className="text-foreground font-medium mb-1">Tips</h4>
                <ul className="list-disc list-inside space-y-1 ml-1">
                  <li>Use 2+ calibration points for best accuracy.</li>
                  <li>ISO container width: <strong>2438mm</strong> (same for 20ft/40ft/45ft).</li>
                  <li>Include the reference as a target too if you want its distance.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
