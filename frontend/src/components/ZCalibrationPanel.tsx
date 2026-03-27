import { useState, useCallback } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Plus, Trash2, Ruler, Video, RefreshCw } from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'
import type { ZCalibrationLabel, InferenceResult, ZCalibration } from '@/types'

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

  const { data: existingCal, isLoading: calLoading } = useQuery({
    queryKey: ['z-calibration', projectName, runName, videoId, inferenceId],
    queryFn: () => api.inference.getZCalibration(projectName, runName, videoId, inferenceId),
  })

  const hasExistingZ = existingCal?.z_calibration?.model != null

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
      await api.inference.saveZCalibration(projectName, runName, videoId, inferenceId, labels)
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
      <div className="flex items-center gap-2">
        <Ruler className="h-4 w-4 text-muted-foreground" />
        <h4 className="text-sm font-medium">Z-Axis Height Estimation</h4>
        {hasExistingZ && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
            Active
          </span>
        )}
      </div>

      {hasExistingZ && existingCal?.z_calibration && (
        <div className="p-3 bg-muted/50 rounded-lg text-xs space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Model</span>
            <span className="font-mono">
              {existingCal.z_calibration.model?.type === 'k_over_s'
                ? `Z = ${existingCal.z_calibration.model.k?.toFixed(0)} / s`
                : `Z = ${existingCal.z_calibration.model?.a?.toFixed(0)} / s + ${existingCal.z_calibration.model?.b?.toFixed(0)}`}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Labels</span>
            <span>{existingCal.z_calibration.labels.length} calibration point(s)</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Class</span>
            <span>{existingCal.z_calibration.class_name}</span>
          </div>
        </div>
      )}

      <div className="space-y-2">
        <p className="text-xs text-muted-foreground">
          Add calibration points: select a frame and enter the known distance from camera (mm).
          {labels.length === 0 && ' 1 point uses Z=k/s model. 2+ points use linear regression.'}
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
    </div>
  )
}
