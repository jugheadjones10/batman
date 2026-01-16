import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Cpu,
  Play,
  Download,
  Check,
  X,
  Clock,
  Loader2,
  ChevronDown,
  BarChart3,
  Zap,
  LineChart,
  ExternalLink,
  Square,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { useToast } from '@/components/ui/Toaster'
import { formatDate, formatNumber } from '@/lib/utils'
import type { TrainingConfig, TrainingRun } from '@/types'

export default function TrainingPage() {
  const { projectName } = useParams<{ projectName: string }>()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  const [runName, setRunName] = useState('')
  const [config, setConfig] = useState<TrainingConfig>({
    base_model: 'yolo11s',
    image_size: 640,
    batch_size: 16,
    epochs: 100,
    lr_preset: 'medium',
    augmentation_preset: 'standard',
    val_split: 0.2,
    test_split: 0.1,
    freeze_backbone: false,
    mixed_precision: true,
    early_stopping_patience: 20,
  })
  const [showAdvanced, setShowAdvanced] = useState(false)

  const { data: project } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ['training-runs', projectName],
    queryFn: () => api.training.listRuns(projectName!),
    enabled: !!projectName,
    refetchInterval: 5000, // Poll for status updates
  })

  const exportMutation = useMutation({
    mutationFn: () => api.training.exportDataset(projectName!),
    onSuccess: (data) => {
      toast({
        title: 'Dataset exported',
        description: `${data.train_images} train, ${data.val_images} val images`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Export failed', description: error.message, type: 'error' })
    },
  })

  const trainMutation = useMutation({
    mutationFn: () =>
      api.training.start(projectName!, {
        name: runName || `run_${Date.now()}`,
        label_iteration_id: project?.current_iteration || 0,
        config,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      toast({ title: 'Training started', type: 'success' })
      setRunName('')
    },
    onError: (error: Error) => {
      toast({ title: 'Training failed to start', description: error.message, type: 'error' })
    },
  })

  const handleStartTraining = async () => {
    if (!project?.annotation_count) {
      toast({ title: 'No annotations', description: 'Add annotations before training', type: 'error' })
      return
    }

    // Export dataset first if not done
    await exportMutation.mutateAsync()

    // Then start training
    trainMutation.mutate()
  }

  return (
    <div className="container max-w-6xl py-8 px-6 lg:px-8">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-3xl font-bold flex items-center gap-3">
            <Cpu className="h-8 w-8 text-primary" />
            Model Training
          </h1>
          <p className="text-muted-foreground mt-1">
            Fine-tune detection models on your labeled data
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Training config */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>
                Configure your training run
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Run name */}
              <div>
                <label className="text-sm font-medium mb-2 block">Run Name</label>
                <Input
                  placeholder="my-detector-v1"
                  value={runName}
                  onChange={(e) => setRunName(e.target.value)}
                />
              </div>

              {/* Base model */}
              <div>
                <label className="text-sm font-medium mb-2 block">Base Model</label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: 'yolo11n', name: 'YOLO11-N', desc: 'Fastest, smallest' },
                    { id: 'yolo11s', name: 'YOLO11-S', desc: 'Good balance' },
                    { id: 'yolo11m', name: 'YOLO11-M', desc: 'More accurate' },
                    { id: 'rfdetr-b', name: 'RF-DETR-B', desc: 'Transformer-based' },
                    { id: 'rfdetr-l', name: 'RF-DETR-L', desc: 'Largest, most accurate' },
                  ].map((model) => (
                    <button
                      key={model.id}
                      onClick={() => setConfig({ ...config, base_model: model.id })}
                      className={`
                        p-3 rounded-lg border text-left transition-colors
                        ${config.base_model === model.id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                        }
                      `}
                    >
                      <div className="font-medium text-sm">{model.name}</div>
                      <div className="text-xs text-muted-foreground">{model.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Basic params */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Epochs</label>
                  <Input
                    type="number"
                    value={config.epochs}
                    onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })}
                    min={1}
                    max={1000}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Batch Size</label>
                  <Input
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}
                    min={1}
                    max={128}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Image Size</label>
                  <select
                    value={config.image_size}
                    onChange={(e) => setConfig({ ...config, image_size: Number(e.target.value) })}
                    className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
                  >
                    <option value={320}>320</option>
                    <option value={416}>416</option>
                    <option value={512}>512</option>
                    <option value={640}>640</option>
                    <option value={800}>800</option>
                    <option value={1024}>1024</option>
                  </select>
                </div>
              </div>

              {/* Learning rate */}
              <div>
                <label className="text-sm font-medium mb-2 block">Learning Rate</label>
                <div className="flex gap-2">
                  {['small', 'medium', 'large'].map((preset) => (
                    <button
                      key={preset}
                      onClick={() => setConfig({ ...config, lr_preset: preset as any })}
                      className={`
                        flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors
                        ${config.lr_preset === preset
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted hover:bg-muted/80'
                        }
                      `}
                    >
                      {preset.charAt(0).toUpperCase() + preset.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Augmentation */}
              <div>
                <label className="text-sm font-medium mb-2 block">Augmentation</label>
                <div className="flex gap-2">
                  {['none', 'light', 'standard', 'heavy'].map((preset) => (
                    <button
                      key={preset}
                      onClick={() => setConfig({ ...config, augmentation_preset: preset as any })}
                      className={`
                        flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors
                        ${config.augmentation_preset === preset
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted hover:bg-muted/80'
                        }
                      `}
                    >
                      {preset.charAt(0).toUpperCase() + preset.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Advanced options */}
              <div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  <ChevronDown
                    className={`h-4 w-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
                  />
                  Advanced Options
                </button>

                {showAdvanced && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-4 space-y-4 pt-4 border-t border-border"
                  >
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">Validation Split</label>
                        <Input
                          type="number"
                          value={config.val_split}
                          onChange={(e) => setConfig({ ...config, val_split: Number(e.target.value) })}
                          min={0.1}
                          max={0.5}
                          step={0.05}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">Early Stopping</label>
                        <Input
                          type="number"
                          value={config.early_stopping_patience}
                          onChange={(e) =>
                            setConfig({ ...config, early_stopping_patience: Number(e.target.value) })
                          }
                          min={5}
                          max={100}
                        />
                      </div>
                    </div>

                    <div className="flex items-center gap-6">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={config.freeze_backbone}
                          onChange={(e) =>
                            setConfig({ ...config, freeze_backbone: e.target.checked })
                          }
                          className="rounded"
                        />
                        <span className="text-sm">Freeze backbone</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={config.mixed_precision}
                          onChange={(e) =>
                            setConfig({ ...config, mixed_precision: e.target.checked })
                          }
                          className="rounded"
                        />
                        <span className="text-sm">Mixed precision</span>
                      </label>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Start button */}
              <div className="flex gap-2 pt-4">
                <Button
                  onClick={handleStartTraining}
                  disabled={trainMutation.isPending || exportMutation.isPending}
                  className="gap-2"
                >
                  {trainMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  Start Training
                </Button>
                <Button
                  variant="outline"
                  onClick={() => exportMutation.mutate()}
                  disabled={exportMutation.isPending}
                  className="gap-2"
                >
                  {exportMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  Export Dataset
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Training runs sidebar */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Runs</CardTitle>
              <CardDescription>
                {runs?.length || 0} total runs
              </CardDescription>
            </CardHeader>
            <CardContent>
              {runsLoading ? (
                <div className="space-y-3">
                  {[1, 2].map((i) => (
                    <div key={i} className="h-20 bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : runs?.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No training runs yet
                </p>
              ) : (
                <div className="space-y-3">
                  {runs?.map((run) => (
                    <TrainingRunCard key={run.id} run={run} projectName={projectName!} />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function TrainingRunCard({ run, projectName }: { run: TrainingRun; projectName: string }) {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const [isStartingTB, setIsStartingTB] = useState(false)

  const statusColors: Record<string, string> = {
    pending: 'text-muted-foreground',
    running: 'text-accent',
    completed: 'text-green-500',
    failed: 'text-destructive',
    stopped: 'text-muted-foreground',
  }

  const statusIcons: Record<string, typeof Clock> = {
    pending: Clock,
    running: Loader2,
    completed: Check,
    failed: X,
    stopped: Square,
  }

  const Icon = statusIcons[run.status] || Clock

  const startTensorBoard = async () => {
    setIsStartingTB(true)
    try {
      const result = await api.training.startTensorBoard(projectName, run.name)
      toast({
        title: 'TensorBoard started',
        description: `Running at ${result.url}`,
        type: 'success',
      })
      // Refresh runs to get the new tensorboard_url
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
    } catch (error: any) {
      toast({
        title: 'Failed to start TensorBoard',
        description: error.message,
        type: 'error',
      })
    } finally {
      setIsStartingTB(false)
    }
  }

  const stopTensorBoard = async () => {
    try {
      await api.training.stopTensorBoard(projectName, run.name)
      toast({ title: 'TensorBoard stopped', type: 'success' })
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
    } catch (error: any) {
      toast({
        title: 'Failed to stop TensorBoard',
        description: error.message,
        type: 'error',
      })
    }
  }

  return (
    <div className="p-4 rounded-lg border border-border hover:border-primary/30 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div>
          <h4 className="font-medium text-sm">{run.name}</h4>
          <p className="text-xs text-muted-foreground">{run.base_model}</p>
        </div>
        <div className={`flex items-center gap-1 ${statusColors[run.status]}`}>
          <Icon className={`h-4 w-4 ${run.status === 'running' ? 'animate-spin' : ''}`} />
          <span className="text-xs capitalize">{run.status}</span>
        </div>
      </div>

      {run.status === 'running' && (
        <div className="mb-2">
          <Progress value={run.progress * 100} className="h-1.5" />
          <p className="text-xs text-muted-foreground mt-1">
            {Math.round(run.progress * 100)}% complete
          </p>
        </div>
      )}

      {run.status === 'completed' && run.metrics && (
        <div className="flex items-center gap-4 text-xs mb-2">
          <span className="flex items-center gap-1">
            <BarChart3 className="h-3 w-3" />
            mAP50: {((run.metrics.mAP50 || 0) * 100).toFixed(1)}%
          </span>
          {run.latency_ms && (
            <span className="flex items-center gap-1">
              <Zap className="h-3 w-3" />
              {run.latency_ms.toFixed(1)}ms
            </span>
          )}
        </div>
      )}

      {/* TensorBoard section - show for running or completed RF-DETR runs */}
      {run.base_model.startsWith('rfdetr') && (run.status === 'running' || run.status === 'completed') && (
        <div className="mt-3 pt-3 border-t border-border">
          {run.tensorboard_url ? (
            <div className="flex items-center gap-2">
              <a
                href={run.tensorboard_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs text-primary hover:underline"
              >
                <LineChart className="h-3.5 w-3.5" />
                TensorBoard
                <ExternalLink className="h-3 w-3" />
              </a>
              <button
                onClick={stopTensorBoard}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive transition-colors ml-auto"
                title="Stop TensorBoard"
              >
                <Square className="h-3 w-3" />
              </button>
            </div>
          ) : (
            <button
              onClick={startTensorBoard}
              disabled={isStartingTB}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors"
            >
              {isStartingTB ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <LineChart className="h-3.5 w-3.5" />
              )}
              {isStartingTB ? 'Starting...' : 'Launch TensorBoard'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

