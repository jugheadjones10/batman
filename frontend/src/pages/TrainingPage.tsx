import { useState } from 'react'
import { useEffect } from 'react'
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
  XCircle,
  Eye,
} from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { Progress } from '@/components/ui/Progress'
import { useToast } from '@/components/ui/Toaster'
import GpuConnectionPanel from '@/components/GpuConnectionPanel'
import LogViewer from '@/components/LogViewer'
import type {
  TrainingConfig,
  GPUConfig,
  DataConfig,
  TrainingSubmitRequest,
  TrainingRun,
  DataSource,
  ManualDataSplitStrategy,
  RFDETRModelSize,
  GPUType,
} from '@/types'

const MODEL_OPTIONS: { id: RFDETRModelSize; name: string; desc: string }[] = [
  { id: 'nano', name: 'RF-DETR Nano', desc: 'Fastest' },
  { id: 'small', name: 'RF-DETR Small', desc: 'Fast & light' },
  { id: 'base', name: 'RF-DETR Base', desc: 'Balanced' },
  { id: 'medium', name: 'RF-DETR Medium', desc: 'More accurate' },
  { id: 'large', name: 'RF-DETR Large', desc: 'Most accurate' },
]

const GPU_OPTIONS: { id: GPUType; name: string; desc: string }[] = [
  { id: 'h200', name: 'H200', desc: '141 GB' },
  { id: 'h100-96', name: 'H100-96', desc: '96 GB' },
  { id: 'h100-47', name: 'H100-47', desc: '47 GB' },
  { id: 'a100-80', name: 'A100-80', desc: '80 GB' },
  { id: 'a100-40', name: 'A100-40', desc: '40 GB' },
  { id: 'nv', name: 'NV', desc: 'Misc GPU' },
]

export default function TrainingPage() {
  const { projectName } = useParams<{ projectName: string }>()
  const queryClient = useQueryClient()
  const { toast } = useToast()

  // Config state
  const [label, setLabel] = useState('')
  const [training, setTraining] = useState<TrainingConfig>({
    model: 'base',
    epochs: 50,
    batch_size: null,
    image_size: 640,
    lr: 1e-4,
    patience: 10,
    grad_accum: 1,
  })
  const [gpu, setGpu] = useState<GPUConfig>({
    gpu_type: 'a100-80',
    num_gpus: 1,
    time_limit: '24:00:00',
  })
  const [data, setData] = useState<DataConfig>({
    sources: null,
    manual_split_strategy: 'train_only',
    manual_datasets: null,
    exclude_manual_datasets: null,
    filter_classes: null,
    max_frames_per_class: null,
    train_split: 0.70,
    val_split: 0.15,
    test_split: 0.15,
  })
  const [inferAfter, setInferAfter] = useState(false)
  const [inferTestOnly, setInferTestOnly] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Data source toggles
  const [includeManualData, setIncludeManualData] = useState(true)
  const [includeImports, setIncludeImports] = useState(true)

  // Log viewer state
  const [selectedRunName, setSelectedRunName] = useState<string | null>(null)

  const { data: project } = useQuery({
    queryKey: ['project', projectName],
    queryFn: () => api.projects.get(projectName!),
    enabled: !!projectName,
  })

  const { data: gpuStatus } = useQuery({
    queryKey: ['gpu-status'],
    queryFn: () => api.gpu.getStatus(),
    refetchInterval: 10000,
  })

  const { data: manualImages } = useQuery({
    queryKey: ['manual-data-images', projectName],
    queryFn: () => api.manualData.listImages(projectName!, 0, 1),
    enabled: !!projectName,
  })

  const { data: manualDatasets } = useQuery({
    queryKey: ['manual-data-datasets', projectName],
    queryFn: () => api.manualData.listDatasets(projectName!),
    enabled: !!projectName,
  })

  const { data: importedDatasets } = useQuery({
    queryKey: ['imported-datasets', projectName],
    queryFn: () => api.import.listDatasets(projectName!),
    enabled: !!projectName,
  })

  const hasManualData = (manualImages?.total ?? 0) > 0
  const hasImports = (importedDatasets?.length ?? 0) > 0
  const gpuConnected = gpuStatus?.connected ?? false

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ['training-runs', projectName],
    queryFn: () => api.training.listRuns(projectName!),
    enabled: !!projectName,
    refetchInterval: 5000,
  })

  const exportMutation = useMutation({
    mutationFn: () => {
      const sources: DataSource[] = []
      if (includeManualData && hasManualData) sources.push('manual_data')
      if (includeImports && hasImports) sources.push('imports')
      return api.training.exportDataset(projectName!, {
        data_sources: sources.length > 0 ? sources : null,
        manual_data_split_strategy: data.manual_split_strategy,
        manual_datasets: data.manual_datasets,
        exclude_manual_datasets: data.exclude_manual_datasets,
      })
    },
    onSuccess: (result) => {
      toast({
        title: 'Dataset exported',
        description: `${result.train_images} train, ${result.val_images} val images`,
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: 'Export failed', description: error.message, type: 'error' })
    },
  })

  const submitMutation = useMutation({
    mutationFn: () => {
      const sources: DataSource[] = []
      if (includeManualData && hasManualData) sources.push('manual_data')
      if (includeImports && hasImports) sources.push('imports')

      const req: TrainingSubmitRequest = {
        label: label || null,
        training,
        gpu,
        data: { ...data, sources: sources.length > 0 ? sources : null },
        infer_after: inferAfter,
        infer_test_only: inferTestOnly,
      }
      return api.training.submit(projectName!, req)
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      toast({ title: 'Training submitted', description: `Job ${result.job_id}`, type: 'success' })
      setSelectedRunName(result.run_name)
      setLabel('')
    },
    onError: (error: Error) => {
      toast({ title: 'Submission failed', description: error.message, type: 'error' })
    },
  })

  // If submit stays pending for a few seconds, tell the user it's pushing data (can be slow)
  useEffect(() => {
    if (!submitMutation.isPending) return
    const t = setTimeout(() => {
      toast({
        title: 'Still working…',
        description: 'Pushing data to cluster. This may take a minute for large projects.',
        type: 'default',
      })
    }, 4000)
    return () => clearTimeout(t)
  }, [submitMutation.isPending, toast])

  const handleSubmit = async () => {
    if (!project?.annotation_count) {
      toast({ title: 'No annotations', description: 'Add annotations before training', type: 'error' })
      return
    }
    const sources: DataSource[] = []
    if (includeManualData && hasManualData) sources.push('manual_data')
    if (includeImports && hasImports) sources.push('imports')
    if (sources.length === 0) {
      toast({ title: 'No data sources', description: 'Enable at least one data source', type: 'error' })
      return
    }
    // Export first, then submit
    await exportMutation.mutateAsync()
    submitMutation.mutate()
  }

  const logsUrl = selectedRunName
    ? api.training.streamLogsUrl(projectName!, selectedRunName)
    : null

  return (
    <div className="container max-w-7xl py-8 px-6 lg:px-8">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="font-display text-3xl font-bold flex items-center gap-3">
            <Cpu className="h-8 w-8 text-primary" />
            Model Training
          </h1>
          <p className="text-muted-foreground mt-1">
            Train RF-DETR models on the GPU cluster
          </p>
        </div>
        <div className="w-72">
          <GpuConnectionPanel />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ─── Section 1: Configuration ─── */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>RF-DETR model training on GPU cluster</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Run label */}
              <div>
                <label className="text-sm font-medium mb-2 block">Run Label (optional)</label>
                <Input
                  placeholder="e.g. experiment-v2"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                />
                <p className="text-xs text-muted-foreground mt-1">Appended to auto-generated run name</p>
              </div>

              {/* Model selection */}
              <div>
                <label className="text-sm font-medium mb-2 block">Model Size</label>
                <div className="grid grid-cols-5 gap-2">
                  {MODEL_OPTIONS.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setTraining({ ...training, model: m.id })}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        training.model === m.id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <div className="font-medium text-sm">{m.name}</div>
                      <div className="text-xs text-muted-foreground">{m.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* GPU config */}
              <div>
                <label className="text-sm font-medium mb-2 block">GPU Configuration</label>
                <div className="grid grid-cols-6 gap-2 mb-3">
                  {GPU_OPTIONS.map((g) => (
                    <button
                      key={g.id}
                      onClick={() => setGpu({ ...gpu, gpu_type: g.id })}
                      className={`p-2.5 rounded-lg border text-center transition-colors ${
                        gpu.gpu_type === g.id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <div className="font-medium text-xs">{g.name}</div>
                      <div className="text-[10px] text-muted-foreground">{g.desc}</div>
                    </button>
                  ))}
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Num GPUs</label>
                    <select
                      value={gpu.num_gpus}
                      onChange={(e) => setGpu({ ...gpu, num_gpus: Number(e.target.value) })}
                      className="w-full h-9 px-3 rounded-md border border-border bg-background text-sm"
                    >
                      {[1, 2, 3, 4].map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Time Limit</label>
                    <select
                      value={gpu.time_limit}
                      onChange={(e) => setGpu({ ...gpu, time_limit: e.target.value })}
                      className="w-full h-9 px-3 rounded-md border border-border bg-background text-sm"
                    >
                      <option value="03:00:00">3 hours</option>
                      <option value="06:00:00">6 hours</option>
                      <option value="12:00:00">12 hours</option>
                      <option value="24:00:00">24 hours</option>
                      <option value="48:00:00">48 hours</option>
                      <option value="72:00:00">72 hours</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Training params */}
              <div className="grid grid-cols-4 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Epochs</label>
                  <Input
                    type="number"
                    value={training.epochs}
                    onChange={(e) => setTraining({ ...training, epochs: Number(e.target.value) })}
                    min={1}
                    max={1000}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Batch Size</label>
                  <Input
                    type="number"
                    value={training.batch_size ?? ''}
                    onChange={(e) =>
                      setTraining({
                        ...training,
                        batch_size: e.target.value ? Number(e.target.value) : null,
                      })
                    }
                    placeholder="auto"
                    min={1}
                    max={128}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Learning Rate</label>
                  <Input
                    type="number"
                    value={training.lr}
                    onChange={(e) => setTraining({ ...training, lr: Number(e.target.value) })}
                    step={0.0001}
                    min={0.00001}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Image Size</label>
                  <select
                    value={training.image_size}
                    onChange={(e) => setTraining({ ...training, image_size: Number(e.target.value) })}
                    className="w-full h-10 px-3 rounded-md border border-border bg-background text-sm"
                  >
                    {[320, 416, 512, 560, 640, 800, 1024].map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Data sources */}
              {(hasManualData || hasImports) && (
                <div>
                  <label className="text-sm font-medium mb-2 block">Data Sources</label>
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-4">
                      {hasManualData && (
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={includeManualData}
                            onChange={(e) => setIncludeManualData(e.target.checked)}
                            className="rounded"
                          />
                          <span className="text-sm">
                            Manual Data
                            <span className="text-muted-foreground ml-1">({manualImages?.total ?? 0} images)</span>
                          </span>
                        </label>
                      )}
                      {hasImports && (
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={includeImports}
                            onChange={(e) => setIncludeImports(e.target.checked)}
                            className="rounded"
                          />
                          <span className="text-sm">
                            Imported Datasets
                            <span className="text-muted-foreground ml-1">({importedDatasets?.length ?? 0} datasets)</span>
                          </span>
                        </label>
                      )}
                    </div>

                    {hasManualData && includeManualData && (
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                          Manual Data Strategy
                        </label>
                        <select
                          value={data.manual_split_strategy}
                          onChange={(e) =>
                            setData({ ...data, manual_split_strategy: e.target.value as ManualDataSplitStrategy })
                          }
                          className="w-full h-9 px-3 rounded-md border border-border bg-background text-sm"
                        >
                          <option value="proportional">Proportional - Distribute across all splits</option>
                          <option value="val_only">Validation Only</option>
                          <option value="train_only">Training Only</option>
                          <option value="all_splits">All Splits</option>
                        </select>
                      </div>
                    )}

                    {/* Manual dataset filter */}
                    {hasManualData &&
                      includeManualData &&
                      manualDatasets?.datasets &&
                      manualDatasets.datasets.length > 1 && (
                        <div>
                          <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                            Filter Manual Datasets (leave empty for all)
                          </label>
                          <div className="flex flex-wrap gap-2">
                            {manualDatasets.datasets.map((ds) => {
                              const selected = data.manual_datasets?.includes(ds.name) ?? false
                              return (
                                <button
                                  key={ds.name}
                                  onClick={() => {
                                    const current = data.manual_datasets ?? []
                                    const next = selected
                                      ? current.filter((n) => n !== ds.name)
                                      : [...current, ds.name]
                                    setData({
                                      ...data,
                                      manual_datasets: next.length > 0 ? next : null,
                                    })
                                  }}
                                  className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
                                    selected
                                      ? 'border-primary bg-primary/10 text-primary'
                                      : 'border-border hover:border-primary/50'
                                  }`}
                                >
                                  {ds.name} ({ds.image_count})
                                </button>
                              )
                            })}
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              )}

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
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">Patience</label>
                        <Input
                          type="number"
                          value={training.patience}
                          onChange={(e) => setTraining({ ...training, patience: Number(e.target.value) })}
                          min={1}
                          max={100}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">Grad Accumulation</label>
                        <Input
                          type="number"
                          value={training.grad_accum}
                          onChange={(e) => setTraining({ ...training, grad_accum: Number(e.target.value) })}
                          min={1}
                          max={16}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">Max Frames/Class</label>
                        <Input
                          type="number"
                          value={data.max_frames_per_class ?? ''}
                          onChange={(e) =>
                            setData({
                              ...data,
                              max_frames_per_class: e.target.value ? Number(e.target.value) : null,
                            })
                          }
                          placeholder="unlimited"
                          min={1}
                        />
                      </div>
                    </div>

                    {/* Class filter */}
                    {project?.classes && project.classes.length > 0 && (
                      <div>
                        <label className="text-sm font-medium mb-2 block">Filter Classes (leave empty for all)</label>
                        <div className="flex flex-wrap gap-2">
                          {project.classes.map((cls) => {
                            const selected = data.filter_classes?.includes(cls) ?? false
                            return (
                              <button
                                key={cls}
                                onClick={() => {
                                  const current = data.filter_classes ?? []
                                  const next = selected
                                    ? current.filter((c) => c !== cls)
                                    : [...current, cls]
                                  setData({ ...data, filter_classes: next.length > 0 ? next : null })
                                }}
                                className={`px-2.5 py-1 text-xs rounded-md border transition-colors ${
                                  selected
                                    ? 'border-primary bg-primary/10 text-primary'
                                    : 'border-border hover:border-primary/50'
                                }`}
                              >
                                {cls}
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* Splits */}
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">Train Split</label>
                        <Input
                          type="number"
                          value={data.train_split}
                          onChange={(e) => setData({ ...data, train_split: Number(e.target.value) })}
                          min={0.1}
                          max={1}
                          step={0.05}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">Val Split</label>
                        <Input
                          type="number"
                          value={data.val_split}
                          onChange={(e) => setData({ ...data, val_split: Number(e.target.value) })}
                          min={0.05}
                          max={0.5}
                          step={0.05}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">Test Split</label>
                        <Input
                          type="number"
                          value={data.test_split}
                          onChange={(e) => setData({ ...data, test_split: Number(e.target.value) })}
                          min={0}
                          max={0.5}
                          step={0.05}
                        />
                      </div>
                    </div>

                    {/* Post-training inference */}
                    <div className="flex items-center gap-6">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={inferAfter}
                          onChange={(e) => setInferAfter(e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-sm">Run inference after training</span>
                      </label>
                      {inferAfter && (
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={inferTestOnly}
                            onChange={(e) => setInferTestOnly(e.target.checked)}
                            className="rounded"
                          />
                          <span className="text-sm">Test-only videos</span>
                        </label>
                      )}
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Action buttons */}
              <div className="flex gap-2 pt-4">
                <Button
                  onClick={handleSubmit}
                  disabled={!gpuConnected || submitMutation.isPending || exportMutation.isPending}
                  className="gap-2"
                >
                  {submitMutation.isPending || exportMutation.isPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  {!gpuConnected ? 'Connect GPU First' : 'Submit Training'}
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

        {/* ─── Section 2: Active/Recent Jobs ─── */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Runs</CardTitle>
              <CardDescription>{runs?.length || 0} total runs</CardDescription>
            </CardHeader>
            <CardContent>
              {runsLoading ? (
                <div className="space-y-3">
                  {[1, 2].map((i) => (
                    <div key={i} className="h-20 bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : runs?.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">No training runs yet</p>
              ) : (
                <div className="space-y-3">
                  {runs?.map((run) => (
                    <TrainingRunCard
                      key={run.id}
                      run={run}
                      projectName={projectName!}
                      isSelected={selectedRunName === run.name}
                      onSelect={() => setSelectedRunName(run.name)}
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ─── Job Logs (full width) ─── */}
      <div className="mt-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Job Logs
              {selectedRunName && (
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  {selectedRunName}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <LogViewer url={logsUrl} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function TrainingRunCard({
  run,
  projectName,
  isSelected,
  onSelect,
}: {
  run: TrainingRun
  projectName: string
  isSelected: boolean
  onSelect: () => void
}) {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const [isStartingTB, setIsStartingTB] = useState(false)

  const cancelMutation = useMutation({
    mutationFn: () => api.training.cancel(projectName, run.name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      toast({ title: 'Job cancelled', type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: 'Cancel failed', description: error.message, type: 'error' })
    },
  })

  const statusColors: Record<string, string> = {
    pending: 'text-muted-foreground',
    queued: 'text-yellow-500',
    running: 'text-accent',
    completed: 'text-green-500',
    failed: 'text-destructive',
    cancelled: 'text-muted-foreground',
    timeout: 'text-orange-500',
  }

  const StatusIcon = {
    pending: Clock,
    queued: Clock,
    running: Loader2,
    completed: Check,
    failed: X,
    cancelled: XCircle,
    timeout: Clock,
  }[run.status] || Clock

  const startTensorBoard = async () => {
    setIsStartingTB(true)
    try {
      const result = await api.training.startTensorBoard(projectName, run.name)
      toast({ title: 'TensorBoard started', description: `Running at ${result.url}`, type: 'success' })
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
    } catch (error: any) {
      toast({ title: 'Failed to start TensorBoard', description: error.message, type: 'error' })
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
      toast({ title: 'Failed to stop TensorBoard', description: error.message, type: 'error' })
    }
  }

  return (
    <div
      className={`p-4 rounded-lg border transition-colors cursor-pointer ${
        isSelected ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/30'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="min-w-0">
          <h4 className="font-medium text-sm truncate">{run.name}</h4>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{run.model}</span>
            {run.gpu_type && (
              <>
                <span>&middot;</span>
                <span>{run.gpu_type.toUpperCase()}</span>
              </>
            )}
            {run.slurm_job_id && (
              <>
                <span>&middot;</span>
                <span>#{run.slurm_job_id}</span>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {(run.status === 'queued' || run.status === 'running' || run.status === 'pending') && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                cancelMutation.mutate()
              }}
              className="text-muted-foreground hover:text-destructive transition-colors"
              title="Cancel"
            >
              <XCircle className="h-4 w-4" />
            </button>
          )}
          <div className={`flex items-center gap-1 ${statusColors[run.status]}`}>
            <StatusIcon className={`h-4 w-4 ${run.status === 'running' ? 'animate-spin' : ''}`} />
            <span className="text-xs capitalize">{run.status}</span>
          </div>
        </div>
      </div>

      {run.status === 'running' && (
        <div className="mb-2">
          <Progress value={run.progress * 100} className="h-1.5" />
          <p className="text-xs text-muted-foreground mt-1">{Math.round(run.progress * 100)}%</p>
        </div>
      )}

      {run.status === 'completed' && run.metrics && (
        <div className="flex items-center gap-4 text-xs mb-2">
          <span className="flex items-center gap-1">
            <BarChart3 className="h-3 w-3" />
            mAP50: {((run.metrics.mAP50 || 0) * 100).toFixed(1)}%
          </span>
          {run.latency_ms != null && (
            <span className="flex items-center gap-1">
              <Zap className="h-3 w-3" />
              {run.latency_ms.toFixed(1)}ms
            </span>
          )}
        </div>
      )}

      {/* TensorBoard */}
      {(run.status === 'running' || run.status === 'completed') && (
        <div className="mt-3 pt-3 border-t border-border">
          {run.tensorboard_url ? (
            <div className="flex items-center gap-2">
              <a
                href={run.tensorboard_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-xs text-primary hover:underline"
                onClick={(e) => e.stopPropagation()}
              >
                <LineChart className="h-3.5 w-3.5" />
                TensorBoard
                <ExternalLink className="h-3 w-3" />
              </a>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  stopTensorBoard()
                }}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-destructive transition-colors ml-auto"
              >
                <Square className="h-3 w-3" />
              </button>
            </div>
          ) : (
            <button
              onClick={(e) => {
                e.stopPropagation()
                startTensorBoard()
              }}
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

      {/* Show logs button for selected */}
      {isSelected && (run.status === 'queued' || run.status === 'running') && (
        <div className="mt-2 flex items-center gap-1 text-xs text-primary">
          <Eye className="h-3 w-3" />
          Viewing logs
        </div>
      )}
    </div>
  )
}
