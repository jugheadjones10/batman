import { useState, useRef, useCallback } from 'react'
import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import {
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
  Monitor,
  Server,
  Pencil,
  Trash2,
  ImageIcon,
  Video,
  FolderDown,
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
  LocalTrainingSubmitRequest,
  TrainingRun,
  DataSource,
  RFDETRModelSize,
  GPUType,
} from '@/types'

const MODEL_OPTIONS: { id: RFDETRModelSize; nameKey: string; descKey: string }[] = [
  { id: 'nano', nameKey: 'RF-DETR Nano', descKey: 'Fastest' },
  { id: 'small', nameKey: 'RF-DETR Small', descKey: 'Fast & light' },
  { id: 'base', nameKey: 'RF-DETR Base', descKey: 'Balanced' },
  { id: 'medium', nameKey: 'RF-DETR Medium', descKey: 'More accurate' },
  { id: 'large', nameKey: 'RF-DETR Large', descKey: 'Most accurate' },
  { id: 'xlarge', nameKey: 'RF-DETR XLarge', descKey: 'Biggest (seg only)' },
]

// RF-DETR-Seg ships only for the sizes listed below; keep in sync with
// _RFDETR_CLASS_TABLE in src/core/trainer.py. The "base" size is intentionally
// excluded — picking it with task=segmentation is remapped to "medium" on the
// backend, so we disable it in the UI to avoid surprising the user.
const SEG_AVAILABLE_SIZES = new Set<RFDETRModelSize>([
  'nano', 'small', 'medium', 'large', 'xlarge',
])

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
  const { t } = useTranslation()

  // Config state
  const [label, setLabel] = useState('')
  const [training, setTraining] = useState<TrainingConfig>({
    model: 'base',
    task: 'detection',
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
    manual_split_strategy: 'proportional',
    manual_datasets: null,
    exclude_manual_datasets: null,
    exclude_videos: null,
    filter_classes: null,
    max_frames_per_class: null,
    train_split: 0.70,
    val_split: 0.15,
    test_split: 0.15,
  })
  const [inferAfter, setInferAfter] = useState(false)
  const [inferTestOnly, setInferTestOnly] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Local GPU vs GPU Cluster
  const [trainMode, setTrainMode] = useState<'local' | 'gpu'>('local')

  // Per-source exclusion sets (excluded items tracked; everything included by default)
  const [excludedManualDatasets, setExcludedManualDatasets] = useState<Set<string>>(new Set())
  const [excludedVideos, setExcludedVideos] = useState<Set<string>>(new Set())
  const [excludedImports, setExcludedImports] = useState<Set<string>>(new Set())

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

  const { data: manualDatasetPreviews } = useQuery({
    queryKey: ['manual-data-previews', projectName, manualDatasets?.datasets?.map((d) => d.name).join(',')],
    queryFn: async () => {
      const datasets = manualDatasets?.datasets ?? []
      const previews: Record<string, string[]> = {}
      await Promise.all(
        datasets.map(async (ds) => {
          const result = await api.manualData.listImages(projectName!, 0, 6, ds.name)
          previews[ds.name] = result.images.map((img) => img.url)
        })
      )
      return previews
    },
    enabled: !!projectName && (manualDatasets?.datasets?.length ?? 0) > 0,
  })

  const { data: importedDatasets } = useQuery({
    queryKey: ['imported-datasets', projectName],
    queryFn: () => api.import.listDatasets(projectName!),
    enabled: !!projectName,
  })

  const { data: videos } = useQuery({
    queryKey: ['videos', projectName],
    queryFn: () => api.videos.list(projectName!),
    enabled: !!projectName,
  })

  const allManualDatasets = manualDatasets?.datasets ?? []
  const hasManualData = (manualImages?.total ?? 0) > 0
  const hasImports = (importedDatasets?.length ?? 0) > 0
  const trainableVideos = videos?.filter((v) => !v.exclude_from_training && (v.annotated_frame_count ?? v.annotation_count) > 0) ?? []
  const hasVideos = trainableVideos.length > 0
  const gpuConnected = gpuStatus?.connected ?? false

  const includeManualData = hasManualData && excludedManualDatasets.size < allManualDatasets.length
  const includeVideos = hasVideos && excludedVideos.size < trainableVideos.length
  const includeImports = hasImports && excludedImports.size < (importedDatasets?.length ?? 0)

  const buildDataOverrides = () => {
    const sources: DataSource[] = []
    if (includeManualData) sources.push('manual_data')
    if (includeImports) sources.push('imports')
    if (includeVideos) sources.push('videos')

    const excManual = excludedManualDatasets.size > 0 ? Array.from(excludedManualDatasets) : null

    // Merge video exclusions and import exclusions (imports use video_id as dir name)
    const allExcVids = new Set(excludedVideos)
    for (const ds of importedDatasets ?? []) {
      if (excludedImports.has(String(ds.video_id))) {
        allExcVids.add(String(ds.video_id))
      }
    }
    const excVids = allExcVids.size > 0 ? Array.from(allExcVids) : null

    return {
      sources: sources.length > 0 ? sources : null,
      exclude_manual_datasets: excManual,
      exclude_videos: excVids,
    }
  }

  const { data: deviceInfo } = useQuery({
    queryKey: ['device-info'],
    queryFn: () => api.device.getInfo(),
    enabled: trainMode === 'local',
  })

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ['training-runs', projectName],
    queryFn: () => api.training.listRuns(projectName!),
    enabled: !!projectName,
    refetchInterval: 5000,
  })

  const exportMutation = useMutation({
    mutationFn: () => {
      const overrides = buildDataOverrides()
      return api.training.exportDataset(projectName!, {
        data_sources: overrides.sources,
        manual_data_split_strategy: data.manual_split_strategy,
        manual_datasets: data.manual_datasets,
        exclude_manual_datasets: overrides.exclude_manual_datasets,
      })
    },
    onSuccess: (result) => {
      toast({
        title: t('training.exported'),
        description: t('training.exportedDesc', { train: result.train_images, val: result.val_images }),
        type: 'success',
      })
    },
    onError: (error: Error) => {
      toast({ title: t('training.exportFailed'), description: error.message, type: 'error' })
    },
  })

  const submitMutation = useMutation({
    mutationFn: () => {
      const overrides = buildDataOverrides()
      const req: TrainingSubmitRequest = {
        label: label || null,
        training,
        gpu,
        data: { ...data, ...overrides },
        infer_after: inferAfter,
        infer_test_only: inferTestOnly,
      }
      return api.training.submit(projectName!, req)
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      toast({ title: t('training.submitted'), description: t('training.submittedDesc', { id: result.job_id }), type: 'success' })
      setSelectedRunName(result.run_name)
      setLabel('')
    },
    onError: (error: Error) => {
      toast({ title: t('training.submissionFailed'), description: error.message, type: 'error' })
    },
  })

  const submitLocalMutation = useMutation({
    mutationFn: () => {
      const overrides = buildDataOverrides()
      const req: LocalTrainingSubmitRequest = {
        label: label || null,
        training,
        data: { ...data, ...overrides },
        infer_after: inferAfter,
        infer_test_only: inferTestOnly,
      }
      return api.training.submitLocal(projectName!, req)
    },
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      toast({ title: t('training.localStarted'), description: result.run_name, type: 'success' })
      setSelectedRunName(result.run_name)
      setLabel('')
    },
    onError: (error: Error) => {
      toast({ title: t('training.startFailed'), description: error.message, type: 'error' })
    },
  })

  // If submit stays pending for a few seconds, tell the user it's pushing data (can be slow)
  useEffect(() => {
    if (!submitMutation.isPending) return
    const timer = setTimeout(() => {
      toast({
        title: t('training.stillWorking'),
        description: t('training.stillWorkingDesc'),
        type: 'default',
      })
    }, 4000)
    return () => clearTimeout(timer)
  }, [submitMutation.isPending, toast])

  const handleSubmit = async () => {
    if (!project?.annotation_count) {
      toast({ title: t('training.noAnnotations'), description: t('training.noAnnotationsDesc'), type: 'error' })
      return
    }
    const { sources } = buildDataOverrides()
    if (!sources || sources.length === 0) {
      toast({ title: t('training.noDataSources'), description: t('training.noDataSourcesDesc'), type: 'error' })
      return
    }
    if (trainMode === 'local') {
      submitLocalMutation.mutate()
    } else {
      await exportMutation.mutateAsync()
      submitMutation.mutate()
    }
  }

  const selectedRun = runs?.find((r) => r.name === selectedRunName)
  const isLocalRun =
    selectedRun?.gpu_type === 'local' || selectedRunName?.startsWith('rfdetr_local_') === true
  const logsUrl = selectedRunName
    ? isLocalRun
      ? api.training.streamLocalLogsUrl(projectName!, selectedRunName)
      : api.training.streamLogsUrl(projectName!, selectedRunName)
    : null

  return (
    <div className="container max-w-[90rem] py-8 px-6 lg:px-8">
      <div className="flex justify-end mb-6">
        <div className="w-72">
          <GpuConnectionPanel />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* ─── Section 1: Configuration ─── */}
        <div className="lg:col-span-3 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>{t('training.configTitle')}</CardTitle>
              <CardDescription>{t('training.configDesc')}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Mode: Local GPU vs GPU Cluster */}
              <div>
                <label className="text-sm font-medium mb-2 block">{t('training.runOn')}</label>
                <div className="flex gap-1 p-1 bg-muted rounded-lg">
                  <button
                    onClick={() => setTrainMode('local')}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-sm font-medium transition-colors ${
                      trainMode === 'local' ? 'bg-background shadow-sm' : 'text-muted-foreground'
                    }`}
                  >
                    <Monitor className="h-3.5 w-3.5" />
                    {t('training.localGPU')}
                  </button>
                  <button
                    onClick={() => setTrainMode('gpu')}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-sm font-medium transition-colors ${
                      trainMode === 'gpu' ? 'bg-background shadow-sm' : 'text-muted-foreground'
                    }`}
                  >
                    <Server className="h-3.5 w-3.5" />
                    {t('training.gpuCluster')}
                  </button>
                </div>
                {trainMode === 'local' && deviceInfo && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {deviceInfo.memory_gb != null
                      ? t('training.deviceWithMem', { name: deviceInfo.name, mem: deviceInfo.memory_gb })
                      : t('training.device', { name: deviceInfo.name })}
                  </p>
                )}
              </div>

              {/* Run label */}
              <div>
                <label className="text-sm font-medium mb-2 block">{t('training.runLabel')}</label>
                <Input
                  placeholder={t('training.runLabelPlaceholder')}
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                />
                <p className="text-xs text-muted-foreground mt-1">{t('training.runLabelHint')}</p>
              </div>

              {/* Task toggle (detection vs segmentation) */}
              <div>
                <label className="text-sm font-medium mb-2 block">Task</label>
                <div className="grid grid-cols-2 gap-2">
                  {([
                    { id: 'detection', name: 'Detection', desc: 'Axis-aligned bboxes' },
                    { id: 'segmentation', name: 'Segmentation', desc: 'Masks + skew angle' },
                  ] as const).map((opt) => (
                    <button
                      key={opt.id}
                      onClick={() => {
                        const nextTask = opt.id
                        // If switching to segmentation and the current size has no
                        // seg variant, auto-pick "medium" which is always available.
                        const nextModel: RFDETRModelSize =
                          nextTask === 'segmentation' && !SEG_AVAILABLE_SIZES.has(training.model)
                            ? 'medium'
                            : training.model
                        setTraining({ ...training, task: nextTask, model: nextModel })
                      }}
                      className={`p-3 rounded-lg border text-left transition-colors ${
                        (training.task ?? 'detection') === opt.id
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <div className="font-medium text-sm">{opt.name}</div>
                      <div className="text-xs text-muted-foreground">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model selection */}
              <div>
                <label className="text-sm font-medium mb-2 block">{t('training.modelSize')}</label>
                <div className="grid grid-cols-6 gap-2">
                  {MODEL_OPTIONS.map((m) => {
                    const isSeg = (training.task ?? 'detection') === 'segmentation'
                    const disabled = isSeg && !SEG_AVAILABLE_SIZES.has(m.id)
                    return (
                      <button
                        key={m.id}
                        disabled={disabled}
                        onClick={() => setTraining({ ...training, model: m.id })}
                        className={`p-3 rounded-lg border text-left transition-colors ${
                          training.model === m.id
                            ? 'border-primary bg-primary/10'
                            : 'border-border hover:border-primary/50'
                        } ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
                        title={disabled ? 'No RF-DETR-Seg variant at this size' : undefined}
                      >
                        <div className="font-medium text-sm">{m.nameKey}</div>
                        <div className="text-xs text-muted-foreground">{m.descKey}</div>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* GPU config (cluster only) */}
              {trainMode === 'gpu' && (
              <div>
                <label className="text-sm font-medium mb-2 block">{t('training.gpuConfig')}</label>
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
                    <label className="text-xs font-medium text-muted-foreground mb-1.5 block">{t('training.numGPUs')}</label>
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
                    <label className="text-xs font-medium text-muted-foreground mb-1.5 block">{t('training.timeLimit')}</label>
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
              )}

              {/* Data sources — granular per-item checklist with previews */}
              {(hasManualData || hasImports || hasVideos) && (
                <div>
                  <label className="text-sm font-medium mb-3 block">Data Sources</label>
                  <div className="space-y-4">
                    {/* ── Manual datasets ── */}
                    {hasManualData && allManualDatasets.length > 0 && (
                      <div>
                        <div className="flex items-center gap-1.5 mb-2">
                          <ImageIcon className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Manual Data</span>
                        </div>
                        <div className="space-y-2">
                          {allManualDatasets.map((ds) => {
                            const excluded = excludedManualDatasets.has(ds.name)
                            const previews = manualDatasetPreviews?.[ds.name] ?? []
                            return (
                              <div key={`manual_${ds.name}`} className={`rounded-lg border p-3 transition-colors ${excluded ? 'opacity-50 border-border' : 'border-border'}`}>
                                <label className="flex items-center gap-2 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={!excluded}
                                    onChange={() => {
                                      const next = new Set(excludedManualDatasets)
                                      if (excluded) next.delete(ds.name)
                                      else next.add(ds.name)
                                      setExcludedManualDatasets(next)
                                    }}
                                    className="rounded"
                                  />
                                  <span className="text-sm font-medium">
                                    {allManualDatasets.length > 1 ? ds.name : 'Manual Data'}
                                  </span>
                                  <span className="text-xs text-muted-foreground">{ds.image_count} images</span>
                                </label>
                                {previews.length > 0 && !excluded && (
                                  <div className="flex gap-1.5 mt-2 overflow-x-auto pb-1">
                                    {previews.map((url, i) => (
                                      <img key={i} src={url} className="h-14 w-14 object-cover rounded flex-shrink-0" loading="lazy" />
                                    ))}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* ── Video frames ── */}
                    {trainableVideos.length > 0 && (
                      <div>
                        <div className="flex items-center gap-1.5 mb-2">
                          <Video className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Video Frames</span>
                        </div>
                        <div className="space-y-2">
                          {trainableVideos.map((v) => {
                            const vid = String(v.id)
                            const excluded = excludedVideos.has(vid)
                            const frameCount = v.annotated_frame_count ?? v.annotation_count
                            const thumbUrl = api.videos.thumbnailUrl(projectName!, vid)
                            return (
                              <div key={`video_${vid}`} className={`rounded-lg border p-3 transition-colors ${excluded ? 'opacity-50 border-border' : 'border-border'}`}>
                                <label className="flex items-center gap-2 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={!excluded}
                                    onChange={() => {
                                      const next = new Set(excludedVideos)
                                      if (excluded) next.delete(vid)
                                      else next.add(vid)
                                      setExcludedVideos(next)
                                    }}
                                    className="rounded"
                                  />
                                  <span className="text-sm font-medium">{v.filename}</span>
                                  <span className="text-xs text-muted-foreground">{frameCount} frames</span>
                                </label>
                                {!excluded && (
                                  <div className="flex gap-1.5 mt-2 overflow-x-auto pb-1">
                                    <img src={thumbUrl} className="h-14 rounded flex-shrink-0" style={{ maxWidth: '120px' }} loading="lazy" />
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      </div>
                    )}

                    {/* ── Imported datasets ── */}
                    {(importedDatasets?.length ?? 0) > 0 && (
                      <div>
                        <div className="flex items-center gap-1.5 mb-2">
                          <FolderDown className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Imported Datasets</span>
                        </div>
                        <div className="space-y-2">
                          {importedDatasets?.map((ds) => {
                            const key = String(ds.video_id)
                            const excluded = excludedImports.has(key)
                            return (
                              <div key={`import_${key}`} className={`rounded-lg border p-3 transition-colors ${excluded ? 'opacity-50 border-border' : 'border-border'}`}>
                                <label className="flex items-center gap-2 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={!excluded}
                                    onChange={() => {
                                      const next = new Set(excludedImports)
                                      if (excluded) next.delete(key)
                                      else next.add(key)
                                      setExcludedImports(next)
                                    }}
                                    className="rounded"
                                  />
                                  <span className="text-sm font-medium">{ds.source}</span>
                                  <span className="text-xs text-muted-foreground">{ds.image_count} images</span>
                                </label>
                                {ds.sample_images.length > 0 && !excluded && (
                                  <div className="flex gap-1.5 mt-2 overflow-x-auto pb-1">
                                    {ds.sample_images.map((url, i) => (
                                      <img key={i} src={url} className="h-14 w-14 object-cover rounded flex-shrink-0" loading="lazy" />
                                    ))}
                                  </div>
                                )}
                              </div>
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
                  {t('training.advancedOptions')}
                </button>

                {showAdvanced && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-4 space-y-4 pt-4 border-t border-border"
                  >
                    {/* Training hyperparameters */}
                    <div className="grid grid-cols-4 gap-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">{t('training.epochs')}</label>
                        <Input
                          type="number"
                          value={training.epochs}
                          onChange={(e) => setTraining({ ...training, epochs: Number(e.target.value) })}
                          min={1}
                          max={1000}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">{t('training.batchSize')}</label>
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
                        <label className="text-sm font-medium mb-2 block">{t('training.learningRate')}</label>
                        <Input
                          type="number"
                          value={training.lr}
                          onChange={(e) => setTraining({ ...training, lr: Number(e.target.value) })}
                          step={0.0001}
                          min={0.00001}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">{t('training.imageSize')}</label>
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

                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="text-sm font-medium mb-2 block">{t('training.patience')}</label>
                        <Input
                          type="number"
                          value={training.patience}
                          onChange={(e) => setTraining({ ...training, patience: Number(e.target.value) })}
                          min={1}
                          max={100}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">{t('training.gradAccumulation')}</label>
                        <Input
                          type="number"
                          value={training.grad_accum}
                          onChange={(e) => setTraining({ ...training, grad_accum: Number(e.target.value) })}
                          min={1}
                          max={16}
                        />
                      </div>
                      <div>
                        <label className="text-sm font-medium mb-2 block">{t('training.maxFramesPerClass')}</label>
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
                        <label className="text-sm font-medium mb-2 block">{t('training.filterClasses')}</label>
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
                        <label className="text-sm font-medium mb-2 block">{t('training.trainSplit')}</label>
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
                        <label className="text-sm font-medium mb-2 block">{t('training.valSplit')}</label>
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
                        <label className="text-sm font-medium mb-2 block">{t('training.testSplit')}</label>
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
                        <span className="text-sm">{t('training.runInferenceAfter')}</span>
                      </label>
                      {inferAfter && (
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={inferTestOnly}
                            onChange={(e) => setInferTestOnly(e.target.checked)}
                            className="rounded"
                          />
                          <span className="text-sm">{t('training.testOnlyVideos')}</span>
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
                  disabled={
                    (trainMode === 'gpu' ? !gpuConnected : false) ||
                    (trainMode === 'local' ? submitLocalMutation.isPending : submitMutation.isPending || exportMutation.isPending)
                  }
                  className="gap-2"
                >
                  {(trainMode === 'local' ? submitLocalMutation.isPending : submitMutation.isPending || exportMutation.isPending) ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  {trainMode === 'gpu'
                    ? (!gpuConnected ? t('common.connectGPUFirst') : t('common.submitTraining'))
                    : (submitLocalMutation.isPending ? t('common.starting') : t('common.startTraining'))}
                </Button>
                {trainMode === 'gpu' && (
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
                  {t('common.exportDataset')}
                </Button>
                )}
              </div>
            </CardContent>
          </Card>

        </div>

        {/* ─── Section 2: Active/Recent Jobs ─── */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>{t('training.runsTitle')}</CardTitle>
              <CardDescription>{t('training.runsCount', { count: runs?.length || 0 })}</CardDescription>
            </CardHeader>
            <CardContent>
              {runsLoading ? (
                <div className="space-y-3">
                  {[1, 2].map((i) => (
                    <div key={i} className="h-20 bg-muted animate-pulse rounded-lg" />
                  ))}
                </div>
              ) : runs?.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">{t('training.noRuns')}</p>
              ) : (
                <div className="space-y-3">
                  {runs?.map((run) => (
                    <TrainingRunCard
                      key={run.id}
                      run={run}
                      projectName={projectName!}
                      isSelected={selectedRunName === run.name}
                      onSelect={() => setSelectedRunName(run.name)}
                      onRenamed={(oldName, newName) => {
                        if (selectedRunName === oldName) setSelectedRunName(newName)
                      }}
                      onDeleted={(name) => {
                        if (selectedRunName === name) setSelectedRunName(null)
                      }}
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
              {t('training.jobLogs')}
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
  onRenamed,
  onDeleted,
}: {
  run: TrainingRun
  projectName: string
  isSelected: boolean
  onSelect: () => void
  onRenamed?: (oldName: string, newName: string) => void
  onDeleted?: (name: string) => void
}) {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const { t } = useTranslation()
  const [isStartingTB, setIsStartingTB] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState(run.name)
  const editInputRef = useRef<HTMLInputElement>(null)

  const renameMutation = useMutation({
    mutationFn: (newName: string) => api.training.renameRun(projectName, run.name, newName),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      queryClient.invalidateQueries({ queryKey: ['inference-results', projectName] })
      toast({ title: t('training.runRenamed'), type: 'success' })
      onRenamed?.(run.name, result.name)
      setIsEditing(false)
    },
    onError: (error: Error) => {
      toast({ title: t('training.renameFailed'), description: error.message, type: 'error' })
      setEditValue(run.name)
      setIsEditing(false)
    },
  })

  const commitRename = useCallback(() => {
    const trimmed = editValue.trim()
    if (!trimmed || trimmed === run.name) {
      setEditValue(run.name)
      setIsEditing(false)
      return
    }
    renameMutation.mutate(trimmed)
  }, [editValue, run.name, renameMutation])

  const cancelMutation = useMutation({
    mutationFn: () => api.training.cancel(projectName, run.name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      toast({ title: t('training.jobCancelled'), type: 'success' })
    },
    onError: (error: Error) => {
      toast({ title: t('training.cancelFailed'), description: error.message, type: 'error' })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => api.training.deleteRun(projectName, run.name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
      queryClient.invalidateQueries({ queryKey: ['inference-results', projectName] })
      toast({ title: t('training.runDeleted'), description: t('training.runDeletedDesc'), type: 'success' })
      onDeleted?.(run.name)
    },
    onError: (error: Error) => {
      toast({ title: t('training.deleteFailed'), description: error.message, type: 'error' })
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
      toast({ title: t('training.tbStarted'), description: t('training.tbStartedDesc', { url: result.url }), type: 'success' })
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
    } catch (error: any) {
      toast({ title: t('training.tbFailed'), description: error.message, type: 'error' })
    } finally {
      setIsStartingTB(false)
    }
  }

  const stopTensorBoard = async () => {
    try {
      await api.training.stopTensorBoard(projectName, run.name)
      toast({ title: t('training.tbStopped'), type: 'success' })
      queryClient.invalidateQueries({ queryKey: ['training-runs', projectName] })
    } catch (error: any) {
      toast({ title: t('training.tbStopFailed'), description: error.message, type: 'error' })
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
          {isEditing ? (
            <input
              ref={editInputRef}
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={(e) => {
                e.stopPropagation()
                if (e.key === 'Enter') commitRename()
                if (e.key === 'Escape') { setEditValue(run.name); setIsEditing(false) }
              }}
              onBlur={commitRename}
              onClick={(e) => e.stopPropagation()}
              className="font-medium text-sm bg-background border border-border rounded px-1.5 py-0.5 w-full outline-none focus:ring-1 focus:ring-primary"
              autoFocus
              disabled={renameMutation.isPending}
            />
          ) : (
            <h4
              className="font-medium text-sm truncate group/name flex items-center gap-1 cursor-text"
              onClick={(e) => { e.stopPropagation(); setEditValue(run.name); setIsEditing(true) }}
              title={t('training.clickToRename')}
            >
              {run.name}
              <Pencil className="h-3 w-3 text-muted-foreground opacity-0 group-hover/name:opacity-100 transition-opacity flex-shrink-0" />
            </h4>
          )}
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
          {run.status !== 'running' && run.status !== 'queued' && run.status !== 'pending' && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                if (confirm(t('training.deleteRunConfirm', { name: run.name }))) {
                  deleteMutation.mutate()
                }
              }}
              className="text-muted-foreground hover:text-destructive transition-colors"
              title="Delete run"
              disabled={deleteMutation.isPending}
            >
              <Trash2 className="h-3.5 w-3.5" />
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
          <p className="text-xs text-muted-foreground mt-1">
            {Math.round(run.progress * 100)}%
            {run.current_epoch != null && run.total_epochs != null && (
              <span className="ml-2">{t('common.epoch', { current: run.current_epoch! + 1, total: run.total_epochs })}</span>
            )}
          </p>
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
              {isStartingTB ? t('training.starting') : t('training.launchTensorboard')}
            </button>
          )}
        </div>
      )}

      {/* Show logs button for selected */}
      {isSelected && (run.status === 'queued' || run.status === 'running') && (
        <div className="mt-2 flex items-center gap-1 text-xs text-primary">
          <Eye className="h-3 w-3" />
          {t('training.viewingLogs')}
        </div>
      )}
    </div>
  )
}
