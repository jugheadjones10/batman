const API_BASE = '/api'

// In dev, video can load from backend directly to avoid proxy streaming issues (set in .env: VITE_API_ORIGIN=http://127.0.0.1:8000)
const getVideoBase = () =>
  (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_API_ORIGIN) || ''

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `Request failed: ${response.status}`)
  }

  return response.json()
}

export const api = {
  // Health check
  health: () => request<{ status: string }>('/health'),

  // Device info (for local GPU training/inference)
  device: {
    getInfo: () => request<import('@/types').DeviceInfo>('/device-info'),
  },

  // Projects
  projects: {
    list: () => request<import('@/types').Project[]>('/projects'),
    get: (name: string) => request<import('@/types').Project>(`/projects/${name}`),
    create: (data: { name: string; description?: string; classes?: string[] }) =>
      request<import('@/types').Project>('/projects', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    updateClasses: (name: string, classes: string[]) =>
      request<import('@/types').Project>(`/projects/${name}/classes`, {
        method: 'PUT',
        body: JSON.stringify(classes),
      }),
    updateConfig: (name: string, config: import('@/types').ProjectConfig) =>
      request<import('@/types').Project>(`/projects/${name}/config`, {
        method: 'PUT',
        body: JSON.stringify(config),
      }),
    updateClassDescriptions: (name: string, descriptions: Record<string, string>) =>
      request<import('@/types').Project>(`/projects/${name}/class-descriptions`, {
        method: 'PUT',
        body: JSON.stringify(descriptions),
      }),
    delete: (name: string) =>
      request<{ message: string }>(`/projects/${name}`, { method: 'DELETE' }),
    iterations: (name: string) =>
      request<import('@/types').LabelIteration[]>(`/projects/${name}/iterations`),
    activateIteration: (name: string, iterationId: number) =>
      request<{ message: string }>(`/projects/${name}/iterations/${iterationId}/activate`, {
        method: 'POST',
      }),
  },

  // Videos
  videos: {
    list: (projectName: string) =>
      request<import('@/types').Video[]>(`/projects/${projectName}/videos`),
    get: (projectName: string, videoId: number | string) =>
      request<import('@/types').Video>(`/projects/${projectName}/videos/${videoId}`),
    upload: async (
      projectName: string,
      file: File,
      options?: { exclude_from_training?: boolean; create_proxy?: boolean }
    ) => {
      const formData = new FormData()
      formData.append('file', file)
      const params = new URLSearchParams()
      if (options?.exclude_from_training !== undefined)
        params.set('exclude_from_training', String(options.exclude_from_training))
      if (options?.create_proxy !== undefined) params.set('create_proxy', String(options.create_proxy))
      const url = `${API_BASE}/projects/${projectName}/videos${params.toString() ? `?${params}` : ''}`
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(error.detail)
      }
      return response.json()
    },
    extractFrames: (
      projectName: string,
      videoId: number | string,
      sampling?: { mode: string; interval: number }
    ) =>
      request<import('@/types').Frame[]>(
        `/projects/${projectName}/videos/${videoId}/extract-frames`,
        {
          method: 'POST',
          body: JSON.stringify(sampling || {}),
        }
      ),
    getFrames: (projectName: string, videoId: number | string) =>
      request<import('@/types').Frame[]>(`/projects/${projectName}/videos/${videoId}/frames`),
    update: (projectName: string, videoId: number | string, data: { exclude_from_training?: boolean }) =>
      request<import('@/types').Video>(`/projects/${projectName}/videos/${videoId}`, {
        method: 'PATCH',
        body: JSON.stringify(data),
      }),
    delete: (projectName: string, videoId: number | string) =>
      request<{ message: string }>(`/projects/${projectName}/videos/${videoId}`, {
        method: 'DELETE',
      }),
    streamUrl: (projectName: string, videoId: number | string, proxy = true) =>
      `${API_BASE}/projects/${projectName}/videos/${videoId}/stream?proxy=${proxy}`,
    frameUrl: (projectName: string, videoId: number | string, frameNumber: number) =>
      `${API_BASE}/projects/${projectName}/videos/${videoId}/frame/${frameNumber}`,
    frameImageUrl: (projectName: string, videoId: number | string, frameId: string) =>
      `${API_BASE}/projects/${projectName}/videos/${videoId}/frames/${encodeURIComponent(frameId)}/image`,
    thumbnailUrl: (projectName: string, videoId: number | string) =>
      `${API_BASE}/projects/${projectName}/videos/${videoId}/thumbnail`,
  },

  // Annotations
  annotations: {
    listForFrame: (projectName: string, frameId: number | string) =>
      request<import('@/types').Annotation[]>(
        `/projects/${projectName}/frames/${frameId}/annotations`
      ),
    create: (projectName: string, data: {
      frame_id: number | string
      class_label_id: number
      box: import('@/types').BoundingBox
      polygon?: number[][]
      track_id?: number
      source?: string
    }) =>
      request<import('@/types').Annotation>(`/projects/${projectName}/annotations`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    update: (projectName: string, annotationId: number, data: Partial<{
      class_label_id: number
      box: import('@/types').BoundingBox
      polygon: number[][]
      track_id: number
    }>) =>
      request<import('@/types').Annotation>(
        `/projects/${projectName}/annotations/${annotationId}`,
        {
          method: 'PUT',
          body: JSON.stringify(data),
        }
      ),
    delete: (projectName: string, annotationId: number) =>
      request<{ message: string }>(`/projects/${projectName}/annotations/${annotationId}`, {
        method: 'DELETE',
      }),
    clearFrame: (projectName: string, frameId: number | string) =>
      request<{ message: string; deleted: number }>(
        `/projects/${projectName}/frames/${frameId}/annotations`,
        { method: 'DELETE' }
      ),
    clearFrames: (projectName: string, frameIds: (number | string)[]) =>
      request<{ message: string; deleted: number; frames_cleared: number }>(
        `/projects/${projectName}/annotations/clear-frames`,
        { method: 'POST', body: JSON.stringify({ frame_ids: frameIds.map(String) }) }
      ),
  },

  // Tracks
  tracks: {
    listForVideo: (projectName: string, videoId: number | string) =>
      request<import('@/types').Track[]>(`/projects/${projectName}/videos/${videoId}/tracks`),
    update: (projectName: string, trackId: number, data: Partial<{
      class_label_id: number
      is_approved: boolean
      needs_review: boolean
    }>) =>
      request<import('@/types').Track>(`/projects/${projectName}/tracks/${trackId}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    split: (projectName: string, trackId: number, splitFrame: number) =>
      request<{ message: string }>(`/projects/${projectName}/tracks/split`, {
        method: 'POST',
        body: JSON.stringify({ track_id: trackId, split_frame: splitFrame }),
      }),
    merge: (projectName: string, sourceTrackId: number, targetTrackId: number) =>
      request<{ message: string }>(`/projects/${projectName}/tracks/merge`, {
        method: 'POST',
        body: JSON.stringify({
          source_track_id: sourceTrackId,
          target_track_id: targetTrackId,
        }),
      }),
  },

  // Labeling
  labeling: {
    autoLabel: (projectName: string, data?: {
      video_ids?: (number | string)[]
      frame_ids?: (number | string)[]
      source_keys?: string[]
      class_descriptions?: Record<string, string>
      confidence?: number
      skip_labeled_frames?: boolean
      use_exemplars?: boolean
      tracking_mode?: string
    }) =>
      request<{ job_id: string }>(`/projects/${projectName}/labeling/auto-label`, {
        method: 'POST',
        body: JSON.stringify(data || {}),
      }),
    getLabelingStatus: (projectName: string, jobId: string) =>
      request<{
        status: string
        progress: number
        frames_processed: number
        total_frames: number
        annotations_created: number
        tracks_created: number
        message: string
      }>(`/projects/${projectName}/labeling/auto-label/${jobId}/status`),
    refine: (projectName: string, data: {
      scope: 'clip_range' | 'touched_tracks' | 'full'
      video_id?: number | string
      track_ids?: number[]
    }) =>
      request<{ job_id: string }>(`/projects/${projectName}/labeling/refine`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    createIteration: (projectName: string, description?: string) =>
      request<{ iteration_id: number }>(`/projects/${projectName}/labeling/create-iteration`, {
        method: 'POST',
        body: JSON.stringify({ description }),
      }),
    getProblemQueue: (projectName: string, videoId?: number | string) =>
      request<import('@/types').ProblemQueueItem[]>(
        `/projects/${projectName}/problem-queue${videoId != null ? `?video_id=${videoId}` : ''}`
      ),
  },

  // GPU cluster connection
  gpu: {
    connect: (password: string) =>
      request<{ status: string; hostname?: string }>('/gpu/connect', {
        method: 'POST',
        body: JSON.stringify({ password }),
      }),
    disconnect: () =>
      request<{ status: string }>('/gpu/disconnect', { method: 'POST' }),
    getStatus: () =>
      request<import('@/types').GPUStatus>('/gpu/status'),
  },

  // Training (RF-DETR, GPU cluster + local)
  training: {
    exportDataset: (projectName: string, config?: import('@/types').DatasetExportConfig) =>
      request<{
        format: string
        output_path: string
        train_images: number
        val_images: number
        test_images: number
        total_annotations: number
        classes: string[]
      }>(`/projects/${projectName}/training/export-dataset`, {
        method: 'POST',
        body: JSON.stringify(config || {}),
      }),
    submit: (projectName: string, data: import('@/types').TrainingSubmitRequest) =>
      request<{ job_id: string; run_name: string }>(
        `/projects/${projectName}/training/submit`,
        { method: 'POST', body: JSON.stringify(data) }
      ),
    submitLocal: (projectName: string, data: import('@/types').LocalTrainingSubmitRequest) =>
      request<{ run_name: string; pid: number; message: string }>(
        `/projects/${projectName}/training/submit-local`,
        { method: 'POST', body: JSON.stringify(data) }
      ),
    cancel: (projectName: string, runName: string) =>
      request<{ status: string }>(`/projects/${projectName}/training/runs/${encodeURIComponent(runName)}/cancel`, {
        method: 'POST',
      }),
    listRuns: (projectName: string) =>
      request<import('@/types').TrainingRun[]>(`/projects/${projectName}/training/runs`),
    streamLogsUrl: (projectName: string, runName: string) =>
      `${API_BASE}/projects/${projectName}/training/runs/${runName}/logs`,
    streamLocalLogsUrl: (projectName: string, runName: string) =>
      `${API_BASE}/projects/${projectName}/training/runs/${runName}/local-logs`,
    startTensorBoard: (projectName: string, runName: string) =>
      request<{ status: string; port: number; url: string }>(
        `/projects/${projectName}/training/runs/${runName}/tensorboard/start`,
        { method: 'POST' }
      ),
    stopTensorBoard: (projectName: string, runName: string) =>
      request<{ status: string }>(
        `/projects/${projectName}/training/runs/${runName}/tensorboard/stop`,
        { method: 'POST' }
      ),
    getTensorBoardStatus: (projectName: string, runName: string) =>
      request<{ running: boolean; port?: number; url?: string }>(
        `/projects/${projectName}/training/runs/${runName}/tensorboard/status`
      ),
    renameRun: (projectName: string, runName: string, newName: string) =>
      request<{ name: string }>(
        `/projects/${projectName}/training/runs/${encodeURIComponent(runName)}/rename`,
        { method: 'PATCH', body: JSON.stringify({ new_name: newName }) }
      ),
    deleteRun: (projectName: string, runName: string) =>
      request<{ message: string }>(
        `/projects/${projectName}/training/runs/${encodeURIComponent(runName)}`,
        { method: 'DELETE' }
      ),
  },

  // Inference
  inference: {
    loadModel: (projectName: string, runName: string, device?: string) =>
      request<{ message: string }>(`/projects/${projectName}/inference/load-model`, {
        method: 'POST',
        body: JSON.stringify({ run_name: runName, device: device ?? undefined }),
      }),
    runOnImage: (projectName: string, frameId: number, config?: {
      confidence_threshold?: number
      iou_threshold?: number
    }) =>
      request<{
        detections: import('@/types').Detection[]
        inference_time_ms: number
      }>(
        `/projects/${projectName}/inference/run-on-image?frame_id=${frameId}&confidence_threshold=${config?.confidence_threshold || 0.5}&iou_threshold=${config?.iou_threshold || 0.45}`,
        { method: 'POST' }
      ),
    runOnVideo: (projectName: string, videoId: number | string, config: import('@/types').InferenceConfig) =>
      request<{
        total_frames: number
        avg_fps: number
        avg_inference_time_ms: number
        results: import('@/types').InferenceResult[]
      }>(`/projects/${projectName}/inference/run-on-video/${videoId}`, {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    /**
     * Run inference with a live progress stream (SSE). Yields each event from
     * the backend so the caller can render a progress bar backed by the actual
     * per-frame progress of the inference worker.
     *
     * Preconditions (no model loaded, missing video, etc.) are returned as
     * HTTP 4xx *before* the stream opens, so failures from bad state throw
     * immediately with the backend detail message.
     */
    runOnVideoWithProgress: async function* (
      projectName: string,
      videoId: number | string,
      config: import('@/types').InferenceConfig,
      signal?: AbortSignal,
    ): AsyncGenerator<import('@/types').InferenceProgressEvent> {
      const response = await fetch(
        `${API_BASE}/projects/${projectName}/inference/run-on-video/${videoId}/stream`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config),
          signal,
        },
      )
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Inference failed' }))
        throw new Error(error.detail || `Request failed: ${response.status}`)
      }
      const reader = response.body?.getReader()
      if (!reader) throw new Error('Streaming not supported')

      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        // SSE events are separated by a blank line (\n\n).
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''
        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (!payload) continue
          try {
            yield JSON.parse(payload) as import('@/types').InferenceProgressEvent
          } catch {
            // Malformed frame — skip but keep streaming.
          }
        }
      }
    },
    exportVideo: (projectName: string, videoId: number | string, config: import('@/types').InferenceConfig) =>
      request<{
        output_path: string
        total_frames: number
        avg_fps: number
      }>(`/projects/${projectName}/inference/export-video/${videoId}`, {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    listResults: (projectName: string) =>
      request<import('@/types').InferenceResultMatrix>(
        `/projects/${projectName}/inference/results`
      ),
    getResult: (projectName: string, runName: string, videoId: string, inferenceId: string) =>
      request<import('@/types').InferenceResultSummary & { frames: import('@/types').InferenceResult[] }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}`
      ),
    videoUrl: (
      projectName: string,
      runName: string,
      videoId: string,
      inferenceId: string,
      variant?: 'z' | 'raw' | 'bytetrack',
    ) => {
      const base = `${getVideoBase() || ''}${API_BASE}/projects/${encodeURIComponent(projectName)}/inference/results/${encodeURIComponent(runName)}/${encodeURIComponent(videoId)}/${encodeURIComponent(inferenceId)}/video`
      return variant ? `${base}?variant=${variant}` : base
    },
    zVideoUrl: (projectName: string, runName: string, videoId: string, inferenceId: string) =>
      `${getVideoBase() || ''}${API_BASE}/projects/${encodeURIComponent(projectName)}/inference/results/${encodeURIComponent(runName)}/${encodeURIComponent(videoId)}/${encodeURIComponent(inferenceId)}/video?variant=z`,
    deleteResult: (projectName: string, runName: string, videoId: string, inferenceId: string) =>
      request<{ message: string }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}`,
        { method: 'DELETE' }
      ),
    extractFrames: async (
      projectName: string,
      runName: string,
      videoId: string,
      inferenceId: string,
      frameNumbers: number[],
    ): Promise<Blob> => {
      const response = await fetch(
        `${API_BASE}/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/extract-frames`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame_numbers: frameNumbers }),
        },
      )
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Extract failed' }))
        throw new Error(error.detail || `Request failed: ${response.status}`)
      }
      return response.blob()
    },
    submitGpu: (projectName: string, data: import('@/types').InferenceGPUSubmitRequest) =>
      request<{ job_id: string; run_name: string }>(
        `/projects/${projectName}/inference/submit-gpu`,
        { method: 'POST', body: JSON.stringify(data) }
      ),
    cancelGpu: (projectName: string, jobName: string) =>
      request<{ status: string }>(
        `/projects/${projectName}/inference/gpu-jobs/${jobName}/cancel`,
        { method: 'POST' }
      ),
    gpuLogsUrl: (projectName: string, jobName: string) =>
      `${API_BASE}/projects/${projectName}/inference/gpu-jobs/${jobName}/logs`,
    getZCalibration: (projectName: string, runName: string, videoId: string, inferenceId: string) =>
      request<{ z_calibration: import('@/types').ZCalibration | null }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/z-calibration`
      ),
    saveZCalibration: (
      projectName: string, runName: string, videoId: string, inferenceId: string,
      labels: import('@/types').ZCalibrationLabel[], referenceClass: string,
      opts?: {
        lengthMm?: number | null
        targetClasses?: string[]
      },
    ) =>
      request<{ message: string; labels_count: number }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/z-calibration`,
        {
          method: 'POST',
          body: JSON.stringify({
            labels,
            reference_class: referenceClass,
            length_mm: opts?.lengthMm ?? null,
            target_classes: opts?.targetClasses ?? [],
          }),
        }
      ),
    applyZEstimation: (projectName: string, runName: string, videoId: string, inferenceId: string) =>
      request<{ message: string; model: import('@/types').ZCalibrationModel }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/z-estimate`,
        { method: 'POST' }
      ),
    exportZVideo: (projectName: string, runName: string, videoId: string, inferenceId: string) =>
      request<{ message: string; output_path: string }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/z-export-video`,
        { method: 'POST' }
      ),
    rerender: (
      projectName: string,
      runName: string,
      videoId: string,
      inferenceId: string,
      renderMode: 'polygon' | 'bbox',
    ) =>
      request<{ message: string; render_mode: string; frames_written: number; output_path: string }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/rerender`,
        { method: 'POST', body: JSON.stringify({ render_mode: renderMode }) }
      ),
    renderComparison: (
      projectName: string,
      runName: string,
      videoId: string,
      inferenceId: string,
      options?: {
        render_mode?: 'polygon' | 'bbox'
        track_activation_threshold?: number
        lost_track_buffer?: number
        minimum_matching_threshold?: number
      },
    ) =>
      request<{
        message: string
        frames_written: number
        has_raw_video: boolean
        has_bytetrack_video: boolean
        bytetrack_config: {
          track_activation_threshold: number
          lost_track_buffer: number
          minimum_matching_threshold: number
        }
      }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/render-comparison`,
        { method: 'POST', body: JSON.stringify(options ?? {}) },
      ),
    getBytetrackFrames: (
      projectName: string,
      runName: string,
      videoId: string,
      inferenceId: string,
      params?: {
        track_activation_threshold?: number
        lost_track_buffer?: number
        minimum_matching_threshold?: number
      },
    ) => {
      const q = new URLSearchParams()
      if (params?.track_activation_threshold != null) {
        q.set('track_activation_threshold', String(params.track_activation_threshold))
      }
      if (params?.lost_track_buffer != null) {
        q.set('lost_track_buffer', String(params.lost_track_buffer))
      }
      if (params?.minimum_matching_threshold != null) {
        q.set('minimum_matching_threshold', String(params.minimum_matching_threshold))
      }
      const qs = q.toString()
      return request<{
        frames: import('@/types').InferenceResult[]
        bytetrack_config: {
          track_activation_threshold: number
          lost_track_buffer: number
          minimum_matching_threshold: number
        }
      }>(
        `/projects/${projectName}/inference/results/${runName}/${videoId}/${inferenceId}/bytetrack-frames${qs ? `?${qs}` : ''}`,
      )
    },
  },

  // Import
  import: {
    fromRoboflow: (projectName: string, config: {
      api_key: string
      workspace: string
      project: string
      version: number
      format?: string
    }) =>
      request<{
        images_imported: number
        annotations_imported: number
        classes_added: string[]
        splits_imported: string[]
        message: string
      }>(`/projects/${projectName}/import/roboflow`, {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    fromRoboflowWithProgress: async function* (
      projectName: string,
      config: {
        api_key: string
        workspace: string
        project: string
        version: number
        format?: string
      },
      onProgress?: (progress: {
        status: 'downloading' | 'processing' | 'complete' | 'error'
        progress: number
        message: string
        images_imported?: number
        annotations_imported?: number
        classes_added?: string[]
        splits_imported?: string[]
      }) => void
    ) {
      const response = await fetch(`${API_BASE}/projects/${projectName}/import/roboflow/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        throw new Error(`Import failed: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6))
            if (onProgress) onProgress(data)
            yield data
          }
        }
      }
    },
    fromLocalCoco: (projectName: string, config: {
      path: string
      split?: string
    }) =>
      request<{
        images_imported: number
        annotations_imported: number
        classes_added: string[]
        message: string
      }>(`/projects/${projectName}/import/local-coco`, {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    listDatasets: (projectName: string) =>
      request<{
        video_id: number | string
        source: string
        image_count: number
        annotation_count: number
        sample_images: string[]
        classes: string[]
      }[]>(`/projects/${projectName}/import/datasets`),
    listImages: (projectName: string, videoId: number | string, offset = 0, limit = 50) =>
      request<{
        total: number
        offset: number
        limit: number
        images: { frame_id: number | string; url: string; original_filename: string; split: string }[]
      }>(`/projects/${projectName}/import/images/${videoId}?offset=${offset}&limit=${limit}`),
    deleteDataset: (projectName: string, videoId: number | string) =>
      request<{ message: string; images_deleted: number; annotations_deleted: number }>(
        `/projects/${projectName}/import/datasets/${videoId}`,
        { method: 'DELETE' }
      ),
    imageUrl: (projectName: string, videoId: number | string, filename: string) =>
      `${API_BASE}/projects/${projectName}/import/image/${videoId}/${filename}`,
  },

  // Manual data (folder of images at project_root/manual_data)
  manualData: {
    sync: (projectName: string) =>
      request<{ images_found: number; images_added: number; images_removed: number; total: number }>(
        `/projects/${projectName}/manual-data/sync`,
        { method: 'POST' }
      ),
    upload: async (projectName: string, files: File[], dataset?: string) => {
      const formData = new FormData()
      files.forEach((f) => formData.append('files', f))
      const url = dataset?.trim()
        ? `/projects/${projectName}/manual-data/upload?dataset=${encodeURIComponent(dataset.trim())}`
        : `/projects/${projectName}/manual-data/upload`
      const response = await fetch(`${API_BASE}${url}`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(error.detail || 'Upload failed')
      }
      return response.json() as Promise<{
        uploaded: number
        dataset: string
        filenames: string[]
        sync: { images_found: number; images_added: number; images_removed: number; total: number } | null
      }>
    },
    listDatasets: (projectName: string) =>
      request<{ datasets: import('@/types').ManualDataset[] }>(
        `/projects/${projectName}/manual-data/datasets`
      ),
    listImages: (projectName: string, offset = 0, limit = 500, dataset?: string) => {
      const params = new URLSearchParams({ offset: String(offset), limit: String(limit) })
      if (dataset) params.set('dataset', dataset)
      return request<{
        total: number
        offset: number
        limit: number
        images: import('@/types').ManualDataImage[]
      }>(`/projects/${projectName}/manual-data/images?${params}`)
    },
    imageUrl: (projectName: string, filename: string) =>
      `${API_BASE}/projects/${projectName}/manual-data/image/${encodeURIComponent(filename)}`,
    deleteImage: (projectName: string, frameId: string) =>
      request<{ deleted: boolean; frame_id: string; annotations_deleted: number }>(
        `/projects/${projectName}/manual-data/images/${encodeURIComponent(frameId)}`,
        { method: 'DELETE' }
      ),
    renameDataset: (projectName: string, datasetName: string, newName: string) =>
      request<{ old_name: string; new_name: string; frames_migrated: number }>(
        `/projects/${projectName}/manual-data/datasets/${encodeURIComponent(datasetName)}`,
        { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ new_name: newName }) }
      ),
  },

  // Class management
  classes: {
    getDetails: (projectName: string) =>
      request<{
        id: number
        name: string
        source: string
        annotation_count: number
        annotation_sources: Record<string, number>
      }[]>(`/projects/${projectName}/classes/details`),
    rename: (projectName: string, oldName: string, newName: string) =>
      request<{ message: string }>(`/projects/${projectName}/classes/rename`, {
        method: 'POST',
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      }),
    merge: (projectName: string, sourceClasses: string[], targetClass: string) =>
      request<{ message: string; annotations_updated: number; classes_removed: string[] }>(
        `/projects/${projectName}/classes/merge`,
        {
          method: 'POST',
          body: JSON.stringify({ source_classes: sourceClasses, target_class: targetClass }),
        }
      ),
    delete: (projectName: string, className: string, deleteAnnotations: boolean = true) =>
      request<{ message: string; annotations_deleted: number }>(
        `/projects/${projectName}/classes/${encodeURIComponent(className)}?delete_annotations=${deleteAnnotations}`,
        { method: 'DELETE' }
      ),
  },
}

