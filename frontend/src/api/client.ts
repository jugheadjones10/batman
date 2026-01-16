const API_BASE = '/api'

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
    get: (projectName: string, videoId: number) =>
      request<import('@/types').Video>(`/projects/${projectName}/videos/${videoId}`),
    upload: async (projectName: string, file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch(`${API_BASE}/projects/${projectName}/videos`, {
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
      videoId: number,
      sampling?: { mode: string; interval: number }
    ) =>
      request<import('@/types').Frame[]>(
        `/projects/${projectName}/videos/${videoId}/extract-frames`,
        {
          method: 'POST',
          body: JSON.stringify(sampling || {}),
        }
      ),
    getFrames: (projectName: string, videoId: number) =>
      request<import('@/types').Frame[]>(`/projects/${projectName}/videos/${videoId}/frames`),
    delete: (projectName: string, videoId: number) =>
      request<{ message: string }>(`/projects/${projectName}/videos/${videoId}`, {
        method: 'DELETE',
      }),
    streamUrl: (projectName: string, videoId: number, proxy = true) =>
      `${API_BASE}/projects/${projectName}/videos/${videoId}/stream?proxy=${proxy}`,
    frameUrl: (projectName: string, videoId: number, frameNumber: number) =>
      `${API_BASE}/projects/${projectName}/videos/${videoId}/frame/${frameNumber}`,
  },

  // Annotations
  annotations: {
    listForFrame: (projectName: string, frameId: number) =>
      request<import('@/types').Annotation[]>(
        `/projects/${projectName}/frames/${frameId}/annotations`
      ),
    create: (projectName: string, data: {
      frame_id: number
      class_label_id: number
      box: import('@/types').BoundingBox
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
  },

  // Tracks
  tracks: {
    listForVideo: (projectName: string, videoId: number) =>
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
      video_ids?: number[]
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
      video_id?: number
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
    getProblemQueue: (projectName: string, videoId?: number) =>
      request<import('@/types').ProblemQueueItem[]>(
        `/projects/${projectName}/problem-queue${videoId ? `?video_id=${videoId}` : ''}`
      ),
  },

  // Training
  training: {
    exportDataset: (projectName: string, config?: {
      format?: string
      include_unapproved?: boolean
    }) =>
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
    start: (projectName: string, data: {
      name: string
      label_iteration_id: number
      config: import('@/types').TrainingConfig
    }) =>
      request<{ run_id: number }>(`/projects/${projectName}/training/start`, {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    listRuns: (projectName: string) =>
      request<import('@/types').TrainingRun[]>(`/projects/${projectName}/training/runs`),
    getRun: (projectName: string, runId: number) =>
      request<import('@/types').TrainingRun>(`/projects/${projectName}/training/runs/${runId}`),
    getProgress: (projectName: string, runId: number) =>
      request<{
        run_id: number
        status: string
        progress: number
        current_epoch: number
        total_epochs: number
        metrics?: object
      }>(`/projects/${projectName}/training/runs/${runId}/progress`),
    // TensorBoard management
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
  },

  // Inference
  inference: {
    loadModel: (projectName: string, runId: number) =>
      request<{ message: string }>(`/projects/${projectName}/inference/load-model`, {
        method: 'POST',
        body: JSON.stringify({ run_id: runId }),
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
    runOnVideo: (projectName: string, videoId: number, config: import('@/types').InferenceConfig) =>
      request<{
        total_frames: number
        avg_fps: number
        avg_inference_time_ms: number
        results: import('@/types').InferenceResult[]
      }>(`/projects/${projectName}/inference/run-on-video/${videoId}`, {
        method: 'POST',
        body: JSON.stringify(config),
      }),
    exportVideo: (projectName: string, videoId: number, config: import('@/types').InferenceConfig) =>
      request<{
        output_path: string
        total_frames: number
        avg_fps: number
      }>(`/projects/${projectName}/inference/export-video/${videoId}`, {
        method: 'POST',
        body: JSON.stringify(config),
      }),
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
        video_id: number
        source: string
        image_count: number
        annotation_count: number
        sample_images: string[]
      }[]>(`/projects/${projectName}/import/datasets`),
    listImages: (projectName: string, videoId: number, offset = 0, limit = 50) =>
      request<{
        total: number
        offset: number
        limit: number
        images: { frame_id: number; url: string; original_filename: string; split: string }[]
      }>(`/projects/${projectName}/import/images/${videoId}?offset=${offset}&limit=${limit}`),
    deleteDataset: (projectName: string, videoId: number) =>
      request<{ message: string; images_deleted: number; annotations_deleted: number }>(
        `/projects/${projectName}/import/datasets/${videoId}`,
        { method: 'DELETE' }
      ),
    imageUrl: (projectName: string, videoId: number, filename: string) =>
      `${API_BASE}/projects/${projectName}/import/image/${videoId}/${filename}`,
  },

  // Class management
  classes: {
    getDetails: (projectName: string) =>
      request<{ id: number; name: string; source: string; annotation_count: number }[]>(
        `/projects/${projectName}/classes/details`
      ),
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
  },
}

