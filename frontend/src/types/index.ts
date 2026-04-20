// Project types
export interface Project {
  name: string
  path: string
  description?: string
  classes: string[]
  class_descriptions?: Record<string, string>
  config: ProjectConfig
  video_count: number
  frame_count: number
  annotation_count: number
  current_iteration: number
  created_at: string
  updated_at: string
}

export interface ProjectConfig {
  sample_mode: 'frames' | 'seconds'
  sample_interval: number
  tracking_mode: 'visible_only' | 'occlusion_tolerant'
  max_age: number
  iou_threshold: number
  min_hits: number
  use_appearance_embedding: boolean
}

// Video types (id: video_1 or legacy 1)
export interface Video {
  id: number | string
  filename: string
  width: number
  height: number
  fps: number
  duration: number
  total_frames: number
  has_proxy: boolean
  frame_count: number
  annotation_count: number
  /** Number of frames that have at least one annotation */
  annotated_frame_count?: number
  exclude_from_training: boolean
  created_at: string
}

export interface ManualDataImage {
  filename: string
  frame_id: string
  dataset: string
  width: number
  height: number
  annotation_count: number
  url: string
}

export interface ManualDataset {
  name: string
  source_key: string
  image_count: number
}

export interface Frame {
  id: number | string
  video_id: number | string
  frame_number: number
  timestamp: number
  image_path: string
  is_approved: boolean
  needs_review: boolean
  annotation_count?: number
}

// Annotation types
export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Annotation {
  id: number
  frame_id: number | string
  class_label_id: number
  class_name: string
  class_color: string
  box: BoundingBox
  track_id?: number
  confidence: number
  source: 'auto' | 'manual' | 'corrected'
  is_exemplar: boolean
  exemplar_type?: 'anchor' | 'correction'
  created_at: string
  updated_at: string
}

export interface Track {
  id: number
  track_id: number
  class_label_id: number
  class_name: string
  class_color: string
  video_id: number | string
  start_frame: number
  end_frame: number
  annotation_count: number
  is_approved: boolean
  needs_review: boolean
}

// Training types (RF-DETR only, GPU cluster execution)
export type RFDETRModelSize = 'nano' | 'small' | 'base' | 'medium' | 'large'
export type GPUType = 'h200' | 'h100-96' | 'h100-47' | 'a100-80' | 'a100-40' | 'nv'
export type DataSource = 'manual_data' | 'imports' | 'videos'
export type ManualDataSplitStrategy = 'proportional' | 'val_only' | 'train_only' | 'train_and_val' | 'all_splits'

export interface TrainingConfig {
  model: RFDETRModelSize
  epochs: number
  batch_size: number | null  // null = auto based on GPU
  image_size: number
  lr: number
  patience: number
  grad_accum: number
}

export interface GPUConfig {
  gpu_type: GPUType
  num_gpus: number
  time_limit: string
}

export interface DataConfig {
  sources?: DataSource[] | null
  manual_split_strategy: ManualDataSplitStrategy
  manual_datasets?: string[] | null
  exclude_manual_datasets?: string[] | null
  exclude_videos?: string[] | null
  filter_classes?: string[] | null
  max_frames_per_class?: number | null
  train_split: number
  val_split: number
  test_split: number
}

export interface TrainingSubmitRequest {
  label?: string | null
  training: TrainingConfig
  gpu: GPUConfig
  data: DataConfig
  infer_after: boolean
  infer_test_only: boolean
}

/** Request to run training locally (e.g. on Windows GPU). No GPU cluster config. */
export interface LocalTrainingSubmitRequest {
  label?: string | null
  training: TrainingConfig
  data: DataConfig
  infer_after: boolean
  infer_test_only: boolean
}

export interface DeviceInfo {
  device: string
  name: string
  memory_gb?: number
}

export interface DatasetExportConfig {
  format?: 'yolo' | 'coco' | 'both'
  include_unapproved?: boolean
  split_by_video?: boolean
  data_sources?: DataSource[] | null
  manual_data_split_strategy?: ManualDataSplitStrategy
  manual_datasets?: string[] | null
  exclude_manual_datasets?: string[] | null
}

export interface TrainingRun {
  id: number
  name: string
  status: 'queued' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout'
  model: string
  gpu_type?: string
  slurm_job_id?: string
  progress: number
  current_epoch?: number | null
  total_epochs?: number | null
  metrics?: {
    mAP50?: number
    'mAP50-95'?: number
    precision?: number
    recall?: number
  }
  checkpoint_path?: string
  latency_ms?: number
  tensorboard_url?: string
  config?: Record<string, unknown>
  started_at?: string
  completed_at?: string
  created_at: string
}

export interface InferenceGPUSubmitRequest {
  run_name?: string | null
  video_ids?: string[] | null
  test_only: boolean
  model: RFDETRModelSize
  confidence: number
  frame_interval: number
  track: boolean
  track_thresh: number
  track_buffer: number
  match_thresh: number
  no_video: boolean
  gpu: GPUConfig
}

export interface GPUStatus {
  connected: boolean
  host: string
  user: string
}

// Labeling types
export interface LabelIteration {
  id: number
  version: number
  description?: string
  total_annotations: number
  total_tracks: number
  approved_frames: number
  is_active: boolean
  created_at: string
}

export interface ProblemQueueItem {
  frame_id: number | string
  frame_number: number
  timestamp: number
  video_id: number | string
  problem_type: string
  severity: number
  description: string
  affected_track_ids: number[]
}

// Inference types
export interface InferenceConfig {
  model_run_id: number
  confidence_threshold: number
  iou_threshold: number
  max_detections: number
  enable_tracking: boolean
  tracking_mode: 'visible_only' | 'occlusion_tolerant'
  detection_interval: number  // Run detection every N frames (1 = every frame, higher = faster)
}

export interface Detection {
  box: BoundingBox
  confidence: number
  class_id: number
  class_name: string
  track_id?: number
  z_mm?: number
}

export interface InferenceResult {
  frame_number: number
  timestamp: number
  detections: Detection[]
  inference_time_ms: number
}

export interface InferenceResultSummary {
  run_name: string
  video_id: string
  inference_id: string
  created_at: string
  config: {
    confidence_threshold: number
    iou_threshold?: number
    frame_interval: number
    tracking: boolean
    tracking_mode: string
  }
  stats: {
    total_frames: number
    keyframes: number
    total_detections: number
    avg_inference_time_ms: number
  }
  has_video: boolean
  has_z_video?: boolean
}

export interface InferenceResultMatrix {
  runs: string[]
  videos: string[]
  results: Record<string, Record<string, InferenceResultSummary[] | null>>
}

export interface ZCalibrationLabel {
  frame_number: number
  z_mm: number
  detection_index: number
}

export interface ZCalibrationModel {
  type: string
  k?: number
  a?: number
  b?: number
  focal_length_px?: number
  targets?: { class_name: string; model: ZCalibrationModel }[]
}

export interface ZCalibrationTarget {
  class_name: string
  real_width_mm: number
}

export interface ZCalibration {
  labels: ZCalibrationLabel[]
  model: ZCalibrationModel | null
  class_name: string
  size_metric: string
  reference_real_width_mm?: number | null
  targets?: ZCalibrationTarget[] | null
  video_resolution?: { width: number; height: number }
}

