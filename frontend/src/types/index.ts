// Project types
export interface Project {
  name: string
  path: string
  description?: string
  classes: string[]
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

// Video types
export interface Video {
  id: number
  filename: string
  width: number
  height: number
  fps: number
  duration: number
  total_frames: number
  has_proxy: boolean
  frame_count: number
  annotation_count: number
  created_at: string
}

export interface Frame {
  id: number
  video_id: number
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
  frame_id: number
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
  video_id: number
  start_frame: number
  end_frame: number
  annotation_count: number
  is_approved: boolean
  needs_review: boolean
}

// Training types
export interface TrainingConfig {
  base_model: string
  image_size: number
  batch_size: number
  epochs: number
  lr_preset: 'small' | 'medium' | 'large'
  augmentation_preset: 'none' | 'light' | 'standard' | 'heavy'
  val_split: number
  test_split: number
  freeze_backbone: boolean
  mixed_precision: boolean
  early_stopping_patience: number
}

export interface TrainingRun {
  id: number
  name: string
  label_iteration_id: number
  base_model: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  metrics?: {
    mAP50?: number
    'mAP50-95'?: number
    precision?: number
    recall?: number
  }
  checkpoint_path?: string
  latency_ms?: number
  tensorboard_url?: string
  started_at?: string
  completed_at?: string
  created_at: string
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
  frame_id: number
  frame_number: number
  timestamp: number
  video_id: number
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
}

export interface InferenceResult {
  frame_number: number
  timestamp: number
  detections: Detection[]
  inference_time_ms: number
}

