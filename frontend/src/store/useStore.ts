import { create } from 'zustand'
import type { Project, Video, Frame, Annotation, Track, BoundingBox } from '@/types'

interface AppState {
  // Current context
  currentProject: Project | null
  currentVideo: Video | null
  currentFrame: Frame | null
  currentFrameIndex: number

  // Editor state
  selectedAnnotationId: number | null
  selectedTrackId: number | null
  isDrawing: boolean
  drawingBox: BoundingBox | null
  selectedClassId: number
  tool: 'select' | 'draw' | 'pan'

  // Video player state
  isPlaying: boolean
  playbackSpeed: number
  volume: number

  // UI state
  sidebarOpen: boolean
  showTracks: boolean
  showProblems: boolean

  // Actions
  setCurrentProject: (project: Project | null) => void
  setCurrentVideo: (video: Video | null) => void
  setCurrentFrame: (frame: Frame | null) => void
  setCurrentFrameIndex: (index: number) => void
  setSelectedAnnotation: (id: number | null) => void
  setSelectedTrack: (id: number | null) => void
  setIsDrawing: (drawing: boolean) => void
  setDrawingBox: (box: BoundingBox | null) => void
  setSelectedClassId: (id: number) => void
  setTool: (tool: 'select' | 'draw' | 'pan') => void
  setIsPlaying: (playing: boolean) => void
  setPlaybackSpeed: (speed: number) => void
  toggleSidebar: () => void
  toggleTracks: () => void
  toggleProblems: () => void
  reset: () => void
}

const initialState = {
  currentProject: null,
  currentVideo: null,
  currentFrame: null,
  currentFrameIndex: 0,
  selectedAnnotationId: null,
  selectedTrackId: null,
  isDrawing: false,
  drawingBox: null,
  selectedClassId: 0,
  tool: 'select' as const,
  isPlaying: false,
  playbackSpeed: 1,
  volume: 1,
  sidebarOpen: true,
  showTracks: true,
  showProblems: false,
}

export const useStore = create<AppState>((set) => ({
  ...initialState,

  setCurrentProject: (project) => set({ currentProject: project }),
  setCurrentVideo: (video) => set({ currentVideo: video }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setCurrentFrameIndex: (index) => set({ currentFrameIndex: index }),
  setSelectedAnnotation: (id) => set({ selectedAnnotationId: id }),
  setSelectedTrack: (id) => set({ selectedTrackId: id }),
  setIsDrawing: (drawing) => set({ isDrawing: drawing }),
  setDrawingBox: (box) => set({ drawingBox: box }),
  setSelectedClassId: (id) => set({ selectedClassId: id }),
  setTool: (tool) => set({ tool }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  toggleTracks: () => set((state) => ({ showTracks: !state.showTracks })),
  toggleProblems: () => set((state) => ({ showProblems: !state.showProblems })),
  reset: () => set(initialState),
}))

