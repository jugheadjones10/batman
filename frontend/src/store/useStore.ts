import { create } from 'zustand'
import type { Project, BoundingBox } from '@/types'

interface AppState {
  // Current context (manual data image navigation)
  currentProject: Project | null
  currentImageIndex: number

  // Editor state
  selectedAnnotationId: number | null
  selectedTrackId: number | null
  isDrawing: boolean
  drawingBox: BoundingBox | null
  selectedClassId: number
  tool: 'select' | 'draw' | 'pan'

  // UI state
  sidebarOpen: boolean
  showTracks: boolean
  showProblems: boolean

  // Actions
  setCurrentProject: (project: Project | null) => void
  setCurrentImageIndex: (index: number) => void
  setSelectedAnnotation: (id: number | null) => void
  setSelectedTrack: (id: number | null) => void
  setIsDrawing: (drawing: boolean) => void
  setDrawingBox: (box: BoundingBox | null) => void
  setSelectedClassId: (id: number) => void
  setTool: (tool: 'select' | 'draw' | 'pan') => void
  toggleSidebar: () => void
  toggleTracks: () => void
  toggleProblems: () => void
  reset: () => void
}

const initialState = {
  currentProject: null,
  currentImageIndex: 0,
  selectedAnnotationId: null,
  selectedTrackId: null,
  isDrawing: false,
  drawingBox: null,
  selectedClassId: 0,
  tool: 'select' as const,
  sidebarOpen: true,
  showTracks: true,
  showProblems: false,
}

export const useStore = create<AppState>((set) => ({
  ...initialState,

  setCurrentProject: (project) => set({ currentProject: project }),
  setCurrentImageIndex: (index) => set({ currentImageIndex: index }),
  setSelectedAnnotation: (id) => set({ selectedAnnotationId: id }),
  setSelectedTrack: (id) => set({ selectedTrackId: id }),
  setIsDrawing: (drawing) => set({ isDrawing: drawing }),
  setDrawingBox: (box) => set({ drawingBox: box }),
  setSelectedClassId: (id) => set({ selectedClassId: id }),
  setTool: (tool) => set({ tool }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  toggleTracks: () => set((state) => ({ showTracks: !state.showTracks })),
  toggleProblems: () => set((state) => ({ showProblems: !state.showProblems })),
  reset: () => set(initialState),
}))

