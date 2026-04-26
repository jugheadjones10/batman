import { useState, useRef, useEffect, useCallback, useImperativeHandle, forwardRef } from 'react'
import type { Annotation, BoundingBox } from '@/types'

const MIN_BOX_NORM = 5 / 1024
const HANDLE_SIZE_NORM = 12 / 1024
const VERTEX_RADIUS_PX = 5
const VERTEX_HIT_RADIUS_PX = 9
const CLOSE_POLYGON_RADIUS_PX = 12
const COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

type Tool = 'bbox' | 'polygon'
type DragMode = 'none' | 'draw' | 'move' | 'resize' | 'move-vertex' | 'move-polygon'
type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | null

type UndoAction =
  | { type: 'create'; annotation: Annotation }
  | {
      type: 'update'
      id: number
      previousBox: BoundingBox
      previousPolygon?: number[][] | null
    }
  | { type: 'delete'; annotation: Annotation }
  | { type: 'restore'; annotation: Annotation }

export interface AnnotationCanvasProps {
  imageUrl: string
  imageWidth: number
  imageHeight: number
  annotations: Annotation[]
  selectedAnnotationId: number | null
  selectedClassId: number
  classes: string[]
  onCreateAnnotation: (box: BoundingBox, classId: number, polygon?: number[][]) => void
  onUpdateAnnotation: (id: number, box: BoundingBox, polygon?: number[][] | null) => void
  onDeleteAnnotation: (id: number) => void
  onSelectAnnotation: (id: number | null) => void
  onAnnotationCreated?: (annotation: Annotation) => void
  onRestoreAnnotation?: (annotation: Annotation) => Promise<Annotation | void>
  disabled?: boolean
}

export interface AnnotationCanvasHandle {
  onAnnotationCreated: (annotation: Annotation) => void
}

function AnnotationCanvasInner(props: AnnotationCanvasProps, ref: React.ForwardedRef<AnnotationCanvasHandle>) {
  const {
    imageUrl,
    imageWidth,
    imageHeight,
    annotations,
    selectedAnnotationId,
    selectedClassId,
    classes,
    onCreateAnnotation,
    onUpdateAnnotation,
    onDeleteAnnotation,
    onSelectAnnotation,
    onRestoreAnnotation,
    disabled = false,
  } = props
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const [dragMode, setDragMode] = useState<DragMode>('none')
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [drawingBox, setDrawingBox] = useState<BoundingBox | null>(null)
  const [resizeHandle, setResizeHandle] = useState<ResizeHandle>(null)
  const [originalBox, setOriginalBox] = useState<BoundingBox | null>(null)
  const [originalPolygon, setOriginalPolygon] = useState<number[][] | null>(null)
  const [optimisticAnnotations, setOptimisticAnnotations] = useState<Annotation[] | null>(null)
  const [tool, setTool] = useState<Tool>('bbox')
  const [draftPolygon, setDraftPolygon] = useState<{ x: number; y: number }[]>([])
  const [draggedVertexIdx, setDraggedVertexIdx] = useState<number | null>(null)

  // Hot-path values stored as refs to avoid re-renders on every mouse move
  const mousePosRef = useRef<{ x: number; y: number } | null>(null)
  const hoveredIdRef = useRef<number | null>(null)
  const rafIdRef = useRef<number>(0)
  const rectCacheRef = useRef<DOMRect | null>(null)

  const undoStackRef = useRef<UndoAction[]>([])
  const redoStackRef = useRef<UndoAction[]>([])
  const cycleClickIndexRef = useRef(0)

  const displayAnnotations = optimisticAnnotations ?? annotations

  // Clear optimistic state when the real annotations prop updates (server responded)
  useEffect(() => {
    setOptimisticAnnotations(null)
  }, [annotations])

  useEffect(() => {
    if (!containerRef.current || imageWidth <= 0 || imageHeight <= 0) return
    const container = containerRef.current
    const aspectRatio = imageWidth / imageHeight
    const maxWidth = container.clientWidth
    const maxHeight = container.clientHeight
    let width = maxWidth
    let height = width / aspectRatio
    if (height > maxHeight) {
      height = maxHeight
      width = height * aspectRatio
    }
    setCanvasSize({ width: Math.floor(width), height: Math.floor(height) })
    rectCacheRef.current = null
  }, [imageWidth, imageHeight])

  // Invalidate cached rect on resize
  useEffect(() => {
    const observer = new ResizeObserver(() => { rectCacheRef.current = null })
    if (canvasRef.current) observer.observe(canvasRef.current)
    return () => observer.disconnect()
  }, [])

  const getRect = useCallback(() => {
    if (!rectCacheRef.current && canvasRef.current) {
      rectCacheRef.current = canvasRef.current.getBoundingClientRect()
    }
    return rectCacheRef.current
  }, [])

  const getNormalizedCoords = useCallback(
    (e: React.MouseEvent) => {
      const rect = getRect()
      if (!rect || canvasSize.width === 0 || canvasSize.height === 0) return null
      return {
        x: (e.clientX - rect.left) / canvasSize.width,
        y: (e.clientY - rect.top) / canvasSize.height,
      }
    },
    [canvasSize, getRect]
  )

  const isInsideBox = useCallback((x: number, y: number, box: BoundingBox) => {
    const left = box.x - box.width / 2
    const right = box.x + box.width / 2
    const top = box.y - box.height / 2
    const bottom = box.y + box.height / 2
    return x >= left && x <= right && y >= top && y <= bottom
  }, [])

  const getHandleSize = useCallback(() => {
    return Math.max(HANDLE_SIZE_NORM, 12 / (canvasSize.width || 1024))
  }, [canvasSize.width])

  const getResizeHandle = useCallback(
    (x: number, y: number, box: BoundingBox): ResizeHandle => {
      const hs = getHandleSize()
      const left = box.x - box.width / 2
      const right = box.x + box.width / 2
      const top = box.y - box.height / 2
      const bottom = box.y + box.height / 2
      const midX = (left + right) / 2
      const midY = (top + bottom) / 2
      if (Math.abs(x - left) <= hs && Math.abs(y - top) <= hs) return 'nw'
      if (Math.abs(x - midX) <= hs && Math.abs(y - top) <= hs) return 'n'
      if (Math.abs(x - right) <= hs && Math.abs(y - top) <= hs) return 'ne'
      if (Math.abs(x - right) <= hs && Math.abs(y - midY) <= hs) return 'e'
      if (Math.abs(x - right) <= hs && Math.abs(y - bottom) <= hs) return 'se'
      if (Math.abs(x - midX) <= hs && Math.abs(y - bottom) <= hs) return 's'
      if (Math.abs(x - left) <= hs && Math.abs(y - bottom) <= hs) return 'sw'
      if (Math.abs(x - left) <= hs && Math.abs(y - midY) <= hs) return 'w'
      return null
    },
    [getHandleSize]
  )

  const isInsidePolygon = useCallback((x: number, y: number, poly: number[][]): boolean => {
    let inside = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0]
      const yi = poly[i][1]
      const xj = poly[j][0]
      const yj = poly[j][1]
      const denom = yj - yi
      if (denom === 0) continue
      const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / denom + xi
      if (intersect) inside = !inside
    }
    return inside
  }, [])

  const getAnnotationsAtPoint = useCallback(
    (x: number, y: number): Annotation[] => {
      const at: Annotation[] = []
      for (let i = displayAnnotations.length - 1; i >= 0; i--) {
        const ann = displayAnnotations[i]
        if (ann.polygon && ann.polygon.length >= 3) {
          if (isInsidePolygon(x, y, ann.polygon)) at.push(ann)
        } else if (isInsideBox(x, y, ann.box)) {
          at.push(ann)
        }
      }
      return at
    },
    [displayAnnotations, isInsideBox, isInsidePolygon]
  )

  const polygonBboxFromPoints = useCallback((pts: { x: number; y: number }[] | number[][]): BoundingBox => {
    let minX = 1
    let minY = 1
    let maxX = 0
    let maxY = 0
    for (const p of pts as Array<any>) {
      const px = Array.isArray(p) ? p[0] : p.x
      const py = Array.isArray(p) ? p[1] : p.y
      if (px < minX) minX = px
      if (px > maxX) maxX = px
      if (py < minY) minY = py
      if (py > maxY) maxY = py
    }
    minX = Math.max(0, Math.min(1, minX))
    minY = Math.max(0, Math.min(1, minY))
    maxX = Math.max(0, Math.min(1, maxX))
    maxY = Math.max(0, Math.min(1, maxY))
    const width = Math.max(MIN_BOX_NORM, maxX - minX)
    const height = Math.max(MIN_BOX_NORM, maxY - minY)
    return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, width, height }
  }, [])

  const getVertexAtPoint = useCallback(
    (x: number, y: number, poly: number[][]): number | null => {
      if (canvasSize.width === 0 || canvasSize.height === 0) return null
      const radiusX = VERTEX_HIT_RADIUS_PX / canvasSize.width
      const radiusY = VERTEX_HIT_RADIUS_PX / canvasSize.height
      for (let i = 0; i < poly.length; i++) {
        const dx = (x - poly[i][0]) / radiusX
        const dy = (y - poly[i][1]) / radiusY
        if (dx * dx + dy * dy <= 1) return i
      }
      return null
    },
    [canvasSize]
  )

  const cycleAnnotationAtPoint = useCallback(
    (x: number, y: number) => {
      const at = getAnnotationsAtPoint(x, y)
      if (at.length === 0) {
        onSelectAnnotation(null)
        cycleClickIndexRef.current = 0
        return
      }
      const idx = cycleClickIndexRef.current % at.length
      onSelectAnnotation(at[idx].id)
      cycleClickIndexRef.current = idx + 1
    },
    [getAnnotationsAtPoint, onSelectAnnotation]
  )

  // Imperative draw - reads refs directly, no React state dependency for hot-path values
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx || canvasSize.width === 0 || canvasSize.height === 0) return
    const w = canvasSize.width
    const h = canvasSize.height
    const hovered = hoveredIdRef.current
    const mp = mousePosRef.current
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const drawBox = (
      box: BoundingBox,
      color: string,
      isSelected: boolean,
      isHovered: boolean,
      label?: string,
      isDrawing = false
    ) => {
      const x = (box.x - box.width / 2) * w
      const y = (box.y - box.height / 2) * h
      const bw = box.width * w
      const bh = box.height * h
      ctx.fillStyle = color + (isSelected ? '30' : isHovered ? '25' : '15')
      ctx.fillRect(x, y, bw, bh)
      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? 2.5 : 1.5
      if (isDrawing) {
        ctx.setLineDash([4, 4])
        ctx.strokeRect(x, y, bw, bh)
        ctx.setLineDash([])
      } else {
        ctx.strokeRect(x, y, bw, bh)
      }
      if (label) {
        ctx.font = '11px system-ui, sans-serif'
        const metrics = ctx.measureText(label)
        const padding = 4
        const labelHeight = 16
        ctx.fillStyle = color
        ctx.fillRect(x, y - labelHeight, metrics.width + padding * 2, labelHeight)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, x + padding, y - 4)
      }
      if (isSelected) {
        const handleSize = 8
        const positions: [number, number][] = [
          [x, y],
          [x + bw / 2, y],
          [x + bw, y],
          [x + bw, y + bh / 2],
          [x + bw, y + bh],
          [x + bw / 2, y + bh],
          [x, y + bh],
          [x, y + bh / 2],
        ]
        ctx.fillStyle = '#fff'
        ctx.strokeStyle = color
        ctx.lineWidth = 1.5
        positions.forEach(([hx, hy]) => {
          ctx.fillRect(hx - handleSize / 2, hy - handleSize / 2, handleSize, handleSize)
          ctx.strokeRect(hx - handleSize / 2, hy - handleSize / 2, handleSize, handleSize)
        })
      }
    }

    const drawPolygon = (polygon: number[][], color: string, isSelected: boolean) => {
      if (!polygon || polygon.length < 3) return
      ctx.beginPath()
      polygon.forEach(([px, py], i) => {
        const cx = px * w
        const cy = py * h
        if (i === 0) ctx.moveTo(cx, cy)
        else ctx.lineTo(cx, cy)
      })
      ctx.closePath()
      ctx.fillStyle = color + (isSelected ? '55' : '35')
      ctx.fill()
      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? 2 : 1.25
      ctx.stroke()
    }

    const drawVertexHandles = (polygon: number[][], color: string) => {
      ctx.fillStyle = '#fff'
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      polygon.forEach(([px, py]) => {
        const cx = px * w
        const cy = py * h
        ctx.beginPath()
        ctx.arc(cx, cy, VERTEX_RADIUS_PX, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      })
    }

    const drawPolygonLabel = (polygon: number[][], color: string, label?: string) => {
      if (!label) return
      let minX = 1
      let minY = 1
      for (const [px, py] of polygon) {
        if (px < minX) minX = px
        if (py < minY) minY = py
      }
      const lx = minX * w
      const ly = minY * h
      ctx.font = '11px system-ui, sans-serif'
      const metrics = ctx.measureText(label)
      const padding = 4
      const labelHeight = 16
      ctx.fillStyle = color
      ctx.fillRect(lx, ly - labelHeight, metrics.width + padding * 2, labelHeight)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, lx + padding, ly - 4)
    }

    displayAnnotations.forEach((ann) => {
      const color = ann.class_color || COLORS[0]
      const isSelected = ann.id === selectedAnnotationId
      const hasPoly = !!(ann.polygon && ann.polygon.length >= 3)
      if (hasPoly) {
        drawPolygon(ann.polygon!, color, isSelected)
        drawPolygonLabel(ann.polygon!, color, ann.class_name)
        if (isSelected) drawVertexHandles(ann.polygon!, color)
      } else {
        drawBox(
          ann.box,
          color,
          isSelected,
          ann.id === hovered,
          ann.class_name
        )
      }
    })
    if (drawingBox) {
      drawBox(drawingBox, COLORS[selectedClassId % COLORS.length], true, false, classes[selectedClassId], true)
    }

    // Draft polygon while drawing
    if (tool === 'polygon' && draftPolygon.length > 0) {
      const color = COLORS[selectedClassId % COLORS.length]
      ctx.strokeStyle = color
      ctx.lineWidth = 1.75
      ctx.setLineDash([6, 4])
      ctx.beginPath()
      draftPolygon.forEach((p, i) => {
        const cx = p.x * w
        const cy = p.y * h
        if (i === 0) ctx.moveTo(cx, cy)
        else ctx.lineTo(cx, cy)
      })
      if (mp) {
        ctx.lineTo(mp.x * w, mp.y * h)
      }
      ctx.stroke()
      ctx.setLineDash([])
      // First-vertex marker (click-to-close hint when >=3 points)
      const first = draftPolygon[0]
      ctx.fillStyle = draftPolygon.length >= 3 ? color : '#fff'
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.arc(first.x * w, first.y * h, CLOSE_POLYGON_RADIUS_PX / 2, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
      // Other vertices
      ctx.fillStyle = '#fff'
      ctx.strokeStyle = color
      for (let i = 1; i < draftPolygon.length; i++) {
        ctx.beginPath()
        ctx.arc(draftPolygon[i].x * w, draftPolygon[i].y * h, VERTEX_RADIUS_PX, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()
      }
    }

    if (mp && dragMode === 'draw' && dragStart) {
      ctx.strokeStyle = 'rgba(255,255,255,0.4)'
      ctx.setLineDash([2, 2])
      ctx.lineWidth = 1
      const px = mp.x * w
      const py = mp.y * h
      ctx.beginPath()
      ctx.moveTo(px, 0)
      ctx.lineTo(px, h)
      ctx.moveTo(0, py)
      ctx.lineTo(w, py)
      ctx.stroke()
      ctx.setLineDash([])
    }

    // Tool indicator (top-right)
    {
      const label =
        tool === 'polygon'
          ? draftPolygon.length > 0
            ? `Polygon · ${draftPolygon.length} pts · Enter close · Esc cancel`
            : 'Polygon (P to switch · click to add vertex)'
          : 'Bbox (P for polygon)'
      ctx.font = '11px system-ui, sans-serif'
      const padding = 6
      const metrics = ctx.measureText(label)
      const boxW = metrics.width + padding * 2
      const boxH = 18
      const bx = w - boxW - 8
      const by = 8
      ctx.fillStyle = 'rgba(0,0,0,0.6)'
      ctx.fillRect(bx, by, boxW, boxH)
      ctx.fillStyle = tool === 'polygon' ? '#4ECDC4' : '#fff'
      ctx.fillText(label, bx + padding, by + 13)
    }
  }, [canvasSize, displayAnnotations, selectedAnnotationId, drawingBox, selectedClassId, classes, dragMode, dragStart, tool, draftPolygon])

  // Schedule a canvas redraw on the next animation frame (coalesces multiple calls)
  const scheduleRedraw = useCallback(() => {
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current)
    rafIdRef.current = requestAnimationFrame(() => {
      rafIdRef.current = 0
      drawCanvas()
    })
  }, [drawCanvas])

  // Redraw whenever React-tracked visual state changes
  useEffect(() => {
    scheduleRedraw()
  }, [scheduleRedraw])

  // Set cursor directly on the DOM element (no React re-render)
  const setCursorDirect = useCallback((value: string) => {
    if (canvasRef.current) canvasRef.current.style.cursor = value
  }, [])

  const updateCursor = useCallback(
    (x: number, y: number) => {
      if (tool === 'polygon') {
        // If hovering selected polygon's vertex, show grab cursor
        if (selectedAnnotationId) {
          const sel = displayAnnotations.find((a) => a.id === selectedAnnotationId)
          if (sel?.polygon && sel.polygon.length >= 3) {
            const vIdx = getVertexAtPoint(x, y, sel.polygon)
            if (vIdx !== null) { setCursorDirect('grab'); return }
            if (isInsidePolygon(x, y, sel.polygon)) { setCursorDirect('move'); return }
          }
        }
        setCursorDirect('crosshair')
        return
      }
      if (selectedAnnotationId) {
        const selectedAnn = displayAnnotations.find((a) => a.id === selectedAnnotationId)
        if (selectedAnn) {
          // Vertex takes priority over bbox handles for polygon annotations
          if (selectedAnn.polygon && selectedAnn.polygon.length >= 3) {
            const vIdx = getVertexAtPoint(x, y, selectedAnn.polygon)
            if (vIdx !== null) { setCursorDirect('grab'); return }
          }
          const handle = getResizeHandle(x, y, selectedAnn.box)
          if (handle === 'nw' || handle === 'se') { setCursorDirect('nwse-resize'); return }
          if (handle === 'ne' || handle === 'sw') { setCursorDirect('nesw-resize'); return }
          if (handle === 'n' || handle === 's') { setCursorDirect('ns-resize'); return }
          if (handle === 'e' || handle === 'w') { setCursorDirect('ew-resize'); return }
          if (selectedAnn.polygon && selectedAnn.polygon.length >= 3) {
            if (isInsidePolygon(x, y, selectedAnn.polygon)) { setCursorDirect('move'); return }
          } else if (isInsideBox(x, y, selectedAnn.box)) { setCursorDirect('move'); return }
        }
      }
      const at = getAnnotationsAtPoint(x, y)
      if (at.length > 0) {
        hoveredIdRef.current = at[0].id
        setCursorDirect('pointer')
        scheduleRedraw()
        return
      }
      if (hoveredIdRef.current !== null) {
        hoveredIdRef.current = null
        scheduleRedraw()
      }
      setCursorDirect('crosshair')
    },
    [tool, selectedAnnotationId, displayAnnotations, getResizeHandle, isInsideBox, isInsidePolygon, getVertexAtPoint, getAnnotationsAtPoint, setCursorDirect, scheduleRedraw]
  )

  const closeDraftPolygon = useCallback(() => {
    if (draftPolygon.length < 3) return
    const poly: number[][] = draftPolygon.map((p) => [
      Math.max(0, Math.min(1, p.x)),
      Math.max(0, Math.min(1, p.y)),
    ])
    const box = polygonBboxFromPoints(poly)
    onCreateAnnotation(box, selectedClassId, poly)
    const placeholder: Annotation = {
      id: -(Date.now()),
      frame_id: 0,
      class_label_id: selectedClassId,
      class_name: classes[selectedClassId] || '',
      class_color: COLORS[selectedClassId % COLORS.length],
      box,
      polygon: poly,
      confidence: 1,
      source: 'manual',
      is_exemplar: false,
      created_at: '',
      updated_at: '',
    }
    setOptimisticAnnotations([...(optimisticAnnotations ?? annotations), placeholder])
    setDraftPolygon([])
    // Return to bbox tool so the user can select/translate the shape immediately
    // without the next click being interpreted as a new polygon vertex.
    setTool('bbox')
  }, [draftPolygon, polygonBboxFromPoints, onCreateAnnotation, selectedClassId, classes, optimisticAnnotations, annotations])

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled) return
      const coords = getNormalizedCoords(e)
      if (!coords) return
      const { x, y } = coords

      // Polygon tool: click = add vertex, close when clicking near first vertex
      if (tool === 'polygon') {
        // Allow dragging vertices of the currently selected polygon even in polygon tool
        if (selectedAnnotationId) {
          const sel = displayAnnotations.find((a) => a.id === selectedAnnotationId)
          if (sel?.polygon && sel.polygon.length >= 3) {
            const vIdx = getVertexAtPoint(x, y, sel.polygon)
            if (vIdx !== null) {
              setDragMode('move-vertex')
              setDraggedVertexIdx(vIdx)
              setDragStart({ x, y })
              setOriginalBox({ ...sel.box })
              setOriginalPolygon(sel.polygon.map((p) => [p[0], p[1]]))
              return
            }
          }
        }
        if (draftPolygon.length >= 3 && canvasSize.width > 0 && canvasSize.height > 0) {
          const first = draftPolygon[0]
          const dxPx = (x - first.x) * canvasSize.width
          const dyPx = (y - first.y) * canvasSize.height
          if (dxPx * dxPx + dyPx * dyPx <= CLOSE_POLYGON_RADIUS_PX * CLOSE_POLYGON_RADIUS_PX) {
            closeDraftPolygon()
            return
          }
        }
        setDraftPolygon([...draftPolygon, { x, y }])
        return
      }

      // Bbox tool
      if (selectedAnnotationId) {
        const selectedAnn = displayAnnotations.find((a) => a.id === selectedAnnotationId)
        if (selectedAnn) {
          // Polygon annotations: vertex drag or whole-polygon drag only (no bbox handles)
          if (selectedAnn.polygon && selectedAnn.polygon.length >= 3) {
            const vIdx = getVertexAtPoint(x, y, selectedAnn.polygon)
            if (vIdx !== null) {
              setDragMode('move-vertex')
              setDraggedVertexIdx(vIdx)
              setDragStart({ x, y })
              setOriginalBox({ ...selectedAnn.box })
              setOriginalPolygon(selectedAnn.polygon.map((p) => [p[0], p[1]]))
              return
            }
            if (isInsidePolygon(x, y, selectedAnn.polygon)) {
              setDragMode('move-polygon')
              setDragStart({ x, y })
              setOriginalBox({ ...selectedAnn.box })
              setOriginalPolygon(selectedAnn.polygon.map((p) => [p[0], p[1]]))
              return
            }
            // Fall through: clicking outside the polygon with a polygon ann selected
            // should behave like empty-space (select other / start drawing).
          } else {
            const handle = getResizeHandle(x, y, selectedAnn.box)
            if (handle) {
              setDragMode('resize')
              setResizeHandle(handle)
              setDragStart({ x, y })
              setOriginalBox({ ...selectedAnn.box })
              return
            }
            if (isInsideBox(x, y, selectedAnn.box)) {
              setDragMode('move')
              setDragStart({ x, y })
              setOriginalBox({ ...selectedAnn.box })
              return
            }
          }
        }
      }
      const at = getAnnotationsAtPoint(x, y)
      if (at.length > 0) {
        cycleAnnotationAtPoint(x, y)
        const ann = at[cycleClickIndexRef.current % at.length] ?? at[0]
        setDragStart({ x, y })
        setOriginalBox({ ...ann.box })
        if (ann.polygon && ann.polygon.length >= 3) {
          setDragMode('move-polygon')
          setOriginalPolygon(ann.polygon.map((p) => [p[0], p[1]]))
        } else {
          setDragMode('move')
        }
        return
      }
      onSelectAnnotation(null)
      cycleClickIndexRef.current = 0
      setDragMode('draw')
      setDragStart({ x, y })
    },
    [disabled, getNormalizedCoords, tool, draftPolygon, canvasSize, closeDraftPolygon, selectedAnnotationId, displayAnnotations, getVertexAtPoint, isInsidePolygon, getResizeHandle, isInsideBox, getAnnotationsAtPoint, cycleAnnotationAtPoint, onSelectAnnotation]
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const coords = getNormalizedCoords(e)
      if (!coords) return
      const { x, y } = coords
      mousePosRef.current = { x, y }
      if (dragMode === 'none') {
        updateCursor(x, y)
        // Keep rubberband line updated while drafting a polygon
        if (tool === 'polygon' && draftPolygon.length > 0) scheduleRedraw()
        return
      }
      if (!dragStart) return
      if (dragMode === 'draw') {
        let left = Math.min(dragStart.x, x)
        let right = Math.max(dragStart.x, x)
        let top = Math.min(dragStart.y, y)
        let bottom = Math.max(dragStart.y, y)
        if (right - left < MIN_BOX_NORM) right = left + MIN_BOX_NORM
        if (bottom - top < MIN_BOX_NORM) bottom = top + MIN_BOX_NORM
        setDrawingBox({
          x: (left + right) / 2,
          y: (top + bottom) / 2,
          width: right - left,
          height: bottom - top,
        })
        return
      }
      if (dragMode === 'move' && originalBox && selectedAnnotationId) {
        const dx = x - dragStart.x
        const dy = y - dragStart.y
        const newX = Math.max(originalBox.width / 2, Math.min(1 - originalBox.width / 2, originalBox.x + dx))
        const newY = Math.max(originalBox.height / 2, Math.min(1 - originalBox.height / 2, originalBox.y + dy))
        const newBox: BoundingBox = { x: newX, y: newY, width: originalBox.width, height: originalBox.height }
        setOptimisticAnnotations((prev) => {
          const base = prev ?? annotations
          return base.map((a) => (a.id === selectedAnnotationId ? { ...a, box: newBox } : a))
        })
        return
      }
      if (dragMode === 'move-vertex' && originalPolygon && selectedAnnotationId && draggedVertexIdx !== null) {
        const nx = Math.max(0, Math.min(1, x))
        const ny = Math.max(0, Math.min(1, y))
        const newPoly = originalPolygon.map((p, i) => (i === draggedVertexIdx ? [nx, ny] : [p[0], p[1]]))
        const newBox = polygonBboxFromPoints(newPoly)
        setOptimisticAnnotations((prev) => {
          const base = prev ?? annotations
          return base.map((a) => (a.id === selectedAnnotationId ? { ...a, box: newBox, polygon: newPoly } : a))
        })
        return
      }
      if (dragMode === 'move-polygon' && originalPolygon && originalBox && selectedAnnotationId) {
        const dx = x - dragStart.x
        const dy = y - dragStart.y
        // Clamp translation so polygon stays within [0,1]
        let minX = 1, minY = 1, maxX = 0, maxY = 0
        for (const p of originalPolygon) {
          if (p[0] < minX) minX = p[0]
          if (p[0] > maxX) maxX = p[0]
          if (p[1] < minY) minY = p[1]
          if (p[1] > maxY) maxY = p[1]
        }
        const clampedDx = Math.max(-minX, Math.min(1 - maxX, dx))
        const clampedDy = Math.max(-minY, Math.min(1 - maxY, dy))
        const newPoly = originalPolygon.map((p) => [p[0] + clampedDx, p[1] + clampedDy])
        const newBox: BoundingBox = {
          x: originalBox.x + clampedDx,
          y: originalBox.y + clampedDy,
          width: originalBox.width,
          height: originalBox.height,
        }
        setOptimisticAnnotations((prev) => {
          const base = prev ?? annotations
          return base.map((a) => (a.id === selectedAnnotationId ? { ...a, box: newBox, polygon: newPoly } : a))
        })
        return
      }
      if (dragMode === 'resize' && originalBox && selectedAnnotationId && resizeHandle) {
        let left = originalBox.x - originalBox.width / 2
        let right = originalBox.x + originalBox.width / 2
        let top = originalBox.y - originalBox.height / 2
        let bottom = originalBox.y + originalBox.height / 2
        if (resizeHandle.includes('w')) left = Math.min(x, right - MIN_BOX_NORM)
        if (resizeHandle.includes('e')) right = Math.max(x, left + MIN_BOX_NORM)
        if (resizeHandle.includes('n')) top = Math.min(y, bottom - MIN_BOX_NORM)
        if (resizeHandle.includes('s')) bottom = Math.max(y, top + MIN_BOX_NORM)
        left = Math.max(0, left)
        right = Math.min(1, right)
        top = Math.max(0, top)
        bottom = Math.min(1, bottom)
        const newBox: BoundingBox = {
          x: (left + right) / 2,
          y: (top + bottom) / 2,
          width: right - left,
          height: bottom - top,
        }
        setOptimisticAnnotations((prev) => {
          const base = prev ?? annotations
          return base.map((a) => (a.id === selectedAnnotationId ? { ...a, box: newBox } : a))
        })
      }
    },
    [getNormalizedCoords, dragMode, dragStart, originalBox, originalPolygon, draggedVertexIdx, selectedAnnotationId, resizeHandle, annotations, updateCursor, tool, draftPolygon, scheduleRedraw, polygonBboxFromPoints]
  )

  const handleMouseUp = useCallback(() => {
    if (dragMode === 'draw' && drawingBox) {
      if (drawingBox.width >= MIN_BOX_NORM && drawingBox.height >= MIN_BOX_NORM) {
        onCreateAnnotation(drawingBox, selectedClassId)
        // Optimistic: show the new box immediately as a placeholder annotation
        const placeholder: Annotation = {
          id: -(Date.now()),
          frame_id: 0,
          class_label_id: selectedClassId,
          class_name: classes[selectedClassId] || '',
          class_color: COLORS[selectedClassId % COLORS.length],
          box: drawingBox,
          confidence: 1,
          source: 'manual',
          is_exemplar: false,
          created_at: '',
          updated_at: '',
        }
        setOptimisticAnnotations([...annotations, placeholder])
      }
      setDrawingBox(null)
    }
    if ((dragMode === 'move' || dragMode === 'resize') && selectedAnnotationId) {
      const current = optimisticAnnotations ?? annotations
      const ann = current.find((a) => a.id === selectedAnnotationId)
      if (ann && originalBox) {
        const changed =
          ann.box.x !== originalBox.x ||
          ann.box.y !== originalBox.y ||
          ann.box.width !== originalBox.width ||
          ann.box.height !== originalBox.height
        if (changed) {
          undoStackRef.current.push({ type: 'update', id: selectedAnnotationId, previousBox: originalBox, previousPolygon: ann.polygon ?? null })
          redoStackRef.current = []
          onUpdateAnnotation(selectedAnnotationId, ann.box, ann.polygon ?? undefined)
        }
      }
    }
    if ((dragMode === 'move-vertex' || dragMode === 'move-polygon') && selectedAnnotationId) {
      const current = optimisticAnnotations ?? annotations
      const ann = current.find((a) => a.id === selectedAnnotationId)
      if (ann && originalBox && originalPolygon) {
        const polyChanged =
          !ann.polygon ||
          ann.polygon.length !== originalPolygon.length ||
          ann.polygon.some((p, i) => p[0] !== originalPolygon[i][0] || p[1] !== originalPolygon[i][1])
        if (polyChanged) {
          undoStackRef.current.push({
            type: 'update',
            id: selectedAnnotationId,
            previousBox: originalBox,
            previousPolygon: originalPolygon,
          })
          redoStackRef.current = []
          onUpdateAnnotation(selectedAnnotationId, ann.box, ann.polygon ?? undefined)
        }
      }
    }
    setDragMode('none')
    setDragStart(null)
    setOriginalBox(null)
    setOriginalPolygon(null)
    setDraggedVertexIdx(null)
    setResizeHandle(null)
  }, [dragMode, drawingBox, selectedClassId, selectedAnnotationId, optimisticAnnotations, annotations, originalBox, originalPolygon, onCreateAnnotation, onUpdateAnnotation, classes])

  const pushDeleteToUndo = useCallback((annotation: Annotation) => {
    undoStackRef.current.push({ type: 'delete', annotation })
    redoStackRef.current = []
  }, [])

  useImperativeHandle(ref, () => ({
    onAnnotationCreated(annotation: Annotation) {
      undoStackRef.current.push({ type: 'create', annotation })
      redoStackRef.current = []
    },
  }), [])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (disabled) return
      if (e.key === 'Escape') {
        if (draftPolygon.length > 0) {
          setDraftPolygon([])
          return
        }
        onSelectAnnotation(null)
        return
      }
      if ((e.key === 'p' || e.key === 'P') && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault()
        setTool((t) => (t === 'polygon' ? 'bbox' : 'polygon'))
        setDraftPolygon([])
        return
      }
      if (e.key === 'Enter' && tool === 'polygon' && draftPolygon.length >= 3) {
        e.preventDefault()
        closeDraftPolygon()
        return
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotationId) {
        e.preventDefault()
        const ann = displayAnnotations.find((a) => a.id === selectedAnnotationId)
        if (ann) {
          pushDeleteToUndo(ann)
          // Optimistic: remove the annotation from display immediately
          setOptimisticAnnotations(displayAnnotations.filter((a) => a.id !== selectedAnnotationId))
          onDeleteAnnotation(selectedAnnotationId)
          onSelectAnnotation(null)
        }
        return
      }
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 'z' && !e.shiftKey) {
          e.preventDefault()
          const action = undoStackRef.current.pop()
          if (!action) return
          if (action.type === 'create') {
            redoStackRef.current.push({ type: 'restore', annotation: action.annotation })
            setOptimisticAnnotations(displayAnnotations.filter((a) => a.id !== action.annotation.id))
            onDeleteAnnotation(action.annotation.id)
            onSelectAnnotation(null)
          } else if (action.type === 'update') {
            const current = displayAnnotations.find((a) => a.id === action.id)
            if (current) {
              redoStackRef.current.push({
                type: 'update',
                id: action.id,
                previousBox: current.box,
                previousPolygon: current.polygon ?? null,
              })
            }
            const prevPoly = action.previousPolygon
            setOptimisticAnnotations(
              displayAnnotations.map((a) =>
                a.id === action.id
                  ? { ...a, box: action.previousBox, polygon: prevPoly === null ? undefined : prevPoly ?? a.polygon }
                  : a
              )
            )
            onUpdateAnnotation(action.id, action.previousBox, prevPoly === null ? undefined : prevPoly)
          } else if (action.type === 'delete' && onRestoreAnnotation) {
            onRestoreAnnotation(action.annotation).then((created) => {
              if (created) redoStackRef.current.push({ type: 'create', annotation: created })
            })
          }
          return
        }
        if (e.shiftKey && e.key.toLowerCase() === 'z') {
          e.preventDefault()
          const action = redoStackRef.current.pop()
          if (!action) return
          if (action.type === 'create') {
            undoStackRef.current.push({ type: 'delete', annotation: action.annotation })
            setOptimisticAnnotations(displayAnnotations.filter((a) => a.id !== action.annotation.id))
            onDeleteAnnotation(action.annotation.id)
          } else if (action.type === 'update') {
            const current = displayAnnotations.find((a) => a.id === action.id)
            if (current) {
              undoStackRef.current.push({
                type: 'update',
                id: action.id,
                previousBox: current.box,
                previousPolygon: current.polygon ?? null,
              })
            }
            const prevPoly = action.previousPolygon
            setOptimisticAnnotations(
              displayAnnotations.map((a) =>
                a.id === action.id
                  ? { ...a, box: action.previousBox, polygon: prevPoly === null ? undefined : prevPoly ?? a.polygon }
                  : a
              )
            )
            onUpdateAnnotation(action.id, action.previousBox, prevPoly === null ? undefined : prevPoly)
          } else if (action.type === 'restore' && onRestoreAnnotation) {
            onRestoreAnnotation(action.annotation).then((created) => {
              if (created) undoStackRef.current.push({ type: 'create', annotation: created })
            })
          } else if (action.type === 'delete') {
            pushDeleteToUndo(action.annotation)
            setOptimisticAnnotations(displayAnnotations.filter((a) => a.id !== action.annotation.id))
            onDeleteAnnotation(action.annotation.id)
          }
        }
        if (e.key === 'a') {
          e.preventDefault()
          if (displayAnnotations.length > 0) onSelectAnnotation(displayAnnotations[displayAnnotations.length - 1].id)
        }
        if (e.key === 'd' && selectedAnnotationId) {
          e.preventDefault()
          const ann = displayAnnotations.find((a) => a.id === selectedAnnotationId)
          if (ann) onCreateAnnotation(ann.box, ann.class_label_id)
        }
      }
    },
    [disabled, selectedAnnotationId, displayAnnotations, onSelectAnnotation, onDeleteAnnotation, onUpdateAnnotation, onCreateAnnotation, onRestoreAnnotation, pushDeleteToUndo, draftPolygon, tool, closeDraftPolygon]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Cleanup rAF on unmount
  useEffect(() => {
    return () => { if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current) }
  }, [])

  if (imageWidth <= 0 || imageHeight <= 0) return null

  return (
    <div ref={containerRef} className="relative flex items-center justify-center w-full h-full min-h-0">
      <div className="relative flex-shrink-0" style={{ width: canvasSize.width, height: canvasSize.height }}>
        <img src={imageUrl} alt="" className="absolute inset-0 w-full h-full object-contain" draggable={false} />
        <canvas
          ref={canvasRef}
          width={canvasSize.width}
          height={canvasSize.height}
          className="absolute inset-0"
          style={{ cursor: disabled ? 'default' : 'crosshair' }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onDoubleClick={() => {
            if (tool === 'polygon' && draftPolygon.length >= 3) closeDraftPolygon()
          }}
          onMouseLeave={() => {
            handleMouseUp()
            mousePosRef.current = null
            hoveredIdRef.current = null
            scheduleRedraw()
          }}
        />
      </div>
    </div>
  )
}

export const AnnotationCanvas = forwardRef(AnnotationCanvasInner)
