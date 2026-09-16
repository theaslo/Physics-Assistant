import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type TouchEvent as ReactTouchEvent,
} from 'react'
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Stack,
  Typography,
} from '@mui/material'
import {
  Close as CloseIcon,
  DeleteOutline as DeleteOutlineIcon,
  Edit as EditIcon,
  AutoFixHigh as EraserIcon,
} from '@mui/icons-material'
import type { StudentDrawing, StudentSketchStroke } from '../stores/chat-store'

const CANVAS_WIDTH = 720
const CANVAS_HEIGHT = 420
const GRID_SIZE = 24
const COLORS = ['#111827', '#1565C0', '#C2410C', '#047857']

type SketchTool = 'pen' | 'eraser'
type Point = { x: number; y: number }

interface SketchPadDialogProps {
  open: boolean
  onClose: () => void
  onAttach: (drawing: StudentDrawing) => void
}

function drawCanvasBackground(ctx: CanvasRenderingContext2D) {
  ctx.save()
  ctx.fillStyle = '#FFFFFF'
  ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT)
  ctx.strokeStyle = '#E5E7EB'
  ctx.lineWidth = 1

  for (let x = 0; x <= CANVAS_WIDTH; x += GRID_SIZE) {
    ctx.beginPath()
    ctx.moveTo(x, 0)
    ctx.lineTo(x, CANVAS_HEIGHT)
    ctx.stroke()
  }

  for (let y = 0; y <= CANVAS_HEIGHT; y += GRID_SIZE) {
    ctx.beginPath()
    ctx.moveTo(0, y)
    ctx.lineTo(CANVAS_WIDTH, y)
    ctx.stroke()
  }

  ctx.restore()
}

export default function SketchPadDialog({ open, onClose, onAttach }: SketchPadDialogProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const isDrawingRef = useRef(false)
  const lastPointRef = useRef<Point | null>(null)
  const pointerEventsSeenRef = useRef(false)
  const strokesRef = useRef<StudentSketchStroke[]>([])
  const currentStrokeRef = useRef<StudentSketchStroke | null>(null)
  const [tool, setTool] = useState<SketchTool>('pen')
  const [strokeColor, setStrokeColor] = useState(COLORS[0])
  const [hasInk, setHasInk] = useState(false)

  const prepareCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    const ratio = window.devicePixelRatio || 1
    canvas.width = CANVAS_WIDTH * ratio
    canvas.height = CANVAS_HEIGHT * ratio
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
    drawCanvasBackground(ctx)
    strokesRef.current = []
    currentStrokeRef.current = null
    isDrawingRef.current = false
    lastPointRef.current = null
    setHasInk(false)
  }

  const resetCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    drawCanvasBackground(ctx)
    strokesRef.current = []
    currentStrokeRef.current = null
    isDrawingRef.current = false
    lastPointRef.current = null
    setHasInk(false)
  }

  useEffect(() => {
    if (!open) return
    prepareCanvas()
  }, [open])

  const getPointFromClient = (clientX: number, clientY: number, canvas: HTMLCanvasElement): Point => {
    const rect = canvas.getBoundingClientRect()
    const rawX = clientX - rect.left
    const rawY = clientY - rect.top

    return {
      x: Math.max(0, Math.min(CANVAS_WIDTH, (rawX / Math.max(rect.width, 1)) * CANVAS_WIDTH)),
      y: Math.max(0, Math.min(CANVAS_HEIGHT, (rawY / Math.max(rect.height, 1)) * CANVAS_HEIGHT)),
    }
  }

  const drawPoint = (point: Point) => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return

    const width = tool === 'eraser' ? 20 : 4
    ctx.save()
    ctx.fillStyle = tool === 'eraser' ? '#FFFFFF' : strokeColor
    ctx.beginPath()
    ctx.arc(point.x, point.y, width / 2, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()
  }

  const drawSegment = (from: Point, to: Point) => {
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return

    ctx.save()
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.strokeStyle = tool === 'eraser' ? '#FFFFFF' : strokeColor
    ctx.lineWidth = tool === 'eraser' ? 20 : 4
    ctx.beginPath()
    ctx.moveTo(from.x, from.y)
    ctx.lineTo(to.x, to.y)
    ctx.stroke()
    ctx.restore()
  }

  const beginStroke = (point: Point) => {
    isDrawingRef.current = true
    lastPointRef.current = point
    currentStrokeRef.current = {
      tool,
      color: tool === 'eraser' ? '#FFFFFF' : strokeColor,
      points: [point],
    }
    drawPoint(point)
    if (tool === 'pen') {
      setHasInk(true)
    }
  }

  const continueStroke = (point: Point) => {
    if (!isDrawingRef.current || !lastPointRef.current) return
    currentStrokeRef.current?.points.push(point)
    drawSegment(lastPointRef.current, point)
    lastPointRef.current = point
  }

  const endStroke = () => {
    if (currentStrokeRef.current && currentStrokeRef.current.points.length > 0) {
      strokesRef.current = [...strokesRef.current, currentStrokeRef.current]
    }
    currentStrokeRef.current = null
    isDrawingRef.current = false
    lastPointRef.current = null
  }

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    pointerEventsSeenRef.current = true
    event.preventDefault()
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Some browsers are conservative about pointer capture inside dialogs.
    }
    beginStroke(getPointFromClient(event.clientX, event.clientY, event.currentTarget))
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return
    event.preventDefault()
    continueStroke(getPointFromClient(event.clientX, event.clientY, event.currentTarget))
  }

  const stopDrawing = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    try {
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId)
      }
    } catch {
      // Pointer capture may not exist on older browser/device combinations.
    }
    endStroke()
  }

  const handleMouseDown = (event: ReactMouseEvent<HTMLCanvasElement>) => {
    if (pointerEventsSeenRef.current) return
    event.preventDefault()
    beginStroke(getPointFromClient(event.clientX, event.clientY, event.currentTarget))
  }

  const handleMouseMove = (event: ReactMouseEvent<HTMLCanvasElement>) => {
    if (pointerEventsSeenRef.current || !isDrawingRef.current) return
    event.preventDefault()
    continueStroke(getPointFromClient(event.clientX, event.clientY, event.currentTarget))
  }

  const handleMouseUp = () => {
    if (pointerEventsSeenRef.current) return
    endStroke()
  }

  const handleTouchStart = (event: ReactTouchEvent<HTMLCanvasElement>) => {
    if (pointerEventsSeenRef.current) return
    const touch = event.touches[0]
    if (!touch) return
    event.preventDefault()
    beginStroke(getPointFromClient(touch.clientX, touch.clientY, event.currentTarget))
  }

  const handleTouchMove = (event: ReactTouchEvent<HTMLCanvasElement>) => {
    if (pointerEventsSeenRef.current || !isDrawingRef.current) return
    const touch = event.touches[0]
    if (!touch) return
    event.preventDefault()
    continueStroke(getPointFromClient(touch.clientX, touch.clientY, event.currentTarget))
  }

  const handleTouchEnd = () => {
    if (pointerEventsSeenRef.current) return
    endStroke()
  }

  const handleAttach = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    onAttach({
      dataUrl: canvas.toDataURL('image/png'),
      title: 'Student sketch',
      width: CANVAS_WIDTH,
      height: CANVAS_HEIGHT,
      createdAt: Date.now(),
      strokes: strokesRef.current.map((stroke) => ({
        ...stroke,
        points: stroke.points.map((point) => ({ ...point })),
      })),
    })
    onClose()
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="md">
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
        Sketch your setup
        <IconButton aria-label="Close sketch pad" onClick={onClose} edge="end">
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <Typography variant="body2" color="text.secondary">
          Use this for axes, force arrows, free-body diagrams, or quick graphs. After attaching it, type the key labels or equations so I can respond accurately.
        </Typography>

        <Box
          sx={{
            mt: 2,
            p: 1,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            bgcolor: '#F8FAFC',
            aspectRatio: `${CANVAS_WIDTH} / ${CANVAS_HEIGHT}`,
            width: '100%',
            maxHeight: 460,
          }}
        >
          <canvas
            ref={canvasRef}
            width={CANVAS_WIDTH}
            height={CANVAS_HEIGHT}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={stopDrawing}
            onPointerCancel={stopDrawing}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
            onTouchCancel={handleTouchEnd}
            style={{
              display: 'block',
              width: '100%',
              height: '100%',
              borderRadius: 12,
              cursor: tool === 'eraser' ? 'cell' : 'crosshair',
              touchAction: 'none',
              userSelect: 'none',
            }}
          />
        </Box>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ mt: 2, alignItems: { sm: 'center' } }}>
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              variant={tool === 'pen' ? 'contained' : 'outlined'}
              startIcon={<EditIcon />}
              onClick={() => setTool('pen')}
            >
              Pen
            </Button>
            <Button
              size="small"
              variant={tool === 'eraser' ? 'contained' : 'outlined'}
              startIcon={<EraserIcon />}
              onClick={() => setTool('eraser')}
            >
              Eraser
            </Button>
          </Stack>

          <Stack direction="row" spacing={0.75} aria-label="Sketch colors">
            {COLORS.map((color) => (
              <IconButton
                key={color}
                aria-label={`Use color ${color}`}
                onClick={() => {
                  setStrokeColor(color)
                  setTool('pen')
                }}
                sx={{
                  width: 34,
                  height: 34,
                  border: '2px solid',
                  borderColor: strokeColor === color && tool === 'pen' ? 'primary.main' : 'transparent',
                }}
              >
                <Box sx={{ width: 18, height: 18, borderRadius: '50%', bgcolor: color }} />
              </IconButton>
            ))}
          </Stack>

          <Button size="small" variant="text" color="inherit" startIcon={<DeleteOutlineIcon />} onClick={resetCanvas}>
            Clear
          </Button>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleAttach} disabled={!hasInk}>
          Attach sketch
        </Button>
      </DialogActions>
    </Dialog>
  )
}
