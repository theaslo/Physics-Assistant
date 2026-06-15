import { useMemo } from 'react'
import { Alert, Box, Chip, Typography } from '@mui/material'
import type { GraphDefinition, GraphPoint, PhysicsGraphPayload } from '../types/graphs'

interface PhysicsGraphPanelProps {
  graphs: PhysicsGraphPayload[]
}

interface ScaledChart {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  xTicks: number[]
  yTicks: number[]
}

const SVG_WIDTH = 680
const SVG_HEIGHT = 280
const PLOT_LEFT = 70
const PLOT_RIGHT = 18
const PLOT_TOP = 28
const PLOT_BOTTOM = 58
const PLOT_WIDTH = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT
const PLOT_HEIGHT = SVG_HEIGHT - PLOT_TOP - PLOT_BOTTOM

export default function PhysicsGraphPanel({ graphs }: PhysicsGraphPanelProps) {
  if (graphs.length === 0) {
    return null
  }

  return (
    <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
      {graphs.map((payload, index) => (
        <Box
          key={`${payload.type}-${index}`}
          sx={{
            border: 1,
            borderColor: 'divider',
            borderRadius: 2,
            bgcolor: 'background.default',
            overflow: 'hidden',
          }}
        >
          <Box sx={{ p: 2, pb: 1 }}>
            <Typography variant="subtitle1" fontWeight={700}>
              {payload.title}
            </Typography>
            {payload.subtitle && (
              <Typography variant="body2" color="text.secondary">
                {payload.subtitle}
              </Typography>
            )}

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1.5 }}>
              {payload.parameters.map((parameter) => (
                <Chip
                  key={parameter.symbol}
                  size="small"
                  label={`${parameter.symbol} = ${formatParameter(parameter.value)}${parameter.unit ? ` ${parameter.unit}` : ''}`}
                  variant="outlined"
                  sx={{ bgcolor: 'background.paper' }}
                />
              ))}
            </Box>

            {payload.warnings && payload.warnings.length > 0 && (
              <Alert severity="warning" sx={{ mt: 1.5 }}>
                {payload.warnings.join(' ')}
              </Alert>
            )}
          </Box>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' },
              gap: 1.5,
              p: 2,
              pt: 1,
            }}
          >
            {payload.graphs.map((graph) => (
              <GraphCard key={graph.id} graph={graph} />
            ))}
          </Box>
        </Box>
      ))}
    </Box>
  )
}

function GraphCard({ graph }: { graph: GraphDefinition }) {
  return (
    <Box
      sx={{
        minWidth: 0,
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        bgcolor: 'background.paper',
        p: 1.5,
      }}
    >
      <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1 }}>
        {graph.title}
      </Typography>
      <GraphSvg graph={graph} />
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
        {graph.series.map((series) => (
          <Box
            key={series.label}
            sx={{ display: 'flex', alignItems: 'center', gap: 0.75, minWidth: 0 }}
          >
            <Box
              sx={{
                width: 18,
                height: 3,
                borderRadius: 1,
                bgcolor: series.color || 'primary.main',
                flexShrink: 0,
              }}
            />
            <Typography variant="caption" color="text.secondary">
              {series.label}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  )
}

function GraphSvg({ graph }: { graph: GraphDefinition }) {
  const chart = useMemo(() => buildScaledChart(graph), [graph])

  if (!chart) {
    return (
      <Box sx={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography variant="body2" color="text.secondary">
          No graph data available.
        </Typography>
      </Box>
    )
  }

  const scaleX = (value: number) =>
    PLOT_LEFT + ((value - chart.xMin) / (chart.xMax - chart.xMin)) * PLOT_WIDTH
  const scaleY = (value: number) =>
    PLOT_TOP + PLOT_HEIGHT - ((value - chart.yMin) / (chart.yMax - chart.yMin)) * PLOT_HEIGHT

  return (
    <Box sx={{ width: '100%', overflow: 'hidden' }}>
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
        role="img"
        aria-label={graph.title}
        style={{ width: '100%', height: 'auto', display: 'block' }}
      >
        <rect x={PLOT_LEFT} y={PLOT_TOP} width={PLOT_WIDTH} height={PLOT_HEIGHT} fill="#ffffff" />

        {chart.yTicks.map((tick) => {
          const y = scaleY(tick)
          return (
            <g key={`y-${tick}`}>
              <line x1={PLOT_LEFT} x2={SVG_WIDTH - PLOT_RIGHT} y1={y} y2={y} stroke="#e5e7eb" />
              <text x={PLOT_LEFT - 10} y={y + 4} textAnchor="end" fontSize="12" fill="#4b5563">
                {formatTick(tick)}
              </text>
            </g>
          )
        })}

        {chart.xTicks.map((tick) => {
          const x = scaleX(tick)
          return (
            <g key={`x-${tick}`}>
              <line x1={x} x2={x} y1={PLOT_TOP} y2={PLOT_TOP + PLOT_HEIGHT} stroke="#eef2f7" />
              <text x={x} y={PLOT_TOP + PLOT_HEIGHT + 24} textAnchor="middle" fontSize="12" fill="#4b5563">
                {formatTick(tick)}
              </text>
            </g>
          )
        })}

        <line x1={PLOT_LEFT} x2={SVG_WIDTH - PLOT_RIGHT} y1={PLOT_TOP + PLOT_HEIGHT} y2={PLOT_TOP + PLOT_HEIGHT} stroke="#111827" strokeWidth="1.4" />
        <line x1={PLOT_LEFT} x2={PLOT_LEFT} y1={PLOT_TOP} y2={PLOT_TOP + PLOT_HEIGHT} stroke="#111827" strokeWidth="1.4" />

        {graph.series.map((series) => (
          <path
            key={series.label}
            d={pointsToPath(series.points, scaleX, scaleY)}
            fill="none"
            stroke={series.color || '#1976d2'}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}

        {graph.series.map((series) =>
          pickMarkerPoints(series.points).map((point) => (
            <circle
              key={`${series.label}-${point.x}-${point.y}`}
              cx={scaleX(point.x)}
              cy={scaleY(point.y)}
              r="3"
              fill={series.color || '#1976d2'}
              stroke="#ffffff"
              strokeWidth="1.5"
            />
          ))
        )}

        <text x={PLOT_LEFT + PLOT_WIDTH / 2} y={SVG_HEIGHT - 12} textAnchor="middle" fontSize="13" fill="#111827">
          {axisLabel(graph.xAxis.label, graph.xAxis.unit)}
        </text>
        <text
          x="18"
          y={PLOT_TOP + PLOT_HEIGHT / 2}
          textAnchor="middle"
          fontSize="13"
          fill="#111827"
          transform={`rotate(-90 18 ${PLOT_TOP + PLOT_HEIGHT / 2})`}
        >
          {axisLabel(graph.yAxis.label, graph.yAxis.unit)}
        </text>
      </svg>
    </Box>
  )
}

function buildScaledChart(graph: GraphDefinition): ScaledChart | null {
  const points = graph.series.flatMap((series) => series.points)
  if (points.length === 0) {
    return null
  }

  const xValues = points.map((point) => point.x)
  const yValues = points.map((point) => point.y)
  const xBounds = expandBounds(Math.min(...xValues), Math.max(...xValues))
  const yBounds = expandBounds(Math.min(...yValues), Math.max(...yValues))

  return {
    xMin: xBounds[0],
    xMax: xBounds[1],
    yMin: yBounds[0],
    yMax: yBounds[1],
    xTicks: makeTicks(xBounds[0], xBounds[1], 5),
    yTicks: makeTicks(yBounds[0], yBounds[1], 5),
  }
}

function expandBounds(min: number, max: number): [number, number] {
  if (min === max) {
    const padding = Math.max(Math.abs(min) * 0.2, 1)
    return [min - padding, max + padding]
  }

  const padding = (max - min) * 0.08
  return [min - padding, max + padding]
}

function makeTicks(min: number, max: number, count: number): number[] {
  if (count <= 1) {
    return [min]
  }

  const step = (max - min) / (count - 1)
  return Array.from({ length: count }, (_, index) => min + step * index)
}

function pointsToPath(
  points: GraphPoint[],
  scaleX: (value: number) => number,
  scaleY: (value: number) => number
) {
  return points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${scaleX(point.x).toFixed(2)} ${scaleY(point.y).toFixed(2)}`)
    .join(' ')
}

function pickMarkerPoints(points: GraphPoint[]): GraphPoint[] {
  if (points.length <= 3) {
    return points
  }

  return [points[0], points[Math.floor(points.length / 2)], points[points.length - 1]]
}

function axisLabel(label: string, unit?: string) {
  return unit ? `${label} (${unit})` : label
}

function formatTick(value: number) {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0)
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(1)
  }
  return value.toFixed(2)
}

function formatParameter(value: number | string) {
  if (typeof value === 'string') {
    return value
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(1)
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2)
  }
  return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
}
