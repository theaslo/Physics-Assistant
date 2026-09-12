import { Box, Paper, Typography, Chip } from '@mui/material'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import type {
  Message,
  DiagramPayload,
  DiagramForce,
  FreeBodyDiagramPayload,
  ForceVectorAdditionPayload,
  ForceComponentsDiagramPayload,
  EquilibriumResidualDiagramPayload,
  InclinedPlaneDiagramPayload,
  TensionSystemDiagramPayload,
  MotionGraphsPlotPayload,
  ProjectileTrajectoryPayload,
  ProjectileVelocityAnimationPayload,
  FreeFallTimelinePayload,
  RelativeMotionIntersectionPayload,
  EnergyBarChartPayload,
  EnergyFlowDiagramPayload,
  WorkAreaUnderCurvePayload,
  BeforeAfterEnergySnapshotPayload,
  MomentumBeforeAfterVectorsPayload,
  ImpulseAreaPlotPayload,
  CollisionStoryboardPayload,
  CenterOfMassTracePayload,
  TorqueLeverDiagramPayload,
  RotationGraphsPlotPayload,
  CircularMotionVectorsPayload,
  RollingEnergySplitPayload,
  ShmSpringAnimationPayload,
  PendulumAnimationPayload,
  TravelingWaveAnimationPayload,
  StandingWaveModeShapePayload,
  InterferenceFringeMapPayload,
  DopplerWavefrontAnimationPayload,
  PvDiagramPayload,
  HeatingCurvePlotPayload,
  HeatTransferPathDiagramPayload,
  CarnotCycleAnimationPayload,
} from '../stores/chat-store'
import { getAgentColor, getAgentIcon } from '../themes/uconn-theme'

interface ChatMessageProps {
  message: Message
}

function ForceVectorSvg({
  title,
  vectors,
  markerId,
  centerLabel,
}: {
  title: string
  vectors: Array<DiagramForce & { color?: string }>
  markerId: string
  centerLabel?: string
}) {
  const width = 340
  const height = 260
  const cx = width / 2
  const cy = height / 2
  const axisLength = 95

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        {title}
      </Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <defs>
          <marker
            id={markerId}
            markerWidth="8"
            markerHeight="8"
            refX="7"
            refY="4"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L8,4 L0,8 Z" fill="#1565C0" />
          </marker>
        </defs>

        <line x1={cx - axisLength} y1={cy} x2={cx + axisLength} y2={cy} stroke="#D1D5DB" strokeWidth="1.2" />
        <line x1={cx} y1={cy + axisLength} x2={cx} y2={cy - axisLength} stroke="#D1D5DB" strokeWidth="1.2" />
        <text x={cx + axisLength - 8} y={cy - 6} fontSize="10" fill="#6B7280">
          +x
        </text>
        <text x={cx + 6} y={cy - axisLength + 10} fontSize="10" fill="#6B7280">
          +y
        </text>
        {centerLabel && (
          <text x={cx - 8} y={cy + 16} fontSize="10" fill="#111827">
            {centerLabel}
          </text>
        )}

        {vectors.map((force, idx) => {
          const radians = (force.angle_deg * Math.PI) / 180
          const scale = Math.min(1.5, Math.max(0.45, force.magnitude_n / 30))
          const length = 70 * scale
          const x2 = cx + Math.cos(radians) * length
          const y2 = cy - Math.sin(radians) * length
          const lx = cx + Math.cos(radians) * (length + 12)
          const ly = cy - Math.sin(radians) * (length + 12)

          return (
            <g key={`${force.name}-${idx}`}>
              <line
                x1={cx}
                y1={cy}
                x2={x2}
                y2={y2}
                stroke={force.color || '#1565C0'}
                strokeWidth="2"
                markerEnd={`url(#${markerId})`}
              />
              <text x={lx} y={ly} fontSize="10" fill="#1F2937" textAnchor="middle">
                {force.name} ({force.magnitude_n.toFixed(1)} N)
              </text>
            </g>
          )
        })}
      </svg>
    </Box>
  )
}

function FreeBodyDiagram({ diagram, markerId }: { diagram: FreeBodyDiagramPayload; markerId: string }) {
  return <ForceVectorSvg title={diagram.title} vectors={diagram.forces} markerId={markerId} centerLabel={diagram.object_name} />
}

function ForceVectorAdditionDiagram({ diagram, markerId }: { diagram: ForceVectorAdditionPayload; markerId: string }) {
  const vectors = [
    ...diagram.vectors.map((v, idx) => ({ ...v, color: idx % 2 ? '#2563EB' : '#0EA5E9' })),
    { ...diagram.resultant, color: '#DC2626' },
  ]
  return <ForceVectorSvg title={diagram.title} vectors={vectors} markerId={markerId} />
}

function ForceComponentsDiagram({ diagram, markerId }: { diagram: ForceComponentsDiagramPayload; markerId: string }) {
  const width = 340
  const height = 260
  const cx = width / 2
  const cy = height / 2
  const v = diagram.vector
  const radians = (v.angle_deg * Math.PI) / 180
  const scale = Math.min(1.5, Math.max(0.45, v.magnitude_n / 30))
  const length = 70 * scale
  const x2 = cx + Math.cos(radians) * length
  const y2 = cy - Math.sin(radians) * length
  const xComp = cx + (v.fx_n / Math.max(Math.abs(v.fx_n) + Math.abs(v.fy_n), 1)) * length
  const yComp = cy - (v.fy_n / Math.max(Math.abs(v.fx_n) + Math.abs(v.fy_n), 1)) * length

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        {diagram.title}
      </Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8 Z" fill="#1565C0" />
          </marker>
        </defs>
        <line x1={cx - 95} y1={cy} x2={cx + 95} y2={cy} stroke="#D1D5DB" strokeWidth="1.2" />
        <line x1={cx} y1={cy + 95} x2={cx} y2={cy - 95} stroke="#D1D5DB" strokeWidth="1.2" />
        <line x1={cx} y1={cy} x2={x2} y2={y2} stroke="#1565C0" strokeWidth="2.2" markerEnd={`url(#${markerId})`} />
        <line x1={cx} y1={cy} x2={xComp} y2={cy} stroke="#059669" strokeWidth="2" strokeDasharray="4 3" />
        <line x1={xComp} y1={cy} x2={xComp} y2={yComp} stroke="#7C3AED" strokeWidth="2" strokeDasharray="4 3" />
        <text x={x2 + 8} y={y2} fontSize="10" fill="#1F2937">F</text>
        <text x={(cx + xComp) / 2} y={cy - 6} fontSize="10" fill="#059669">Fx={v.fx_n.toFixed(1)}N</text>
        <text x={xComp + 6} y={(cy + yComp) / 2} fontSize="10" fill="#7C3AED">Fy={v.fy_n.toFixed(1)}N</text>
      </svg>
    </Box>
  )
}

function EquilibriumDiagram({ diagram, markerId }: { diagram: EquilibriumResidualDiagramPayload; markerId: string }) {
  const vectors: Array<DiagramForce & { color?: string }> = [
    ...diagram.forces.map((f) => ({ ...f, color: '#2563EB' })),
    { ...diagram.net_force, color: '#DC2626' },
  ]
  if (diagram.balancing_force) {
    vectors.push({ ...diagram.balancing_force, color: '#059669' })
  }

  return (
    <Box>
      <ForceVectorSvg title={diagram.title} vectors={vectors} markerId={markerId} />
      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
        {diagram.is_equilibrium ? 'System is in equilibrium.' : 'Red = net force, Green = balancing force.'}
      </Typography>
    </Box>
  )
}

function InclinedPlaneDiagram({ diagram, markerId }: { diagram: InclinedPlaneDiagramPayload; markerId: string }) {
  const width = 360
  const height = 240

  const theta = (diagram.angle_deg * Math.PI) / 180
  const baseX = 60
  const baseY = 190
  const inclineLength = 220
  const rise = inclineLength * Math.sin(theta)
  const run = inclineLength * Math.cos(theta)
  const topX = baseX + run
  const topY = baseY - rise

  const t = 0.58
  const cx = baseX + run * t
  const cy = baseY - rise * t
  const boxW = 26
  const boxH = 20
  const boxX = cx - boxW / 2
  const boxY = cy - boxH / 2

  // Derive direction vectors from rendered ramp geometry to avoid visual drift.
  const rampDx = topX - baseX
  const rampDy = topY - baseY
  const rampMag = Math.hypot(rampDx, rampDy) || 1
  const uphill = { x: rampDx / rampMag, y: rampDy / rampMag }
  const downhill = { x: -uphill.x, y: -uphill.y }
  const normalOut = { x: uphill.y, y: -uphill.x }

  const maxForce = Math.max(
    1,
    diagram.weight_n,
    diagram.normal_n,
    diagram.weight_parallel_n,
    diagram.friction_n || 0
  )
  const scaledLen = (magnitude: number) => 34 + (52 * magnitude) / maxForce

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8 Z" fill="#1565C0" />
          </marker>
        </defs>
        <polygon points={`${baseX},${baseY} ${topX},${topY} ${topX},${baseY}`} fill="#F3F4F6" stroke="#9CA3AF" strokeWidth="1.2" />
        <rect
          x={boxX}
          y={boxY}
          width={boxW}
          height={boxH}
          fill="#FFFFFF"
          stroke="#374151"
          strokeWidth="1.2"
          transform={`rotate(${-diagram.angle_deg} ${cx} ${cy})`}
        />

        {/* Weight: always vertical downward */}
        <line
          x1={cx}
          y1={cy}
          x2={cx}
          y2={cy + scaledLen(diagram.weight_n)}
          stroke="#DC2626"
          strokeWidth="2"
          markerEnd={`url(#${markerId})`}
        />
        <text x={cx + 6} y={cy + scaledLen(diagram.weight_n) + 2} fontSize="10" fill="#DC2626">
          W {diagram.weight_n.toFixed(1)}N
        </text>

        {/* Normal: perpendicular outward from the incline */}
        <line
          x1={cx}
          y1={cy}
          x2={cx + normalOut.x * scaledLen(diagram.normal_n)}
          y2={cy + normalOut.y * scaledLen(diagram.normal_n)}
          stroke="#059669"
          strokeWidth="2"
          markerEnd={`url(#${markerId})`}
        />
        <text
          x={cx + normalOut.x * (scaledLen(diagram.normal_n) + 10)}
          y={cy + normalOut.y * (scaledLen(diagram.normal_n) + 10)}
          fontSize="10"
          fill="#059669"
        >
          N {diagram.normal_n.toFixed(1)}N
        </text>

        {/* Gravity component parallel to incline: downhill */}
        <line
          x1={cx}
          y1={cy}
          x2={cx + downhill.x * scaledLen(diagram.weight_parallel_n)}
          y2={cy + downhill.y * scaledLen(diagram.weight_parallel_n)}
          stroke="#2563EB"
          strokeWidth="2"
          markerEnd={`url(#${markerId})`}
        />
        <text
          x={cx + downhill.x * (scaledLen(diagram.weight_parallel_n) + 12)}
          y={cy + downhill.y * (scaledLen(diagram.weight_parallel_n) + 12)}
          fontSize="10"
          fill="#2563EB"
        >
          W∥ {diagram.weight_parallel_n.toFixed(1)}N
        </text>

        {diagram.has_friction && diagram.friction_n !== undefined && (
          <>
            {/* Friction opposes downhill tendency: uphill */}
            <line
              x1={cx}
              y1={cy}
              x2={cx + uphill.x * scaledLen(diagram.friction_n)}
              y2={cy + uphill.y * scaledLen(diagram.friction_n)}
              stroke="#7C3AED"
              strokeWidth="2"
              markerEnd={`url(#${markerId})`}
            />
            <text
              x={cx + uphill.x * (scaledLen(diagram.friction_n) + 10)}
              y={cy + uphill.y * (scaledLen(diagram.friction_n) + 10)}
              fontSize="10"
              fill="#7C3AED"
            >
              f {diagram.friction_n.toFixed(1)}N
            </text>
          </>
        )}

        <text x={baseX + 6} y={baseY - 8} fontSize="11" fill="#111827">θ={diagram.angle_deg.toFixed(1)}°</text>
      </svg>
    </Box>
  )
}

function TensionSystemDiagram({ diagram, markerId }: { diagram: TensionSystemDiagramPayload; markerId: string }) {
  const width = 360
  const height = 250
  const masses = diagram.masses_kg
  const weights = diagram.weights_n
  const tension = diagram.tension_n ?? 0

  if (diagram.system_type === 'single_mass_vertical' || diagram.system_type === 'single_mass_angled') {
    const anchorX = 180
    const anchorY = 36
    const blockW = 38
    const blockH = 28
    const angleDeg = diagram.angles_deg[0] ?? 0
    const ropeLen = 110
    const ropeDx = diagram.system_type === 'single_mass_angled' ? Math.sin((angleDeg * Math.PI) / 180) * ropeLen : 0
    const ropeDy = Math.cos((angleDeg * Math.PI) / 180) * ropeLen
    const bx = anchorX + ropeDx
    const by = anchorY + ropeDy

    return (
      <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
        <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
          <defs>
            <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L8,4 L0,8 Z" fill="#1565C0" />
            </marker>
          </defs>

          <line x1={90} y1={anchorY - 12} x2={270} y2={anchorY - 12} stroke="#9CA3AF" strokeWidth="3" />
          <circle cx={anchorX} cy={anchorY} r={3} fill="#4B5563" />
          <line x1={anchorX} y1={anchorY} x2={bx} y2={by} stroke="#4B5563" strokeWidth="2" />

          <rect x={bx - blockW / 2} y={by - blockH / 2} width={blockW} height={blockH} fill="#FFFFFF" stroke="#374151" strokeWidth="1.2" />
          <text x={bx} y={by + 4} fontSize="10" fill="#111827" textAnchor="middle">{`${masses[0]?.toFixed(1) ?? '?'} kg`}</text>

          <line x1={bx} y1={by} x2={bx - (bx - anchorX) * 0.45} y2={by - (by - anchorY) * 0.45} stroke="#2563EB" strokeWidth="2" markerEnd={`url(#${markerId})`} />
          <text x={bx - (bx - anchorX) * 0.52} y={by - (by - anchorY) * 0.52 - 4} fontSize="10" fill="#2563EB" textAnchor="middle">
            T {tension.toFixed(1)}N
          </text>

          <line x1={bx} y1={by} x2={bx} y2={by + 56} stroke="#DC2626" strokeWidth="2" markerEnd={`url(#${markerId})`} />
          <text x={bx + 6} y={by + 62} fontSize="10" fill="#DC2626">W {weights[0]?.toFixed(1) ?? '?'}N</text>

          {diagram.system_type === 'single_mass_angled' && diagram.horizontal_component_n !== undefined && (
            <text x={14} y={228} fontSize="10" fill="#6B7280">
              {`Components: Tx=${diagram.horizontal_component_n.toFixed(1)}N, Ty=${(diagram.vertical_component_n ?? 0).toFixed(1)}N`}
            </text>
          )}
        </svg>
      </Box>
    )
  }

  const pulleyX = 180
  const pulleyY = 48
  const pulleyR = 18
  const leftX = 114
  const rightX = 246
  const blockW = 36
  const blockH = 28
  const leftBy = 158
  const rightBy = 158

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8 Z" fill="#1565C0" />
          </marker>
        </defs>

        <line x1={80} y1={16} x2={280} y2={16} stroke="#9CA3AF" strokeWidth="3" />
        <line x1={pulleyX} y1={16} x2={pulleyX} y2={pulleyY - pulleyR} stroke="#4B5563" strokeWidth="2" />
        <circle cx={pulleyX} cy={pulleyY} r={pulleyR} fill="#F9FAFB" stroke="#4B5563" strokeWidth="2" />

        <line x1={leftX} y1={pulleyY} x2={leftX} y2={leftBy - blockH / 2} stroke="#4B5563" strokeWidth="2" />
        <line x1={rightX} y1={pulleyY} x2={rightX} y2={rightBy - blockH / 2} stroke="#4B5563" strokeWidth="2" />
        <line x1={leftX} y1={pulleyY} x2={pulleyX - pulleyR} y2={pulleyY} stroke="#4B5563" strokeWidth="2" />
        <line x1={pulleyX + pulleyR} y1={pulleyY} x2={rightX} y2={pulleyY} stroke="#4B5563" strokeWidth="2" />

        <rect x={leftX - blockW / 2} y={leftBy - blockH / 2} width={blockW} height={blockH} fill="#FFFFFF" stroke="#374151" strokeWidth="1.2" />
        <rect x={rightX - blockW / 2} y={rightBy - blockH / 2} width={blockW} height={blockH} fill="#FFFFFF" stroke="#374151" strokeWidth="1.2" />
        <text x={leftX} y={leftBy + 4} fontSize="10" fill="#111827" textAnchor="middle">{`${masses[0]?.toFixed(1) ?? '?'} kg`}</text>
        <text x={rightX} y={rightBy + 4} fontSize="10" fill="#111827" textAnchor="middle">{`${masses[1]?.toFixed(1) ?? '?'} kg`}</text>

        <line x1={leftX} y1={leftBy} x2={leftX} y2={leftBy - 44} stroke="#2563EB" strokeWidth="2" markerEnd={`url(#${markerId})`} />
        <line x1={rightX} y1={rightBy} x2={rightX} y2={rightBy - 44} stroke="#2563EB" strokeWidth="2" markerEnd={`url(#${markerId})`} />
        <text x={leftX + 6} y={leftBy - 48} fontSize="10" fill="#2563EB">T {tension.toFixed(1)}N</text>

        <line x1={leftX} y1={leftBy} x2={leftX} y2={leftBy + 46} stroke="#DC2626" strokeWidth="2" markerEnd={`url(#${markerId})`} />
        <line x1={rightX} y1={rightBy} x2={rightX} y2={rightBy + 46} stroke="#DC2626" strokeWidth="2" markerEnd={`url(#${markerId})`} />
        <text x={leftX + 6} y={leftBy + 54} fontSize="10" fill="#DC2626">W1 {weights[0]?.toFixed(1) ?? '?'}N</text>
        <text x={rightX + 6} y={rightBy + 54} fontSize="10" fill="#DC2626">W2 {weights[1]?.toFixed(1) ?? '?'}N</text>

        {diagram.direction === 'mass_1_down' && <text x={70} y={128} fontSize="12" fill="#111827">↓</text>}
        {diagram.direction === 'mass_2_down' && <text x={286} y={128} fontSize="12" fill="#111827">↓</text>}
        {diagram.acceleration_mps2 !== undefined && (
          <text x={14} y={232} fontSize="10" fill="#6B7280">a = {diagram.acceleration_mps2.toFixed(2)} m/s²</text>
        )}
      </svg>
    </Box>
  )
}

function MotionGraphsDiagram({ diagram }: { diagram: MotionGraphsPlotPayload }) {
  const width = 360
  const height = 240
  const pad = 26
  const plotW = width - pad * 2
  const plotH = height - pad * 2
  const n = Math.max(1, diagram.times_s.length - 1)

  const toPath = (series: number[]) => {
    const min = Math.min(...series)
    const max = Math.max(...series)
    const span = Math.max(1e-6, max - min)
    return series
      .map((v, i) => {
        const x = pad + (plotW * i) / n
        const y = pad + plotH - ((v - min) / span) * plotH
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')
  }

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title} ({diagram.motion_type})</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <rect x={pad} y={pad} width={plotW} height={plotH} fill="#F9FAFB" stroke="#D1D5DB" />
        <path d={toPath(diagram.position_series)} stroke="#2563EB" strokeWidth="2" fill="none" />
        <path d={toPath(diagram.velocity_series)} stroke="#059669" strokeWidth="2" fill="none" />
        <path d={toPath(diagram.acceleration_series)} stroke="#DC2626" strokeWidth="2" fill="none" />
        <text x={pad} y={height - 6} fontSize="10" fill="#6B7280">t</text>
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#2563EB">x(t)</text>
        <text x={pad + 44} y={pad + 12} fontSize="10" fill="#059669">v(t)</text>
        <text x={pad + 84} y={pad + 12} fontSize="10" fill="#DC2626">a(t)</text>
      </svg>
    </Box>
  )
}

function ProjectileTrajectoryDiagram({ diagram }: { diagram: ProjectileTrajectoryPayload }) {
  const width = 360
  const height = 240
  const pad = 26
  const points = diagram.trajectory_points
  const maxX = Math.max(1, ...points.map((p) => p.x_m))
  const maxY = Math.max(1, ...points.map((p) => p.y_m))

  const path = points
    .map((p, i) => {
      const x = pad + (p.x_m / maxX) * (width - 2 * pad)
      const y = height - pad - (p.y_m / maxY) * (height - 2 * pad)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={path} stroke="#2563EB" strokeWidth="2.2" fill="none" />
        <circle
          cx={pad + ((points[0]?.x_m || 0) / maxX) * (width - 2 * pad)}
          cy={height - pad - ((points[0]?.y_m || 0) / maxY) * (height - 2 * pad)}
          r="3"
          fill="#059669"
        />
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#111827">max h: {diagram.max_height_m.toFixed(1)} m</text>
      </svg>
    </Box>
  )
}

function ProjectileVelocityAnimationDiagram({ diagram, markerId }: { diagram: ProjectileVelocityAnimationPayload; markerId: string }) {
  const width = 360
  const height = 240
  const pad = 26
  const frames = diagram.frames
  const last = frames[Math.floor(frames.length / 2)] || frames[0]
  const maxX = Math.max(1, ...frames.map((f) => f.x_m))
  const maxY = Math.max(1, ...frames.map((f) => f.y_m))

  const path = frames
    .map((p, i) => {
      const x = pad + (p.x_m / maxX) * (width - 2 * pad)
      const y = height - pad - (p.y_m / maxY) * (height - 2 * pad)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  const px = pad + ((last?.x_m || 0) / maxX) * (width - 2 * pad)
  const py = height - pad - ((last?.y_m || 0) / maxY) * (height - 2 * pad)
  const vScale = 0.9
  const vx = (last?.vx_mps || 0) * vScale
  const vy = (last?.vy_mps || 0) * vScale

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8 Z" fill="#1565C0" />
          </marker>
        </defs>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={path} stroke="#2563EB" strokeWidth="2" fill="none" />
        <circle cx={px} cy={py} r="3.2" fill="#DC2626" />
        <line
          x1={px}
          y1={py}
          x2={px + vx}
          y2={py - vy}
          stroke="#059669"
          strokeWidth="2"
          markerEnd={`url(#${markerId})`}
        />
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#111827">
          t={last?.t_s.toFixed(2)}s, v={last?.speed_mps.toFixed(1)} m/s
        </text>
      </svg>
    </Box>
  )
}

function FreeFallTimelineDiagram({ diagram }: { diagram: FreeFallTimelinePayload }) {
  const width = 360
  const height = 240
  const pad = 26
  const points = diagram.timeline_points
  const maxT = Math.max(1, diagram.total_time_s)
  const maxH = Math.max(1, ...points.map((p) => p.h_m))

  const path = points
    .map((p, i) => {
      const x = pad + (p.t_s / maxT) * (width - 2 * pad)
      const y = height - pad - (Math.max(0, p.h_m) / maxH) * (height - 2 * pad)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={path} stroke="#DC2626" strokeWidth="2.2" fill="none" />
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#111827">
          t={diagram.total_time_s.toFixed(2)}s, v={diagram.final_v_mps.toFixed(2)}m/s
        </text>
      </svg>
    </Box>
  )
}

function RelativeMotionDiagram({ diagram }: { diagram: RelativeMotionIntersectionPayload }) {
  const width = 360
  const height = 240
  const pad = 26
  const times = [...diagram.object1.points.map((p) => p.t_s), ...diagram.object2.points.map((p) => p.t_s)]
  const xs = [...diagram.object1.points.map((p) => p.x_m), ...diagram.object2.points.map((p) => p.x_m)]
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const spanX = Math.max(1e-6, maxX - minX)
  const maxT = Math.max(1, ...times)
  const midT = maxT / 2
  const midX = minX + spanX / 2

  const mkPath = (pts: Array<{ t_s: number; x_m: number }>) =>
    pts
      .map((p, i) => {
        const x = pad + (p.t_s / maxT) * (width - 2 * pad)
        const y = height - pad - ((p.x_m - minX) / spanX) * (height - 2 * pad)
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <rect x={pad} y={pad} width={width - 2 * pad} height={height - 2 * pad} fill="#F9FAFB" stroke="#D1D5DB" />
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={mkPath(diagram.object1.points)} stroke="#2563EB" strokeWidth="2.2" fill="none" />
        <path d={mkPath(diagram.object2.points)} stroke="#7C3AED" strokeWidth="2.2" fill="none" />
        {diagram.meeting_time_s !== undefined && diagram.meeting_time_s !== null && diagram.meeting_position_m !== undefined && diagram.meeting_position_m !== null && (
          <circle
            cx={pad + (diagram.meeting_time_s / maxT) * (width - 2 * pad)}
            cy={height - pad - ((diagram.meeting_position_m - minX) / spanX) * (height - 2 * pad)}
            r="3.5"
            fill="#DC2626"
          />
        )}
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#2563EB">obj1 x(t)</text>
        <text x={pad + 56} y={pad + 12} fontSize="10" fill="#7C3AED">obj2 x(t)</text>
        {diagram.meeting_time_s !== undefined && diagram.meeting_time_s !== null && (
          <text x={pad + 118} y={pad + 12} fontSize="10" fill="#DC2626">meeting</text>
        )}

        {/* Dynamic axis tick labels with units */}
        <text x={pad - 8} y={height - pad + 12} fontSize="9" fill="#6B7280" textAnchor="end">{minX.toFixed(1)} m</text>
        <text x={pad - 8} y={height - pad - ((midX - minX) / spanX) * (height - 2 * pad) + 3} fontSize="9" fill="#6B7280" textAnchor="end">{midX.toFixed(1)} m</text>
        <text x={pad - 8} y={pad + 3} fontSize="9" fill="#6B7280" textAnchor="end">{maxX.toFixed(1)} m</text>
        <text x={pad} y={height - pad + 14} fontSize="9" fill="#6B7280" textAnchor="middle">0.0 s</text>
        <text x={pad + (midT / maxT) * (width - 2 * pad)} y={height - pad + 14} fontSize="9" fill="#6B7280" textAnchor="middle">{midT.toFixed(1)} s</text>
        <text x={width - pad} y={height - pad + 14} fontSize="9" fill="#6B7280" textAnchor="middle">{maxT.toFixed(1)} s</text>
        <text x={width - pad - 2} y={height - 6} fontSize="10" fill="#374151" textAnchor="end">Time, t (s)</text>
        <text x={6} y={pad - 6} fontSize="10" fill="#374151">Position, x (m)</text>
      </svg>
    </Box>
  )
}

function EnergyBarChartDiagram({ diagram }: { diagram: EnergyBarChartPayload }) {
  const width = 360
  const height = 240
  const baseY = 200
  const barW = 24
  const maxE = Math.max(
    1,
    diagram.initial.kinetic_j,
    diagram.initial.potential_j,
    diagram.initial.elastic_j,
    diagram.final.kinetic_j,
    diagram.final.potential_j,
    diagram.final.elastic_j
  )
  const toH = (e: number) => (e / maxE) * 120

  const bars = [
    { x: 48, e: diagram.initial.kinetic_j, c: '#2563EB', label: 'Ki' },
    { x: 84, e: diagram.initial.potential_j, c: '#059669', label: 'Pgi' },
    { x: 120, e: diagram.initial.elastic_j, c: '#7C3AED', label: 'Psi' },
    { x: 196, e: diagram.final.kinetic_j, c: '#2563EB', label: 'Kf' },
    { x: 232, e: diagram.final.potential_j, c: '#059669', label: 'Pgf' },
    { x: 268, e: diagram.final.elastic_j, c: '#7C3AED', label: 'Psf' },
  ]

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={34} y1={baseY} x2={330} y2={baseY} stroke="#9CA3AF" />
        {bars.map((b) => {
          const h = toH(b.e)
          return (
            <g key={b.label}>
              <rect x={b.x} y={baseY - h} width={barW} height={h} fill={b.c} opacity="0.85" />
              <text x={b.x + barW / 2} y={baseY + 13} fontSize="9" fill="#374151" textAnchor="middle">{b.label}</text>
            </g>
          )
        })}
        <text x={72} y={26} fontSize="10" fill="#111827" textAnchor="middle">Initial</text>
        <text x={220} y={26} fontSize="10" fill="#111827" textAnchor="middle">Final</text>
        <text x={302} y={26} fontSize="9" fill="#374151" textAnchor="end">
          ΔE={(diagram.difference_j ?? (diagram.initial.total_j - diagram.final.total_j)).toFixed(2)} J
        </text>
      </svg>
    </Box>
  )
}

function EnergyFlowDiagram({ diagram }: { diagram: EnergyFlowDiagramPayload }) {
  const width = 360
  const height = 210
  const initial = Number.isFinite(diagram.initial_mechanical_j) ? Math.max(0, diagram.initial_mechanical_j) : 0
  const final = Number.isFinite(diagram.final_mechanical_j) ? Math.max(0, diagram.final_mechanical_j) : 0
  const dissipated = Number.isFinite(diagram.dissipated_j) ? Math.max(0, diagram.dissipated_j) : 0
  const maxE = Math.max(1, initial, final, dissipated)
  const s = 180 / maxE

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <rect x={20} y={56} width={initial * s} height={28} fill="#2563EB" opacity="0.85" />
        <text x={22} y={50} fontSize="10" fill="#1F2937">Initial: {initial.toFixed(1)} J</text>
        <rect x={20} y={108} width={final * s} height={28} fill="#059669" opacity="0.85" />
        <text x={22} y={102} fontSize="10" fill="#1F2937">Final: {final.toFixed(1)} J</text>
        <rect x={20} y={160} width={dissipated * s} height={20} fill="#DC2626" opacity="0.85" />
        <text x={22} y={154} fontSize="10" fill="#1F2937">Dissipated: {dissipated.toFixed(1)} J</text>
        <text x={332} y={22} fontSize="10" fill="#374151" textAnchor="end">
          η={(diagram.efficiency_percent ?? 0).toFixed(1)}%
        </text>
      </svg>
    </Box>
  )
}

function WorkAreaUnderCurveDiagram({ diagram }: { diagram: WorkAreaUnderCurvePayload }) {
  const width = 360
  const height = 220
  const pad = 30
  const maxX = Math.max(1, diagram.displacement_m)
  const maxY = Math.max(1, Math.abs(diagram.force_parallel_n))
  const y0 = height - pad
  const yValue = y0 - (Math.abs(diagram.force_parallel_n) / maxY) * 120
  const xEnd = pad + (diagram.displacement_m / maxX) * (width - 2 * pad)
  const barY = Math.min(y0, yValue)
  const barH = Math.max(2, Math.abs(y0 - yValue))
  const barX = Math.min(pad, xEnd)
  const barW = Math.max(2, Math.abs(xEnd - pad))

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={y0} x2={width - pad} y2={y0} stroke="#9CA3AF" />
        <line x1={pad} y1={pad - 4} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <rect x={barX} y={barY} width={barW} height={barH} fill="#93C5FD" opacity="0.55" />
        <line x1={pad} y1={yValue} x2={xEnd} y2={yValue} stroke="#2563EB" strokeWidth="2" />
        <text x={pad + 4} y={pad + 10} fontSize="10" fill="#111827">
          F∥={diagram.force_parallel_n.toFixed(2)} N, d={diagram.displacement_m.toFixed(2)} m
        </text>
        <text x={pad + 4} y={pad + 24} fontSize="10" fill="#111827">W={diagram.work_j.toFixed(2)} J</text>
      </svg>
    </Box>
  )
}

function BeforeAfterEnergySnapshotDiagram({ diagram }: { diagram: BeforeAfterEnergySnapshotPayload }) {
  const width = 360
  const height = 220
  const hasPoints = (diagram.points?.length ?? 0) > 1

  if (hasPoints) {
    const points = diagram.points || []
    const pad = 28
    const plotW = width - 2 * pad
    const plotH = height - 2 * pad
    const maxIdx = Math.max(1, ...points.map((p) => p.point_index))
    const maxTot = Math.max(1, ...points.map((p) => p.total_j))
    const path = points
      .map((p, i) => {
        const x = pad + (p.point_index / maxIdx) * plotW
        const y = pad + plotH - (p.total_j / maxTot) * plotH
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')

    return (
      <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
        <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
          <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
          <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
          <path d={path} stroke="#7C3AED" strokeWidth="2.2" fill="none" />
          <text x={pad + 4} y={pad + 11} fontSize="10" fill="#111827">Total mechanical energy by point</text>
        </svg>
      </Box>
    )
  }

  const initTotal = diagram.initial?.total_j ?? 0
  const finalTotal = diagram.final?.total_j ?? 0
  const maxE = Math.max(1, initTotal, finalTotal)
  const h1 = (initTotal / maxE) * 120
  const h2 = (finalTotal / maxE) * 120

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={54} y1={186} x2={308} y2={186} stroke="#9CA3AF" />
        <rect x={96} y={186 - h1} width={42} height={h1} fill="#2563EB" opacity="0.85" />
        <rect x={212} y={186 - h2} width={42} height={h2} fill="#059669" opacity="0.85" />
        <text x={117} y={200} fontSize="10" fill="#111827" textAnchor="middle">Before</text>
        <text x={233} y={200} fontSize="10" fill="#111827" textAnchor="middle">After</text>
        <text x={117} y={186 - h1 - 6} fontSize="10" fill="#111827" textAnchor="middle">{initTotal.toFixed(1)} J</text>
        <text x={233} y={186 - h2 - 6} fontSize="10" fill="#111827" textAnchor="middle">{finalTotal.toFixed(1)} J</text>
      </svg>
    </Box>
  )
}

function MomentumBeforeAfterVectorsDiagram({ diagram }: { diagram: MomentumBeforeAfterVectorsPayload }) {
  const width = 360
  const height = 230
  const cx = 180
  const cy1 = 78
  const cy2 = 166
  const maxP = Math.max(
    1,
    ...diagram.before.map((o) => Math.abs(o.momentum_kg_mps)),
    ...diagram.after.map((o) => Math.abs(o.momentum_kg_mps))
  )
  const scale = 90 / maxP

  const drawRow = (y: number, objs: Array<{ name: string; momentum_kg_mps: number }>, color: string) =>
    objs.map((o, idx) => {
      const p = o.momentum_kg_mps
      const x2 = cx + p * scale
      const labelY = y - 12 + idx * 14
      return (
        <g key={`${o.name}-${idx}-${y}`}>
          <line x1={cx} y1={y} x2={x2} y2={y} stroke={color} strokeWidth="2.1" />
          <text x={x2 + (p >= 0 ? 4 : -4)} y={labelY} fontSize="9" fill="#111827" textAnchor={p >= 0 ? 'start' : 'end'}>
            {`${o.name}: ${p.toFixed(1)} kg·m/s`}
          </text>
        </g>
      )
    })

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={36} y1={cy1} x2={324} y2={cy1} stroke="#D1D5DB" />
        <line x1={36} y1={cy2} x2={324} y2={cy2} stroke="#D1D5DB" />
        <line x1={cx} y1={46} x2={cx} y2={198} stroke="#9CA3AF" strokeDasharray="4 4" />
        <text x={12} y={cy1 + 4} fontSize="10" fill="#111827">Before</text>
        <text x={12} y={cy2 + 4} fontSize="10" fill="#111827">After</text>
        {drawRow(cy1, diagram.before, '#2563EB')}
        {drawRow(cy2, diagram.after.map((o) => ({ ...o, momentum_kg_mps: o.momentum_kg_mps })), '#059669')}
        <text x={200} y={220} fontSize="10" fill="#374151">
          Δp_total={(diagram.totals.difference_kg_mps).toFixed(3)} kg·m/s
        </text>
      </svg>
    </Box>
  )
}

function ImpulseAreaPlotDiagram({ diagram }: { diagram: ImpulseAreaPlotPayload }) {
  const width = 360
  const height = 220
  const pad = 28
  const maxT = Math.max(1e-6, Math.abs(diagram.time_s))
  const maxF = Math.max(1e-6, Math.abs(diagram.force_n))
  const x0 = pad
  const y0 = height - pad
  const x1 = pad + (Math.abs(diagram.time_s) / maxT) * (width - 2 * pad)
  const y1 = y0 - (Math.abs(diagram.force_n) / maxF) * 120

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={y0} x2={width - pad} y2={y0} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <rect x={x0} y={Math.min(y0, y1)} width={Math.max(2, x1 - x0)} height={Math.max(2, Math.abs(y0 - y1))} fill="#93C5FD" opacity="0.55" />
        <line x1={x0} y1={y1} x2={x1} y2={y1} stroke="#2563EB" strokeWidth="2.2" />
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#111827">F={diagram.force_n.toFixed(2)} N</text>
        <text x={pad + 4} y={pad + 24} fontSize="10" fill="#111827">Δt={diagram.time_s.toFixed(2)} s</text>
        <text x={pad + 4} y={pad + 36} fontSize="10" fill="#111827">J={diagram.impulse_ns.toFixed(2)} N·s</text>
      </svg>
    </Box>
  )
}

function CollisionStoryboardDiagram({ diagram }: { diagram: CollisionStoryboardPayload }) {
  const width = 360
  const height = 220
  const y = 120

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={30} y1={y} x2={330} y2={y} stroke="#D1D5DB" />
        <rect x={52} y={y - 24} width={36} height={20} fill="#BFDBFE" stroke="#1E40AF" />
        <rect x={132} y={y + 4} width={36} height={20} fill="#FBCFE8" stroke="#9D174D" />
        <circle cx={206} cy={y} r={5} fill="#DC2626" />
        <rect x={250} y={y - 10} width={52} height={20} fill="#BBF7D0" stroke="#166534" />
        <text x={52} y={y - 30} fontSize="9" fill="#1F2937">{diagram.objects[0]?.name || 'Obj1'}</text>
        <text x={132} y={y + 38} fontSize="9" fill="#1F2937">{diagram.objects[1]?.name || 'Obj2'}</text>
        <text x={244} y={y - 18} fontSize="9" fill="#1F2937">{`After: ${diagram.combined_after.speed_mps.toFixed(2)} m/s`}</text>
        <text x={12} y={26} fontSize="10" fill="#111827">{`ΔKE=${diagram.energy.dissipated_j.toFixed(1)} J`}</text>
      </svg>
    </Box>
  )
}

function CenterOfMassTraceDiagram({ diagram }: { diagram: CenterOfMassTracePayload }) {
  const width = 360
  const height = 230
  const pad = 26
  const pts = diagram.trace_points
  const maxX = Math.max(1e-6, ...pts.map((p) => p.x_m), 1)
  const maxY = Math.max(1e-6, ...pts.map((p) => p.y_m), 1)

  const path = pts
    .map((p, i) => {
      const x = pad + (p.x_m / maxX) * (width - 2 * pad)
      const y = height - pad - (p.y_m / maxY) * (height - 2 * pad)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={path} stroke="#7C3AED" strokeWidth="2.2" fill="none" />
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#111827">
          vCOM=({diagram.vcom_x_mps.toFixed(2)}, {diagram.vcom_y_mps.toFixed(2)}) m/s
        </text>
      </svg>
    </Box>
  )
}

function TorqueLeverDiagram({ diagram, markerId }: { diagram: TorqueLeverDiagramPayload; markerId: string }) {
  const width = 360
  const height = 220
  const cx = 96
  const cy = 130
  const lever = 120
  const angleRad = (diagram.angle_deg * Math.PI) / 180
  const fx = Math.cos(angleRad) * 70
  const fy = -Math.sin(angleRad) * 70

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8 Z" fill="#2563EB" />
          </marker>
        </defs>
        <line x1={cx} y1={cy} x2={cx + lever} y2={cy} stroke="#374151" strokeWidth="4" />
        <circle cx={cx} cy={cy} r={5} fill="#111827" />
        <line x1={cx + lever} y1={cy} x2={cx + lever + fx} y2={cy + fy} stroke="#2563EB" strokeWidth="2.5" markerEnd={`url(#${markerId})`} />
        <path d={`M ${cx + lever - 20} ${cy} A 20 20 0 0 1 ${cx + lever - 20 * Math.cos(angleRad)} ${cy - 20 * Math.sin(angleRad)}`} fill="none" stroke="#9CA3AF" />
        <text x={12} y={18} fontSize="10" fill="#111827">r={diagram.radius_m.toFixed(2)} m</text>
        <text x={12} y={32} fontSize="10" fill="#111827">F={diagram.force_n.toFixed(2)} N @ {diagram.angle_deg.toFixed(1)}°</text>
        <text x={12} y={46} fontSize="10" fill="#111827">F⊥={diagram.force_perpendicular_n.toFixed(2)} N</text>
        <text x={12} y={60} fontSize="10" fill="#111827">τ={diagram.torque_nm.toFixed(2)} N·m ({diagram.direction})</text>
      </svg>
    </Box>
  )
}

function RotationGraphsDiagram({ diagram }: { diagram: RotationGraphsPlotPayload }) {
  const width = 360
  const height = 230
  const pad = 26
  const tMax = Math.max(1e-6, ...diagram.times_s)
  const allVals = [...diagram.theta_series_rad, ...diagram.omega_series_rad_s, ...diagram.alpha_series_rad_s2]
  const yMin = Math.min(...allVals, 0)
  const yMax = Math.max(...allVals, 1e-6)
  const yRange = Math.max(1e-6, yMax - yMin)
  const sx = (t: number) => pad + (t / tMax) * (width - 2 * pad)
  const sy = (v: number) => height - pad - ((v - yMin) / yRange) * (height - 2 * pad)

  const mkPath = (vals: number[]) => vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${sx(diagram.times_s[i]).toFixed(2)},${sy(v).toFixed(2)}`).join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={mkPath(diagram.theta_series_rad)} stroke="#2563EB" strokeWidth="2" fill="none" />
        <path d={mkPath(diagram.omega_series_rad_s)} stroke="#059669" strokeWidth="2" fill="none" />
        <path d={mkPath(diagram.alpha_series_rad_s2)} stroke="#DC2626" strokeWidth="2" fill="none" />
        <text x={pad + 4} y={pad + 12} fontSize="10" fill="#2563EB">θ(t)</text>
        <text x={pad + 40} y={pad + 12} fontSize="10" fill="#059669">ω(t)</text>
        <text x={pad + 76} y={pad + 12} fontSize="10" fill="#DC2626">α(t)</text>
      </svg>
    </Box>
  )
}

function CircularMotionVectorsDiagram({ diagram, markerId }: { diagram: CircularMotionVectorsPayload; markerId: string }) {
  const width = 360
  const height = 230
  const cx = 180
  const cy = 120
  const r = 62
  const px = cx + r
  const py = cy

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <defs>
          <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L8,4 L0,8 Z" fill="#2563EB" />
          </marker>
        </defs>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#9CA3AF" />
        <circle cx={px} cy={py} r={4} fill="#111827" />
        <line x1={px} y1={py} x2={px} y2={py - 44} stroke="#2563EB" strokeWidth="2.2" markerEnd={`url(#${markerId})`} />
        <line x1={px} y1={py} x2={cx} y2={cy} stroke="#DC2626" strokeWidth="2.2" markerEnd={`url(#${markerId})`} />
        <text x={px + 8} y={py - 40} fontSize="10" fill="#2563EB">v</text>
        <text x={cx + r / 2 - 8} y={cy - 8} fontSize="10" fill="#DC2626">aₙ</text>
        <text x={12} y={18} fontSize="10" fill="#111827">r={diagram.radius_m.toFixed(2)} m, v={diagram.speed_mps.toFixed(2)} m/s</text>
        <text x={12} y={32} fontSize="10" fill="#111827">ω={diagram.omega_rad_s.toFixed(2)} rad/s, T={diagram.period_s.toFixed(2)} s</text>
        <text x={12} y={46} fontSize="10" fill="#111827">aₙ={diagram.centripetal_acc_mps2.toFixed(2)} m/s²</text>
      </svg>
    </Box>
  )
}

function RollingEnergySplitDiagram({ diagram }: { diagram: RollingEnergySplitPayload }) {
  const width = 360
  const height = 220
  const pad = 28
  const chartH = 140
  const maxE = Math.max(1e-6, diagram.total_ke_j)
  const bars = [
    { label: 'Trans', value: diagram.translational_ke_j, color: '#2563EB' },
    { label: 'Rot', value: diagram.rotational_ke_j, color: '#059669' },
    { label: 'Total', value: diagram.total_ke_j, color: '#7C3AED' },
  ]

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        {bars.map((b, i) => {
          const bw = 48
          const gap = 30
          const x = pad + 26 + i * (bw + gap)
          const bh = (b.value / maxE) * chartH
          const y = height - pad - bh
          return (
            <g key={b.label}>
              <rect x={x} y={y} width={bw} height={bh} fill={b.color} opacity="0.8" />
              <text x={x + bw / 2} y={height - pad + 14} textAnchor="middle" fontSize="10" fill="#111827">{b.label}</text>
            </g>
          )
        })}
        <text x={12} y={18} fontSize="10" fill="#111827">v={diagram.velocity_mps.toFixed(2)} m/s, ω={diagram.omega_rad_s.toFixed(2)} rad/s</text>
      </svg>
    </Box>
  )
}

function ShmSpringAnimationDiagram({ diagram }: { diagram: ShmSpringAnimationPayload }) {
  const frame = diagram.frames[Math.floor(diagram.frames.length / 4)] || diagram.frames[0]
  const width = 360
  const height = 180
  const cx = 180
  const y = 96
  const px = cx + (frame?.x_m ?? 0) * 180
  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={24} y1={y} x2={336} y2={y} stroke="#E5E7EB" />
        <line x1={cx} y1={y - 18} x2={cx} y2={y + 18} stroke="#9CA3AF" strokeDasharray="3 3" />
        <line x1={36} y1={y} x2={px} y2={y} stroke="#2563EB" strokeWidth="2.4" />
        <rect x={px - 14} y={y - 12} width={28} height={24} fill="#BFDBFE" stroke="#1E40AF" />
        <text x={12} y={20} fontSize="10" fill="#111827">A={diagram.amplitude_m.toFixed(2)} m, T={diagram.period_s.toFixed(2)} s</text>
      </svg>
    </Box>
  )
}

function PendulumAnimationDiagram({ diagram }: { diagram: PendulumAnimationPayload }) {
  const frame = diagram.frames[Math.floor(diagram.frames.length / 4)] || diagram.frames[0]
  const width = 360
  const height = 220
  const ox = 180
  const oy = 34
  const scale = 90 / Math.max(0.25, diagram.length_m)
  const bx = ox + (frame?.x_m ?? 0) * scale
  const by = oy - (frame?.y_m ?? -diagram.length_m) * scale

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <circle cx={ox} cy={oy} r={4} fill="#111827" />
        <line x1={ox} y1={oy} x2={bx} y2={by} stroke="#2563EB" strokeWidth="2.4" />
        <circle cx={bx} cy={by} r={10} fill="#BFDBFE" stroke="#1E40AF" />
        <text x={12} y={20} fontSize="10" fill="#111827">L={diagram.length_m.toFixed(2)} m, T={diagram.period_s.toFixed(2)} s</text>
        <text x={12} y={34} fontSize="10" fill="#111827">θmax={diagram.theta_max_deg.toFixed(1)}°</text>
      </svg>
    </Box>
  )
}

function TravelingWaveAnimationDiagram({ diagram }: { diagram: TravelingWaveAnimationPayload }) {
  const width = 360
  const height = 220
  const pad = 24
  const frame = diagram.frames[Math.floor(diagram.frames.length / 4)] || diagram.frames[0]
  const samples = frame?.samples || []
  const xMax = Math.max(1e-6, ...samples.map((p) => p.x_m))
  const yMax = Math.max(1e-6, diagram.amplitude_m)
  const sx = (x: number) => pad + (x / xMax) * (width - 2 * pad)
  const sy = (y: number) => height / 2 - (y / yMax) * 70
  const d = samples.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x_m).toFixed(2)},${sy(p.y_m).toFixed(2)}`).join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} stroke="#9CA3AF" strokeDasharray="4 3" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={d} stroke="#2563EB" strokeWidth="2.2" fill="none" />
        <text x={12} y={18} fontSize="10" fill="#111827">
          v={diagram.velocity_mps.toFixed(2)} m/s, f={diagram.frequency_hz.toFixed(2)} Hz, λ={diagram.wavelength_m.toFixed(3)} m
        </text>
      </svg>
    </Box>
  )
}

function StandingWaveModeShapeDiagram({ diagram }: { diagram: StandingWaveModeShapePayload }) {
  const width = 360
  const height = 230
  const pad = 24
  const mode = diagram.mode_shapes[1] || diagram.mode_shapes[0]
  const samples = mode?.samples || []
  const sx = (x: number) => pad + (x / Math.max(1e-6, diagram.length_m)) * (width - 2 * pad)
  const sy = (y: number) => height / 2 - y * 62
  const d = samples.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x_m).toFixed(2)},${sy(p.y_norm).toFixed(2)}`).join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} stroke="#9CA3AF" strokeDasharray="4 3" />
        <path d={d} stroke="#059669" strokeWidth="2.2" fill="none" />
        <text x={12} y={18} fontSize="10" fill="#111827">
          {diagram.system_type.replace('_', ' ')}, f₁={diagram.fundamental_hz.toFixed(2)} Hz, mode n={mode?.n ?? 1}
        </text>
      </svg>
    </Box>
  )
}

function InterferenceFringeMapDiagram({ diagram }: { diagram: InterferenceFringeMapPayload }) {
  const width = 360
  const height = 230
  const pad = 24
  const yMax = Math.max(1e-9, ...diagram.samples.map((s) => Math.abs(s.y_m)))
  const sx = (i: number) => pad + (i / Math.max(1, diagram.samples.length - 1)) * (width - 2 * pad)
  const sy = (v: number) => height - pad - v * (height - 2 * pad)
  const d = diagram.samples.map((s, i) => `${i === 0 ? 'M' : 'L'}${sx(i).toFixed(2)},${sy(s.intensity_norm).toFixed(2)}`).join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={d} stroke="#7C3AED" strokeWidth="2.2" fill="none" />
        <text x={12} y={18} fontSize="10" fill="#111827">
          class={diagram.classification}, λ={diagram.wavelength_m.toExponential(2)} m, d={diagram.slit_separation_m.toExponential(2)} m
        </text>
        <text x={12} y={32} fontSize="10" fill="#111827">
          screen ±{yMax.toFixed(3)} m @ D={diagram.screen_distance_m.toFixed(2)} m
        </text>
      </svg>
    </Box>
  )
}

function DopplerWavefrontAnimationDiagram({ diagram }: { diagram: DopplerWavefrontAnimationPayload }) {
  const width = 360
  const height = 220
  const frame = diagram.frames[Math.floor(diagram.frames.length / 3)] || diagram.frames[0]
  const sx = (x: number) => 180 + x * 24
  const sy = (_: number) => 118

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        {(frame?.fronts || []).map((f, idx) => (
          <circle
            key={`${idx}-${f.radius_m}`}
            cx={sx(f.x_m)}
            cy={sy(0)}
            r={Math.max(0.5, f.radius_m * 22)}
            fill="none"
            stroke="#93C5FD"
            strokeWidth="1.1"
            opacity="0.7"
          />
        ))}
        <circle cx={sx(frame?.source_x_m ?? -2)} cy={sy(0)} r={6} fill="#2563EB" />
        <circle cx={sx(frame?.observer_x_m ?? 2)} cy={sy(0)} r={6} fill="#DC2626" />
        <text x={12} y={18} fontSize="10" fill="#111827">
          fₛ={diagram.source_frequency_hz.toFixed(1)} Hz, f'={diagram.observed_frequency_hz.toFixed(1)} Hz, Δf={diagram.frequency_shift_hz.toFixed(1)} Hz
        </text>
      </svg>
    </Box>
  )
}

function PvDiagram({ diagram }: { diagram: PvDiagramPayload }) {
  const width = 360
  const height = 240
  const pad = 26
  const maxV = Math.max(1e-9, diagram.axis?.max_volume_m3 ?? diagram.state.volume_m3 * 1.2)
  const maxP = Math.max(1.0, diagram.axis?.max_pressure_pa ?? diagram.state.pressure_pa * 1.2)
  const sx = (v: number) => pad + (v / maxV) * (width - 2 * pad)
  const sy = (p: number) => height - pad - (p / maxP) * (height - 2 * pad)

  const stateX = sx(diagram.state.volume_m3)
  const stateY = sy(diagram.state.pressure_pa)

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        {diagram.isotherms.map((iso, idx) => {
          const d = iso.points
            .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.volume_m3).toFixed(2)},${sy(p.pressure_pa).toFixed(2)}`)
            .join(' ')
          const color = idx === 1 ? '#2563EB' : '#93C5FD'
          return <path key={`iso-${idx}`} d={d} stroke={color} strokeWidth={idx === 1 ? 2.2 : 1.2} fill="none" />
        })}
        <circle cx={stateX} cy={stateY} r={4.2} fill="#DC2626" />
        <text x={stateX + 6} y={stateY - 6} fontSize="10" fill="#111827">State</text>
        <text x={12} y={18} fontSize="10" fill="#111827">P={diagram.state.pressure_pa.toFixed(0)} Pa</text>
        <text x={12} y={32} fontSize="10" fill="#111827">V={diagram.state.volume_m3.toExponential(2)} m³, T={diagram.state.temperature_k.toFixed(1)} K</text>
      </svg>
    </Box>
  )
}

function HeatingCurvePlotDiagram({ diagram }: { diagram: HeatingCurvePlotPayload }) {
  const width = 360
  const height = 230
  const pad = 24
  const maxQ = Math.max(1e-6, Math.abs(diagram.heat_j))
  const maxDT = Math.max(1e-6, Math.abs(diagram.delta_t_k))
  const sx = (q: number) => pad + (q / maxQ) * (width - 2 * pad)
  const sy = (dT: number) => height - pad - (dT / maxDT) * (height - 2 * pad)
  const d = diagram.q_vs_delta_t
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.q_j).toFixed(2)},${sy(p.delta_t_k).toFixed(2)}`)
    .join(' ')

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={d} stroke="#EF4444" strokeWidth="2.2" fill="none" />
        <text x={12} y={18} fontSize="10" fill="#111827">Q={diagram.heat_j.toFixed(2)} J, ΔT={diagram.delta_t_k.toFixed(2)} K</text>
        <text x={12} y={32} fontSize="10" fill="#111827">m={diagram.mass_kg.toFixed(3)} kg, c={diagram.specific_heat_j_per_kgk.toFixed(1)} J/(kg·K)</text>
      </svg>
    </Box>
  )
}

function HeatTransferPathDiagram({ diagram }: { diagram: HeatTransferPathDiagramPayload }) {
  const width = 360
  const height = 220
  const x0 = 70
  const y0 = 82
  const blockW = 220
  const blockH = 74

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <rect x={x0} y={y0} width={blockW} height={blockH} rx={8} fill="#E5E7EB" stroke="#9CA3AF" />
        <line x1={x0 - 40} y1={y0 + blockH / 2} x2={x0} y2={y0 + blockH / 2} stroke="#DC2626" strokeWidth="3" />
        <line x1={x0 + blockW} y1={y0 + blockH / 2} x2={x0 + blockW + 40} y2={y0 + blockH / 2} stroke="#2563EB" strokeWidth="3" />
        <text x={24} y={y0 + blockH / 2 - 8} fontSize="10" fill="#111827">Hot</text>
        <text x={24} y={y0 + blockH / 2 + 8} fontSize="10" fill="#111827">{diagram.hot_side_temp_c.toFixed(1)}°C</text>
        <text x={x0 + blockW + 8} y={y0 + blockH / 2 - 8} fontSize="10" fill="#111827">Cold</text>
        <text x={x0 + blockW + 8} y={y0 + blockH / 2 + 8} fontSize="10" fill="#111827">{diagram.cold_side_temp_c.toFixed(1)}°C</text>
        <text x={12} y={18} fontSize="10" fill="#111827">Q̇={diagram.heat_rate_w.toFixed(2)} W, ΔT={diagram.delta_t_k.toFixed(2)} K</text>
        <text x={12} y={32} fontSize="10" fill="#111827">k={diagram.conductivity_w_mk.toFixed(2)} W/(m·K), L={diagram.thickness_m.toFixed(3)} m</text>
      </svg>
    </Box>
  )
}

function CarnotCycleAnimationDiagram({ diagram }: { diagram: CarnotCycleAnimationPayload }) {
  const width = 360
  const height = 240
  const pad = 24
  const sx = (v: number) => pad + v * (width - 2 * pad)
  const sy = (p: number) => height - pad - p * (height - 2 * pad)
  const poly = diagram.cycle_points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.v_norm).toFixed(2)},${sy(p.p_norm).toFixed(2)}`)
    .join(' ') + ' Z'
  const frame = diagram.frames[Math.floor(diagram.frames.length / 3)] || diagram.frames[0]

  return (
    <Box sx={{ mt: 1.5, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{diagram.title}</Typography>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={diagram.title}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} stroke="#9CA3AF" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} stroke="#9CA3AF" />
        <path d={poly} fill="rgba(14,165,233,0.12)" stroke="#0EA5E9" strokeWidth="2.2" />
        {diagram.cycle_points.map((p) => (
          <text key={p.label} x={sx(p.v_norm) + 4} y={sy(p.p_norm) - 4} fontSize="10" fill="#111827">{p.label}</text>
        ))}
        {frame && <circle cx={sx(frame.v_norm)} cy={sy(frame.p_norm)} r={4.2} fill="#DC2626" />}
        <text x={12} y={18} fontSize="10" fill="#111827">η={diagram.efficiency_percent.toFixed(2)}%, Th={diagram.t_hot_k.toFixed(1)} K, Tc={diagram.t_cold_k.toFixed(1)} K</text>
      </svg>
    </Box>
  )
}

function ForcesDiagramRenderer({ diagram, markerId }: { diagram: DiagramPayload; markerId: string }) {
  switch (diagram.type) {
    case 'free_body_diagram':
      return <FreeBodyDiagram diagram={diagram} markerId={markerId} />
    case 'force_vector_addition':
      return <ForceVectorAdditionDiagram diagram={diagram} markerId={markerId} />
    case 'force_components_diagram':
      return <ForceComponentsDiagram diagram={diagram} markerId={markerId} />
    case 'equilibrium_residual_vector':
      return <EquilibriumDiagram diagram={diagram} markerId={markerId} />
    case 'inclined_plane_diagram':
      return <InclinedPlaneDiagram diagram={diagram} markerId={markerId} />
    case 'tension_system_diagram':
      return <TensionSystemDiagram diagram={diagram} markerId={markerId} />
    case 'motion_graphs_plot':
      return <MotionGraphsDiagram diagram={diagram} />
    case 'projectile_trajectory':
      return <ProjectileTrajectoryDiagram diagram={diagram} />
    case 'projectile_velocity_animation':
      return <ProjectileVelocityAnimationDiagram diagram={diagram} markerId={markerId} />
    case 'free_fall_timeline':
      return <FreeFallTimelineDiagram diagram={diagram} />
    case 'relative_motion_intersection':
      return <RelativeMotionDiagram diagram={diagram} />
    case 'energy_bar_chart':
      return <EnergyBarChartDiagram diagram={diagram} />
    case 'energy_flow_diagram':
      return <EnergyFlowDiagram diagram={diagram} />
    case 'work_area_under_curve':
      return <WorkAreaUnderCurveDiagram diagram={diagram} />
    case 'before_after_energy_snapshot':
      return <BeforeAfterEnergySnapshotDiagram diagram={diagram} />
    case 'momentum_before_after_vectors':
      return <MomentumBeforeAfterVectorsDiagram diagram={diagram} />
    case 'impulse_area_plot':
      return <ImpulseAreaPlotDiagram diagram={diagram} />
    case 'collision_storyboard':
      return <CollisionStoryboardDiagram diagram={diagram} />
    case 'center_of_mass_trace':
      return <CenterOfMassTraceDiagram diagram={diagram} />
    case 'torque_lever_diagram':
      return <TorqueLeverDiagram diagram={diagram} markerId={markerId} />
    case 'rotation_graphs_plot':
      return <RotationGraphsDiagram diagram={diagram} />
    case 'circular_motion_vectors':
      return <CircularMotionVectorsDiagram diagram={diagram} markerId={markerId} />
    case 'rolling_energy_split':
      return <RollingEnergySplitDiagram diagram={diagram} />
    case 'shm_spring_animation':
      return <ShmSpringAnimationDiagram diagram={diagram} />
    case 'pendulum_animation':
      return <PendulumAnimationDiagram diagram={diagram} />
    case 'traveling_wave_animation':
      return <TravelingWaveAnimationDiagram diagram={diagram} />
    case 'standing_wave_mode_shape':
      return <StandingWaveModeShapeDiagram diagram={diagram} />
    case 'interference_fringe_map':
      return <InterferenceFringeMapDiagram diagram={diagram} />
    case 'doppler_wavefront_animation':
      return <DopplerWavefrontAnimationDiagram diagram={diagram} />
    case 'pv_diagram':
      return <PvDiagram diagram={diagram} />
    case 'heating_curve_plot':
      return <HeatingCurvePlotDiagram diagram={diagram} />
    case 'heat_transfer_path_diagram':
      return <HeatTransferPathDiagram diagram={diagram} />
    case 'carnot_cycle_animation':
      return <CarnotCycleAnimationDiagram diagram={diagram} />
    default:
      return null
  }
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const agentColor = getAgentColor(message.agentId)
  const agentIcon = getAgentIcon(message.agentId)
  const hasDiagram = !isUser && !!message.diagram
  const markerId = `arrowhead-${message.id}`

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          alignItems: 'flex-start',
          maxWidth: '80%',
        }}
      >
        {/* Avatar */}
        <Box
          sx={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: isUser ? 'primary.main' : agentColor,
            color: 'white',
            fontSize: '1.2rem',
            flexShrink: 0,
            mx: 1,
          }}
        >
          {isUser ? '👨‍🎓' : agentIcon}
        </Box>

        {/* Message Content */}
        <Paper
          elevation={1}
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: isUser ? 'primary.main' : 'background.paper',
            color: isUser ? 'primary.contrastText' : 'text.primary',
            borderTopRightRadius: isUser ? 0 : 16,
            borderTopLeftRadius: isUser ? 16 : 0,
          }}
        >
          {/* Markdown content with LaTeX math rendering */}
          <Box
            sx={{
              '& p': { m: 0, mb: 1 },
              '& p:last-child': { mb: 0 },
              '& code': {
                bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'grey.100',
                px: 0.5,
                borderRadius: 0.5,
                fontFamily: 'monospace',
              },
              '& pre': {
                bgcolor: isUser ? 'rgba(255,255,255,0.1)' : 'grey.100',
                p: 1,
                borderRadius: 1,
                overflow: 'auto',
              },
              '& ul, & ol': { pl: 2, mb: 1 },
              '& li': { mb: 0.5 },
              // KaTeX math styling
              '& .katex': {
                fontSize: '1.1em',
              },
              '& .katex-display': {
                margin: '0.5em 0',
                overflow: 'auto',
              },
            }}
          >
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {message.content}
            </ReactMarkdown>
          </Box>

          {hasDiagram && message.diagram && (
            <ForcesDiagramRenderer diagram={message.diagram} markerId={markerId} />
          )}

          {/* Tools used indicator */}
          {message.toolsUsed && message.toolsUsed.length > 0 && (
            <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {message.toolsUsed.map((tool, i) => (
                <Chip
                  key={i}
                  label={tool}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          )}

          {/* Timestamp */}
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              mt: 1,
              opacity: 0.7,
              textAlign: isUser ? 'right' : 'left',
            }}
          >
            {new Date(message.timestamp).toLocaleTimeString()}
          </Typography>
        </Paper>
      </Box>
    </Box>
  )
}
