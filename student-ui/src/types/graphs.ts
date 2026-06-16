export interface GraphAxis {
  label: string
  unit?: string
}

export interface GraphPoint {
  x: number
  y: number
}

export interface GraphSeries {
  label: string
  unit?: string
  color?: string
  points: GraphPoint[]
}

export interface GraphDefinition {
  id: string
  title: string
  xAxis: GraphAxis
  yAxis: GraphAxis
  series: GraphSeries[]
}

export interface PhysicsGraphParameter {
  symbol: string
  label: string
  value: number | string
  unit?: string
  assumed?: boolean
}

export interface PhysicsGraphPayload {
  type: 'kinematics_1d' | 'projectile_motion' | 'kinematics_piecewise'
  title: string
  subtitle?: string
  motionType?: string
  parameters: PhysicsGraphParameter[]
  warnings?: string[]
  graphs: GraphDefinition[]
  source?: {
    problem?: string
    equations?: string[]
  }
}
