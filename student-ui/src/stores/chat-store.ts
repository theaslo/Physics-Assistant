import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  agentId: string
  toolsUsed?: string[]
  reasoning?: string
  diagram?: DiagramPayload
}

export interface DiagramForce {
  name: string
  magnitude_n: number
  angle_deg: number
  fx_n: number
  fy_n: number
  direction_label?: string
}

export interface FreeBodyDiagramPayload {
  type: 'free_body_diagram'
  title: string
  object_name: string
  forces: DiagramForce[]
  net_force?: {
    fx_n: number
    fy_n: number
    magnitude_n: number
    angle_deg: number
  }
}

export interface ForceVectorAdditionPayload {
  type: 'force_vector_addition'
  title: string
  vectors: DiagramForce[]
  resultant: DiagramForce
}

export interface ForceComponentsDiagramPayload {
  type: 'force_components_diagram'
  title: string
  vector: DiagramForce
}

export interface EquilibriumResidualDiagramPayload {
  type: 'equilibrium_residual_vector'
  title: string
  forces: DiagramForce[]
  net_force: DiagramForce
  balancing_force?: DiagramForce | null
  is_equilibrium: boolean
}

export interface InclinedPlaneDiagramPayload {
  type: 'inclined_plane_diagram'
  title: string
  mass_kg: number
  angle_deg: number
  coefficient_friction: number
  weight_n: number
  weight_parallel_n: number
  weight_perpendicular_n: number
  normal_n: number
  net_down_n: number
  has_friction: boolean
  friction_n?: number
}

export interface TensionSystemDiagramPayload {
  type: 'tension_system_diagram'
  title: string
  system_type:
    | 'single_mass_vertical'
    | 'single_mass_angled'
    | 'two_mass_atwood'
    | 'two_mass_balanced'
    | 'two_mass_angled'
    | 'multi_mass'
  gravity: number
  masses_kg: number[]
  angles_deg: number[]
  weights_n: number[]
  tension_n?: number
  vertical_component_n?: number
  horizontal_component_n?: number
  acceleration_mps2?: number
  direction?: 'mass_1_down' | 'mass_2_down' | 'balanced'
  total_weight_n?: number
  note?: string
}

export interface MotionGraphsPlotPayload {
  type: 'motion_graphs_plot'
  title: string
  motion_type: string
  times_s: number[]
  position_series: number[]
  velocity_series: number[]
  acceleration_series: number[]
}

export interface ProjectileTrajectoryPayload {
  type: 'projectile_trajectory'
  title: string
  launch: {
    v0_mps: number
    angle_deg: number
    h0_m: number
    x0_m: number
  }
  gravity_mps2: number
  v0x_mps: number
  v0y_mps: number
  max_height_m: number
  time_to_max_s: number
  flight_time_s?: number | null
  range_m?: number | null
  impact_speed_mps?: number | null
  trajectory_points: Array<{
    t_s: number
    x_m: number
    y_m: number
  }>
}

export interface ProjectileVelocityAnimationPayload {
  type: 'projectile_velocity_animation'
  title: string
  launch: {
    v0_mps: number
    angle_deg: number
    h0_m: number
    x0_m: number
  }
  gravity_mps2: number
  flight_time_s: number
  frames: Array<{
    t_s: number
    x_m: number
    y_m: number
    vx_mps: number
    vy_mps: number
    speed_mps: number
  }>
}

export interface FreeFallTimelinePayload {
  type: 'free_fall_timeline'
  title: string
  gravity_mps2: number
  h0_m: number
  v0_mps: number
  final_h_m: number
  final_v_mps: number
  total_time_s: number
  max_height_m?: number | null
  time_to_max_s?: number | null
  timeline_points: Array<{
    t_s: number
    h_m: number
    v_mps: number
  }>
}

export interface RelativeMotionIntersectionPayload {
  type: 'relative_motion_intersection'
  title: string
  object1: {
    x0_m: number
    v_mps: number
    points: Array<{ t_s: number; x_m: number }>
  }
  object2: {
    x0_m: number
    v_mps: number
    points: Array<{ t_s: number; x_m: number }>
  }
  relative_velocity_mps: number
  meeting_time_s?: number | null
  meeting_position_m?: number | null
}

export interface EnergyBarChartPayload {
  type: 'energy_bar_chart'
  title: string
  initial: {
    kinetic_j: number
    potential_j: number
    elastic_j: number
    total_j: number
  }
  final: {
    kinetic_j: number
    potential_j: number
    elastic_j: number
    total_j: number
  }
  difference_j?: number
  conserved?: boolean
}

export interface EnergyFlowDiagramPayload {
  type: 'energy_flow_diagram'
  title: string
  initial_mechanical_j: number
  final_mechanical_j: number
  friction_work_j: number
  dissipated_j: number
  efficiency_percent?: number
}

export interface WorkAreaUnderCurvePayload {
  type: 'work_area_under_curve'
  title: string
  force_n: number
  displacement_m: number
  angle_deg: number
  force_parallel_n: number
  force_perpendicular_n: number
  work_j: number
  area_points: Array<{
    x_m: number
    f_parallel_n: number
  }>
}

export interface BeforeAfterEnergySnapshotPayload {
  type: 'before_after_energy_snapshot'
  title: string
  initial?: {
    kinetic_j: number
    potential_j: number
    elastic_j: number
    total_j: number
    velocity_mps?: number
  }
  final?: {
    kinetic_j: number
    potential_j: number
    elastic_j: number
    total_j: number
    velocity_mps?: number
  }
  delta?: {
    work_net_j?: number
    delta_ke_j?: number
  }
  scenario?: string
  points?: Array<{
    point_index: number
    height_m: number
    velocity_mps: number
    potential_j: number
    kinetic_j: number
    total_j: number
  }>
}

export interface MomentumBeforeAfterVectorsPayload {
  type: 'momentum_before_after_vectors'
  title: string
  collision_type: string
  before: Array<{
    name: string
    mass_kg: number
    velocity_mps: number
    momentum_kg_mps: number
  }>
  after: Array<{
    name: string
    mass_kg: number
    velocity_mps: number | null
    momentum_kg_mps: number
  }>
  totals: {
    initial_kg_mps: number
    final_kg_mps: number
    difference_kg_mps: number
  }
}

export interface ImpulseAreaPlotPayload {
  type: 'impulse_area_plot'
  title: string
  force_n: number
  time_s: number
  impulse_ns: number
  initial_momentum?: number | null
  final_momentum?: number | null
  area_points: Array<{
    t_s: number
    force_n: number
  }>
}

export interface CollisionStoryboardPayload {
  type: 'collision_storyboard'
  title: string
  collision_type: string
  objects: Array<{
    name: string
    mass_kg: number
    speed_mps: number
    direction_deg: number
  }>
  stages: Array<{
    label: string
    time_s: number
  }>
  combined_after: {
    mass_kg: number
    speed_mps: number
    direction_deg: number
  }
  energy: {
    initial_ke_j: number
    final_ke_j: number
    dissipated_j: number
  }
}

export interface CenterOfMassTracePayload {
  type: 'center_of_mass_trace'
  title: string
  total_mass_kg: number
  vcom_x_mps: number
  vcom_y_mps: number
  initial_total_momentum: {
    px_kg_mps: number
    py_kg_mps: number
    magnitude_kg_mps: number
    angle_deg: number
  }
  trace_points: Array<{
    t_s: number
    x_m: number
    y_m: number
  }>
}

export interface TorqueLeverDiagramPayload {
  type: 'torque_lever_diagram'
  title: string
  force_n: number
  radius_m: number
  angle_deg: number
  force_perpendicular_n: number
  torque_nm: number
  direction: 'clockwise' | 'counterclockwise'
}

export interface RotationGraphsPlotPayload {
  type: 'rotation_graphs_plot'
  title: string
  times_s: number[]
  theta_series_rad: number[]
  omega_series_rad_s: number[]
  alpha_series_rad_s2: number[]
}

export interface CircularMotionVectorsPayload {
  type: 'circular_motion_vectors'
  title: string
  radius_m: number
  speed_mps: number
  omega_rad_s: number
  period_s: number
  frequency_hz: number
  centripetal_acc_mps2: number
  mass_kg?: number
  centripetal_force_n?: number
}

export interface RollingEnergySplitPayload {
  type: 'rolling_energy_split'
  title: string
  object_type: string
  mass_kg: number
  radius_m: number
  velocity_mps: number
  omega_rad_s: number
  translational_ke_j: number
  rotational_ke_j: number
  total_ke_j: number
  incline_angle_deg?: number
  acceleration_mps2?: number
  time_s?: number
}

export interface ShmSpringAnimationPayload {
  type: 'shm_spring_animation'
  title: string
  amplitude_m: number
  omega_rad_s: number
  period_s: number
  frequency_hz: number
  frames: Array<{
    t_s: number
    x_m: number
    v_mps: number
    a_mps2: number
  }>
}

export interface PendulumAnimationPayload {
  type: 'pendulum_animation'
  title: string
  length_m: number
  period_s: number
  frequency_hz: number
  theta_max_deg: number
  frames: Array<{
    t_s: number
    theta_deg: number
    x_m: number
    y_m: number
  }>
}

export interface TravelingWaveAnimationPayload {
  type: 'traveling_wave_animation'
  title: string
  velocity_mps: number
  frequency_hz: number
  wavelength_m: number
  amplitude_m: number
  frames: Array<{
    t_s: number
    samples: Array<{
      x_m: number
      y_m: number
    }>
  }>
}

export interface StandingWaveModeShapePayload {
  type: 'standing_wave_mode_shape'
  title: string
  system_type: string
  length_m: number
  velocity_mps: number
  fundamental_hz: number
  harmonics: Array<{
    n: number
    frequency_hz: number
    wavelength_m: number
  }>
  mode_shapes: Array<{
    n: number
    samples: Array<{
      x_m: number
      y_norm: number
    }>
  }>
}

export interface InterferenceFringeMapPayload {
  type: 'interference_fringe_map'
  title: string
  wavelength_m: number
  slit_separation_m: number
  screen_distance_m: number
  classification: 'constructive' | 'destructive' | 'partial' | string
  samples: Array<{
    y_m: number
    intensity_norm: number
  }>
}

export interface DopplerWavefrontAnimationPayload {
  type: 'doppler_wavefront_animation'
  title: string
  source_frequency_hz: number
  source_velocity_mps: number
  observer_velocity_mps: number
  medium_velocity_mps: number
  approaching: boolean
  observed_frequency_hz: number
  frequency_shift_hz: number
  frames: Array<{
    t_s: number
    source_x_m: number
    observer_x_m: number
    fronts: Array<{
      x_m: number
      radius_m: number
    }>
  }>
}

export interface PvDiagramPayload {
  type: 'pv_diagram'
  title: string
  state: {
    pressure_pa: number
    volume_m3: number
    moles: number
    temperature_k: number
  }
  pv_points: Array<{
    label: string
    pressure_pa: number
    volume_m3: number
  }>
  isotherms: Array<{
    temperature_k: number
    points: Array<{
      volume_m3: number
      pressure_pa: number
    }>
  }>
  axis?: {
    max_pressure_pa: number
    max_volume_m3: number
  }
}

export interface HeatingCurvePlotPayload {
  type: 'heating_curve_plot'
  title: string
  mass_kg: number
  specific_heat_j_per_kgk: number
  delta_t_k: number
  heat_j: number
  q_vs_delta_t: Array<{
    q_j: number
    delta_t_k: number
  }>
}

export interface HeatTransferPathDiagramPayload {
  type: 'heat_transfer_path_diagram'
  title: string
  conductivity_w_mk: number
  area_m2: number
  thickness_m: number
  delta_t_k: number
  heat_rate_w: number
  thermal_resistance_k_per_w: number
  hot_side_temp_c: number
  cold_side_temp_c: number
}

export interface CarnotCycleAnimationPayload {
  type: 'carnot_cycle_animation'
  title: string
  t_hot_k: number
  t_cold_k: number
  efficiency: number
  efficiency_percent: number
  q_hot_j?: number | null
  cycle_points: Array<{
    label: string
    v_norm: number
    p_norm: number
  }>
  frames: Array<{
    step: number
    v_norm: number
    p_norm: number
  }>
}

export type DiagramPayload =
  | FreeBodyDiagramPayload
  | ForceVectorAdditionPayload
  | ForceComponentsDiagramPayload
  | EquilibriumResidualDiagramPayload
  | InclinedPlaneDiagramPayload
  | TensionSystemDiagramPayload
  | MotionGraphsPlotPayload
  | ProjectileTrajectoryPayload
  | ProjectileVelocityAnimationPayload
  | FreeFallTimelinePayload
  | RelativeMotionIntersectionPayload
  | EnergyBarChartPayload
  | EnergyFlowDiagramPayload
  | WorkAreaUnderCurvePayload
  | BeforeAfterEnergySnapshotPayload
  | MomentumBeforeAfterVectorsPayload
  | ImpulseAreaPlotPayload
  | CollisionStoryboardPayload
  | CenterOfMassTracePayload
  | TorqueLeverDiagramPayload
  | RotationGraphsPlotPayload
  | CircularMotionVectorsPayload
  | RollingEnergySplitPayload
  | ShmSpringAnimationPayload
  | PendulumAnimationPayload
  | TravelingWaveAnimationPayload
  | StandingWaveModeShapePayload
  | InterferenceFringeMapPayload
  | DopplerWavefrontAnimationPayload
  | PvDiagramPayload
  | HeatingCurvePlotPayload
  | HeatTransferPathDiagramPayload
  | CarnotCycleAnimationPayload

export interface Agent {
  agent_id: string
  name: string
  description: string
  icon: string
}

export interface User {
  username: string
  name: string
  role: string
}

export interface PendingKnowledgeCheck {
  checkId: string
  agentId: string
  conceptTag: string
  questionId?: string | null
  question: string
  options: Array<{
    id: string
    text: string
  }>
  confidence: number
  threshold: number
  reasonTags?: string[]
  originalProblem: string
}

interface ChatState {
  // Auth
  isAuthenticated: boolean
  user: User | null

  // Agents
  agents: Agent[]
  selectedAgent: string | null

  // Messages (per agent)
  messagesByAgent: Record<string, Message[]>

  // UI State
  loading: boolean
  error: string | null
  sidebarOpen: boolean
  pendingKnowledgeCheck: PendingKnowledgeCheck | null

  // Actions
  login: (user: User) => void
  logout: () => void
  setAgents: (agents: Agent[]) => void
  selectAgent: (agentId: string) => void
  addMessage: (message: Message) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  toggleSidebar: () => void
  clearMessages: (agentId?: string) => void
  setPendingKnowledgeCheck: (check: PendingKnowledgeCheck | null) => void
}

export const useStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // Initial state
      isAuthenticated: false,
      user: null,
      agents: [],
      selectedAgent: null,
      messagesByAgent: {},
      loading: false,
      error: null,
      sidebarOpen: true,
      pendingKnowledgeCheck: null,

      // Actions
      login: (user) => set({ isAuthenticated: true, user }),

      logout: () =>
        set({
          isAuthenticated: false,
          user: null,
          selectedAgent: null,
          messagesByAgent: {},
          pendingKnowledgeCheck: null,
        }),

      setAgents: (agents) => set({ agents }),

      selectAgent: (agentId) => {
        set({ selectedAgent: agentId, error: null })
      },

      addMessage: (message) => {
        const { messagesByAgent } = get()
        const agentMessages = messagesByAgent[message.agentId] || []
        set({
          messagesByAgent: {
            ...messagesByAgent,
            [message.agentId]: [...agentMessages, message],
          },
        })
      },

      setLoading: (loading) => set({ loading }),

      setError: (error) => set({ error }),

      toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

      clearMessages: (agentId) => {
        if (agentId) {
          const { messagesByAgent } = get()
          const newMessages = { ...messagesByAgent }
          delete newMessages[agentId]
          set({ messagesByAgent: newMessages })
        } else {
          set({ messagesByAgent: {} })
        }
      },

      setPendingKnowledgeCheck: (check) => set({ pendingKnowledgeCheck: check }),
    }),
    {
      name: 'physics-chat-storage',
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        user: state.user,
        selectedAgent: state.selectedAgent,
        messagesByAgent: state.messagesByAgent,
        sidebarOpen: state.sidebarOpen,
      }),
    }
  )
)

// Selector hooks for common queries
export const useMessages = () => {
  const selectedAgent = useStore((state) => state.selectedAgent)
  const messagesByAgent = useStore((state) => state.messagesByAgent)
  return selectedAgent ? messagesByAgent[selectedAgent] || [] : []
}

export const useSelectedAgentInfo = () => {
  const selectedAgent = useStore((state) => state.selectedAgent)
  const agents = useStore((state) => state.agents)
  return agents.find((a) => a.agent_id === selectedAgent) || null
}
