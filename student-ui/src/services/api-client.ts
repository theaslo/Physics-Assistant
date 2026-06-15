import axios, { AxiosInstance } from 'axios'
import type { PhysicsGraphPayload } from '../types/graphs'

// Types
export interface AgentInfo {
  agent_id: string
  name: string
  description: string
}

export interface AgentListResponse {
  available_agents: AgentInfo[]
  active_agents: string[]
}

export interface SolveRequest {
  problem: string
  context?: Record<string, unknown>
  user_id?: string
  session_id?: string
}

export interface SolveResponse {
  success: boolean
  agent_id: string
  problem: string
  solution?: string
  reasoning?: string
  tools_used?: string[]
  execution_time_ms?: number
  metadata?: SolveMetadata
  error?: string
}

export interface SolveMetadata extends Record<string, unknown> {
  graphs?: PhysicsGraphPayload[]
  graph_warnings?: string[]
  graph_errors?: string[]
  graph_inputs?: Record<string, unknown>
}

export interface HealthResponse {
  status: string
  active_agents: number
  agent_keys: string[]
}

class PhysicsAPIClient {
  private api: AxiosInstance

  constructor() {
    this.api = axios.create({
      baseURL: '/api',
      timeout: 60000, // 60 second timeout for agent responses
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  // Health check
  async checkHealth(): Promise<HealthResponse> {
    const response = await this.api.get<HealthResponse>('/health')
    return response.data
  }

  // List available agents
  async listAgents(): Promise<AgentListResponse> {
    const response = await this.api.get<AgentListResponse>('/agents/list')
    return response.data
  }

  // Create/initialize an agent
  async createAgent(agentId: string): Promise<{ success: boolean; message: string }> {
    const response = await this.api.post('/agent/create', {
      agent_id: agentId,
      use_direct_tools: true,
      enable_rag: true,
    })
    return response.data
  }

  // Send message to agent
  async sendMessage(
    agentId: string,
    message: string,
    userId: string = 'react_user'
  ): Promise<SolveResponse> {
    const response = await this.api.post<SolveResponse>(`/agent/${agentId}/solve`, {
      problem: message,
      user_id: userId,
    })
    return response.data
  }

  // Get agent capabilities
  async getAgentCapabilities(agentId: string): Promise<Record<string, unknown>> {
    const response = await this.api.get(`/agent/${agentId}/capabilities`)
    return response.data
  }
}

// Export singleton instance
export const apiClient = new PhysicsAPIClient()
