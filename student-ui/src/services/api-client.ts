import axios, { AxiosInstance } from 'axios'
import type { DiagramPayload } from '../stores/chat-store'

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

export interface KnowledgeTransferQuestion {
  status: 'question_required'
  check_id: string
  agent_id: string
  concept_tag: string
  question_id?: string | null
  question: string
  options: Array<{
    id: string
    text: string
  }>
  confidence: number
  threshold: number
  reason_tags?: string[]
}

export interface KnowledgeTransferResult {
  status: 'answer_processed' | 'remediation_required'
  check_id: string
  agent_id?: string
  concept_tag: string
  was_correct: boolean
  confidence: number
  threshold: number
  guidance?: string
  remediation?: {
    prompt: string
    next_step_prompt: string
    can_continue?: boolean
  }
}

export interface SolveResponse {
  success: boolean
  agent_id: string
  problem: string
  solution?: string
  reasoning?: string
  tools_used?: string[]
  execution_time_ms?: number
  diagram?: DiagramPayload
  hitl?: KnowledgeTransferQuestion | KnowledgeTransferResult
  metadata?: Record<string, unknown>
  error?: string
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
      timeout: 300000, // 5 minute timeout for agent responses
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
    const response = await this.api.get<AgentListResponse>('/agents/list', {
      timeout: 8000,
    })
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
    userId: string = 'react_user',
    context?: Record<string, unknown>
  ): Promise<SolveResponse> {
    const response = await this.api.post<SolveResponse>(`/agent/${agentId}/solve`, {
      problem: message,
      user_id: userId,
      context,
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
