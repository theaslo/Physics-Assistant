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

export interface KnowledgeTransferChoice {
  id: string
  text: string
}

export interface KnowledgeTransferQuestion {
  id: string
  question_key: string
  agent_type: string
  topic: string
  leg: number
  question_type: string
  question_text: string
  choices: KnowledgeTransferChoice[]
  metadata: Record<string, unknown>
}

export interface KnowledgeTransferPickResponse {
  status: string
  question: KnowledgeTransferQuestion
}

export interface KnowledgeTransferAttemptRequest {
  user_id: string
  question_id: string
  session_id?: string
  agent_type?: string
  selected_choice_id?: string
  answer_text?: string
  metadata?: Record<string, unknown>
}

export interface KnowledgeTransferAttemptResponse {
  status: string
  attempt_id: string
  question_id: string
  agent_type: string
  is_correct: boolean
  proceed: boolean
  can_retry: boolean
  attempt_count: number
  max_attempts: number
  max_attempts_reached: boolean
  feedback: string
  next_question: KnowledgeTransferQuestion | null
}

export interface KnowledgeTransferAggregateResponse {
  total_attempts: number
  correct_attempts: number
  incorrect_attempts: number
  accuracy: number
  by_agent: Array<{
    agent_type: string
    total_attempts: number
    correct_attempts: number
    incorrect_attempts: number
    accuracy: number
  }>
}

class PhysicsAPIClient {
  private api: AxiosInstance
  private databaseApi: AxiosInstance

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

    this.databaseApi = axios.create({
      baseURL: '/api/database',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.databaseApi.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('Database API Error:', error.response?.data || error.message)
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

  async pickKnowledgeTransferQuestion(
    agentType: string,
    leg: number = 1,
    topic?: string
  ): Promise<KnowledgeTransferPickResponse> {
    const response = await this.databaseApi.get<KnowledgeTransferPickResponse>(
      '/knowledge-transfer/questions/pick',
      {
        params: {
          agent_type: agentType,
          leg,
          ...(topic ? { topic } : {}),
        },
      }
    )
    return response.data
  }

  async submitKnowledgeTransferAttempt(
    attempt: KnowledgeTransferAttemptRequest
  ): Promise<KnowledgeTransferAttemptResponse> {
    const response = await this.databaseApi.post<KnowledgeTransferAttemptResponse>(
      '/knowledge-transfer/attempts',
      attempt
    )
    return response.data
  }

  async getKnowledgeTransferAggregate(
    userId?: string,
    agentType?: string
  ): Promise<KnowledgeTransferAggregateResponse> {
    const response = await this.databaseApi.get<KnowledgeTransferAggregateResponse>(
      '/knowledge-transfer/attempts/aggregate',
      {
        params: {
          ...(userId ? { user_id: userId } : {}),
          ...(agentType ? { agent_type: agentType } : {}),
        },
      }
    )
    return response.data
  }
}

// Export singleton instance
export const apiClient = new PhysicsAPIClient()
