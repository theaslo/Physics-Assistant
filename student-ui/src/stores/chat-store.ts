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
}

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

      // Actions
      login: (user) => set({ isAuthenticated: true, user }),

      logout: () =>
        set({
          isAuthenticated: false,
          user: null,
          selectedAgent: null,
          messagesByAgent: {},
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
