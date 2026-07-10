import { useEffect, useCallback } from 'react'
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useMediaQuery,
  useTheme,
  CircularProgress,
  Alert,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Science as ScienceIcon,
} from '@mui/icons-material'
import { useStore, useMessages, useSelectedAgentInfo } from '../stores/chat-store'
import { apiClient } from '../services/api-client'
import AgentSelector from '../components/AgentSelector'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import WelcomeMessage from '../components/WelcomeMessage'
import PhysicsConstants from '../components/PhysicsConstants'
import { getAgentIcon } from '../themes/uconn-theme'

const DRAWER_WIDTH = 280
const FULL_SOLUTION_REQUEST_RE = /\b(show\s+me\s+the\s+full\s+solution|full\s+solution|worked\s+solution|worked\s+example)\b/i
const PHYSICS_PROBLEM_CUES_RE =
  /\b(find|calculate|determine|compute|solve|draw|graph|how\s+(far|long|fast|high|much)|what\s+is)\b/i
const PHYSICS_CONTEXT_RE =
  /\b(m\/s|m\/s\^?2|newton|force|velocity|acceleration|distance|displacement|graph|time|seconds?|runner|car|ball|block|charge|circuit|field|energy|momentum|torque|wave|lens|projectile)\b/i

function looksLikePhysicsProblem(text: string) {
  return PHYSICS_PROBLEM_CUES_RE.test(text) && PHYSICS_CONTEXT_RE.test(text)
}

function inferActiveProblem(
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  currentMessage: string
) {
  const candidates = [
    ...messages,
    { role: 'user' as const, content: currentMessage },
  ]

  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    const candidate = candidates[index]
    if (candidate.role !== 'user') continue

    const content = candidate.content.trim()
    if (!content || FULL_SOLUTION_REQUEST_RE.test(content)) continue
    if (looksLikePhysicsProblem(content)) return content
  }

  return undefined
}

export default function ChatPage() {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))

  const {
    user,
    selectedAgent,
    sidebarOpen,
    loading,
    error,
    setAgents,
    addMessage,
    setLoading,
    setError,
    toggleSidebar,
    logout,
  } = useStore()

  const messages = useMessages()
  const selectedAgentInfo = useSelectedAgentInfo()

  // Fetch agents on mount
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await apiClient.listAgents()
        const agentList = response.available_agents.map((a) => ({
          agent_id: a.agent_id,
          name: a.name,
          description: a.description,
          icon: getAgentIcon(a.agent_id),
        }))
        setAgents(agentList)
      } catch (err) {
        console.error('Failed to fetch agents:', err)
        // Fallback to hardcoded agents
        setAgents([
          { agent_id: 'forces_agent', name: 'Forces Agent', description: 'Force analysis and Newton\'s laws', icon: '⚖️' },
          { agent_id: 'kinematics_agent', name: 'Kinematics Agent', description: 'Motion analysis and projectile motion', icon: '🚀' },
          { agent_id: 'math_agent', name: 'Math Agent', description: 'Mathematical calculations and algebra', icon: '🔢' },
          { agent_id: 'momentum_agent', name: 'Momentum Agent', description: 'Momentum and collision analysis', icon: '💥' },
          { agent_id: 'energy_agent', name: 'Energy Agent', description: 'Work, energy, and conservation', icon: '⚡' },
          { agent_id: 'angular_motion_agent', name: 'Angular Motion Agent', description: 'Rotational motion and torque', icon: '🌀' },
        ])
      }
    }
    fetchAgents()
  }, [setAgents])

  // Handle sending messages
  const handleSendMessage = useCallback(
    async (message: string) => {
      if (!selectedAgent || !message.trim()) return

      // Add user message
      const userMessage = {
        id: `user-${Date.now()}`,
        role: 'user' as const,
        content: message,
        timestamp: Date.now(),
        agentId: selectedAgent,
      }
      addMessage(userMessage)
      setLoading(true)
      setError(null)

      try {
        const conversationContext = {
          active_problem: inferActiveProblem(messages, message),
          recent_conversation: [...messages, userMessage].slice(-20).map((msg) => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
          })),
          total_messages: messages.length + 1,
        }

        // Send message
        const response = await apiClient.sendMessage(
          selectedAgent,
          message,
          user?.username || 'react_user',
          conversationContext
        )

        if (response.success && response.solution) {
          const assistantMessage = {
            id: `assistant-${Date.now()}`,
            role: 'assistant' as const,
            content: response.solution,
            timestamp: Date.now(),
            agentId: selectedAgent,
            toolsUsed: response.tools_used,
            reasoning: response.reasoning,
            graphs: response.metadata?.graphs,
            graphWarnings: response.metadata?.graph_warnings,
            graphErrors: response.metadata?.graph_errors,
          }
          addMessage(assistantMessage)
        } else {
          setError(response.error || 'Failed to get response from agent')
        }
      } catch (err) {
        console.error('Send message error:', err)
        setError('Failed to connect to the physics assistant. Please try again.')
      } finally {
        setLoading(false)
      }
    },
    [selectedAgent, user, messages, addMessage, setLoading, setError]
  )

  // Sidebar content
  const sidebarContent = (
    <Box sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* User Info */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Welcome,
        </Typography>
        <Typography variant="h6">{user?.name || 'Student'}</Typography>
      </Box>

      {/* Agent Selector */}
      <AgentSelector />

      {/* Physics Constants */}
      <PhysicsConstants />

      {/* Spacer */}
      <Box sx={{ flexGrow: 1 }} />

      {/* Logout */}
      <Box
        component="button"
        onClick={logout}
        sx={{
          width: '100%',
          p: 1.5,
          border: 'none',
          borderRadius: 2,
          bgcolor: 'error.light',
          color: 'error.contrastText',
          cursor: 'pointer',
          '&:hover': { bgcolor: 'error.main' },
        }}
      >
        Logout
      </Box>
    </Box>
  )

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          zIndex: theme.zIndex.drawer + 1,
          width: { md: sidebarOpen ? `calc(100% - ${DRAWER_WIDTH}px)` : '100%' },
          ml: { md: sidebarOpen ? `${DRAWER_WIDTH}px` : 0 },
          transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            edge="start"
            onClick={toggleSidebar}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <ScienceIcon sx={{ mr: 1 }} />
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Physics Assistant
          </Typography>
          {selectedAgentInfo && (
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              {selectedAgentInfo.icon} {selectedAgentInfo.name}
            </Typography>
          )}
        </Toolbar>
      </AppBar>

      {/* Sidebar Drawer */}
      <Drawer
        variant={isMobile ? 'temporary' : 'persistent'}
        open={sidebarOpen}
        onClose={toggleSidebar}
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
          },
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}
        {sidebarContent}
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          overflow: 'hidden',
          ml: { md: sidebarOpen ? 0 : `-${DRAWER_WIDTH}px` },
          transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar /> {/* Spacer for AppBar */}

        {/* Chat Messages Area */}
        <Box
          sx={{
            flexGrow: 1,
            overflow: 'auto',
            p: 2,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {!selectedAgent ? (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                textAlign: 'center',
              }}
            >
              <ScienceIcon sx={{ fontSize: 80, color: 'primary.light', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                Welcome to Physics Assistant
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Select a physics agent from the sidebar to start chatting
              </Typography>
            </Box>
          ) : (
            <>
              {/* Welcome message if no messages yet */}
              {messages.length === 0 && selectedAgentInfo && (
                <WelcomeMessage agent={selectedAgentInfo} onExampleClick={handleSendMessage} />
              )}

              {/* Chat messages */}
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}

              {/* Loading indicator */}
              {loading && (
                <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                  <CircularProgress size={24} />
                  <Typography variant="body2" sx={{ ml: 1 }}>
                    Thinking...
                  </Typography>
                </Box>
              )}

              {/* Error message */}
              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
            </>
          )}
        </Box>

        {/* Chat Input */}
        {selectedAgent && (
          <ChatInput onSend={handleSendMessage} disabled={loading} />
        )}
      </Box>
    </Box>
  )
}
