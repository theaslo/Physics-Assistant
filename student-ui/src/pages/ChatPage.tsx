import { useEffect, useCallback, useState } from 'react'
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
import { apiClient, type SolveResponse } from '../services/api-client'
import AgentSelector from '../components/AgentSelector'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import WelcomeMessage from '../components/WelcomeMessage'
import PhysicsConstants from '../components/PhysicsConstants'
import KnowledgeCheckPanel from '../components/KnowledgeCheckPanel'
import { getAgentIcon } from '../themes/uconn-theme'

const DRAWER_WIDTH = 280

export default function ChatPage() {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('md'))
  const [isKnowledgeCheckProcessing, setIsKnowledgeCheckProcessing] = useState(false)

  const {
    user,
    selectedAgent,
    sidebarOpen,
    loading,
    error,
    setAgents,
    selectAgent,
    addMessage,
    setLoading,
    setError,
    toggleSidebar,
    logout,
    pendingKnowledgeCheck,
    setPendingKnowledgeCheck,
  } = useStore()

  const messages = useMessages()
  const selectedAgentInfo = useSelectedAgentInfo()

  const appendAssistantMessage = useCallback(
    (agentId: string, response: SolveResponse) => {
      const solutionText = (response.solution || '').trim()
      if (!solutionText) {
        return
      }
      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant' as const,
        content: solutionText,
        timestamp: Date.now(),
        agentId,
        toolsUsed: response.tools_used,
        reasoning: response.reasoning,
        diagram: response.diagram,
      }
      addMessage(assistantMessage)
    },
    [addMessage]
  )

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
        if (!selectedAgent && agentList.length > 0) {
          selectAgent(agentList[0].agent_id)
        }
      } catch (err) {
        console.error('Failed to fetch agents:', err)
        // Fallback to hardcoded agents
        const fallbackAgents = [
          { agent_id: 'forces_agent', name: 'Forces Agent', description: 'Force analysis and Newton\'s laws', icon: '⚖️' },
          { agent_id: 'kinematics_agent', name: 'Kinematics Agent', description: 'Motion analysis and projectile motion', icon: '🚀' },
          { agent_id: 'math_agent', name: 'Math Agent', description: 'Mathematical calculations and algebra', icon: '🔢' },
          { agent_id: 'momentum_agent', name: 'Momentum Agent', description: 'Momentum and collision analysis', icon: '💥' },
          { agent_id: 'energy_agent', name: 'Energy Agent', description: 'Work, energy, and conservation', icon: '⚡' },
          { agent_id: 'angular_motion_agent', name: 'Angular Motion Agent', description: 'Rotational motion and torque', icon: '🌀' },
          { agent_id: 'thermodynamics_agent', name: 'Thermodynamics Agent', description: 'Heat, temperature, and entropy', icon: '🌡️' },
          { agent_id: 'waves_agent', name: 'Waves Agent', description: 'Wave motion, sound, and oscillations', icon: '🌊' },
          { agent_id: 'electromagnetism_agent', name: 'Electromagnetism Agent', description: 'Electricity, magnetism, and circuits', icon: '⚡' },
          { agent_id: 'optics_agent', name: 'Optics Agent', description: 'Light, lenses, and mirrors', icon: '🔍' },
          { agent_id: 'modern_physics_agent', name: 'Modern Physics Agent', description: 'Relativity and quantum concepts', icon: '🧪' },
        ]
        setAgents(fallbackAgents)
        if (!selectedAgent && fallbackAgents.length > 0) {
          selectAgent(fallbackAgents[0].agent_id)
        }
      }
    }
    fetchAgents()
  }, [setAgents, selectAgent, selectedAgent])

  // Handle sending messages
  const handleSendMessage = useCallback(
    async (message: string) => {
      if (!message.trim()) return
      if (pendingKnowledgeCheck) {
        setError('Complete the knowledge check before sending a new prompt.')
        return
      }
      if (!selectedAgent) {
        setError('Select a physics agent before sending a message.')
        return
      }

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
        // Ensure agent is created
        await apiClient.createAgent(selectedAgent)

        // Send message
        const response = await apiClient.sendMessage(
          selectedAgent,
          message,
          user?.username || 'react_user'
        )

        if (response.hitl?.status === 'question_required') {
          setPendingKnowledgeCheck({
            checkId: response.hitl.check_id,
            agentId: response.hitl.agent_id,
            conceptTag: response.hitl.concept_tag,
            questionId: response.hitl.question_id,
            question: response.hitl.question,
            options: response.hitl.options,
            confidence: response.hitl.confidence,
            threshold: response.hitl.threshold,
            reasonTags: response.hitl.reason_tags,
            originalProblem: message,
          })
          return
        }

        if (response.success) {
          appendAssistantMessage(selectedAgent, response)
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
    [
      selectedAgent,
      user,
      pendingKnowledgeCheck,
      addMessage,
      appendAssistantMessage,
      setLoading,
      setError,
      setPendingKnowledgeCheck,
    ]
  )

  const handleKnowledgeCheckSelection = useCallback(
    async (selectedOptionId: string) => {
      if (!pendingKnowledgeCheck) return

      const activeCheck = pendingKnowledgeCheck
      const agentId = activeCheck.agentId || selectedAgent
      if (!agentId) {
        setError('Select a physics agent before submitting knowledge-check answer.')
        return
      }

      // Close the panel immediately after a one-shot selection.
      // Backend validation and follow-up solve happen in the background.
      setPendingKnowledgeCheck(null)
      setIsKnowledgeCheckProcessing(true)
      setLoading(true)
      setError(null)

      try {
        const response = await apiClient.sendMessage(
          agentId,
          activeCheck.originalProblem,
          user?.username || 'react_user',
          {
            knowledge_transfer_response: {
              check_id: activeCheck.checkId,
              selected_option_id: selectedOptionId,
            },
          }
        )

        if (response.hitl?.status === 'question_required') {
          setPendingKnowledgeCheck({
            checkId: response.hitl.check_id,
            agentId: response.hitl.agent_id,
            conceptTag: response.hitl.concept_tag,
            questionId: response.hitl.question_id,
            question: response.hitl.question,
            options: response.hitl.options,
            confidence: response.hitl.confidence,
            threshold: response.hitl.threshold,
            reasonTags: response.hitl.reason_tags,
            originalProblem: activeCheck.originalProblem,
          })
          return
        }
        if (response.success) {
          appendAssistantMessage(agentId, response)
        } else {
          setError(response.error || 'Failed to process knowledge-check answer')
        }
      } catch (err) {
        console.error('Knowledge-check submission error:', err)
        setError('Failed to submit knowledge-check answer. Please try again.')
      } finally {
        setIsKnowledgeCheckProcessing(false)
        setLoading(false)
      }
    },
    [
      pendingKnowledgeCheck,
      selectedAgent,
      user,
      appendAssistantMessage,
      setLoading,
      setError,
      setPendingKnowledgeCheck,
    ]
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

              {isKnowledgeCheckProcessing && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  Checking answer and solving...
                </Alert>
              )}
            </>
          )}
        </Box>

        {/* Chat Input */}
        {selectedAgent && (
          <ChatInput onSend={handleSendMessage} disabled={loading || Boolean(pendingKnowledgeCheck)} />
        )}
      </Box>

      <Drawer
        anchor={isMobile ? 'bottom' : 'right'}
        open={Boolean(pendingKnowledgeCheck)}
        onClose={() => {}}
        ModalProps={{ keepMounted: true }}
        PaperProps={{
          sx: isMobile
            ? {
                width: '100%',
                height: '100vh',
                maxHeight: '100vh',
              }
            : {
                width: 420,
                maxWidth: '100%',
              },
        }}
      >
        {pendingKnowledgeCheck && (
          <KnowledgeCheckPanel
            check={pendingKnowledgeCheck}
            onSelectOption={handleKnowledgeCheckSelection}
            disabled={loading}
          />
        )}
      </Drawer>
    </Box>
  )
}
