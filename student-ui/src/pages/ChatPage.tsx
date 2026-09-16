import { useEffect, useCallback, useState } from 'react'
import {
  Box,
  Button,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useMediaQuery,
  useTheme,
  CircularProgress,
  Alert,
  Paper,
  Stack,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Science as ScienceIcon,
} from '@mui/icons-material'
import { useStore, useMessages, useSelectedAgentInfo, type StudentDrawing } from '../stores/chat-store'
import { apiClient, type SolveResponse } from '../services/api-client'
import AgentSelector from '../components/AgentSelector'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import WelcomeMessage from '../components/WelcomeMessage'
import PhysicsConstants from '../components/PhysicsConstants'
import KnowledgeCheckPanel from '../components/KnowledgeCheckPanel'
import { getAgentIcon } from '../themes/uconn-theme'

const DRAWER_WIDTH = 280
const NEXT_STEP_PATTERN = /\b(next|step|hint|continue)\b/i
const FULL_SOLUTION_PATTERN = /\b(full solution|show(?: me)?(?: the)? solution|give(?: me)?(?: the)? solution|i'?m lost|i am lost|completely lost|really lost|stuck)\b/i

function isNextStepRequest(message: string): boolean {
  return NEXT_STEP_PATTERN.test(message.trim())
}

function isFullSolutionRequest(message: string): boolean {
  return FULL_SOLUTION_PATTERN.test(message.trim())
}

function serializeDrawingForAnalysis(drawing?: StudentDrawing) {
  if (!drawing) return undefined

  return {
    title: drawing.title,
    width: drawing.width,
    height: drawing.height,
    created_at: drawing.createdAt,
    strokes: drawing.strokes || [],
  }
}

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
    pendingHitlRemediation,
    setPendingHitlRemediation,
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
          { agent_id: 'math_agent', name: 'Math Agent', description: 'Physics algebra practice, calculations, and units', icon: '🔢' },
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
    async (message: string, drawing?: StudentDrawing) => {
      const trimmedMessage = message.trim()
      const hasDrawing = Boolean(drawing)
      if (!trimmedMessage && !hasDrawing) return
      if (pendingKnowledgeCheck) {
        setError('Complete the knowledge check before sending a new prompt.')
        return
      }
      const remediationFollowUp = pendingHitlRemediation ? pendingHitlRemediation : null
      const agentForMessage = remediationFollowUp?.agentId || selectedAgent
      const drawingAnalysisPayload = serializeDrawingForAnalysis(drawing)
      if (!agentForMessage) {
        setError('Select a physics agent before sending a message.')
        return
      }

      // Add user message
      const userMessage = {
        id: `user-${Date.now()}`,
        role: 'user' as const,
        content: trimmedMessage || 'Attached a sketch of my work.',
        timestamp: Date.now(),
        agentId: agentForMessage,
        drawing,
      }
      addMessage(userMessage)

      if (!trimmedMessage && hasDrawing && !remediationFollowUp) {
        const assistantMessage = {
          id: `assistant-${Date.now() + 1}`,
          role: 'assistant' as const,
          content: (
            'I received your sketch. To analyze it accurately, please type the problem statement plus the key labels, axes, forces, or equations shown in the sketch.'
          ),
          timestamp: Date.now(),
          agentId: agentForMessage,
        }
        addMessage(assistantMessage)
        return
      }

      setLoading(true)
      setError(null)

      try {
        if (remediationFollowUp) {
          const advancesStep = isNextStepRequest(trimmedMessage)
          const wantsFullSolution = isFullSolutionRequest(trimmedMessage)
          const nextStepIndex = remediationFollowUp.nextStepIndex
          const response = await apiClient.sendMessage(
            agentForMessage,
            remediationFollowUp.originalProblem,
            user?.username || 'react_user',
            {
              knowledge_transfer_remediation_followup: {
                check_id: remediationFollowUp.checkId,
                concept_tag: remediationFollowUp.conceptTag,
                original_problem: remediationFollowUp.originalProblem,
                student_message: trimmedMessage || 'Attached a sketch of my work.',
                has_drawing: hasDrawing,
                drawing: drawingAnalysisPayload,
                step_index: nextStepIndex,
                mode: wantsFullSolution ? 'full_solution' : advancesStep ? 'next_step' : 'clarify',
              },
            }
          )

          if (response.success) {
            appendAssistantMessage(agentForMessage, response)
            if (
              response.hitl?.status === 'remediation_complete' ||
              response.hitl?.status === 'full_solution_requested'
            ) {
              setPendingHitlRemediation(null)
            } else {
              const returnedStepIndex =
                response.hitl?.status === 'remediation_followup' && typeof response.hitl.step_index === 'number'
                  ? response.hitl.step_index
                  : nextStepIndex
              setPendingHitlRemediation({
                ...remediationFollowUp,
                nextStepIndex: returnedStepIndex,
              })
            }
          } else {
            setError(response.error || 'Failed to process the next-step request')
          }
          return
        }

        setPendingHitlRemediation(null)

        // Ensure agent is created
        await apiClient.createAgent(agentForMessage)

        // Send message
        const response = await apiClient.sendMessage(
          agentForMessage,
          trimmedMessage,
          user?.username || 'react_user',
          drawingAnalysisPayload
            ? {
                student_drawing: drawingAnalysisPayload,
              }
            : undefined
        )

        if (response.hitl?.status === 'question_required') {
          setPendingHitlRemediation(null)
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
            originalProblem: trimmedMessage,
          })
          return
        }

        if (response.success) {
          appendAssistantMessage(agentForMessage, response)
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
      pendingHitlRemediation,
      addMessage,
      appendAssistantMessage,
      setLoading,
      setError,
      setPendingKnowledgeCheck,
      setPendingHitlRemediation,
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
          setPendingHitlRemediation(null)
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
          if (response.hitl?.status === 'remediation_required') {
            setPendingHitlRemediation({
              checkId: activeCheck.checkId,
              agentId,
              conceptTag: activeCheck.conceptTag,
              originalProblem: activeCheck.originalProblem,
              nextStepIndex: 0,
            })
          } else {
            setPendingHitlRemediation(null)
          }
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
      setPendingHitlRemediation,
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
        {selectedAgent && pendingHitlRemediation && (
          <Paper
            elevation={0}
            sx={{
              mx: 2,
              mb: 1,
              p: 1.5,
              border: '1px solid',
              borderColor: 'warning.light',
              borderRadius: 2,
              bgcolor: 'rgba(237, 108, 2, 0.08)',
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 800 }}>
              Step-by-step check is active
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
              Send your work to have it checked, ask for one more hint, or choose the full solution if you are stuck.
            </Typography>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} sx={{ mt: 1 }}>
              <Button
                size="small"
                variant="outlined"
                disabled={loading || Boolean(pendingKnowledgeCheck)}
                onClick={() => handleSendMessage('next step')}
              >
                Give me the next step
              </Button>
              <Button
                size="small"
                color="warning"
                variant="contained"
                disabled={loading || Boolean(pendingKnowledgeCheck)}
                onClick={() => handleSendMessage("I'm lost. Please show the full solution.")}
              >
                I'm lost, show full solution
              </Button>
            </Stack>
          </Paper>
        )}
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
