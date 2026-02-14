import { Box, Paper, Typography, Chip } from '@mui/material'
import { Agent } from '../stores/chat-store'
import { getAgentColor } from '../themes/uconn-theme'

interface WelcomeMessageProps {
  agent: Agent
  onExampleClick?: (question: string) => void
}

// Example questions per agent
const EXAMPLE_QUESTIONS: Record<string, string[]> = {
  forces_agent: [
    'A 5kg box on a 30° incline with friction coefficient 0.3',
    'Add forces: 10N at 30°, 15N at 120°, 8N at 270°',
    'Calculate spring force with k=200 N/m, compressed 0.05m',
  ],
  kinematics_agent: [
    'Car accelerates from rest at 3 m/s² for 5 seconds',
    'Ball thrown at 30 m/s at 45° from 10m height',
    'Object dropped from 50m - how long to fall?',
  ],
  math_agent: [
    'Solve x² + 5x + 6 = 0 using the quadratic formula',
    'What is sin(45°) and cos(45°)?',
    'Calculate log₁₀(100) and ln(e²)',
  ],
  momentum_agent: [
    'Calculate momentum of 5kg object moving at 10 m/s',
    '2kg ball at 8 m/s collides with 3kg ball at rest',
    'Car crash: 1500kg at 20 m/s hits 1200kg at 15 m/s',
  ],
  energy_agent: [
    'Calculate kinetic energy of 5kg object at 10 m/s',
    'Ball lifted 10m high - what is the potential energy?',
    'Work done pushing 100N force over 5m distance',
  ],
  angular_motion_agent: [
    'Calculate moment of inertia of 2kg rod, 1.5m long',
    'Cylinder rolls down 30° incline, mass=5kg, radius=0.3m',
    'Figure skater spins faster when pulling arms in - why?',
  ],
}

const HELP_TOPICS: Record<string, string[]> = {
  forces_agent: ['Newton\'s laws of motion', 'Free body diagrams', 'Friction and tension problems'],
  kinematics_agent: ['Position, velocity, acceleration', 'Motion graphs and equations', 'Projectile motion'],
  math_agent: ['Vector operations', 'Trigonometry', 'Unit conversions and algebra'],
  momentum_agent: ['Linear momentum', 'Collision analysis', 'Impulse and momentum conservation'],
  energy_agent: ['Work and energy calculations', 'Conservation of energy', 'Power and efficiency'],
  angular_motion_agent: ['Rotational motion', 'Torque and angular momentum', 'Moment of inertia'],
}

export default function WelcomeMessage({ agent, onExampleClick }: WelcomeMessageProps) {
  const examples = EXAMPLE_QUESTIONS[agent.agent_id] || []
  const topics = HELP_TOPICS[agent.agent_id] || []
  const agentColor = getAgentColor(agent.agent_id)

  return (
    <Box sx={{ mb: 3 }}>
      <Paper
        elevation={1}
        sx={{
          p: 3,
          borderRadius: 2,
          borderLeft: `4px solid ${agentColor}`,
          bgcolor: `${agentColor}08`,
        }}
      >
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: '50%',
              bgcolor: agentColor,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '1.5rem',
            }}
          >
            {agent.icon}
          </Box>
          <Box>
            <Typography variant="h6">{agent.name}</Typography>
            <Typography variant="body2" color="text.secondary">
              {agent.description}
            </Typography>
          </Box>
        </Box>

        {/* Help Topics */}
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          I can help you with:
        </Typography>
        <Box component="ul" sx={{ mt: 1, mb: 2, pl: 2 }}>
          {topics.map((topic, i) => (
            <Typography component="li" variant="body2" key={i} sx={{ mb: 0.5 }}>
              {topic}
            </Typography>
          ))}
        </Box>

        {/* Example Questions */}
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Example questions:
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1 }}>
          {examples.map((example, i) => (
            <Chip
              key={i}
              label={example}
              variant="outlined"
              size="small"
              clickable={!!onExampleClick}
              onClick={() => onExampleClick?.(example)}
              sx={{
                height: 'auto',
                py: 0.5,
                cursor: onExampleClick ? 'pointer' : 'default',
                '& .MuiChip-label': {
                  whiteSpace: 'normal',
                  display: 'block',
                },
                '&:hover': onExampleClick ? {
                  bgcolor: `${agentColor}20`,
                  borderColor: agentColor,
                } : {},
              }}
            />
          ))}
        </Box>

        {/* Getting Started */}
        <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="body2" color="text.secondary">
            <strong>How to get started:</strong> Ask me a specific physics question,
            describe a problem you need help with, or click one of the examples above.
          </Typography>
        </Box>
      </Paper>
    </Box>
  )
}
