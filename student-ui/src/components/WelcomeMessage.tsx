import { Box, Button, Paper, Typography } from '@mui/material'
import { Agent } from '../stores/chat-store'
import { getAgentColor } from '../themes/uconn-theme'

interface WelcomeMessageProps {
  agent: Agent
  onExampleClick?: (question: string) => void
}

// Example questions per agent
const EXAMPLE_QUESTIONS: Record<string, string[]> = {
  forces_agent: [
    'A 5.0 kg box slides down a 30° incline with coefficient of kinetic friction μk = 0.30. Draw the free-body diagram and find the acceleration down the ramp.',
    'A 12 kg sled is pulled across level snow by a 40 N rope at 25° above the horizontal. If μk = 0.10, find the normal force and the sled’s acceleration.',
    'A spring with k = 200 N/m is compressed by 0.050 m. What force does the spring exert on the block, and in which direction does it act?',
  ],
  kinematics_agent: [
    'A car starts from rest and accelerates uniformly at 3.0 m/s² for 5.0 s. Find its final velocity and displacement during this time.',
    'A ball is thrown from a 10 m high balcony at 30 m/s and 45° above the horizontal. Find the time to hit the ground, horizontal range, and impact speed.',
    'A stone is dropped from rest from a 50 m cliff. Ignoring air resistance, how long does it take to hit the ground and what is its impact velocity?',
  ],
  math_agent: [
    'Give me 5 physics algebra practice exercises on rearranging formulas, without answers.',
    'Give me 4 kinematics algebra exercises where I solve for a variable before substituting numbers.',
    'Resolve a 25 N vector directed 40° above the +x axis into its x and y components.',
    'Convert 72 km/h to m/s, then find how far an object travels in 8.0 s at that speed.',
  ],
  momentum_agent: [
    'A 5.0 kg object moves at 10 m/s east. Calculate its momentum, including magnitude and direction.',
    'A 2.0 kg cart moving at 4.0 m/s collides with and sticks to a 3.0 kg cart initially at rest. Find the final velocity.',
    'A 1200 kg car moving east at 18 m/s collides with a 1500 kg car moving west at 12 m/s, and they stick together. Find their final velocity.',
  ],
  energy_agent: [
    'A 5.0 kg object moving at 10 m/s speeds up to 14 m/s. How much net work was done on the object?',
    'A 0.50 kg ball is lifted 2.0 m above the floor and released from rest. Ignoring air resistance, what speed does it have just before hitting the floor?',
    'A 100 N horizontal force pushes a box 5.0 m across a floor while friction does -80 J of work. What is the net work on the box?',
  ],
  angular_motion_agent: [
    'A uniform 2.0 kg rod is 1.5 m long and rotates about its center. Calculate its moment of inertia.',
    'A solid cylinder rolls without slipping down a 30° incline from rest. Find its acceleration and compare it with a sliding block.',
    'A skater reduces their moment of inertia from 4.0 kg·m² to 2.0 kg·m² while spinning at 1.5 rad/s. Find the new angular speed.',
  ],
  thermodynamics_agent: [
    'A 2.0 kg sample of water is heated from 20°C to 80°C. Using c = 4186 J/(kg·°C), how much heat is added?',
    'An ideal gas has pressure 200 kPa, volume 0.010 m³, and temperature 300 K. How many moles of gas are present?',
    'A gas expands from 0.020 m³ to 0.060 m³ at a constant pressure of 150 kPa. How much work does the gas do?',
  ],
  waves_agent: [
    'A wave on a string has frequency 12 Hz and wavelength 0.80 m. Find the wave speed and period.',
    'A 440 Hz sound wave travels through air at 343 m/s. What is its wavelength?',
    'Two speakers emit the same tone in phase. At a point where the path difference is 0.50 m and λ = 1.0 m, is the interference constructive or destructive?',
  ],
  electromagnetism_agent: [
    'A 2.0 Ω resistor is connected to a 12 V battery. Find the current through the resistor and the power dissipated.',
    'A charge of +3.0 μC is placed 0.20 m from a charge of -5.0 μC. Find the magnitude and direction of the electric force on the +3.0 μC charge.',
    'A wire carries 4.0 A through a 0.50 T magnetic field. If 0.30 m of wire is perpendicular to the field, find the magnetic force.',
  ],
  optics_agent: [
    'An object is placed 30 cm in front of a converging lens with focal length 10 cm. Find the image distance and magnification.',
    'Light passes from air into glass with index of refraction 1.50 at an incident angle of 35°. Find the refracted angle.',
    'A concave mirror has focal length 15 cm. An object is placed 45 cm from the mirror. Find the image location and whether it is real or virtual.',
  ],
  modern_physics_agent: [
    'A photon has wavelength 500 nm. Calculate its frequency and energy in electron volts.',
    'An electron is accelerated through a potential difference of 150 V. Find its kinetic energy in joules and electron volts.',
    'A radioactive sample has a half-life of 6.0 hours. If it starts with 80 g, how much remains after 18 hours?',
  ],
}

const HELP_TOPICS: Record<string, string[]> = {
  forces_agent: ['Newton\'s laws of motion', 'Free body diagrams', 'Friction and tension problems'],
  kinematics_agent: ['Position, velocity, acceleration', 'Motion graphs and equations', 'Projectile motion'],
  math_agent: ['Physics algebra practice', 'Vector operations', 'Unit conversions and trigonometry'],
  momentum_agent: ['Linear momentum', 'Collision analysis', 'Impulse and momentum conservation'],
  energy_agent: ['Work and energy calculations', 'Conservation of energy', 'Power and efficiency'],
  angular_motion_agent: ['Rotational motion', 'Torque and angular momentum', 'Moment of inertia'],
  thermodynamics_agent: ['Heat transfer', 'Ideal gases', 'Work done by gases'],
  waves_agent: ['Wave speed and wavelength', 'Sound waves', 'Interference'],
  electromagnetism_agent: ['Circuits', 'Electric forces', 'Magnetic forces'],
  optics_agent: ['Lenses and mirrors', 'Refraction', 'Image formation'],
  modern_physics_agent: ['Photons', 'Electron energy', 'Radioactive decay'],
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
            <Button
              key={i}
              variant="outlined"
              size="small"
              onClick={() => onExampleClick?.(example)}
              disabled={!onExampleClick}
              sx={{
                height: 'auto',
                justifyContent: 'flex-start',
                py: 1,
                px: 1.25,
                borderRadius: 2,
                cursor: onExampleClick ? 'pointer' : 'default',
                lineHeight: 1.35,
                textAlign: 'left',
                textTransform: 'none',
                '&:hover': onExampleClick ? {
                  bgcolor: `${agentColor}20`,
                  borderColor: agentColor,
                } : {},
              }}
            >
              {example}
            </Button>
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
