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
import type { Message } from '../stores/chat-store'
import { apiClient } from '../services/api-client'
import AgentSelector from '../components/AgentSelector'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'
import WelcomeMessage from '../components/WelcomeMessage'
import PhysicsConstants from '../components/PhysicsConstants'
import HumanInTheLoopDialog, {
  HitlQuestion,
  HitlResult,
} from '../components/HumanInTheLoopDialog'
import { getAgentIcon } from '../themes/uconn-theme'

const DRAWER_WIDTH = 280
const FULL_SOLUTION_REQUEST_RE = /\b(show\s+me\s+the\s+full\s+solution|full\s+solution|worked\s+solution|worked\s+example)\b/i
const PHYSICS_PROBLEM_CUES_RE =
  /\b(find|calculate|determine|compute|solve|draw|graph|how\s+(far|long|fast|high|much)|what\s+is)\b/i
const PHYSICS_CONTEXT_RE =
  /\b(m\/s|m\/s\^?2|newton|force|friction|incline|spring|velocity|acceleration|distance|displacement|graph|time|seconds?|runner|car|ball|block|charge|circuit|voltage|current|resistor|field|energy|work|power|momentum|collision|impulse|torque|wave|frequency|wavelength|lens|projectile|heat|temperature|gas|pressure)\b/i

type ConversationMessage = {
  role: 'user' | 'assistant'
  content: string
  timestamp?: number
}

type HitlSession = {
  agentId: string
  agentName: string
  message: string
  activeProblem?: string
  userMessage: Message
  priorMessages: ConversationMessage[]
  questions: HitlQuestion[]
}

type SendToAgentInput = {
  agentId: string
  message: string
  userMessage: Message
  priorMessages: ConversationMessage[]
  activeProblem?: string
  guidedTutoring?: Record<string, unknown>
}

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

function buildKinematicsQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const constantVelocity = /\b(constant velocity|constant speed|no acceleration)\b/.test(normalized)
  const asksForFinalVelocity = /\b(final velocity|final speed|velocity after|speed after|how fast)\b/.test(normalized)
  const asksForDisplacement = /\b(displacement|distance|how far|position|travel(?:ed|led)?)\b/.test(normalized)
  const noTimeGiven = /\bwithout time|no time|not given time\b/.test(normalized)
  const freeFall = /\b(dropped|falls?|free fall|gravity|highest point|maximum height|thrown up|thrown downward)\b/.test(normalized)
  const graph = /\b(graph|position-time|velocity-time|acceleration-time)\b/.test(normalized)
  const projectile = /\b(projectile|launched|thrown|kicked|angle)\b/.test(normalized)

  if (projectile) {
    return [
      {
        id: 'kinematics-projectile-components',
        title: 'Equation Check',
        prompt: 'Before solving projectile motion, which setup keeps the horizontal and vertical motion separate?',
        choices: [
          { id: 'components', text: 'Split v0 into v0x and v0y, then use x- and y-equations separately.' },
          { id: 'single_axis', text: 'Use one one-dimensional equation for the whole path.' },
        ],
        correctChoiceId: 'components',
        feedback: 'Projectile motion needs separate horizontal and vertical equations because gravity acts vertically.',
      },
      {
        id: 'kinematics-projectile-vertical',
        title: 'Follow-Up Check',
        prompt: 'Which vertical equation includes the effect of gravity during the flight?',
        choices: [
          { id: 'vertical_position', text: 'y = y0 + v0y*t - 1/2*g*t^2' },
          { id: 'horizontal_position', text: 'x = x0 + v0x*t' },
        ],
        correctChoiceId: 'vertical_position',
        feedback: 'Gravity changes vertical motion, so the vertical position equation needs the -1/2*g*t^2 term.',
      },
      {
        id: 'kinematics-projectile-acceleration',
        title: 'Second Follow-Up',
        prompt: 'Ignoring air resistance, what acceleration should be used for the horizontal direction?',
        choices: [
          { id: 'zero', text: 'a_x = 0' },
          { id: 'gravity', text: 'a_x = -g' },
        ],
        correctChoiceId: 'zero',
        feedback: 'Gravity acts vertically, so horizontal acceleration is zero when air resistance is ignored.',
      },
    ]
  }

  if (graph) {
    return [
      {
        id: 'kinematics-graph-slope',
        title: 'Graph Check',
        prompt: 'On a position-time graph, what does the slope represent?',
        choices: [
          { id: 'velocity', text: 'Velocity' },
          { id: 'acceleration', text: 'Acceleration' },
        ],
        correctChoiceId: 'velocity',
        feedback: 'The slope of position vs. time is velocity.',
      },
      {
        id: 'kinematics-graph-area',
        title: 'Follow-Up Check',
        prompt: 'On a velocity-time graph, what does the area under the curve represent?',
        choices: [
          { id: 'displacement', text: 'Displacement' },
          { id: 'force', text: 'Net force' },
        ],
        correctChoiceId: 'displacement',
        feedback: 'Area under a velocity-time graph gives displacement.',
      },
      {
        id: 'kinematics-graph-acceleration',
        title: 'Second Follow-Up',
        prompt: 'On a velocity-time graph, what does the slope represent?',
        choices: [
          { id: 'acceleration', text: 'Acceleration' },
          { id: 'position', text: 'Position' },
        ],
        correctChoiceId: 'acceleration',
        feedback: 'The slope of velocity vs. time is acceleration.',
      },
    ]
  }

  if (constantVelocity) {
    return [
      {
        id: 'kinematics-constant-velocity',
        title: 'Equation Check',
        prompt: 'For constant velocity motion, which equation should be used first?',
        choices: [
          { id: 'constant_velocity', text: 'Delta x = v*t' },
          { id: 'accelerated_motion', text: 'Delta x = v0*t + 1/2*a*t^2 with nonzero a' },
        ],
        correctChoiceId: 'constant_velocity',
        feedback: 'Constant velocity means acceleration is zero, so displacement is velocity times time.',
      },
      {
        id: 'kinematics-constant-velocity-acceleration',
        title: 'Follow-Up Check',
        prompt: 'If an object moves at constant velocity, what is its acceleration?',
        choices: [
          { id: 'zero', text: '0 m/s^2' },
          { id: 'same_as_velocity', text: 'The same number as its velocity' },
        ],
        correctChoiceId: 'zero',
        feedback: 'Constant velocity means the velocity is not changing, so acceleration is zero.',
      },
    ]
  }

  if (freeFall) {
    return [
      {
        id: 'kinematics-free-fall-acceleration',
        title: 'Model Check',
        prompt: 'For free fall near Earth, what acceleration should be used if upward is positive?',
        choices: [
          { id: 'negative_g', text: 'a = -g = -9.8 m/s^2' },
          { id: 'zero_at_top', text: 'a = 0 at the top of the motion' },
        ],
        correctChoiceId: 'negative_g',
        feedback: 'Gravity still accelerates the object downward, even at the highest point.',
      },
      {
        id: 'kinematics-free-fall-top',
        title: 'Follow-Up Check',
        prompt: 'At the highest point of vertical motion, which quantity is momentarily zero?',
        choices: [
          { id: 'velocity', text: 'Vertical velocity' },
          { id: 'acceleration', text: 'Acceleration' },
        ],
        correctChoiceId: 'velocity',
        feedback: 'At the top, vertical velocity is zero for an instant, but acceleration is still downward.',
      },
      {
        id: 'kinematics-free-fall-equation',
        title: 'Second Follow-Up',
        prompt: 'Which equation connects height change, initial vertical velocity, acceleration, and time?',
        choices: [
          { id: 'vertical_position', text: 'Delta y = v0y*t + 1/2*a*t^2' },
          { id: 'horizontal_speed', text: 'v_x = Delta x / t' },
        ],
        correctChoiceId: 'vertical_position',
        feedback: 'Vertical position changes are modeled with the constant-acceleration position equation.',
      },
    ]
  }

  if (noTimeGiven) {
    return [
      {
        id: 'kinematics-no-time',
        title: 'Equation Check',
        prompt: 'If time is not given and the problem relates velocity, acceleration, and displacement, which equation is usually best?',
        choices: [
          { id: 'velocity_displacement', text: 'v^2 = v0^2 + 2*a*Delta x' },
          { id: 'velocity_time', text: 'v = v0 + a*t' },
        ],
        correctChoiceId: 'velocity_displacement',
        feedback: 'The velocity-displacement equation avoids introducing time as another unknown.',
      },
      {
        id: 'kinematics-no-time-knowns',
        title: 'Follow-Up Check',
        prompt: 'Before using that equation, what should be checked?',
        choices: [
          { id: 'signs_units', text: 'Signs, units, and which velocity is initial or final' },
          { id: 'mass_first', text: 'Mass and net force first' },
        ],
        correctChoiceId: 'signs_units',
        feedback: 'The equation is only useful after the variables and signs are assigned consistently.',
      },
    ]
  }

  if (asksForFinalVelocity && !asksForDisplacement) {
    return [
      {
        id: 'kinematics-final-velocity',
        title: 'Equation Check',
        prompt: 'If time and acceleration are known, which equation directly gives final velocity?',
        choices: [
          { id: 'velocity_time', text: 'v = v0 + a*t' },
          { id: 'displacement_time', text: 'Delta x = v0*t + 1/2*a*t^2' },
        ],
        correctChoiceId: 'velocity_time',
        feedback: 'Final velocity after a known time comes from v = v0 + a*t.',
      },
      {
        id: 'kinematics-sign-convention',
        title: 'Follow-Up Check',
        prompt: 'What must be decided before substituting signs into the equation?',
        choices: [
          { id: 'axis_signs', text: 'Choose the positive direction and keep v0, a, and v signs consistent.' },
          { id: 'mass_force', text: 'Find the mass and net force first.' },
        ],
        correctChoiceId: 'axis_signs',
        feedback: 'Kinematics equations depend on a consistent sign convention for velocity and acceleration.',
      },
      {
        id: 'kinematics-final-velocity-units',
        title: 'Second Follow-Up',
        prompt: 'What unit should final velocity have in SI units?',
        choices: [
          { id: 'meters_per_second', text: 'm/s' },
          { id: 'meters_per_second_squared', text: 'm/s^2' },
        ],
        correctChoiceId: 'meters_per_second',
        feedback: 'Velocity is measured in meters per second; acceleration uses meters per second squared.',
      },
    ]
  }

  return [
    {
      id: 'kinematics-displacement-time',
      title: 'Equation Check',
      prompt: asksForDisplacement
        ? 'For displacement after a known time with constant acceleration, which equation is the best first choice?'
        : 'For constant-acceleration motion, which equation uses position, time, initial velocity, and acceleration?',
      choices: [
        { id: 'displacement_time', text: 'Delta x = v0*t + 1/2*a*t^2' },
        { id: 'velocity_displacement', text: 'v^2 = v0^2 + 2*a*Delta x' },
      ],
      correctChoiceId: 'displacement_time',
      feedback: 'Use the displacement-time equation when time is part of the setup.',
    },
    {
      id: 'kinematics-knowns',
      title: 'Follow-Up Check',
      prompt: 'Before using the equation, which information should be identified?',
      choices: [
        { id: 'knowns_unknown', text: 'Known values, target unknown, units, and sign convention.' },
        { id: 'force_diagram', text: 'A force diagram and coefficient of friction.' },
      ],
      correctChoiceId: 'knowns_unknown',
      feedback: 'The setup step is to list knowns/unknowns with units and signs before calculating.',
    },
    {
      id: 'kinematics-displacement-units',
      title: 'Second Follow-Up',
      prompt: 'What unit should displacement have in SI units?',
      choices: [
        { id: 'meters', text: 'Meters' },
        { id: 'meters_per_second', text: 'Meters per second' },
      ],
      correctChoiceId: 'meters',
      feedback: 'Displacement is a length, so the SI unit is meters.',
    },
  ]
}

function buildForcesQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const spring = /\b(spring|hooke|compressed|stretched|k\s*=|n\/m)\b/.test(normalized)
  const friction = /\b(friction|mu|coefficient|rough|sliding)\b/.test(normalized)
  const incline = /\b(incline|inclined|ramp|slope|angle)\b/.test(normalized)
  const equilibrium = /\b(equilibrium|balanced|static|at rest|constant velocity|tension)\b/.test(normalized)
  const weightNormal = /\b(weight|normal force|normal|vertical force)\b/.test(normalized)

  if (spring) {
    return [
      {
        id: 'forces-spring-law',
        title: 'Model Check',
        prompt: 'For a spring with known spring constant and compression, which model fits the situation?',
        choices: [
          { id: 'hooke', text: 'F_s = k*x' },
          { id: 'newton_second', text: 'sum F = m*a' },
        ],
        correctChoiceId: 'hooke',
        feedback: "A compressed or stretched spring with k and x points to Hooke's law first.",
      },
      {
        id: 'forces-spring-units',
        title: 'Follow-Up Check',
        prompt: 'What unit should the spring-force answer have?',
        choices: [
          { id: 'newtons', text: 'Newtons, because (N/m)*m gives N.' },
          { id: 'joules', text: 'Joules, because compression stores energy.' },
        ],
        correctChoiceId: 'newtons',
        feedback: 'Spring force is measured in newtons. Joules would be the unit for spring potential energy.',
      },
      {
        id: 'forces-spring-direction',
        title: 'Second Follow-Up',
        prompt: 'What direction is the restoring spring force relative to the compression or stretch?',
        choices: [
          { id: 'opposite', text: 'Opposite the displacement from equilibrium' },
          { id: 'same', text: 'Same direction as the displacement from equilibrium' },
        ],
        correctChoiceId: 'opposite',
        feedback: 'A spring force is restoring, so it points opposite the displacement from equilibrium.',
      },
    ]
  }

  if (friction) {
    return [
      {
        id: 'forces-friction-model',
        title: 'Model Check',
        prompt: 'If a block is already sliding on a rough surface, which friction model should be used?',
        choices: [
          { id: 'kinetic', text: 'f_k = mu_k*N' },
          { id: 'static_max', text: 'f_s <= mu_s*N' },
        ],
        correctChoiceId: 'kinetic',
        feedback: 'Sliding motion uses kinetic friction. Static friction applies before slipping begins.',
      },
      {
        id: 'forces-friction-direction',
        title: 'Follow-Up Check',
        prompt: 'Which way does friction point?',
        choices: [
          { id: 'opposes_motion', text: 'Opposite the motion or tendency to slip' },
          { id: 'same_as_motion', text: 'In the same direction as motion' },
        ],
        correctChoiceId: 'opposes_motion',
        feedback: 'Friction opposes relative motion or the tendency for surfaces to slip.',
      },
      {
        id: 'forces-friction-normal',
        title: 'Second Follow-Up',
        prompt: 'What force is needed before calculating friction with f = mu*N?',
        choices: [
          { id: 'normal', text: 'The normal force' },
          { id: 'velocity', text: 'The velocity' },
        ],
        correctChoiceId: 'normal',
        feedback: 'Friction depends on the normal force pressing the surfaces together.',
      },
    ]
  }

  if (incline) {
    return [
      {
        id: 'forces-incline-components',
        title: 'Setup Check',
        prompt: 'For a block on an incline, which weight component points down the ramp?',
        choices: [
          { id: 'mg_sin', text: 'mg*sin(theta)' },
          { id: 'mg_cos', text: 'mg*cos(theta)' },
        ],
        correctChoiceId: 'mg_sin',
        feedback: 'The component parallel to the incline is mg*sin(theta).',
      },
      {
        id: 'forces-incline-normal',
        title: 'Follow-Up Check',
        prompt: 'If there is no acceleration perpendicular to the incline, what is the normal force?',
        choices: [
          { id: 'mg_cos', text: 'N = mg*cos(theta)' },
          { id: 'mg_sin', text: 'N = mg*sin(theta)' },
        ],
        correctChoiceId: 'mg_cos',
        feedback: 'The normal force balances the perpendicular component of weight, mg*cos(theta).',
      },
      {
        id: 'forces-incline-axis',
        title: 'Second Follow-Up',
        prompt: 'Which axes usually make an incline problem easiest?',
        choices: [
          { id: 'tilted_axes', text: 'x along the ramp and y perpendicular to the ramp' },
          { id: 'horizontal_vertical', text: 'Always horizontal and vertical axes' },
        ],
        correctChoiceId: 'tilted_axes',
        feedback: 'Axes aligned with the ramp reduce the number of force components.',
      },
    ]
  }

  if (equilibrium) {
    return [
      {
        id: 'forces-equilibrium-net-force',
        title: 'Model Check',
        prompt: 'If an object is in equilibrium, what must be true about the net force?',
        choices: [
          { id: 'zero', text: 'The net force is zero' },
          { id: 'ma', text: 'The net force must equal the largest force' },
        ],
        correctChoiceId: 'zero',
        feedback: 'Equilibrium means acceleration is zero, so the net force is zero.',
      },
      {
        id: 'forces-equilibrium-equations',
        title: 'Follow-Up Check',
        prompt: 'Which equations should be written for a 2D equilibrium problem?',
        choices: [
          { id: 'sum_forces_zero', text: 'sum F_x = 0 and sum F_y = 0' },
          { id: 'kinematics', text: 'v = v0 + a*t and Delta x = v0*t + 1/2*a*t^2' },
        ],
        correctChoiceId: 'sum_forces_zero',
        feedback: 'For equilibrium, write net-force equations along each axis.',
      },
      {
        id: 'forces-equilibrium-diagram',
        title: 'Second Follow-Up',
        prompt: 'What should be drawn before solving for tensions or support forces?',
        choices: [
          { id: 'free_body', text: 'A free-body diagram of the object' },
          { id: 'motion_graph', text: 'A velocity-time graph' },
        ],
        correctChoiceId: 'free_body',
        feedback: 'Tension and support-force problems are clearest after a free-body diagram.',
      },
    ]
  }

  if (weightNormal) {
    return [
      {
        id: 'forces-weight',
        title: 'Setup Check',
        prompt: 'Which expression gives the weight of an object near Earth?',
        choices: [
          { id: 'mg', text: 'W = m*g' },
          { id: 'ma_horizontal', text: 'W = m*a_x' },
        ],
        correctChoiceId: 'mg',
        feedback: 'Weight is the gravitational force, W = m*g, directed downward.',
      },
      {
        id: 'forces-normal-not-always-mg',
        title: 'Follow-Up Check',
        prompt: 'When does the normal force equal mg?',
        choices: [
          { id: 'level_no_vertical_accel', text: 'On a level surface with no vertical acceleration and no extra vertical forces' },
          { id: 'always', text: 'Always, in every forces problem' },
        ],
        correctChoiceId: 'level_no_vertical_accel',
        feedback: 'The normal force equals mg only in special cases, not always.',
      },
    ]
  }

  return [
    {
      id: 'forces-free-body',
      title: 'Setup Check',
      prompt: 'Before writing equations for a forces problem, what should come first?',
      choices: [
        { id: 'free_body', text: 'Isolate the object, draw a free-body diagram, and choose axes.' },
        { id: 'kinematics', text: 'Choose a kinematics equation before listing forces.' },
      ],
      correctChoiceId: 'free_body',
      feedback: 'Forces problems start by identifying forces on the isolated object.',
    },
    {
      id: 'forces-newton-axis',
      title: 'Follow-Up Check',
      prompt: 'After the axes are chosen, what equation represents Newtons second law along x?',
      choices: [
        { id: 'sum_fx', text: 'sum F_x = m*a_x' },
        { id: 'velocity_squared', text: 'v_f^2 = v_i^2 + 2*a*Delta x' },
      ],
      correctChoiceId: 'sum_fx',
      feedback: "Newton's second law compares net force along an axis with mass times acceleration along that axis.",
    },
  ]
}

function buildEnergyQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const spring = /\b(spring|compressed|stretched|elastic)\b/.test(normalized)
  const work = /\b(work|force.*distance|angle|cos)\b/.test(normalized)
  const power = /\b(power|watt|rate)\b/.test(normalized)

  if (spring) {
    return [
      {
        id: 'energy-spring-energy',
        title: 'Model Check',
        prompt: 'Which expression gives elastic potential energy stored in a spring?',
        choices: [
          { id: 'spring_energy', text: 'U_s = 1/2*k*x^2' },
          { id: 'spring_force', text: 'F_s = k*x' },
        ],
        correctChoiceId: 'spring_energy',
        feedback: 'Spring force uses k*x, but spring energy uses one half k x squared.',
      },
      {
        id: 'energy-spring-units',
        title: 'Follow-Up Check',
        prompt: 'What unit should spring potential energy have?',
        choices: [
          { id: 'joules', text: 'Joules' },
          { id: 'newtons', text: 'Newtons' },
        ],
        correctChoiceId: 'joules',
        feedback: 'Energy is measured in joules. Newtons measure force.',
      },
    ]
  }

  if (work) {
    return [
      {
        id: 'energy-work-equation',
        title: 'Equation Check',
        prompt: 'For a constant force applied through a displacement at angle theta, which equation gives work?',
        choices: [
          { id: 'work_cos', text: 'W = F*d*cos(theta)' },
          { id: 'work_sin', text: 'W = F*d*sin(theta)' },
        ],
        correctChoiceId: 'work_cos',
        feedback: 'Work uses the component of force along the displacement, so the cosine factor appears.',
      },
      {
        id: 'energy-work-sign',
        title: 'Follow-Up Check',
        prompt: 'When is work negative?',
        choices: [
          { id: 'opposite_motion', text: 'When the force component is opposite the displacement' },
          { id: 'large_force', text: 'Whenever the force is large' },
        ],
        correctChoiceId: 'opposite_motion',
        feedback: 'Work sign depends on the force component relative to displacement.',
      },
    ]
  }

  if (power) {
    return [
      {
        id: 'energy-power-definition',
        title: 'Equation Check',
        prompt: 'Which equation defines average power?',
        choices: [
          { id: 'work_time', text: 'P = W/t' },
          { id: 'force_distance', text: 'P = F*d' },
        ],
        correctChoiceId: 'work_time',
        feedback: 'Power is the rate of energy transfer or work done per time.',
      },
      {
        id: 'energy-power-units',
        title: 'Follow-Up Check',
        prompt: 'What is one watt equal to?',
        choices: [
          { id: 'joule_per_second', text: '1 J/s' },
          { id: 'newton_per_meter', text: '1 N/m' },
        ],
        correctChoiceId: 'joule_per_second',
        feedback: 'A watt is one joule per second.',
      },
    ]
  }

  return [
    {
      id: 'energy-conservation-model',
      title: 'Model Check',
      prompt: 'If non-conservative work is negligible, which idea should usually be used?',
      choices: [
        { id: 'mechanical_energy', text: 'Mechanical energy is conserved' },
        { id: 'momentum', text: 'Momentum is always conserved instead' },
      ],
      correctChoiceId: 'mechanical_energy',
      feedback: 'Without non-conservative work, mechanical energy can be conserved.',
    },
    {
      id: 'energy-state-setup',
      title: 'Follow-Up Check',
      prompt: 'What should be identified before writing an energy equation?',
      choices: [
        { id: 'initial_final', text: 'Initial state, final state, and energy forms present' },
        { id: 'net_force_first', text: 'Only the net force at one instant' },
      ],
      correctChoiceId: 'initial_final',
      feedback: 'Energy problems compare initial and final states.',
    },
    {
      id: 'energy-kinetic',
      title: 'Second Follow-Up',
      prompt: 'Which expression gives translational kinetic energy?',
      choices: [
        { id: 'kinetic', text: 'K = 1/2*m*v^2' },
        { id: 'potential', text: 'U_g = m*g*h' },
      ],
      correctChoiceId: 'kinetic',
      feedback: 'Kinetic energy depends on mass and speed squared.',
    },
  ]
}

function buildMomentumQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const impulse = /\b(impulse|time interval|delta t|force.*time)\b/.test(normalized)
  const collision = /\b(collision|collide|stick|elastic|inelastic|explosion)\b/.test(normalized)

  if (impulse) {
    return [
      {
        id: 'momentum-impulse',
        title: 'Model Check',
        prompt: 'Which relationship connects impulse and momentum change?',
        choices: [
          { id: 'impulse_momentum', text: 'J = Delta p = F_avg*Delta t' },
          { id: 'work_energy', text: 'W = Delta K = F*d' },
        ],
        correctChoiceId: 'impulse_momentum',
        feedback: 'Impulse is the change in momentum.',
      },
      {
        id: 'momentum-impulse-units',
        title: 'Follow-Up Check',
        prompt: 'Which unit is equivalent to impulse?',
        choices: [
          { id: 'newton_second', text: 'N*s' },
          { id: 'joule', text: 'J' },
        ],
        correctChoiceId: 'newton_second',
        feedback: 'Impulse can be measured in newton-seconds, equivalent to kg*m/s.',
      },
    ]
  }

  if (collision) {
    return [
      {
        id: 'momentum-collision-system',
        title: 'Setup Check',
        prompt: 'Before using momentum conservation in a collision, what should be checked?',
        choices: [
          { id: 'isolated', text: 'The system is isolated or external impulse is negligible' },
          { id: 'energy_always', text: 'Kinetic energy is always conserved' },
        ],
        correctChoiceId: 'isolated',
        feedback: 'Momentum conservation requires an isolated system or negligible external impulse.',
      },
      {
        id: 'momentum-inelastic',
        title: 'Follow-Up Check',
        prompt: 'If two carts stick together after colliding, what type of collision is it?',
        choices: [
          { id: 'perfectly_inelastic', text: 'Perfectly inelastic' },
          { id: 'elastic', text: 'Elastic' },
        ],
        correctChoiceId: 'perfectly_inelastic',
        feedback: 'Objects that stick together after collision have a perfectly inelastic collision.',
      },
      {
        id: 'momentum-vector-signs',
        title: 'Second Follow-Up',
        prompt: 'What is important when writing the momentum equation in one dimension?',
        choices: [
          { id: 'direction_signs', text: 'Choose a positive direction and keep velocity signs consistent' },
          { id: 'speed_only', text: 'Use only positive speeds for every object' },
        ],
        correctChoiceId: 'direction_signs',
        feedback: 'Momentum is directional, so velocity signs matter.',
      },
    ]
  }

  return [
    {
      id: 'momentum-definition',
      title: 'Equation Check',
      prompt: 'Which expression defines linear momentum?',
      choices: [
        { id: 'mv', text: 'p = m*v' },
        { id: 'ma', text: 'p = m*a' },
      ],
      correctChoiceId: 'mv',
      feedback: 'Linear momentum is mass times velocity.',
    },
    {
      id: 'momentum-units',
      title: 'Follow-Up Check',
      prompt: 'What unit should momentum have?',
      choices: [
        { id: 'kg_m_s', text: 'kg*m/s' },
        { id: 'newtons', text: 'N' },
      ],
      correctChoiceId: 'kg_m_s',
      feedback: 'Momentum has units of mass times velocity: kg*m/s.',
    },
  ]
}

function buildElectromagnetismQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const circuit = /\b(circuit|resistor|battery|voltage|current|ohm|series|parallel)\b/.test(normalized)
  const coulomb = /\b(charge|coulomb|electric force|electric field|point charge)\b/.test(normalized)

  if (circuit) {
    return [
      {
        id: 'em-circuit-ohm',
        title: 'Equation Check',
        prompt: 'For a simple resistor with voltage and resistance known, which equation gives current?',
        choices: [
          { id: 'ohms_law', text: 'I = V/R' },
          { id: 'power', text: 'I = P*t' },
        ],
        correctChoiceId: 'ohms_law',
        feedback: "Ohm's law gives current from voltage and resistance.",
      },
      {
        id: 'em-series-current',
        title: 'Follow-Up Check',
        prompt: 'In a series circuit, which quantity is the same through each resistor?',
        choices: [
          { id: 'current', text: 'Current' },
          { id: 'voltage', text: 'Voltage across each resistor' },
        ],
        correctChoiceId: 'current',
        feedback: 'Series elements share the same current; voltage divides across them.',
      },
      {
        id: 'em-parallel-voltage',
        title: 'Second Follow-Up',
        prompt: 'In a parallel circuit, which quantity is the same across each branch?',
        choices: [
          { id: 'voltage', text: 'Voltage' },
          { id: 'current', text: 'Current through each branch' },
        ],
        correctChoiceId: 'voltage',
        feedback: 'Parallel branches share the same voltage; current divides between branches.',
      },
    ]
  }

  if (coulomb) {
    return [
      {
        id: 'em-coulomb-law',
        title: 'Equation Check',
        prompt: 'For the electric force between two point charges, which model applies?',
        choices: [
          { id: 'coulomb', text: "F = k*|q1*q2|/r^2" },
          { id: 'ohms_law', text: 'V = I*R' },
        ],
        correctChoiceId: 'coulomb',
        feedback: "Coulomb's law models force between point charges.",
      },
      {
        id: 'em-force-direction',
        title: 'Follow-Up Check',
        prompt: 'What determines whether the electric force is attractive or repulsive?',
        choices: [
          { id: 'charge_signs', text: 'The signs of the two charges' },
          { id: 'charge_masses', text: 'The masses of the charges' },
        ],
        correctChoiceId: 'charge_signs',
        feedback: 'Like charges repel and opposite charges attract.',
      },
    ]
  }

  return [
    {
      id: 'em-field-definition',
      title: 'Setup Check',
      prompt: 'Which relationship connects electric force and electric field for a test charge?',
      choices: [
        { id: 'force_field', text: 'F = q*E' },
        { id: 'force_mass', text: 'F = m*g' },
      ],
      correctChoiceId: 'force_field',
      feedback: 'Electric field is force per unit charge, so F = qE.',
    },
    {
      id: 'em-units-field',
      title: 'Follow-Up Check',
      prompt: 'Which unit can be used for electric field?',
      choices: [
        { id: 'newton_per_coulomb', text: 'N/C' },
        { id: 'newton_meter', text: 'N*m' },
      ],
      correctChoiceId: 'newton_per_coulomb',
      feedback: 'Electric field can be measured in newtons per coulomb.',
    },
  ]
}

function buildWavesQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const standing = /\b(standing wave|harmonic|node|antinode|string|pipe)\b/.test(normalized)

  if (standing) {
    return [
      {
        id: 'waves-standing-boundaries',
        title: 'Model Check',
        prompt: 'Before choosing a standing-wave equation, what should be identified?',
        choices: [
          { id: 'boundary_conditions', text: 'Boundary conditions such as open/closed ends or fixed/free ends' },
          { id: 'mass_only', text: 'Only the mass of the wave source' },
        ],
        correctChoiceId: 'boundary_conditions',
        feedback: 'Standing-wave patterns depend on boundary conditions.',
      },
      {
        id: 'waves-string-harmonic',
        title: 'Follow-Up Check',
        prompt: 'For a string fixed at both ends, what wavelengths are allowed?',
        choices: [
          { id: 'two_l_over_n', text: 'lambda_n = 2L/n' },
          { id: 'l_over_n', text: 'lambda_n = L/n for every case' },
        ],
        correctChoiceId: 'two_l_over_n',
        feedback: 'A fixed-fixed string fits n half-wavelengths in length L, so lambda = 2L/n.',
      },
    ]
  }

  return [
    {
      id: 'waves-speed',
      title: 'Equation Check',
      prompt: 'Which equation relates wave speed, frequency, and wavelength?',
      choices: [
        { id: 'wave_speed', text: 'v = f*lambda' },
        { id: 'acceleration', text: 'v = v0 + a*t' },
      ],
      correctChoiceId: 'wave_speed',
      feedback: 'Wave speed equals frequency times wavelength.',
    },
    {
      id: 'waves-frequency-wavelength',
      title: 'Follow-Up Check',
      prompt: 'If wave speed is fixed and wavelength increases, what happens to frequency?',
      choices: [
        { id: 'decreases', text: 'Frequency decreases' },
        { id: 'increases', text: 'Frequency increases' },
      ],
      correctChoiceId: 'decreases',
      feedback: 'For fixed wave speed, frequency and wavelength are inversely related.',
    },
  ]
}

function buildThermodynamicsQuestions(problem: string): HitlQuestion[] {
  const normalized = problem.toLowerCase()
  const gas = /\b(gas|pressure|volume|temperature|moles|ideal)\b/.test(normalized)
  const heat = /\b(heat|specific heat|temperature change|thermal)\b/.test(normalized)

  if (gas) {
    return [
      {
        id: 'thermo-ideal-gas',
        title: 'Model Check',
        prompt: 'For pressure, volume, moles, and temperature of an ideal gas, which equation is the starting point?',
        choices: [
          { id: 'ideal_gas', text: 'P*V = n*R*T' },
          { id: 'newton_second', text: 'sum F = m*a' },
        ],
        correctChoiceId: 'ideal_gas',
        feedback: 'The ideal gas law relates pressure, volume, amount of gas, and temperature.',
      },
      {
        id: 'thermo-temperature-units',
        title: 'Follow-Up Check',
        prompt: 'What temperature unit should be used in the ideal gas law?',
        choices: [
          { id: 'kelvin', text: 'Kelvin' },
          { id: 'celsius', text: 'Celsius without conversion' },
        ],
        correctChoiceId: 'kelvin',
        feedback: 'Gas-law calculations require absolute temperature in kelvin.',
      },
    ]
  }

  if (heat) {
    return [
      {
        id: 'thermo-specific-heat',
        title: 'Equation Check',
        prompt: 'Which equation gives heat for a material with mass, specific heat, and temperature change?',
        choices: [
          { id: 'specific_heat', text: 'Q = m*c*Delta T' },
          { id: 'work', text: 'W = F*d*cos(theta)' },
        ],
        correctChoiceId: 'specific_heat',
        feedback: 'Specific heat problems use Q = mc Delta T.',
      },
      {
        id: 'thermo-heat-sign',
        title: 'Follow-Up Check',
        prompt: 'If heat enters the system, what sign is Q usually assigned?',
        choices: [
          { id: 'positive', text: 'Positive' },
          { id: 'negative', text: 'Negative' },
        ],
        correctChoiceId: 'positive',
        feedback: 'With the common convention, heat added to the system is positive.',
      },
    ]
  }

  return [
    {
      id: 'thermo-first-law',
      title: 'Model Check',
      prompt: 'Which law connects internal energy, heat, and work?',
      choices: [
        { id: 'first_law', text: 'Delta U = Q - W' },
        { id: 'ohms_law', text: 'V = I*R' },
      ],
      correctChoiceId: 'first_law',
      feedback: 'The first law of thermodynamics connects internal energy, heat, and work.',
    },
    {
      id: 'thermo-process',
      title: 'Follow-Up Check',
      prompt: 'Before applying the first law, what should be identified?',
      choices: [
        { id: 'process', text: 'The process type and sign convention for heat/work' },
        { id: 'projectile_angle', text: 'The projectile launch angle' },
      ],
      correctChoiceId: 'process',
      feedback: 'Thermodynamics problems depend strongly on process type and sign convention.',
    },
  ]
}

function normalizeAgentId(agentId: string) {
  let normalized = agentId.trim().toLowerCase().replace(/-/g, '_')

  if (normalized.startsWith('physics_')) {
    normalized = normalized.slice('physics_'.length)
  }

  if (!normalized.endsWith('_agent')) {
    normalized = `${normalized}_agent`
  }

  return normalized
}

function buildHitlQuestions(agentId: string, problem: string): HitlQuestion[] {
  const normalizedAgentId = normalizeAgentId(agentId)

  if (normalizedAgentId === 'kinematics_agent') {
    return buildKinematicsQuestions(problem)
  }

  if (normalizedAgentId === 'forces_agent') {
    return buildForcesQuestions(problem)
  }

  if (normalizedAgentId === 'energy_agent') {
    return buildEnergyQuestions(problem)
  }

  if (normalizedAgentId === 'momentum_agent') {
    return buildMomentumQuestions(problem)
  }

  if (normalizedAgentId === 'electromagnetism_agent') {
    return buildElectromagnetismQuestions(problem)
  }

  if (normalizedAgentId === 'waves_agent') {
    return buildWavesQuestions(problem)
  }

  if (normalizedAgentId === 'thermodynamics_agent') {
    return buildThermodynamicsQuestions(problem)
  }

  return []
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
  const [hitlSession, setHitlSession] = useState<HitlSession | null>(null)

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

  const sendMessageToAgent = useCallback(
    async ({
      agentId,
      message,
      userMessage,
      priorMessages,
      activeProblem,
      guidedTutoring,
    }: SendToAgentInput) => {
      setLoading(true)
      setError(null)

      try {
        const conversationMessages = [...priorMessages, userMessage]
        const conversationContext: Record<string, unknown> = {
          active_problem: activeProblem || inferActiveProblem(priorMessages, message),
          recent_conversation: conversationMessages.slice(-20).map((msg) => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp,
          })),
          total_messages: conversationMessages.length,
        }

        if (guidedTutoring) {
          conversationContext.guided_tutoring = guidedTutoring
        }

        const response = await apiClient.sendMessage(
          agentId,
          message,
          user?.username || 'react_user',
          conversationContext
        )

        if (response.success && response.solution) {
          const assistantMessage: Message = {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: response.solution,
            timestamp: Date.now(),
            agentId,
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
    [user, addMessage, setLoading, setError]
  )

  // Handle sending messages
  const handleSendMessage = useCallback(
    async (message: string) => {
      const trimmedMessage = message.trim()
      if (!selectedAgent || !trimmedMessage) return

      const userMessage: Message = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: trimmedMessage,
        timestamp: Date.now(),
        agentId: selectedAgent,
      }
      const priorMessages = messages
      const activeProblem = inferActiveProblem(priorMessages, trimmedMessage)
      const hitlQuestions = looksLikePhysicsProblem(trimmedMessage)
        ? buildHitlQuestions(selectedAgent, activeProblem || trimmedMessage)
        : []

      addMessage(userMessage)
      setError(null)

      if (hitlQuestions.length > 0) {
        setLoading(false)
        setHitlSession({
          agentId: selectedAgent,
          agentName: selectedAgentInfo?.name || selectedAgent,
          message: trimmedMessage,
          activeProblem,
          userMessage,
          priorMessages,
          questions: hitlQuestions,
        })
        return
      }

      await sendMessageToAgent({
        agentId: selectedAgent,
        message: trimmedMessage,
        userMessage,
        priorMessages,
        activeProblem,
      })
    },
    [
      selectedAgent,
      selectedAgentInfo,
      messages,
      addMessage,
      setError,
      setLoading,
      sendMessageToAgent,
    ]
  )

  const handleHitlCancel = useCallback(() => {
    setHitlSession(null)
    setLoading(false)
  }, [setLoading])

  const handleHitlComplete = useCallback(
    (results: HitlResult[]) => {
      if (!hitlSession) return

      setHitlSession(null)
      void sendMessageToAgent({
        agentId: hitlSession.agentId,
        message: hitlSession.message,
        userMessage: hitlSession.userMessage,
        priorMessages: hitlSession.priorMessages,
        activeProblem: hitlSession.activeProblem,
        guidedTutoring: {
          full_solution_allowed: true,
          reason: 'hitl_checkpoint_correct',
          hitl_results: results,
          checkpoint_question_ids: results.map((result) => result.questionId),
        },
      })
    },
    [hitlSession, sendMessageToAgent]
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
          <ChatInput onSend={handleSendMessage} disabled={loading || Boolean(hitlSession)} />
        )}
      </Box>

      <HumanInTheLoopDialog
        open={Boolean(hitlSession)}
        agentName={hitlSession?.agentName || 'Physics Assistant'}
        questions={hitlSession?.questions || []}
        onCancel={handleHitlCancel}
        onComplete={handleHitlComplete}
      />
    </Box>
  )
}
