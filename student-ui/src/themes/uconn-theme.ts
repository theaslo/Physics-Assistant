import { createTheme } from '@mui/material/styles'

// UConn Brand Colors
const UCONN_NAVY = '#000E2F'
const UCONN_BLUE = '#0076CE'
const UCONN_SILVER = '#A2AAAD'
const UCONN_WHITE = '#FFFFFF'

// Physics agent colors
export const agentColors: Record<string, string> = {
  // Physics 101
  forces_agent: '#388e3c',           // Green
  kinematics_agent: '#1976d2',       // Blue
  energy_agent: '#f57c00',           // Orange
  momentum_agent: '#7b1fa2',         // Purple
  angular_motion_agent: '#d32f2f',   // Red
  math_agent: '#455a64',             // Grey-blue
  // Physics 102
  thermodynamics_agent: '#e65100',   // Deep Orange
  waves_agent: '#00838f',            // Cyan
  // Physics 201
  electromagnetism_agent: '#fbc02d', // Yellow/Gold
  // Physics 202
  optics_agent: '#29b6f6',           // Light Blue
  modern_physics_agent: '#9c27b0',   // Deep Purple
}

export const agentIcons: Record<string, string> = {
  // Physics 101
  forces_agent: '⚖️',
  kinematics_agent: '🚀',
  energy_agent: '⚡',
  momentum_agent: '💥',
  angular_motion_agent: '🌀',
  math_agent: '🔢',
  // Physics 102
  thermodynamics_agent: '🔥',
  waves_agent: '🌊',
  // Physics 201
  electromagnetism_agent: '🧲',
  // Physics 202
  optics_agent: '🔦',
  modern_physics_agent: '⚛️',
}

export const uconnTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: UCONN_NAVY,
      light: UCONN_BLUE,
      dark: '#000819',
      contrastText: UCONN_WHITE,
    },
    secondary: {
      main: UCONN_BLUE,
      light: '#4da3e8',
      dark: '#005a9e',
      contrastText: UCONN_WHITE,
    },
    background: {
      default: '#f5f7fa',
      paper: UCONN_WHITE,
    },
    text: {
      primary: UCONN_NAVY,
      secondary: UCONN_SILVER,
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      color: UCONN_NAVY,
    },
    h2: {
      fontWeight: 600,
      color: UCONN_NAVY,
    },
    h3: {
      fontWeight: 600,
      color: UCONN_NAVY,
    },
    h4: {
      fontWeight: 600,
      color: UCONN_NAVY,
    },
    h5: {
      fontWeight: 500,
      color: UCONN_NAVY,
    },
    h6: {
      fontWeight: 500,
      color: UCONN_NAVY,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          fontWeight: 500,
        },
        containedPrimary: {
          backgroundColor: UCONN_NAVY,
          '&:hover': {
            backgroundColor: '#001a4d',
          },
        },
        containedSecondary: {
          backgroundColor: UCONN_BLUE,
          '&:hover': {
            backgroundColor: '#005a9e',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0, 14, 47, 0.08)',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: UCONN_NAVY,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: UCONN_WHITE,
          borderRight: `1px solid ${UCONN_SILVER}30`,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
})

export function getAgentColor(agentId: string): string {
  return agentColors[agentId as keyof typeof agentColors] || UCONN_BLUE
}

export function getAgentIcon(agentId: string): string {
  return agentIcons[agentId] || '🤖'
}
