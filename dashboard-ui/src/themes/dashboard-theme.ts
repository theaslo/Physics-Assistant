/**
 * Material-UI Theme Configuration for Dashboard
 * Provides light and dark themes with custom colors and typography
 */

import { createTheme, ThemeOptions } from '@mui/material/styles';
import { alpha } from '@mui/material/styles';

// ============================================================================
// Color Palette
// ============================================================================

const colors = {
  primary: {
    50: '#e3f2fd',
    100: '#bbdefb',
    200: '#90caf9',
    300: '#64b5f6',
    400: '#42a5f5',
    500: '#2196f3',
    600: '#1e88e5',
    700: '#1976d2',
    800: '#1565c0',
    900: '#0d47a1',
  },
  secondary: {
    50: '#f3e5f5',
    100: '#e1bee7',
    200: '#ce93d8',
    300: '#ba68c8',
    400: '#ab47bc',
    500: '#9c27b0',
    600: '#8e24aa',
    700: '#7b1fa2',
    800: '#6a1b9a',
    900: '#4a148c',
  },
  success: {
    50: '#e8f5e8',
    100: '#c8e6c9',
    200: '#a5d6a7',
    300: '#81c784',
    400: '#66bb6a',
    500: '#4caf50',
    600: '#43a047',
    700: '#388e3c',
    800: '#2e7d32',
    900: '#1b5e20',
  },
  warning: {
    50: '#fff8e1',
    100: '#ffecb3',
    200: '#ffe082',
    300: '#ffd54f',
    400: '#ffca28',
    500: '#ffc107',
    600: '#ffb300',
    700: '#ffa000',
    800: '#ff8f00',
    900: '#ff6f00',
  },
  error: {
    50: '#ffebee',
    100: '#ffcdd2',
    200: '#ef9a9a',
    300: '#e57373',
    400: '#ef5350',
    500: '#f44336',
    600: '#e53935',
    700: '#d32f2f',
    800: '#c62828',
    900: '#b71c1c',
  },
  info: {
    50: '#e1f5fe',
    100: '#b3e5fc',
    200: '#81d4fa',
    300: '#4fc3f7',
    400: '#29b6f6',
    500: '#03a9f4',
    600: '#039be5',
    700: '#0288d1',
    800: '#0277bd',
    900: '#01579b',
  },
  grey: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#eeeeee',
    300: '#e0e0e0',
    400: '#bdbdbd',
    500: '#9e9e9e',
    600: '#757575',
    700: '#616161',
    800: '#424242',
    900: '#212121',
  },
};

// ============================================================================
// Physics-specific Colors
// ============================================================================

const physicsColors = {
  kinematics: '#1976d2',
  forces: '#388e3c',
  energy: '#f57c00',
  momentum: '#7b1fa2',
  angular: '#d32f2f',
  math: '#455a64',
};

// ============================================================================
// Chart Colors
// ============================================================================

const chartColors = {
  primary: [
    '#1976d2',
    '#388e3c',
    '#f57c00',
    '#7b1fa2',
    '#d32f2f',
    '#455a64',
    '#00796b',
    '#5d4037',
  ],
  gradient: [
    'rgba(25, 118, 210, 0.8)',
    'rgba(56, 142, 60, 0.8)',
    'rgba(245, 124, 0, 0.8)',
    'rgba(123, 31, 162, 0.8)',
    'rgba(211, 47, 47, 0.8)',
    'rgba(69, 90, 100, 0.8)',
  ],
};

// ============================================================================
// Typography
// ============================================================================

const typography = {
  fontFamily: [
    '-apple-system',
    'BlinkMacSystemFont',
    '"Segoe UI"',
    'Roboto',
    '"Helvetica Neue"',
    'Arial',
    'sans-serif',
    '"Apple Color Emoji"',
    '"Segoe UI Emoji"',
    '"Segoe UI Symbol"',
  ].join(','),
  h1: {
    fontSize: '2.125rem',
    fontWeight: 600,
    lineHeight: 1.2,
  },
  h2: {
    fontSize: '1.875rem',
    fontWeight: 600,
    lineHeight: 1.3,
  },
  h3: {
    fontSize: '1.5rem',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  h4: {
    fontSize: '1.25rem',
    fontWeight: 600,
    lineHeight: 1.4,
  },
  h5: {
    fontSize: '1.125rem',
    fontWeight: 600,
    lineHeight: 1.5,
  },
  h6: {
    fontSize: '1rem',
    fontWeight: 600,
    lineHeight: 1.5,
  },
  body1: {
    fontSize: '1rem',
    lineHeight: 1.5,
  },
  body2: {
    fontSize: '0.875rem',
    lineHeight: 1.4,
  },
  button: {
    textTransform: 'none' as const,
    fontWeight: 500,
  },
  caption: {
    fontSize: '0.75rem',
    lineHeight: 1.4,
  },
};

// ============================================================================
// Component Overrides
// ============================================================================

const getComponentOverrides = (isDark: boolean) => ({
  MuiCssBaseline: {
    styleOverrides: {
      body: {
        scrollbarWidth: 'thin',
        scrollbarColor: isDark ? '#6b6b6b #2b2b2b' : '#c1c1c1 #f1f1f1',
        '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
          width: 8,
        },
        '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
          borderRadius: 8,
          backgroundColor: isDark ? '#6b6b6b' : '#c1c1c1',
          minHeight: 24,
        },
        '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
          backgroundColor: isDark ? '#2b2b2b' : '#f1f1f1',
        },
      },
    },
  },
  MuiCard: {
    styleOverrides: {
      root: {
        borderRadius: 12,
        boxShadow: isDark 
          ? '0 4px 20px rgba(0, 0, 0, 0.3)'
          : '0 4px 20px rgba(0, 0, 0, 0.1)',
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: {
        backgroundImage: 'none',
      },
      elevation1: {
        boxShadow: isDark 
          ? '0 2px 8px rgba(0, 0, 0, 0.3)'
          : '0 2px 8px rgba(0, 0, 0, 0.1)',
      },
    },
  },
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: 8,
        textTransform: 'none',
        fontWeight: 500,
      },
      contained: {
        boxShadow: 'none',
        '&:hover': {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
        },
      },
    },
  },
  MuiChip: {
    styleOverrides: {
      root: {
        borderRadius: 6,
      },
    },
  },
  MuiAppBar: {
    styleOverrides: {
      root: {
        backgroundColor: isDark ? '#1e1e1e' : '#ffffff',
        color: isDark ? '#ffffff' : '#000000',
        boxShadow: isDark 
          ? '0 1px 3px rgba(0, 0, 0, 0.3)'
          : '0 1px 3px rgba(0, 0, 0, 0.1)',
      },
    },
  },
  MuiDrawer: {
    styleOverrides: {
      paper: {
        borderRight: `1px solid ${isDark ? '#333' : '#e0e0e0'}`,
      },
    },
  },
  MuiTableHead: {
    styleOverrides: {
      root: {
        backgroundColor: isDark ? alpha('#ffffff', 0.05) : alpha('#000000', 0.02),
      },
    },
  },
});

// ============================================================================
// Light Theme
// ============================================================================

const lightThemeOptions: ThemeOptions = {
  palette: {
    mode: 'light',
    primary: {
      main: colors.primary[600],
      light: colors.primary[300],
      dark: colors.primary[800],
    },
    secondary: {
      main: colors.secondary[600],
      light: colors.secondary[300],
      dark: colors.secondary[800],
    },
    success: {
      main: colors.success[600],
      light: colors.success[300],
      dark: colors.success[800],
    },
    warning: {
      main: colors.warning[600],
      light: colors.warning[300],
      dark: colors.warning[800],
    },
    error: {
      main: colors.error[600],
      light: colors.error[300],
      dark: colors.error[800],
    },
    info: {
      main: colors.info[600],
      light: colors.info[300],
      dark: colors.info[800],
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.6)',
    },
  },
  typography,
  shape: {
    borderRadius: 8,
  },
  components: getComponentOverrides(false),
};

// ============================================================================
// Dark Theme
// ============================================================================

const darkThemeOptions: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: {
      main: colors.primary[400],
      light: colors.primary[200],
      dark: colors.primary[600],
    },
    secondary: {
      main: colors.secondary[400],
      light: colors.secondary[200],
      dark: colors.secondary[600],
    },
    success: {
      main: colors.success[400],
      light: colors.success[200],
      dark: colors.success[600],
    },
    warning: {
      main: colors.warning[400],
      light: colors.warning[200],
      dark: colors.warning[600],
    },
    error: {
      main: colors.error[400],
      light: colors.error[200],
      dark: colors.error[600],
    },
    info: {
      main: colors.info[400],
      light: colors.info[200],
      dark: colors.info[600],
    },
    background: {
      default: '#121212',
      paper: '#1e1e1e',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  typography,
  shape: {
    borderRadius: 8,
  },
  components: getComponentOverrides(true),
};

// ============================================================================
// Create Themes
// ============================================================================

export const lightTheme = createTheme(lightThemeOptions);
export const darkTheme = createTheme(darkThemeOptions);

// ============================================================================
// Theme Hook
// ============================================================================

export const useTheme = (mode: 'light' | 'dark') => {
  return mode === 'dark' ? darkTheme : lightTheme;
};

// ============================================================================
// Custom Theme Augmentation
// ============================================================================

declare module '@mui/material/styles' {
  interface Theme {
    custom: {
      physics: typeof physicsColors;
      charts: typeof chartColors;
    };
  }

  interface ThemeOptions {
    custom?: {
      physics?: typeof physicsColors;
      charts?: typeof chartColors;
    };
  }
}

// Add custom properties to themes
lightTheme.custom = {
  physics: physicsColors,
  charts: chartColors,
};

darkTheme.custom = {
  physics: physicsColors,
  charts: chartColors,
};

// ============================================================================
// Utility Functions
// ============================================================================

export const getPhysicsColor = (agentType: string): string => {
  return physicsColors[agentType as keyof typeof physicsColors] || physicsColors.math;
};

export const getChartColor = (index: number): string => {
  return chartColors.primary[index % chartColors.primary.length];
};

export const getGradientColor = (index: number): string => {
  return chartColors.gradient[index % chartColors.gradient.length];
};

export const generateColorPalette = (count: number): string[] => {
  const colors = [];
  for (let i = 0; i < count; i++) {
    colors.push(getChartColor(i));
  }
  return colors;
};

// Default theme export
export const dashboardTheme = lightTheme;

export default { lightTheme, darkTheme, useTheme, dashboardTheme };