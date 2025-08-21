/**
 * Dashboard Sidebar Navigation Component
 * Provides navigation for different dashboard views and sections
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Chip,
  Paper,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  School as StudentsIcon,
  Class as ClassesIcon,
  Analytics as AnalyticsIcon,
  TrendingUp as TrendsIcon,
  Assessment as ReportsIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  Speed as PerformanceIcon,
  Group as GroupIcon,
  Person as PersonIcon,
  Timeline as TimelineIcon,
  BarChart as ChartIcon,
  CloudDownload as ExportIcon,
  Psychology as PredictiveIcon,
  AutoFixHigh as AutoInsightsIcon,
} from '@mui/icons-material';

import { useDashboardStore, useRealtimeState } from '../../stores/dashboard-store';

// ============================================================================
// Types
// ============================================================================

interface DashboardSidebarProps {
  onItemClick?: () => void;
}

interface NavigationItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  path: string;
  badge?: string | number;
  disabled?: boolean;
  children?: NavigationItem[];
}

// ============================================================================
// Navigation Configuration
// ============================================================================

const navigationItems: NavigationItem[] = [
  {
    id: 'overview',
    label: 'Dashboard Overview',
    icon: <DashboardIcon />,
    path: '/dashboard',
  },
  {
    id: 'analytics',
    label: 'Analytics',
    icon: <AnalyticsIcon />,
    path: '/analytics',
    children: [
      {
        id: 'performance',
        label: 'Performance Metrics',
        icon: <PerformanceIcon />,
        path: '/analytics/performance',
      },
      {
        id: 'trends',
        label: 'Trends & Insights',
        icon: <TrendsIcon />,
        path: '/analytics/trends',
      },
      {
        id: 'charts',
        label: 'Data Visualization',
        icon: <ChartIcon />,
        path: '/analytics/charts',
      },
      {
        id: 'advanced',
        label: 'Advanced Analytics',
        icon: <PredictiveIcon />,
        path: '/analytics/advanced',
        badge: 'NEW',
      },
    ],
  },
  {
    id: 'students',
    label: 'Student Analytics',
    icon: <StudentsIcon />,
    path: '/students',
    children: [
      {
        id: 'progress',
        label: 'Progress Tracking',
        icon: <TimelineIcon />,
        path: '/students/progress',
      },
      {
        id: 'individual',
        label: 'Individual Insights',
        icon: <PersonIcon />,
        path: '/students/individual',
      },
      {
        id: 'concepts',
        label: 'Concept Mastery',
        icon: <Assessment as any />,
        path: '/students/concepts',
      },
    ],
  },
  {
    id: 'classes',
    label: 'Class Management',
    icon: <ClassesIcon />,
    path: '/classes',
    children: [
      {
        id: 'overview',
        label: 'Class Overview',
        icon: <GroupIcon />,
        path: '/classes/overview',
      },
      {
        id: 'performance',
        label: 'Class Performance',
        icon: <BarChart as any />,
        path: '/classes/performance',
      },
    ],
  },
  {
    id: 'reports',
    label: 'Reports & Export',
    icon: <ReportsIcon />,
    path: '/reports',
    children: [
      {
        id: 'generate',
        label: 'Generate Reports',
        icon: <Assessment as any />,
        path: '/reports/generate',
      },
      {
        id: 'export',
        label: 'Data Export',
        icon: <ExportIcon />,
        path: '/reports/export',
      },
    ],
  },
];

const secondaryItems: NavigationItem[] = [
  {
    id: 'settings',
    label: 'Settings',
    icon: <SettingsIcon />,
    path: '/settings',
  },
  {
    id: 'help',
    label: 'Help & Support',
    icon: <HelpIcon />,
    path: '/help',
  },
];

// ============================================================================
// Main Component
// ============================================================================

const DashboardSidebar: React.FC<DashboardSidebarProps> = ({ onItemClick }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const { ui: { selectedView }, setSelectedView } = useDashboardStore();
  const realtimeState = useRealtimeState();

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleItemClick = (item: NavigationItem) => {
    const viewMap = {
      'overview': 'overview' as const,
      'students': 'students' as const,
      'classes': 'classes' as const,
      'analytics': 'analytics' as const,
    };

    if (viewMap[item.id]) {
      setSelectedView(viewMap[item.id]);
    }

    // Navigate to the item's path using React Router
    if (item.path) {
      navigate(item.path);
    }

    onItemClick?.();
  };

  // ============================================================================
  // Render Methods
  // ============================================================================

  const renderNavigationItem = (item: NavigationItem, level: number = 0) => {
    const isSelected = selectedView === item.id || 
      (item.children && item.children.some(child => selectedView === child.id));
    
    const isChildSelected = item.children?.some(child => selectedView === child.id);

    return (
      <React.Fragment key={item.id}>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => handleItemClick(item)}
            disabled={item.disabled}
            sx={{
              pl: 2 + level * 2,
              pr: 2,
              py: 1.5,
              borderRadius: 1,
              mx: 1,
              mb: 0.5,
              backgroundColor: isSelected
                ? alpha(theme.palette.primary.main, 0.1)
                : 'transparent',
              color: isSelected
                ? theme.palette.primary.main
                : theme.palette.text.primary,
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.05),
              },
              '&.Mui-disabled': {
                opacity: 0.5,
              },
            }}
          >
            <ListItemIcon
              sx={{
                color: 'inherit',
                minWidth: 40,
              }}
            >
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                variant: 'body2',
                fontWeight: isSelected ? 600 : 400,
              }}
            />
            {item.badge && (
              <Chip
                label={item.badge}
                size="small"
                color={isSelected ? 'primary' : 'default'}
                sx={{
                  height: 20,
                  fontSize: '0.75rem',
                }}
              />
            )}
          </ListItemButton>
        </ListItem>

        {/* Render children if expanded */}
        {item.children && (isSelected || isChildSelected) && (
          <Box>
            {item.children.map(child => renderNavigationItem(child, level + 1))}
          </Box>
        )}
      </React.Fragment>
    );
  };

  const renderConnectionStatus = () => (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        m: 1,
        backgroundColor: alpha(
          realtimeState.connected ? theme.palette.success.main : theme.palette.error.main,
          0.1
        ),
        border: `1px solid ${alpha(
          realtimeState.connected ? theme.palette.success.main : theme.palette.error.main,
          0.2
        )}`,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: realtimeState.connected 
              ? theme.palette.success.main 
              : theme.palette.error.main,
            mr: 1,
          }}
        />
        <Typography variant="caption" fontWeight="medium">
          {realtimeState.connected ? 'Connected' : 'Disconnected'}
        </Typography>
      </Box>
      <Typography variant="caption" color="text.secondary">
        {realtimeState.connected 
          ? 'Real-time updates active'
          : 'Connection lost - trying to reconnect'
        }
      </Typography>
    </Paper>
  );

  const renderStatsCard = () => (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        m: 1,
        backgroundColor: alpha(theme.palette.primary.main, 0.05),
        border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
      }}
    >
      <Typography variant="h6" gutterBottom>
        Quick Stats
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Active Sessions
          </Typography>
          <Typography variant="caption" fontWeight="medium">
            24
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Success Rate
          </Typography>
          <Typography variant="caption" fontWeight="medium" color="success.main">
            87%
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Avg Response
          </Typography>
          <Typography variant="caption" fontWeight="medium">
            245ms
          </Typography>
        </Box>
      </Box>
    </Paper>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, pb: 1 }}>
        <Typography variant="h6" fontWeight="bold" color="primary">
          Physics Assistant
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Analytics Dashboard
        </Typography>
      </Box>

      <Divider />

      {/* Main Navigation */}
      <Box sx={{ flex: 1, overflow: 'auto', py: 1 }}>
        <List dense>
          {navigationItems.map(item => renderNavigationItem(item))}
        </List>

        <Divider sx={{ my: 2 }} />

        {/* Secondary Navigation */}
        <List dense>
          {secondaryItems.map(item => renderNavigationItem(item))}
        </List>
      </Box>

      {/* Bottom Section */}
      <Box sx={{ p: 1 }}>
        {/* Quick Stats */}
        {renderStatsCard()}

        {/* Connection Status */}
        {renderConnectionStatus()}
      </Box>
    </Box>
  );
};

export default DashboardSidebar;