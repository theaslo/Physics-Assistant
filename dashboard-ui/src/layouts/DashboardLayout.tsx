/**
 * Main Dashboard Layout Component
 * Provides the overall structure with sidebar, header, and content area
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  Tooltip,
  useMediaQuery,
  useTheme,
  Divider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
  AccountCircle as AccountCircleIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  Refresh as RefreshIcon,
  CloudOff as OfflineIcon,
  Cloud as OnlineIcon,
} from '@mui/icons-material';

import { useDashboardStore, useRealtimeState } from '../stores/dashboard-store';
import DashboardSidebar from '../components/navigation/DashboardSidebar';
import AlertSnackbar from '../components/common/AlertSnackbar';
import ConnectionStatus from '../components/common/ConnectionStatus';

// ============================================================================
// Types
// ============================================================================

interface DashboardLayoutProps {
  children: React.ReactNode;
}

// ============================================================================
// Constants
// ============================================================================

const DRAWER_WIDTH = 280;
const MOBILE_DRAWER_WIDTH = 260;

// ============================================================================
// Main Component
// ============================================================================

const DashboardLayout: React.FC<DashboardLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Store state
  const {
    ui: { sidebarOpen, theme: themeMode, autoRefresh, refreshInterval },
    setSidebarOpen,
    setTheme,
    setAutoRefresh,
    clearErrors,
  } = useDashboardStore();

  const realtimeState = useRealtimeState();

  // Local state
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [notificationAnchor, setNotificationAnchor] = useState<null | HTMLElement>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  // ============================================================================
  // Auto-refresh Logic
  // ============================================================================

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      handleRefresh();
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleDrawerToggle = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleNotificationMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchor(event.currentTarget);
  };

  const handleNotificationMenuClose = () => {
    setNotificationAnchor(null);
  };

  const handleThemeToggle = () => {
    setTheme(themeMode === 'light' ? 'dark' : 'light');
  };

  const handleAutoRefreshToggle = () => {
    setAutoRefresh(!autoRefresh);
  };

  const handleRefresh = async () => {
    if (refreshing) return;
    
    setRefreshing(true);
    setLastRefresh(new Date());
    
    try {
      // Clear any previous errors
      clearErrors();
      
      // Trigger refresh of dashboard data
      // This would typically call the API to refresh data
      // For now, we'll just simulate a refresh
      await new Promise(resolve => setTimeout(resolve, 1000));
      
    } catch (error) {
      console.error('Failed to refresh dashboard:', error);
    } finally {
      setRefreshing(false);
    }
  };

  // ============================================================================
  // Computed Values
// ============================================================================

  const drawerWidth = isMobile ? MOBILE_DRAWER_WIDTH : DRAWER_WIDTH;
  const isMenuOpen = Boolean(anchorEl);
  const isNotificationMenuOpen = Boolean(notificationAnchor);

  // ============================================================================
  // Render Methods
  // ============================================================================

  const renderProfileMenu = () => (
    <Menu
      anchorEl={anchorEl}
      anchorOrigin={{
        vertical: 'bottom',
        horizontal: 'right',
      }}
      keepMounted
      transformOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      open={isMenuOpen}
      onClose={handleProfileMenuClose}
    >
      <MenuItem onClick={handleProfileMenuClose}>
        <AccountCircleIcon sx={{ mr: 1 }} />
        Profile
      </MenuItem>
      <MenuItem onClick={handleProfileMenuClose}>
        <SettingsIcon sx={{ mr: 1 }} />
        Settings
      </MenuItem>
      <Divider />
      <MenuItem>
        <FormControlLabel
          control={
            <Switch
              checked={themeMode === 'dark'}
              onChange={handleThemeToggle}
              size="small"
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {themeMode === 'dark' ? <DarkModeIcon sx={{ mr: 1 }} /> : <LightModeIcon sx={{ mr: 1 }} />}
              Dark Mode
            </Box>
          }
        />
      </MenuItem>
      <MenuItem>
        <FormControlLabel
          control={
            <Switch
              checked={autoRefresh}
              onChange={handleAutoRefreshToggle}
              size="small"
            />
          }
          label="Auto Refresh"
        />
      </MenuItem>
    </Menu>
  );

  const renderNotificationMenu = () => (
    <Menu
      anchorEl={notificationAnchor}
      anchorOrigin={{
        vertical: 'bottom',
        horizontal: 'right',
      }}
      keepMounted
      transformOrigin={{
        vertical: 'top',
        horizontal: 'right',
      }}
      open={isNotificationMenuOpen}
      onClose={handleNotificationMenuClose}
      sx={{ mt: 1 }}
    >
      {realtimeState.alerts.length === 0 ? (
        <MenuItem disabled>
          <Typography variant="body2" color="text.secondary">
            No new notifications
          </Typography>
        </MenuItem>
      ) : (
        realtimeState.alerts.slice(0, 5).map((alert, index) => (
          <MenuItem key={`${alert.timestamp}-${index}`} onClick={handleNotificationMenuClose}>
            <Box>
              <Typography variant="body2" fontWeight="medium">
                {alert.title}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {alert.message}
              </Typography>
              <Typography variant="caption" color="text.secondary" display="block">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </Typography>
            </Box>
          </MenuItem>
        ))
      )}
      {realtimeState.alerts.length > 5 && (
        <MenuItem onClick={handleNotificationMenuClose}>
          <Typography variant="body2" color="primary">
            View all notifications
          </Typography>
        </MenuItem>
      )}
    </Menu>
  );

  const renderAppBar = () => (
    <AppBar
      position="fixed"
      sx={{
        width: { md: sidebarOpen ? `calc(100% - ${drawerWidth}px)` : '100%' },
        ml: { md: sidebarOpen ? `${drawerWidth}px` : 0 },
        transition: theme.transitions.create(['width', 'margin'], {
          easing: theme.transitions.easing.sharp,
          duration: theme.transitions.duration.leavingScreen,
        }),
      }}
    >
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="toggle drawer"
          edge="start"
          onClick={handleDrawerToggle}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>

        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
          Physics Assistant Dashboard
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Connection Status */}
          <ConnectionStatus />

          {/* Refresh Button */}
          <Tooltip title={`Last refresh: ${lastRefresh?.toLocaleTimeString() || 'Never'}`}>
            <IconButton
              color="inherit"
              onClick={handleRefresh}
              disabled={refreshing}
              sx={{
                animation: refreshing ? 'spin 1s linear infinite' : 'none',
                '@keyframes spin': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' },
                },
              }}
            >
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton
              color="inherit"
              onClick={handleNotificationMenuOpen}
            >
              <Badge badgeContent={realtimeState.unreadAlerts} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* Profile Menu */}
          <Tooltip title="User menu">
            <IconButton
              onClick={handleProfileMenuOpen}
              color="inherit"
              sx={{ p: 0.5 }}
            >
              <Avatar sx={{ width: 32, height: 32 }}>
                <AccountCircleIcon />
              </Avatar>
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
  );

  const renderDrawer = () => (
    <Box
      component="nav"
      sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
    >
      <Drawer
        variant={isMobile ? 'temporary' : 'persistent'}
        open={sidebarOpen}
        onClose={handleDrawerToggle}
        ModalProps={{
          keepMounted: true, // Better mobile performance
        }}
        sx={{
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
            borderRight: `1px solid ${theme.palette.divider}`,
          },
        }}
      >
        <DashboardSidebar onItemClick={isMobile ? handleDrawerToggle : undefined} />
      </Drawer>
    </Box>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      {renderAppBar()}

      {/* Drawer */}
      {renderDrawer()}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: sidebarOpen ? `calc(100% - ${drawerWidth}px)` : '100%' },
          ml: { md: sidebarOpen ? 0 : `-${drawerWidth}px` },
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
          }),
          minHeight: '100vh',
          backgroundColor: theme.palette.background.default,
        }}
      >
        {/* Toolbar spacer */}
        <Toolbar />
        
        {/* Page Content */}
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      </Box>

      {/* Menus */}
      {renderProfileMenu()}
      {renderNotificationMenu()}

      {/* Alert Snackbar */}
      <AlertSnackbar />
    </Box>
  );
};

export default DashboardLayout;