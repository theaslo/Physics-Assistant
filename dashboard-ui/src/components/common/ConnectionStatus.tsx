/**
 * Connection Status Component
 * Shows the real-time connection status and health information
 */

import React from 'react';
import {
  Box,
  Chip,
  Tooltip,
  Typography,
  Popover,
  Paper,
  Divider,
  useTheme,
} from '@mui/material';
import {
  Circle as CircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';

import { useRealtimeState } from '../../stores/dashboard-store';

// ============================================================================
// Types
// ============================================================================

interface ConnectionStatusProps {
  showDetails?: boolean;
  variant?: 'chip' | 'icon' | 'full';
}

// ============================================================================
// Main Component
// ============================================================================

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  showDetails = true,
  variant = 'chip',
}) => {
  const theme = useTheme();
  const realtimeState = useRealtimeState();
  const [anchorEl, setAnchorEl] = React.useState<HTMLElement | null>(null);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const getStatusConfig = () => {
    switch (realtimeState.connectionStatus) {
      case 'connected':
        return {
          color: theme.palette.success.main,
          label: 'Connected',
          icon: <CheckCircleIcon />,
          description: 'Real-time connection is active',
        };
      case 'connecting':
        return {
          color: theme.palette.warning.main,
          label: 'Connecting',
          icon: <WarningIcon />,
          description: 'Establishing connection...',
        };
      case 'disconnected':
        return {
          color: theme.palette.error.main,
          label: 'Disconnected',
          icon: <ErrorIcon />,
          description: 'Connection lost - data may be outdated',
        };
      case 'error':
        return {
          color: theme.palette.error.main,
          label: 'Error',
          icon: <ErrorIcon />,
          description: 'Connection error - check network',
        };
      default:
        return {
          color: theme.palette.grey[500],
          label: 'Unknown',
          icon: <CircleIcon />,
          description: 'Connection status unknown',
        };
    }
  };

  const getLastHeartbeatText = () => {
    if (!realtimeState.lastHeartbeat) {
      return 'No heartbeat received';
    }

    const lastHeartbeat = new Date(realtimeState.lastHeartbeat);
    const now = new Date();
    const diffMs = now.getTime() - lastHeartbeat.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);

    if (diffSeconds < 60) {
      return `${diffSeconds}s ago`;
    } else if (diffSeconds < 3600) {
      return `${Math.floor(diffSeconds / 60)}m ago`;
    } else {
      return lastHeartbeat.toLocaleTimeString();
    }
  };

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    if (showDetails) {
      setAnchorEl(event.currentTarget);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  // ============================================================================
  // Render Methods
  // ============================================================================

  const statusConfig = getStatusConfig();

  const renderStatusIcon = () => (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        color: statusConfig.color,
        cursor: showDetails ? 'pointer' : 'default',
      }}
      onClick={handleClick}
    >
      <CircleIcon sx={{ fontSize: 12, mr: 0.5 }} />
      {variant === 'full' && (
        <Typography variant="caption" sx={{ color: statusConfig.color }}>
          {statusConfig.label}
        </Typography>
      )}
    </Box>
  );

  const renderStatusChip = () => (
    <Chip
      icon={<CircleIcon sx={{ fontSize: '0.75rem' }} />}
      label={statusConfig.label}
      size="small"
      variant="outlined"
      onClick={showDetails ? handleClick : undefined}
      sx={{
        color: statusConfig.color,
        borderColor: statusConfig.color,
        cursor: showDetails ? 'pointer' : 'default',
        '& .MuiChip-icon': {
          color: statusConfig.color,
        },
      }}
    />
  );

  const renderPopoverContent = () => (
    <Paper sx={{ p: 2, maxWidth: 300 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Box sx={{ color: statusConfig.color, mr: 1 }}>
          {statusConfig.icon}
        </Box>
        <Typography variant="subtitle2" fontWeight="bold">
          Connection Status
        </Typography>
      </Box>

      <Typography variant="body2" color="text.secondary" gutterBottom>
        {statusConfig.description}
      </Typography>

      <Divider sx={{ my: 1.5 }} />

      {/* Connection Details */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Status:
          </Typography>
          <Typography variant="caption" sx={{ color: statusConfig.color }}>
            {statusConfig.label}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Last heartbeat:
          </Typography>
          <Typography variant="caption">
            {getLastHeartbeatText()}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Alerts:
          </Typography>
          <Typography variant="caption">
            {realtimeState.alerts.length} active
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="caption" color="text.secondary">
            Unread:
          </Typography>
          <Typography variant="caption" color="error.main">
            {realtimeState.unreadAlerts}
          </Typography>
        </Box>
      </Box>

      {/* Action Buttons */}
      {realtimeState.connectionStatus === 'disconnected' && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Connection lost. The dashboard will automatically attempt to reconnect.
          </Typography>
        </Box>
      )}
    </Paper>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <>
      <Tooltip title={showDetails ? 'Click for details' : statusConfig.description}>
        {variant === 'chip' ? renderStatusChip() : renderStatusIcon()}
      </Tooltip>

      {showDetails && (
        <Popover
          open={Boolean(anchorEl)}
          anchorEl={anchorEl}
          onClose={handleClose}
          anchorOrigin={{
            vertical: 'bottom',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'top',
            horizontal: 'right',
          }}
        >
          {renderPopoverContent()}
        </Popover>
      )}
    </>
  );
};

export default ConnectionStatus;