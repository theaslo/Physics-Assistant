/**
 * Alert Snackbar Component
 * Displays real-time alerts and notifications
 */

import React, { useState, useEffect } from 'react';
import {
  Snackbar,
  Alert,
  AlertTitle,
  IconButton,
  Slide,
  SlideProps,
} from '@mui/material';
import {
  Close as CloseIcon,
} from '@mui/icons-material';

import { useDashboardStore, useRealtimeState } from '../../stores/dashboard-store';
import { Alert as AlertType } from '../../types/api';

// ============================================================================
// Types
// ============================================================================

interface AlertSnackbarProps {
  maxVisible?: number;
  autoHideDuration?: number;
}

interface AlertItem extends AlertType {
  id: string;
  visible: boolean;
}

// ============================================================================
// Transition Component
// ============================================================================

function SlideTransition(props: SlideProps) {
  return <Slide {...props} direction="up" />;
}

// ============================================================================
// Main Component
// ============================================================================

const AlertSnackbar: React.FC<AlertSnackbarProps> = ({
  maxVisible = 3,
  autoHideDuration = 6000,
}) => {
  const { removeAlert } = useDashboardStore();
  const realtimeState = useRealtimeState();
  
  // Local state for managing visible alerts
  const [visibleAlerts, setVisibleAlerts] = useState<AlertItem[]>([]);

  // ============================================================================
  // Effects
  // ============================================================================

  // Update visible alerts when store alerts change
  useEffect(() => {
    const newAlerts = realtimeState.alerts
      .slice(0, maxVisible)
      .map((alert, index) => ({
        ...alert,
        id: `${alert.timestamp}-${index}`,
        visible: true,
      }));

    setVisibleAlerts(newAlerts);
  }, [realtimeState.alerts, maxVisible]);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleClose = (alertId: string) => {
    // Mark alert as not visible
    setVisibleAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, visible: false } : alert
      )
    );

    // Remove from store after animation
    setTimeout(() => {
      removeAlert(alertId);
    }, 300);
  };

  const handleExited = (alertId: string) => {
    // Remove from local state after transition
    setVisibleAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  // ============================================================================
  // Render Methods
  // ============================================================================

  const getSeverityIcon = (severity: AlertType['severity']) => {
    // Icons are automatically handled by MUI Alert component
    return undefined;
  };

  const renderAlert = (alert: AlertItem, index: number) => (
    <Snackbar
      key={alert.id}
      open={alert.visible}
      autoHideDuration={autoHideDuration}
      onClose={() => handleClose(alert.id)}
      onExited={() => handleExited(alert.id)}
      TransitionComponent={SlideTransition}
      anchorOrigin={{ 
        vertical: 'bottom', 
        horizontal: 'right' 
      }}
      sx={{
        position: 'fixed',
        bottom: 16 + (index * 80), // Stack alerts vertically
        right: 16,
        zIndex: (theme) => theme.zIndex.snackbar + index,
      }}
    >
      <Alert
        severity={alert.severity}
        onClose={() => handleClose(alert.id)}
        variant="filled"
        sx={{
          minWidth: 300,
          maxWidth: 400,
          '& .MuiAlert-message': {
            overflow: 'hidden',
          },
        }}
        action={
          <IconButton
            aria-label="close"
            color="inherit"
            size="small"
            onClick={() => handleClose(alert.id)}
          >
            <CloseIcon fontSize="inherit" />
          </IconButton>
        }
      >
        {alert.title && (
          <AlertTitle>
            {alert.title}
          </AlertTitle>
        )}
        {alert.message}
        {alert.entity_id && (
          <div style={{ fontSize: '0.75rem', marginTop: 4, opacity: 0.8 }}>
            Entity: {alert.entity_id}
          </div>
        )}
      </Alert>
    </Snackbar>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  if (visibleAlerts.length === 0) {
    return null;
  }

  return (
    <>
      {visibleAlerts.map((alert, index) => renderAlert(alert, index))}
    </>
  );
};

export default AlertSnackbar;