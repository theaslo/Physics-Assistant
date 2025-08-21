import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  Badge,
  Divider,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  Notifications as NotificationsIcon,
  TrendingUp as TrendingUpIcon,
  Person as PersonIcon,
  School as SchoolIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

// Components
import TimeSeriesChart from '../components/charts/TimeSeriesChart';
import MetricCard from '../components/widgets/MetricCard';
import RealTimeActivityFeed from '../components/widgets/RealTimeActivityFeed';
import LiveStudentProgress from '../components/widgets/LiveStudentProgress';
import SystemHealthIndicator from '../components/widgets/SystemHealthIndicator';

// Services and Types
import { useWebSocket } from '../hooks/useWebSocket';
import { useDashboardStore } from '../stores/dashboard-store';
import {
  WebSocketMessage,
  MetricsUpdate,
  Alert as AlertType,
  StudentProgressUpdate,
} from '../types/api';

// Utilities
import { formatNumber, formatPercentage, formatDuration } from '../utils/formatters';

const RealTimeDashboard: React.FC = () => {
  const { addAlert, clearAlerts } = useDashboardStore();
  
  // Local state
  const [isLive, setIsLive] = useState(true);
  const [alerts, setAlerts] = useState<AlertType[]>([]);
  const [realtimeMetrics, setRealtimeMetrics] = useState({
    activeUsers: 0,
    totalInteractions: 0,
    systemLoad: 0,
    responseTime: 0,
    successRate: 0,
  });
  const [activityFeed, setActivityFeed] = useState<any[]>([]);
  const [studentUpdates, setStudentUpdates] = useState<StudentProgressUpdate[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<any[]>([]);
  
  // Refs for managing data retention
  const maxDataPoints = useRef(50);
  const maxActivityItems = useRef(20);
  const maxAlerts = useRef(10);

  // WebSocket connection
  const { 
    isConnected, 
    lastMessage, 
    connectionError,
    connect,
    disconnect 
  } = useWebSocket('ws://localhost:8002/ws/realtime');

  // Effects
  useEffect(() => {
    if (isLive && !isConnected) {
      connect();
    } else if (!isLive && isConnected) {
      disconnect();
    }
  }, [isLive, isConnected, connect, disconnect]);

  useEffect(() => {
    if (lastMessage) {
      handleWebSocketMessage(lastMessage);
    }
  }, [lastMessage]);

  // Initialize with mock data
  useEffect(() => {
    initializeMockData();
    
    // Simulate real-time updates when WebSocket is not available
    if (!isConnected && isLive) {
      const interval = setInterval(() => {
        simulateRealtimeUpdate();
      }, 2000);
      
      return () => clearInterval(interval);
    }
  }, [isConnected, isLive]);

  // Handlers
  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'metrics_update':
        handleMetricsUpdate(message.data as MetricsUpdate);
        break;
      case 'alert':
        handleNewAlert(message.data as AlertType);
        break;
      case 'student_progress':
        handleStudentProgress(message.data as StudentProgressUpdate);
        break;
      case 'heartbeat':
        // Handle heartbeat to maintain connection
        break;
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const handleMetricsUpdate = (update: MetricsUpdate) => {
    setRealtimeMetrics(prev => ({
      ...prev,
      activeUsers: update.active_users_count,
      totalInteractions: update.recent_interactions,
      ...update.summary,
    }));
    
    // Add to time series
    const newDataPoint = {
      timestamp: new Date().toISOString(),
      activeUsers: update.active_users_count,
      interactions: update.recent_interactions,
      value: update.active_users_count, // Default value for single series
    };
    
    setTimeSeriesData(prev => {
      const updated = [...prev, newDataPoint];
      return updated.slice(-maxDataPoints.current);
    });
  };

  const handleNewAlert = (alert: AlertType) => {
    setAlerts(prev => {
      const updated = [alert, ...prev];
      return updated.slice(0, maxAlerts.current);
    });
    
    addAlert(alert.message, alert.severity);
  };

  const handleStudentProgress = (update: StudentProgressUpdate) => {
    setStudentUpdates(prev => {
      const updated = [update, ...prev];
      return updated.slice(0, 10); // Keep last 10 updates
    });
    
    // Add to activity feed
    const activityItem = {
      id: Date.now().toString(),
      timestamp: new Date().toISOString(),
      type: 'student_progress',
      message: `Student ${update.user_id} made progress`,
      data: update,
    };
    
    setActivityFeed(prev => {
      const updated = [activityItem, ...prev];
      return updated.slice(0, maxActivityItems.current);
    });
  };

  const toggleLiveMode = () => {
    setIsLive(!isLive);
  };

  const handleRefresh = () => {
    if (isConnected) {
      // Request fresh data from WebSocket
      // In a real implementation, this would send a message to the server
    } else {
      simulateRealtimeUpdate();
    }
  };

  const clearAllAlerts = () => {
    setAlerts([]);
    clearAlerts();
  };

  // Mock data and simulation functions
  const initializeMockData = () => {
    const now = new Date();
    const initialData = [];
    
    for (let i = 20; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 30000); // 30 second intervals
      initialData.push({
        timestamp: timestamp.toISOString(),
        activeUsers: 15 + Math.floor(Math.random() * 10),
        interactions: 5 + Math.floor(Math.random() * 15),
        value: 15 + Math.floor(Math.random() * 10),
      });
    }
    
    setTimeSeriesData(initialData);
    setRealtimeMetrics({
      activeUsers: 23,
      totalInteractions: 147,
      systemLoad: 45.2,
      responseTime: 125,
      successRate: 96.8,
    });
  };

  const simulateRealtimeUpdate = () => {
    // Simulate metrics update
    const metricsUpdate = {
      activeUsers: Math.max(10, realtimeMetrics.activeUsers + Math.floor(Math.random() * 6) - 3),
      totalInteractions: realtimeMetrics.totalInteractions + Math.floor(Math.random() * 5),
      systemLoad: Math.max(0, Math.min(100, realtimeMetrics.systemLoad + (Math.random() - 0.5) * 5)),
      responseTime: Math.max(50, realtimeMetrics.responseTime + Math.floor((Math.random() - 0.5) * 20)),
      successRate: Math.max(85, Math.min(100, realtimeMetrics.successRate + (Math.random() - 0.5) * 2)),
    };
    
    setRealtimeMetrics(metricsUpdate);
    
    // Add new data point
    const newDataPoint = {
      timestamp: new Date().toISOString(),
      activeUsers: metricsUpdate.activeUsers,
      interactions: Math.floor(Math.random() * 8),
      value: metricsUpdate.activeUsers,
    };
    
    setTimeSeriesData(prev => {
      const updated = [...prev, newDataPoint];
      return updated.slice(-maxDataPoints.current);
    });
    
    // Occasionally add alerts
    if (Math.random() < 0.1) {
      const alertTypes: AlertType['severity'][] = ['info', 'warning', 'error', 'success'];
      const randomAlert: AlertType = {
        severity: alertTypes[Math.floor(Math.random() * alertTypes.length)],
        title: 'System Update',
        message: `Random system event detected at ${new Date().toLocaleTimeString()}`,
        timestamp: new Date().toISOString(),
      };
      
      handleNewAlert(randomAlert);
    }
  };

  const getAlertIcon = (severity: AlertType['severity']) => {
    switch (severity) {
      case 'error': return <ErrorIcon color="error" />;
      case 'warning': return <WarningIcon color="warning" />;
      case 'success': return <CheckCircleIcon color="success" />;
      default: return <InfoIcon color="info" />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Real-Time Dashboard
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={isLive}
                onChange={toggleLiveMode}
                color="primary"
              />
            }
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {isLive ? <PlayIcon color="success" /> : <PauseIcon />}
                Live Mode
              </Box>
            }
          />
          
          <SystemHealthIndicator
            status={isConnected ? 'healthy' : 'down'}
            label={isConnected ? 'Connected' : 'Disconnected'}
          />
          
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Connection Status */}
      {connectionError && (
        <Alert severity="error" sx={{ mb: 2 }}>
          WebSocket Connection Error: {connectionError}. Displaying simulated data.
        </Alert>
      )}

      {/* Real-time Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2.4}>
          <MetricCard
            title="Active Users"
            value={realtimeMetrics.activeUsers}
            unit=""
            icon={<PersonIcon />}
            trend={2.1}
            color="primary"
            isLive={isLive}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2.4}>
          <MetricCard
            title="Total Interactions"
            value={realtimeMetrics.totalInteractions}
            unit=""
            icon={<SchoolIcon />}
            trend={5.8}
            color="success"
            isLive={isLive}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2.4}>
          <MetricCard
            title="System Load"
            value={realtimeMetrics.systemLoad}
            unit="%"
            icon={<SpeedIcon />}
            trend={-1.2}
            color="warning"
            isLive={isLive}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2.4}>
          <MetricCard
            title="Response Time"
            value={realtimeMetrics.responseTime}
            unit="ms"
            icon={<SpeedIcon />}
            trend={-3.5}
            color="info"
            isLive={isLive}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={2.4}>
          <MetricCard
            title="Success Rate"
            value={realtimeMetrics.successRate}
            unit="%"
            icon={<CheckCircleIcon />}
            trend={0.8}
            color="success"
            isLive={isLive}
          />
        </Grid>
      </Grid>

      {/* Real-time Charts and Activity */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Live Activity Metrics
              </Typography>
              <Chip
                label={isLive ? 'LIVE' : 'PAUSED'}
                color={isLive ? 'success' : 'default'}
                variant={isLive ? 'filled' : 'outlined'}
                size="small"
              />
            </Box>
            <TimeSeriesChart
              data={timeSeriesData}
              xKey="timestamp"
              yKey="activeUsers"
              title="Active Users"
              color="#1976d2"
              isRealTime={isLive}
              showMultipleLines={true}
            />
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Live Alerts
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Badge badgeContent={alerts.length} color="error">
                  <NotificationsIcon />
                </Badge>
                {alerts.length > 0 && (
                  <Button size="small" onClick={clearAllAlerts}>
                    Clear All
                  </Button>
                )}
              </Box>
            </Box>
            
            <List sx={{ maxHeight: 320, overflow: 'auto' }}>
              {alerts.length === 0 ? (
                <ListItem>
                  <ListItemText primary="No alerts" secondary="System running smoothly" />
                </ListItem>
              ) : (
                alerts.map((alert, index) => (
                  <React.Fragment key={index}>
                    <ListItem>
                      <ListItemIcon>
                        {getAlertIcon(alert.severity)}
                      </ListItemIcon>
                      <ListItemText
                        primary={alert.title}
                        secondary={
                          <Box>
                            <Typography variant="body2" color="textSecondary">
                              {alert.message}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              {new Date(alert.timestamp).toLocaleTimeString()}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                    {index < alerts.length - 1 && <Divider />}
                  </React.Fragment>
                ))
              )}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Activity Feed and Student Progress */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Real-Time Activity Feed
            </Typography>
            <RealTimeActivityFeed activities={activityFeed} isLive={isLive} />
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Live Student Progress
            </Typography>
            <LiveStudentProgress updates={studentUpdates} isLive={isLive} />
          </Paper>
        </Grid>
        
        {/* System Performance Summary */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Real-Time System Summary
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Connection Status
                    </Typography>
                    <Typography variant="h6" color={isConnected ? 'success.main' : 'error.main'}>
                      {isConnected ? 'Connected' : 'Disconnected'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      WebSocket connection to real-time server
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Data Points
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {timeSeriesData.length}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Real-time metrics collected
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Update Frequency
                    </Typography>
                    <Typography variant="h6" color="info.main">
                      {isLive ? '2s' : 'Paused'}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Real-time refresh rate
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Last Update
                    </Typography>
                    <Typography variant="h6" color="text.primary">
                      {new Date().toLocaleTimeString()}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Most recent data refresh
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RealTimeDashboard;