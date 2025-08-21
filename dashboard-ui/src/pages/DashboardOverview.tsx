/**
 * Dashboard Overview Page
 * Main dashboard view with key metrics, charts, and real-time updates
 */

import React, { useEffect, useState } from 'react';
import {
  Grid,
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Button,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  TrendingUp as TrendingUpIcon,
  Group as GroupIcon,
  Speed as SpeedIcon,
  CheckCircle as SuccessIcon,
  Timeline as TimelineIcon,
  School as SchoolIcon,
  Assignment as AssignmentIcon,
  Psychology as BrainIcon,
} from '@mui/icons-material';

import { useDashboardStore, useSummaryData } from '../stores/dashboard-store';
import { apiClient } from '../services/api-client';
import { useWebSocket } from '../services/websocket-client';
import MetricCard from '../components/widgets/MetricCard';
import TimeSeriesChart from '../components/charts/TimeSeriesChart';
import BarChart from '../components/charts/BarChart';
import { TimeSeriesDataPoint } from '../types/api';

// ============================================================================
// Types
// ============================================================================

interface DashboardOverviewProps {
  // Add any props if needed
}

// ============================================================================
// Main Component
// ============================================================================

const DashboardOverview: React.FC<DashboardOverviewProps> = () => {
  const theme = useTheme();
  
  // Store state
  const {
    loading,
    errors,
    ui: { selectedTimeRange, autoRefresh },
    setLoading,
    setError,
    setSummary,
    setTimeSeries,
    updateLastUpdated,
    setConnectionStatus,
    addAlert,
  } = useDashboardStore();

  const summaryData = useSummaryData();

  // Local state
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesDataPoint[]>([]);
  const [agentData, setAgentData] = useState<any[]>([]);
  const [refreshing, setRefreshing] = useState(false);

  // ============================================================================
  // WebSocket Integration
  // ============================================================================

  const { isConnected, send } = useWebSocket({
    url: apiClient.createWebSocketURL('/dashboard/ws', 'dashboard_overview'),
    enabled: true,
    callbacks: {
      onOpen: () => {
        setConnectionStatus('connected');
        console.log('Dashboard WebSocket connected');
      },
      onClose: () => {
        setConnectionStatus('disconnected');
        console.log('Dashboard WebSocket disconnected');
      },
      onError: (error) => {
        setConnectionStatus('error');
        console.error('Dashboard WebSocket error:', error);
      },
      onMetricsUpdate: (data) => {
        console.log('Received metrics update:', data);
        // Update summary data with real-time metrics
        if (summaryData) {
          setSummary({
            ...summaryData,
            summary: {
              ...summaryData.summary,
              ...data.summary,
            },
          });
        }
      },
      onAlert: (alert) => {
        addAlert(alert);
      },
    },
  });

  // ============================================================================
  // Data Fetching
  // ============================================================================

  const fetchDashboardData = async () => {
    try {
      setLoading('summary', true);
      setError('summary', null);

      // Fetch dashboard summary
      const summary = await apiClient.getDashboardSummary({
        preset: selectedTimeRange,
      });
      setSummary(summary);

      // Fetch time series data
      setLoading('timeSeries', true);
      const timeSeries = await apiClient.getTimeSeriesData({
        metrics: ['interaction_count', 'success_rate', 'avg_response_time'],
        time_range: { preset: selectedTimeRange },
        granularity: selectedTimeRange === '1h' ? '5m' : selectedTimeRange === '24h' ? '1h' : '1d',
      });
      setTimeSeries(timeSeries);
      setTimeSeriesData(timeSeries.data);

      // Process agent breakdown data
      if (summary.agent_breakdown) {
        setAgentData(summary.agent_breakdown.map(item => ({
          name: item.agent_type,
          value: item.interaction_count,
          agent_type: item.agent_type,
        })));
      }

      updateLastUpdated();
    } catch (error: any) {
      console.error('Failed to fetch dashboard data:', error);
      setError('summary', error.detail || error.message || 'Failed to load data');
    } finally {
      setLoading('summary', false);
      setLoading('timeSeries', false);
    }
  };

  const handleRefresh = async () => {
    if (refreshing) return;
    
    setRefreshing(true);
    try {
      await fetchDashboardData();
    } finally {
      setRefreshing(false);
    }
  };

  // ============================================================================
  // Effects
  // ============================================================================

  useEffect(() => {
    fetchDashboardData();
  }, [selectedTimeRange]);

  // Auto-refresh logic
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchDashboardData();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, selectedTimeRange]);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const metrics = React.useMemo(() => {
    if (!summaryData) return [];

    const { summary } = summaryData;
    
    return [
      {
        title: 'Total Interactions',
        value: summary.total_interactions,
        icon: <TimelineIcon />,
        color: 'primary' as const,
        trend: {
          value: 12.5,
          label: '+12.5%',
          direction: 'up' as const,
        },
      },
      {
        title: 'Active Users',
        value: summary.active_users,
        icon: <GroupIcon />,
        color: 'info' as const,
        trend: {
          value: 8.3,
          label: '+8.3%',
          direction: 'up' as const,
        },
      },
      {
        title: 'Success Rate',
        value: `${(summary.success_rate * 100).toFixed(1)}%`,
        icon: <SuccessIcon />,
        color: summary.success_rate >= 0.8 ? 'success' : 'warning' as const,
        trend: {
          value: 2.1,
          label: '+2.1%',
          direction: 'up' as const,
        },
      },
      {
        title: 'Avg Response Time',
        value: `${summary.avg_response_time.toFixed(0)}ms`,
        icon: <SpeedIcon />,
        color: summary.avg_response_time <= 300 ? 'success' : 'warning' as const,
        trend: {
          value: -5.2,
          label: '-5.2%',
          direction: 'down' as const,
        },
      },
    ];
  }, [summaryData]);

  // ============================================================================
  // Render Methods
  // ============================================================================

  const renderHeader = () => (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Dashboard Overview
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time analytics for Physics Assistant platform
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Chip
            icon={<div style={{ 
              width: 8, 
              height: 8, 
              borderRadius: '50%', 
              backgroundColor: isConnected ? theme.palette.success.main : theme.palette.error.main 
            }} />}
            label={isConnected ? 'Live' : 'Offline'}
            size="small"
            variant="outlined"
            sx={{ 
              color: isConnected ? theme.palette.success.main : theme.palette.error.main,
              borderColor: isConnected ? theme.palette.success.main : theme.palette.error.main,
            }}
          />
          
          <Tooltip title="Refresh data">
            <IconButton
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
        </Box>
      </Box>
    </Box>
  );

  const renderMetricCards = () => (
    <Grid container spacing={3} sx={{ mb: 3 }}>
      {metrics.map((metric, index) => (
        <Grid item xs={12} sm={6} md={3} key={metric.title}>
          <MetricCard
            title={metric.title}
            value={metric.value}
            icon={metric.icon}
            color={metric.color}
            trend={metric.trend}
            loading={loading.summary}
            error={errors.summary}
          />
        </Grid>
      ))}
    </Grid>
  );

  const renderCharts = () => (
    <Grid container spacing={3} sx={{ mb: 3 }}>
      {/* Time Series Chart */}
      <Grid item xs={12} lg={8}>
        <TimeSeriesChart
          data={timeSeriesData}
          metrics={['interaction_count', 'success_rate', 'avg_response_time']}
          title="Performance Metrics Over Time"
          height={400}
          loading={loading.timeSeries}
          error={errors.timeSeries}
          showBrush={selectedTimeRange !== '1h'}
          onExport={() => {
            // TODO: Implement export functionality
            console.log('Export time series data');
          }}
        />
      </Grid>

      {/* Agent Usage Chart */}
      <Grid item xs={12} lg={4}>
        <BarChart
          data={agentData}
          dataKey="value"
          categoryKey="name"
          title="Agent Usage"
          height={400}
          colorMode="physics"
          loading={loading.summary}
          error={errors.summary}
          onBarClick={(data) => {
            console.log('Clicked agent:', data);
          }}
          onExport={() => {
            console.log('Export agent data');
          }}
        />
      </Grid>
    </Grid>
  );

  const renderQuickStats = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <SchoolIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Student Activity
            </Typography>
            
            {summaryData && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Students Active Today:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {summaryData.summary.active_users}
                  </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Most Used Agent:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {agentData.length > 0 ? agentData[0].name : 'N/A'}
                  </Typography>
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" color="text.secondary">
                    Agents Available:
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {summaryData.summary.agents_used}
                  </Typography>
                </Box>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <BrainIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              System Health
            </Typography>
            
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  Connection Status:
                </Typography>
                <Chip
                  label={isConnected ? 'Connected' : 'Disconnected'}
                  size="small"
                  color={isConnected ? 'success' : 'error'}
                  variant="outlined"
                />
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Last Update:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {summaryData?.cache_info.generated_at 
                    ? new Date(summaryData.cache_info.generated_at).toLocaleTimeString()
                    : 'Never'
                  }
                </Typography>
              </Box>
              
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">
                  Auto Refresh:
                </Typography>
                <Typography variant="body2" fontWeight="medium">
                  {autoRefresh ? 'Enabled' : 'Disabled'}
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <Box>
      {renderHeader()}
      {renderMetricCards()}
      {renderCharts()}
      {renderQuickStats()}
    </Box>
  );
};

export default DashboardOverview;