import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  CircularProgress,
  Alert,
  Tooltip,
  IconButton,
  LinearProgress,
  Chip,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Cached as CachedIcon,
} from '@mui/icons-material';
import { useQuery, useQueryClient } from '@tanstack/react-query';

// Components
import TimeSeriesChart from '../components/charts/TimeSeriesChart';
import BarChart from '../components/charts/BarChart';
import MetricCard from '../components/widgets/MetricCard';
import SystemHealthIndicator from '../components/widgets/SystemHealthIndicator';
import CacheMetricsCard from '../components/widgets/CacheMetricsCard';

// Services and Types
import { apiClient } from '../services/api-client';
import { useDashboardStore } from '../stores/dashboard-store';
import {
  HealthCheck,
  CacheStatistics,
  TimeSeriesRequest,
  ServiceStatus,
} from '../types/api';

// Utilities
import { formatBytes, formatPercentage, formatNumber, formatDuration } from '../utils/formatters';

const SystemMetrics: React.FC = () => {
  const queryClient = useQueryClient();
  const { filters, setFilters, loading, error, setLoading, setError } = useDashboardStore();
  
  // Local state
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000); // 5 seconds
  const [exportLoading, setExportLoading] = useState(false);

  // Queries
  const { data: healthCheck, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ['health-check'],
    queryFn: () => apiClient.getHealthCheck(),
    refetchInterval: autoRefresh ? refreshInterval : false,
  });

  const { data: cacheStats, isLoading: cacheLoading, error: cacheError } = useQuery({
    queryKey: ['cache-statistics'],
    queryFn: () => apiClient.getCacheStatistics(),
    refetchInterval: autoRefresh ? refreshInterval : false,
  });

  // Mock system metrics for demonstration
  const mockSystemMetrics = {
    cpu_usage: 45.2,
    memory_usage: 68.7,
    disk_usage: 34.1,
    network_latency: 12.5,
    active_connections: 147,
    requests_per_second: 23.8,
    error_rate: 0.03,
    uptime: 99.97,
  };

  // Handlers
  const handleRefresh = async () => {
    setLoading(true);
    try {
      await queryClient.invalidateQueries({ queryKey: ['health-check'] });
      await queryClient.invalidateQueries({ queryKey: ['cache-statistics'] });
    } catch (err) {
      setError('Failed to refresh system metrics');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    setExportLoading(true);
    try {
      // Export system metrics and health data
      const exportData = {
        health_check: healthCheck,
        cache_statistics: cacheStats,
        system_metrics: mockSystemMetrics,
        timestamp: new Date().toISOString(),
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json',
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `system-metrics-${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to export data');
    } finally {
      setExportLoading(false);
    }
  };

  const handleCacheInvalidation = async (pattern?: string) => {
    try {
      await apiClient.invalidateCache(pattern);
      await queryClient.invalidateQueries({ queryKey: ['cache-statistics'] });
    } catch (err) {
      setError('Failed to invalidate cache');
    }
  };

  const handleCacheWarming = async () => {
    try {
      await apiClient.warmCache();
      await queryClient.invalidateQueries({ queryKey: ['cache-statistics'] });
    } catch (err) {
      setError('Failed to warm cache');
    }
  };

  // Helper functions
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'down': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleIcon />;
      case 'degraded': return <WarningIcon />;
      case 'down': return <ErrorIcon />;
      default: return <CheckCircleIcon />;
    }
  };

  // Render loading state
  if (healthLoading || cacheLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading system metrics...
        </Typography>
      </Box>
    );
  }

  // Render error state
  if (healthError || cacheError || error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Error loading system data: {healthError?.message || cacheError?.message || error}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          System Metrics
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel>Interval</InputLabel>
            <Select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(e.target.value as number)}
              label="Interval"
              disabled={!autoRefresh}
            >
              <MenuItem value={5000}>5s</MenuItem>
              <MenuItem value={10000}>10s</MenuItem>
              <MenuItem value={30000}>30s</MenuItem>
              <MenuItem value={60000}>1m</MenuItem>
            </Select>
          </FormControl>
          
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={exportLoading}
          >
            {exportLoading ? 'Exporting...' : 'Export'}
          </Button>
        </Box>
      </Box>

      {/* System Health Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Health Status
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={2}>
                <SystemHealthIndicator
                  status={healthCheck?.status || 'healthy'}
                  label="Overall Status"
                  size="large"
                />
              </Grid>
              
              <Grid item xs={12} md={10}>
                <Grid container spacing={2}>
                  {healthCheck?.services && Object.entries(healthCheck.services).map(([service, status]) => (
                    <Grid item xs={6} sm={3} key={service}>
                      <Card variant="outlined">
                        <CardContent sx={{ textAlign: 'center', py: 1 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                            {getStatusIcon(status)}
                          </Box>
                          <Typography variant="body2" color="textSecondary">
                            {service.replace('_', ' ').toUpperCase()}
                          </Typography>
                          <Chip
                            label={status}
                            size="small"
                            color={getStatusColor(status) as any}
                            variant="outlined"
                          />
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      {/* Key System Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="CPU Usage"
            value={mockSystemMetrics.cpu_usage}
            unit="%"
            icon={<SpeedIcon />}
            trend={-2.1}
            color="primary"
            showProgress={true}
            progressValue={mockSystemMetrics.cpu_usage}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Memory Usage"
            value={mockSystemMetrics.memory_usage}
            unit="%"
            icon={<MemoryIcon />}
            trend={1.5}
            color="warning"
            showProgress={true}
            progressValue={mockSystemMetrics.memory_usage}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Disk Usage"
            value={mockSystemMetrics.disk_usage}
            unit="%"
            icon={<StorageIcon />}
            trend={0.8}
            color="success"
            showProgress={true}
            progressValue={mockSystemMetrics.disk_usage}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Network Latency"
            value={mockSystemMetrics.network_latency}
            unit="ms"
            icon={<NetworkIcon />}
            trend={-0.3}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Performance Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Connections"
            value={mockSystemMetrics.active_connections}
            unit=""
            icon={<NetworkIcon />}
            trend={4.2}
            color="primary"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Requests/Second"
            value={mockSystemMetrics.requests_per_second}
            unit="req/s"
            icon={<SpeedIcon />}
            trend={7.8}
            color="success"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Error Rate"
            value={mockSystemMetrics.error_rate}
            unit="%"
            icon={<ErrorIcon />}
            trend={-15.2}
            color="error"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Uptime"
            value={mockSystemMetrics.uptime}
            unit="%"
            icon={<CheckCircleIcon />}
            trend={0.1}
            color="success"
          />
        </Grid>
      </Grid>

      {/* Cache Management */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Cache Statistics
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  size="small"
                  startIcon={<CachedIcon />}
                  onClick={() => handleCacheInvalidation()}
                >
                  Clear All
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={handleCacheWarming}
                >
                  Warm Cache
                </Button>
              </Box>
            </Box>
            
            {cacheStats && (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <CacheMetricsCard
                    title="Memory Cache"
                    cacheInfo={cacheStats.cache_statistics.memory}
                    totalRequests={cacheStats.total_cache_requests}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <CacheMetricsCard
                    title="Redis Cache"
                    cacheInfo={cacheStats.cache_statistics.redis}
                    totalRequests={cacheStats.total_cache_requests}
                  />
                </Grid>
              </Grid>
            )}
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: 300 }}>
            <Typography variant="h6" gutterBottom>
              Resource Usage Breakdown
            </Typography>
            <BarChart
              data={[
                { name: 'CPU', value: mockSystemMetrics.cpu_usage, color: '#1976d2' },
                { name: 'Memory', value: mockSystemMetrics.memory_usage, color: '#ff9800' },
                { name: 'Disk', value: mockSystemMetrics.disk_usage, color: '#4caf50' },
                { name: 'Network', value: mockSystemMetrics.network_latency * 2, color: '#f44336' },
              ]}
              xKey="name"
              yKey="value"
              showLabels={true}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* System Performance Timeline */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Performance Timeline (Last 24 Hours)
            </Typography>
            <TimeSeriesChart
              data={generateMockTimeSeriesData()}
              xKey="timestamp"
              yKey="value"
              title="System Performance Metrics"
              color="#1976d2"
              showMultipleLines={true}
            />
          </Paper>
        </Grid>
        
        {/* System Alerts */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Alerts & Notifications
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Alert severity="warning" sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">
                    Memory Usage Above 65%
                  </Typography>
                  <Typography variant="body2">
                    Current memory usage is at {mockSystemMetrics.memory_usage}%. 
                    Consider optimizing or scaling resources.
                  </Typography>
                </Alert>
                
                <Alert severity="info" sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">
                    Cache Performance Optimal
                  </Typography>
                  <Typography variant="body2">
                    Cache hit rate is above 85%. System performance is optimal.
                  </Typography>
                </Alert>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Alert severity="success" sx={{ mb: 1 }}>
                  <Typography variant="subtitle2">
                    Network Latency Low
                  </Typography>
                  <Typography variant="body2">
                    Network response times are within acceptable range at {mockSystemMetrics.network_latency}ms.
                  </Typography>
                </Alert>
                
                <Alert severity="error">
                  <Typography variant="subtitle2">
                    Error Rate Spike
                  </Typography>
                  <Typography variant="body2">
                    Minor increase in error rate detected. Monitoring for resolution.
                  </Typography>
                </Alert>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

// Helper function to generate mock time series data
const generateMockTimeSeriesData = () => {
  const data = [];
  const now = new Date();
  
  for (let i = 23; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
    data.push({
      timestamp: timestamp.toISOString(),
      cpu: 30 + Math.random() * 40,
      memory: 50 + Math.random() * 30,
      network: 10 + Math.random() * 20,
      value: 30 + Math.random() * 40, // Default line for single series
    });
  }
  
  return data;
};

export default SystemMetrics;