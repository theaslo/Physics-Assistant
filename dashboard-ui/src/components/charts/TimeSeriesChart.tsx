/**
 * Time Series Chart Component
 * Displays time-based data with multiple metrics and interactive features
 */

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
  ReferenceLine,
} from 'recharts';
import {
  Card,
  CardContent,
  CardHeader,
  Box,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Fullscreen as FullscreenIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';

import { TimeSeriesDataPoint, MetricType } from '../../types/api';
import { getChartColor } from '../../themes/dashboard-theme';

// ============================================================================
// Types
// ============================================================================

export interface TimeSeriesChartProps {
  data: TimeSeriesDataPoint[];
  metrics: MetricType[];
  title?: string;
  height?: number;
  showBrush?: boolean;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  loading?: boolean;
  error?: string;
  onDataPointClick?: (data: TimeSeriesDataPoint) => void;
  onExport?: () => void;
  customColors?: string[];
}

interface TooltipData {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    payload: TimeSeriesDataPoint;
    color: string;
  }>;
  label?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

const formatMetricLabel = (metric: MetricType): string => {
  const labels: Record<MetricType, string> = {
    interaction_count: 'Interactions',
    success_rate: 'Success Rate',
    avg_response_time: 'Avg Response Time',
    unique_users: 'Unique Users',
    agent_usage: 'Agent Usage',
    error_rate: 'Error Rate',
  };
  return labels[metric] || metric;
};

const formatMetricValue = (value: number, metric: MetricType): string => {
  switch (metric) {
    case 'success_rate':
    case 'error_rate':
      return `${(value * 100).toFixed(1)}%`;
    case 'avg_response_time':
      return `${value.toFixed(0)}ms`;
    case 'interaction_count':
    case 'unique_users':
    case 'agent_usage':
      return value.toLocaleString();
    default:
      return value.toString();
  }
};

const getMetricUnit = (metric: MetricType): string => {
  switch (metric) {
    case 'success_rate':
    case 'error_rate':
      return '%';
    case 'avg_response_time':
      return 'ms';
    default:
      return '';
  }
};

// ============================================================================
// Custom Tooltip Component
// ============================================================================

const CustomTooltip: React.FC<TooltipData> = ({ active, payload, label }) => {
  const theme = useTheme();

  if (!active || !payload || !payload.length) {
    return null;
  }

  const timestamp = new Date(label || '');
  const formattedTime = timestamp.toLocaleString();

  return (
    <Card
      elevation={8}
      sx={{
        minWidth: 200,
        backgroundColor: alpha(theme.palette.background.paper, 0.95),
        backdropFilter: 'blur(8px)',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Typography variant="caption" color="text.secondary" gutterBottom>
          {formattedTime}
        </Typography>
        {payload.map((entry) => (
          <Box key={entry.dataKey} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                backgroundColor: entry.color,
                borderRadius: '50%',
                mr: 1,
              }}
            />
            <Typography variant="body2" sx={{ flex: 1 }}>
              {formatMetricLabel(entry.dataKey as MetricType)}:
            </Typography>
            <Typography variant="body2" fontWeight="medium">
              {formatMetricValue(entry.value, entry.dataKey as MetricType)}
            </Typography>
          </Box>
        ))}
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  data,
  metrics,
  title = 'Time Series Data',
  height = 400,
  showBrush = true,
  showGrid = true,
  showLegend = true,
  showTooltip = true,
  loading = false,
  error,
  onDataPointClick,
  onExport,
  customColors,
}) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const chartData = useMemo(() => {
    return data.map(point => ({
      ...point,
      timestamp: new Date(point.timestamp).getTime(),
      formattedTime: new Date(point.timestamp).toLocaleTimeString(),
    }));
  }, [data]);

  const colors = useMemo(() => {
    if (customColors && customColors.length >= metrics.length) {
      return customColors;
    }
    return metrics.map((_, index) => getChartColor(index));
  }, [metrics, customColors]);

  const hasData = chartData.length > 0;

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleExport = () => {
    onExport?.();
    handleMenuClose();
  };

  const handleDataPointClick = (data: any) => {
    if (onDataPointClick && data?.payload) {
      onDataPointClick(data.payload);
    }
  };

  // ============================================================================
  // Render Methods
  // ============================================================================

  const renderHeader = () => (
    <CardHeader
      title={
        <Typography variant="h6" component="div">
          {title}
        </Typography>
      }
      action={
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Metric chips */}
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {metrics.map((metric, index) => (
              <Chip
                key={metric}
                label={formatMetricLabel(metric)}
                size="small"
                variant="outlined"
                sx={{
                  borderColor: colors[index],
                  color: colors[index],
                  fontSize: '0.7rem',
                }}
              />
            ))}
          </Box>

          {/* Menu button */}
          <IconButton onClick={handleMenuOpen} size="small">
            <MoreVertIcon />
          </IconButton>
        </Box>
      }
      sx={{ pb: 1 }}
    />
  );

  const renderMenu = () => (
    <Menu
      anchorEl={anchorEl}
      open={Boolean(anchorEl)}
      onClose={handleMenuClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      transformOrigin={{ vertical: 'top', horizontal: 'right' }}
    >
      <MenuItem onClick={handleExport}>
        <DownloadIcon sx={{ mr: 1 }} />
        Export Data
      </MenuItem>
      <MenuItem onClick={handleMenuClose}>
        <FullscreenIcon sx={{ mr: 1 }} />
        Fullscreen
      </MenuItem>
      <MenuItem onClick={handleMenuClose}>
        <ZoomInIcon sx={{ mr: 1 }} />
        Zoom In
      </MenuItem>
    </Menu>
  );

  const renderChart = () => {
    if (loading) {
      return (
        <Box
          sx={{
            height,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'text.secondary',
          }}
        >
          <Typography>Loading chart data...</Typography>
        </Box>
      );
    }

    if (error) {
      return (
        <Box
          sx={{
            height,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'error.main',
          }}
        >
          <Typography>Error loading chart: {error}</Typography>
        </Box>
      );
    }

    if (!hasData) {
      return (
        <Box
          sx={{
            height,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'text.secondary',
          }}
        >
          <Typography>No data available</Typography>
        </Box>
      );
    }

    return (
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: showBrush ? 60 : 5 }}
          onClick={handleDataPointClick}
        >
          {showGrid && (
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke={alpha(theme.palette.text.primary, 0.1)}
            />
          )}
          
          <XAxis
            dataKey="timestamp"
            type="number"
            scale="time"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(value) => new Date(value).toLocaleTimeString()}
            stroke={theme.palette.text.secondary}
            fontSize={12}
          />
          
          <YAxis
            stroke={theme.palette.text.secondary}
            fontSize={12}
            tickFormatter={(value) => {
              if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
              if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
              return value.toString();
            }}
          />
          
          {showTooltip && <Tooltip content={<CustomTooltip />} />}
          
          {showLegend && (
            <Legend
              verticalAlign="top"
              height={36}
              iconType="line"
              wrapperStyle={{ paddingBottom: '20px' }}
            />
          )}

          {metrics.map((metric, index) => (
            <Line
              key={metric}
              type="monotone"
              dataKey={metric}
              stroke={colors[index]}
              strokeWidth={2}
              dot={{ fill: colors[index], strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: colors[index], strokeWidth: 2 }}
              name={formatMetricLabel(metric)}
              connectNulls={false}
            />
          ))}

          {/* Reference lines for important thresholds */}
          {metrics.includes('success_rate') && (
            <ReferenceLine 
              y={0.8} 
              stroke={theme.palette.warning.main} 
              strokeDasharray="5 5"
              label="Target: 80%"
            />
          )}

          {showBrush && (
            <Brush
              dataKey="timestamp"
              height={30}
              stroke={theme.palette.primary.main}
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    );
  };

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <>
      <Card>
        {renderHeader()}
        <CardContent sx={{ pt: 0 }}>
          {renderChart()}
        </CardContent>
      </Card>
      {renderMenu()}
    </>
  );
};

export default TimeSeriesChart;