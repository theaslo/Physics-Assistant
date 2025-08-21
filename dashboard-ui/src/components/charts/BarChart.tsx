/**
 * Bar Chart Component
 * Displays categorical data with customizable styling and interactions
 */

import React, { useMemo } from 'react';
import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
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
  useTheme,
  alpha,
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  Download as DownloadIcon,
  Fullscreen as FullscreenIcon,
} from '@mui/icons-material';

import { getChartColor, getPhysicsColor } from '../../themes/dashboard-theme';

// ============================================================================
// Types
// ============================================================================

export interface BarChartData {
  [key: string]: string | number;
}

export interface BarChartProps {
  data: BarChartData[];
  dataKey: string;
  categoryKey: string;
  title?: string;
  height?: number;
  showGrid?: boolean;
  showLegend?: boolean;
  showTooltip?: boolean;
  orientation?: 'vertical' | 'horizontal';
  loading?: boolean;
  error?: string;
  onBarClick?: (data: BarChartData) => void;
  onExport?: () => void;
  customColors?: string[];
  colorMode?: 'default' | 'physics' | 'gradient';
  maxBars?: number;
}

interface TooltipData {
  active?: boolean;
  payload?: Array<{
    dataKey: string;
    value: number;
    payload: BarChartData;
    color: string;
  }>;
  label?: string;
}

// ============================================================================
// Custom Tooltip Component
// ============================================================================

const CustomTooltip: React.FC<TooltipData> = ({ active, payload, label }) => {
  const theme = useTheme();

  if (!active || !payload || !payload.length) {
    return null;
  }

  const data = payload[0];

  return (
    <Card
      elevation={8}
      sx={{
        minWidth: 150,
        backgroundColor: alpha(theme.palette.background.paper, 0.95),
        backdropFilter: 'blur(8px)',
      }}
    >
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Typography variant="body2" fontWeight="medium" gutterBottom>
          {label}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              backgroundColor: data.color,
              borderRadius: 1,
              mr: 1,
            }}
          />
          <Typography variant="body2">
            {typeof data.value === 'number' 
              ? data.value.toLocaleString() 
              : data.value
            }
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const BarChart: React.FC<BarChartProps> = ({
  data,
  dataKey,
  categoryKey,
  title = 'Bar Chart',
  height = 400,
  showGrid = true,
  showLegend = false,
  showTooltip = true,
  orientation = 'vertical',
  loading = false,
  error,
  onBarClick,
  onExport,
  customColors,
  colorMode = 'default',
  maxBars = 20,
}) => {
  const theme = useTheme();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  // ============================================================================
  // Computed Values
  // ============================================================================

  const chartData = useMemo(() => {
    let processedData = [...data];
    
    // Limit number of bars if specified
    if (maxBars && processedData.length > maxBars) {
      processedData = processedData.slice(0, maxBars);
    }
    
    return processedData;
  }, [data, maxBars]);

  const colors = useMemo(() => {
    if (customColors && customColors.length >= chartData.length) {
      return customColors;
    }

    if (colorMode === 'physics') {
      return chartData.map((item) => {
        const category = item[categoryKey] as string;
        return getPhysicsColor(category.toLowerCase());
      });
    }

    if (colorMode === 'gradient') {
      return chartData.map((_, index) => {
        const ratio = index / Math.max(chartData.length - 1, 1);
        const startColor = theme.palette.primary.light;
        const endColor = theme.palette.primary.dark;
        return `color-mix(in srgb, ${startColor} ${(1 - ratio) * 100}%, ${endColor} ${ratio * 100}%)`;
      });
    }

    return chartData.map((_, index) => getChartColor(index));
  }, [chartData, categoryKey, colorMode, customColors, theme]);

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

  const handleBarClick = (data: any) => {
    if (onBarClick && data) {
      onBarClick(data);
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
        <IconButton onClick={handleMenuOpen} size="small">
          <MoreVertIcon />
        </IconButton>
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

    const isHorizontal = orientation === 'horizontal';

    return (
      <ResponsiveContainer width="100%" height={height}>
        <RechartsBarChart
          data={chartData}
          layout={isHorizontal ? 'horizontal' : 'vertical'}
          margin={{ 
            top: 5, 
            right: 30, 
            left: isHorizontal ? 80 : 20, 
            bottom: isHorizontal ? 5 : 50 
          }}
        >
          {showGrid && (
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke={alpha(theme.palette.text.primary, 0.1)}
            />
          )}
          
          {isHorizontal ? (
            <>
              <XAxis
                type="number"
                stroke={theme.palette.text.secondary}
                fontSize={12}
              />
              <YAxis
                type="category"
                dataKey={categoryKey}
                stroke={theme.palette.text.secondary}
                fontSize={12}
                width={80}
              />
            </>
          ) : (
            <>
              <XAxis
                dataKey={categoryKey}
                stroke={theme.palette.text.secondary}
                fontSize={12}
                angle={-45}
                textAnchor="end"
                height={50}
              />
              <YAxis
                stroke={theme.palette.text.secondary}
                fontSize={12}
              />
            </>
          )}
          
          {showTooltip && <Tooltip content={<CustomTooltip />} />}
          
          {showLegend && <Legend />}

          <Bar 
            dataKey={dataKey} 
            onClick={handleBarClick}
            cursor="pointer"
            radius={[4, 4, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={colors[index]} 
              />
            ))}
          </Bar>
        </RechartsBarChart>
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
          
          {/* Data summary */}
          {hasData && (
            <Box sx={{ mt: 2, pt: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
              <Typography variant="caption" color="text.secondary">
                Showing {chartData.length} {chartData.length === 1 ? 'item' : 'items'}
                {maxBars && data.length > maxBars && ` (limited from ${data.length})`}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
      {renderMenu()}
    </>
  );
};

export default BarChart;