/**
 * Metric Card Widget Component
 * Displays key metrics with optional trend indicators and charts
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Skeleton,
  useTheme,
  alpha,
  IconButton,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  Info as InfoIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

// ============================================================================
// Types
// ============================================================================

export interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  icon?: React.ReactNode;
  trend?: number;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  loading?: boolean;
  error?: string;
  onClick?: () => void;
  onRefresh?: () => void;
  tooltip?: string;
  size?: 'small' | 'medium' | 'large';
  variant?: 'default' | 'outlined' | 'filled';
  sparklineData?: number[];
  showProgress?: boolean;
  progressValue?: number;
  isLive?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

const formatValue = (value: string | number): string => {
  if (typeof value === 'number') {
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    } else if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    } else if (value % 1 !== 0) {
      return value.toFixed(2);
    }
    return value.toString();
  }
  return value;
};

const getTrendColor = (direction: 'up' | 'down' | 'flat', theme: any) => {
  switch (direction) {
    case 'up':
      return theme.palette.success.main;
    case 'down':
      return theme.palette.error.main;
    case 'flat':
      return theme.palette.warning.main;
    default:
      return theme.palette.text.secondary;
  }
};

const getTrendIcon = (direction: 'up' | 'down' | 'flat') => {
  switch (direction) {
    case 'up':
      return <TrendingUpIcon fontSize="small" />;
    case 'down':
      return <TrendingDownIcon fontSize="small" />;
    case 'flat':
      return <TrendingFlatIcon fontSize="small" />;
    default:
      return null;
  }
};

// ============================================================================
// Sparkline Component
// ============================================================================

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}

const Sparkline: React.FC<SparklineProps> = ({
  data,
  width = 80,
  height = 24,
  color = '#1976d2',
}) => {
  if (!data || data.length < 2) {
    return null;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * width;
    const y = height - ((value - min) / range) * height;
    return `${x},${y}`;
  }).join(' ');

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  subtitle,
  icon,
  trend,
  color = 'primary',
  loading = false,
  error,
  onClick,
  onRefresh,
  tooltip,
  size = 'medium',
  variant = 'default',
  sparklineData,
  showProgress = false,
  progressValue,
  isLive = false,
}) => {
  const theme = useTheme();

  // ============================================================================
  // Computed Styles
  // ============================================================================

  const cardSx = {
    cursor: onClick ? 'pointer' : 'default',
    transition: 'all 0.2s ease-in-out',
    height: '100%',
    ...(onClick && {
      '&:hover': {
        transform: 'translateY(-2px)',
        boxShadow: theme.shadows[4],
      },
    }),
    ...(variant === 'outlined' && {
      border: `1px solid ${theme.palette.divider}`,
    }),
    ...(variant === 'filled' && {
      backgroundColor: alpha(theme.palette[color].main, 0.1),
      border: `1px solid ${alpha(theme.palette[color].main, 0.2)}`,
    }),
  };

  const iconColor = theme.palette[color].main;
  const trendColor = trend ? getTrendColor(trend.direction, theme) : undefined;

  const padding = {
    small: 1.5,
    medium: 2,
    large: 3,
  }[size];

  // ============================================================================
  // Render Methods
  // ============================================================================

  const renderHeader = () => (
    <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
        {icon && (
          <Box sx={{ color: iconColor, mr: 1, display: 'flex', alignItems: 'center' }}>
            {icon}
          </Box>
        )}
        <Typography
          variant={size === 'small' ? 'caption' : 'body2'}
          color="text.secondary"
          sx={{ fontWeight: 500 }}
        >
          {title}
        </Typography>
        {tooltip && (
          <Tooltip title={tooltip}>
            <InfoIcon sx={{ fontSize: 16, ml: 0.5, color: 'text.disabled' }} />
          </Tooltip>
        )}
      </Box>
      
      {onRefresh && (
        <IconButton size="small" onClick={onRefresh} sx={{ ml: 1 }}>
          <RefreshIcon fontSize="small" />
        </IconButton>
      )}
    </Box>
  );

  const renderValue = () => {
    if (loading) {
      return (
        <Skeleton
          variant="text"
          width="60%"
          height={size === 'large' ? 48 : size === 'medium' ? 40 : 32}
        />
      );
    }

    if (error) {
      return (
        <Typography variant="body2" color="error">
          Error loading data
        </Typography>
      );
    }

    return (
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
        <Typography
          variant={size === 'large' ? 'h4' : size === 'medium' ? 'h5' : 'h6'}
          fontWeight="bold"
          color="text.primary"
          sx={{ lineHeight: 1.2 }}
        >
          {formatValue(value)}
        </Typography>
        {unit && (
          <Typography
            variant={size === 'large' ? 'h6' : 'body2'}
            color="text.secondary"
          >
            {unit}
          </Typography>
        )}
        {isLive && (
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: 'success.main',
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%': { opacity: 1 },
                '50%': { opacity: 0.5 },
                '100%': { opacity: 1 },
              },
            }}
          />
        )}
      </Box>
    );
  };

  const renderSubtitle = () => {
    if (!subtitle && !trend && !sparklineData && !showProgress) {
      return null;
    }

    const trendDirection = trend ? (trend > 0 ? 'up' : trend < 0 ? 'down' : 'flat') : 'flat';
    const formattedTrend = trend ? `${trend > 0 ? '+' : ''}${trend.toFixed(1)}%` : '';
    const trendColor = getTrendColor(trendDirection, theme);

    return (
      <Box sx={{ mt: 1 }}>
        {showProgress && progressValue !== undefined && (
          <Box sx={{ mb: 1 }}>
            <LinearProgress
              variant="determinate"
              value={Math.min(Math.max(progressValue, 0), 100)}
              color={color}
              sx={{ height: 6, borderRadius: 3 }}
            />
          </Box>
        )}
        
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
            {subtitle && (
              <Typography variant="caption" color="text.secondary">
                {subtitle}
              </Typography>
            )}
            
            {trend !== undefined && (
              <Chip
                icon={getTrendIcon(trendDirection)}
                label={formattedTrend}
                size="small"
                variant="outlined"
                sx={{
                  ml: subtitle ? 1 : 0,
                  height: 20,
                  fontSize: '0.7rem',
                  color: trendColor,
                  borderColor: trendColor,
                  '& .MuiChip-icon': {
                    color: trendColor,
                  },
                }}
              />
            )}
          </Box>

          {sparklineData && (
            <Box sx={{ ml: 1 }}>
              <Sparkline
                data={sparklineData}
                color={iconColor}
                width={size === 'large' ? 100 : 80}
                height={size === 'large' ? 30 : 24}
              />
            </Box>
          )}
        </Box>
      </Box>
    );
  };

  const renderLoadingState = () => (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Skeleton variant="circular" width={20} height={20} sx={{ mr: 1 }} />
        <Skeleton variant="text" width="40%" />
      </Box>
      <Skeleton variant="text" width="60%" height={40} />
      <Skeleton variant="text" width="30%" height={20} sx={{ mt: 1 }} />
    </Box>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <Card sx={cardSx} onClick={onClick}>
      <CardContent sx={{ p: padding, '&:last-child': { pb: padding } }}>
        {loading ? (
          renderLoadingState()
        ) : (
          <>
            {renderHeader()}
            {renderValue()}
            {renderSubtitle()}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default MetricCard;