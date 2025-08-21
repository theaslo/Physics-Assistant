import React from 'react';
import { Box, Typography, Chip } from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  RadioButtonUnchecked as RadioButtonUncheckedIcon,
} from '@mui/icons-material';

interface SystemHealthIndicatorProps {
  status: 'healthy' | 'degraded' | 'down' | 'unknown';
  label: string;
  size?: 'small' | 'medium' | 'large';
}

const SystemHealthIndicator: React.FC<SystemHealthIndicatorProps> = ({ 
  status, 
  label, 
  size = 'medium' 
}) => {
  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'healthy':
        return {
          color: 'success' as const,
          icon: <CheckCircleIcon />,
          text: 'Healthy',
          bgColor: '#e8f5e8',
          borderColor: '#4caf50',
        };
      case 'degraded':
        return {
          color: 'warning' as const,
          icon: <WarningIcon />,
          text: 'Degraded',
          bgColor: '#fff3e0',
          borderColor: '#ff9800',
        };
      case 'down':
        return {
          color: 'error' as const,
          icon: <ErrorIcon />,
          text: 'Down',
          bgColor: '#ffebee',
          borderColor: '#f44336',
        };
      default:
        return {
          color: 'default' as const,
          icon: <RadioButtonUncheckedIcon />,
          text: 'Unknown',
          bgColor: '#f5f5f5',
          borderColor: '#9e9e9e',
        };
    }
  };

  const config = getStatusConfig(status);
  
  const sizeConfig = {
    small: { iconSize: 16, fontSize: '0.75rem', padding: '4px 8px' },
    medium: { iconSize: 20, fontSize: '0.875rem', padding: '8px 12px' },
    large: { iconSize: 24, fontSize: '1rem', padding: '12px 16px' },
  };

  const currentSize = sizeConfig[size];

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        padding: currentSize.padding,
        backgroundColor: config.bgColor,
        border: `1px solid ${config.borderColor}`,
        borderRadius: 2,
        minWidth: size === 'large' ? 120 : 80,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          '& svg': {
            fontSize: currentSize.iconSize,
            color: config.borderColor,
          },
        }}
      >
        {config.icon}
      </Box>
      
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="caption"
          sx={{
            fontSize: currentSize.fontSize,
            fontWeight: 500,
            color: 'text.primary',
            display: 'block',
            lineHeight: 1.2,
          }}
        >
          {label}
        </Typography>
        
        <Chip
          label={config.text}
          size="small"
          color={config.color}
          variant="outlined"
          sx={{
            height: 16,
            fontSize: '0.625rem',
            '& .MuiChip-label': {
              px: 0.5,
            },
          }}
        />
      </Box>
    </Box>
  );
};

export default SystemHealthIndicator;