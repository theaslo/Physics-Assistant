import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  LinearProgress, 
  Chip,
  Divider
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { CacheInfo } from '../../types/api';

interface CacheMetricsCardProps {
  title: string;
  cacheInfo: CacheInfo;
  totalRequests: number;
}

const CacheMetricsCard: React.FC<CacheMetricsCardProps> = ({ 
  title, 
  cacheInfo, 
  totalRequests 
}) => {
  const hitRate = cacheInfo.hit_rate * 100;
  const isConnected = cacheInfo.connected !== false;
  
  const getHitRateColor = (rate: number) => {
    if (rate >= 80) return 'success';
    if (rate >= 60) return 'warning';
    return 'error';
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          {title.includes('Memory') ? (
            <MemoryIcon color="primary" sx={{ mr: 1 }} />
          ) : (
            <StorageIcon color="secondary" sx={{ mr: 1 }} />
          )}
          <Typography variant="h6" component="h3">
            {title}
          </Typography>
          <Box sx={{ ml: 'auto' }}>
            <Chip
              label={isConnected ? 'Connected' : 'Disconnected'}
              size="small"
              color={isConnected ? 'success' : 'error'}
              variant="outlined"
            />
          </Box>
        </Box>

        {/* Hit Rate */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" color="textSecondary">
              Hit Rate
            </Typography>
            <Typography variant="body2" fontWeight="medium">
              {hitRate.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={hitRate}
            color={getHitRateColor(hitRate) as any}
            sx={{ height: 6, borderRadius: 3 }}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Cache Statistics */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {cacheInfo.size !== undefined && cacheInfo.max_size !== undefined && (
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="textSecondary">
                Size Usage
              </Typography>
              <Typography variant="body2">
                {cacheInfo.size} / {cacheInfo.max_size}
              </Typography>
            </Box>
          )}
          
          {cacheInfo.memory_usage && (
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="textSecondary">
                Memory Usage
              </Typography>
              <Typography variant="body2">
                {cacheInfo.memory_usage}
              </Typography>
            </Box>
          )}
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="body2" color="textSecondary">
              Total Requests
            </Typography>
            <Typography variant="body2">
              {totalRequests.toLocaleString()}
            </Typography>
          </Box>
        </Box>

        {/* Performance Indicator */}
        <Box sx={{ mt: 2, p: 1, backgroundColor: 'background.default', borderRadius: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SpeedIcon color="action" sx={{ fontSize: 16 }} />
            <Typography variant="caption" color="textSecondary">
              Performance: 
            </Typography>
            <Chip
              label={hitRate >= 80 ? 'Excellent' : hitRate >= 60 ? 'Good' : 'Needs Improvement'}
              size="small"
              color={getHitRateColor(hitRate) as any}
              variant="filled"
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default CacheMetricsCard;