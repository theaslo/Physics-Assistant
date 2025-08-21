import React from 'react';
import { 
  Box, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemIcon,
  Typography,
  Chip,
  Avatar,
  LinearProgress,
  Divider
} from '@mui/material';
import {
  Person as PersonIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
} from '@mui/icons-material';
import { StudentProgressUpdate } from '../../types/api';

interface LiveStudentProgressProps {
  updates: StudentProgressUpdate[];
  isLive: boolean;
}

const LiveStudentProgress: React.FC<LiveStudentProgressProps> = ({ updates, isLive }) => {
  const getTrendIcon = (change: number) => {
    if (change > 0) return <TrendingUpIcon color="success" />;
    if (change < 0) return <TrendingDownIcon color="error" />;
    return <TrendingFlatIcon color="action" />;
  };

  const getTrendColor = (change: number) => {
    if (change > 0) return 'success';
    if (change < 0) return 'error';
    return 'default';
  };

  const formatChange = (change: number) => {
    const sign = change > 0 ? '+' : '';
    return `${sign}${change.toFixed(1)}%`;
  };

  const getInitials = (userId: string) => {
    const parts = userId.split('_');
    if (parts.length > 1) {
      return parts[1].substring(0, 2).toUpperCase();
    }
    return userId.substring(0, 2).toUpperCase();
  };

  return (
    <Box sx={{ height: '320px', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2">
          Student Progress Updates
        </Typography>
        <Chip
          label={isLive ? 'LIVE' : 'PAUSED'}
          size="small"
          color={isLive ? 'success' : 'default'}
          variant={isLive ? 'filled' : 'outlined'}
        />
      </Box>
      
      {updates.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body2" color="textSecondary">
            {isLive ? 'Waiting for progress updates...' : 'No recent progress updates'}
          </Typography>
        </Box>
      ) : (
        <List dense sx={{ p: 0 }}>
          {updates.map((update, index) => {
            const overallChange = update.progress_change?.overall_score || 0;
            const engagementChange = update.progress_change?.engagement_score || 0;
            
            return (
              <React.Fragment key={`${update.user_id}-${index}`}>
                <ListItem
                  sx={{
                    px: 0,
                    py: 1.5,
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    },
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 40 }}>
                    <Avatar
                      sx={{
                        width: 36,
                        height: 36,
                        bgcolor: 'primary.light',
                        fontSize: '0.75rem',
                      }}
                    >
                      {getInitials(update.user_id)}
                    </Avatar>
                  </ListItemIcon>
                  
                  <ListItemText
                    primary={
                      <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Typography variant="body2" fontWeight="medium">
                            {update.user_id}
                          </Typography>
                          {getTrendIcon(overallChange)}
                          <Typography 
                            variant="caption" 
                            color={getTrendColor(overallChange) === 'success' ? 'success.main' : 
                                   getTrendColor(overallChange) === 'error' ? 'error.main' : 'text.secondary'}
                          >
                            {formatChange(overallChange)}
                          </Typography>
                        </Box>
                        
                        {/* Progress Indicators */}
                        <Box sx={{ mb: 1 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                            <Typography variant="caption" color="textSecondary">
                              Overall Progress
                            </Typography>
                            <Typography variant="caption">
                              {((update.progress_change?.overall_score || 0) + 75).toFixed(1)}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={(update.progress_change?.overall_score || 0) + 75}
                            color="primary"
                            sx={{ height: 4, borderRadius: 2 }}
                          />
                        </Box>
                        
                        {update.progress_change?.engagement_score !== undefined && (
                          <Box>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                              <Typography variant="caption" color="textSecondary">
                                Engagement
                              </Typography>
                              <Typography variant="caption">
                                {((update.progress_change?.engagement_score || 0) + 70).toFixed(1)}%
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={(update.progress_change?.engagement_score || 0) + 70}
                              color="secondary"
                              sx={{ height: 4, borderRadius: 2 }}
                            />
                          </Box>
                        )}
                      </Box>
                    }
                    secondary={
                      <Box sx={{ mt: 1 }}>
                        {update.concept_updates && update.concept_updates.length > 0 && (
                          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                            {update.concept_updates.slice(0, 2).map((concept, idx) => (
                              <Chip
                                key={idx}
                                label={concept.concept.replace('_', ' ')}
                                size="small"
                                variant="outlined"
                                color="primary"
                                sx={{ fontSize: '0.625rem', height: 18 }}
                              />
                            ))}
                            {update.concept_updates.length > 2 && (
                              <Chip
                                label={`+${update.concept_updates.length - 2} more`}
                                size="small"
                                variant="outlined"
                                sx={{ fontSize: '0.625rem', height: 18 }}
                              />
                            )}
                          </Box>
                        )}
                        
                        <Typography variant="caption" color="textSecondary" sx={{ display: 'block', mt: 0.5 }}>
                          Updated just now
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
                {index < updates.length - 1 && (
                  <Divider variant="inset" component="li" />
                )}
              </React.Fragment>
            );
          })}
        </List>
      )}
      
      {isLive && updates.length > 0 && (
        <Box sx={{ p: 1, textAlign: 'center' }}>
          <Typography variant="caption" color="success.main">
            • Live progress tracking active
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default LiveStudentProgress;