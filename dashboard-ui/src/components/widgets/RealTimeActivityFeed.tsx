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
  Divider
} from '@mui/material';
import {
  Person as PersonIcon,
  School as SchoolIcon,
  Psychology as PsychologyIcon,
  CheckCircle as CheckCircleIcon,
  TrendingUp as TrendingUpIcon,
  Chat as ChatIcon,
} from '@mui/icons-material';

interface Activity {
  id: string;
  timestamp: string;
  type: 'student_progress' | 'interaction' | 'system_event' | 'agent_usage';
  message: string;
  data?: any;
}

interface RealTimeActivityFeedProps {
  activities: Activity[];
  isLive: boolean;
}

const RealTimeActivityFeed: React.FC<RealTimeActivityFeedProps> = ({ activities, isLive }) => {
  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'student_progress':
        return <TrendingUpIcon color="success" />;
      case 'interaction':
        return <ChatIcon color="primary" />;
      case 'system_event':
        return <CheckCircleIcon color="info" />;
      case 'agent_usage':
        return <PsychologyIcon color="secondary" />;
      default:
        return <PersonIcon />;
    }
  };

  const getActivityColor = (type: string) => {
    switch (type) {
      case 'student_progress': return 'success';
      case 'interaction': return 'primary';
      case 'system_event': return 'info';
      case 'agent_usage': return 'secondary';
      default: return 'default';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) { // Less than 1 minute
      return 'Just now';
    } else if (diff < 3600000) { // Less than 1 hour
      return `${Math.floor(diff / 60000)}m ago`;
    } else {
      return date.toLocaleTimeString();
    }
  };

  return (
    <Box sx={{ height: '320px', overflow: 'auto' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2">
          Activity Feed
        </Typography>
        <Chip
          label={isLive ? 'LIVE' : 'PAUSED'}
          size="small"
          color={isLive ? 'success' : 'default'}
          variant={isLive ? 'filled' : 'outlined'}
        />
      </Box>
      
      {activities.length === 0 ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="body2" color="textSecondary">
            {isLive ? 'Waiting for activity...' : 'No recent activity'}
          </Typography>
        </Box>
      ) : (
        <List dense sx={{ p: 0 }}>
          {activities.map((activity, index) => (
            <React.Fragment key={activity.id}>
              <ListItem
                sx={{
                  px: 0,
                  py: 1,
                  '&:hover': {
                    backgroundColor: 'action.hover',
                  },
                }}
              >
                <ListItemIcon sx={{ minWidth: 36 }}>
                  <Avatar
                    sx={{
                      width: 32,
                      height: 32,
                      bgcolor: `${getActivityColor(activity.type)}.light`,
                    }}
                  >
                    {React.cloneElement(getActivityIcon(activity.type), {
                      sx: { fontSize: 16 }
                    })}
                  </Avatar>
                </ListItemIcon>
                
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                        {activity.message}
                      </Typography>
                      <Chip
                        label={activity.type.replace('_', ' ')}
                        size="small"
                        color={getActivityColor(activity.type) as any}
                        variant="outlined"
                        sx={{ fontSize: '0.625rem', height: 18 }}
                      />
                    </Box>
                  }
                  secondary={
                    <Typography variant="caption" color="textSecondary">
                      {formatTimestamp(activity.timestamp)}
                    </Typography>
                  }
                />
              </ListItem>
              {index < activities.length - 1 && (
                <Divider variant="inset" component="li" />
              )}
            </React.Fragment>
          ))}
        </List>
      )}
      
      {isLive && activities.length > 0 && (
        <Box sx={{ p: 1, textAlign: 'center' }}>
          <Typography variant="caption" color="success.main">
            • Live updates active
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default RealTimeActivityFeed;