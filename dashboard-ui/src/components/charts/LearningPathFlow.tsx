import React from 'react';
import { Box, Typography, Chip, Paper } from '@mui/material';
import { 
  CheckCircle as CheckCircleIcon,
  RadioButtonUnchecked as RadioButtonUncheckedIcon,
  TrendingUp as TrendingUpIcon
} from '@mui/icons-material';
import { ConceptMastery } from '../../types/api';

interface LearningPathFlowProps {
  concepts: ConceptMastery[];
  studentId: string;
}

const LearningPathFlow: React.FC<LearningPathFlowProps> = ({ concepts, studentId }) => {
  // Sort concepts by mastery score to show learning progression (handle both field names)
  const sortedConcepts = [...concepts].sort((a: any, b: any) =>
    (b.mastery_score ?? b.mastery ?? 0) - (a.mastery_score ?? a.mastery ?? 0)
  );

  const getStatusIcon = (masteryScore: number) => {
    if (masteryScore >= 0.8) {
      return <CheckCircleIcon color="success" />;
    } else if (masteryScore >= 0.6) {
      return <TrendingUpIcon color="warning" />;
    } else {
      return <RadioButtonUncheckedIcon color="action" />;
    }
  };

  const getStatusColor = (masteryScore: number) => {
    if (masteryScore >= 0.8) return 'success';
    if (masteryScore >= 0.6) return 'warning';
    return 'default';
  };

  const getStatusText = (masteryScore: number) => {
    if (masteryScore >= 0.8) return 'Mastered';
    if (masteryScore >= 0.6) return 'In Progress';
    return 'Not Started';
  };

  return (
    <Box sx={{ width: '100%', height: '240px', overflow: 'auto' }}>
      <Typography variant="subtitle2" gutterBottom>
        Learning Path Progress for {studentId}
      </Typography>
      
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {sortedConcepts.map((concept: any, index) => {
          const score = concept.mastery_score ?? concept.mastery ?? 0;
          const name = concept.concept || concept.name || 'Unknown';
          return (
            <Paper
              key={name}
              elevation={1}
              sx={{
                p: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                border: score >= 0.8 ? '2px solid #4caf50' : '1px solid #e0e0e0',
              }}
            >
              {getStatusIcon(score)}

              <Box sx={{ flex: 1 }}>
                <Typography variant="subtitle2">
                  {name.replace(/_/g, ' ').toUpperCase()}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Mastery: {Math.round(score * 100)}%
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip
                  label={getStatusText(score)}
                  size="small"
                  color={getStatusColor(score) as any}
                  variant="outlined"
                />

                <Box
                  sx={{
                    width: 60,
                    height: 6,
                    backgroundColor: '#e0e0e0',
                    borderRadius: 3,
                    overflow: 'hidden',
                  }}
                >
                  <Box
                    sx={{
                      width: `${score * 100}%`,
                      height: '100%',
                      backgroundColor: score >= 0.8 ? '#4caf50' :
                                     score >= 0.6 ? '#ff9800' : '#f44336',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </Box>
              </Box>
            </Paper>
          );
        })}
        
        {sortedConcepts.length === 0 && (
          <Typography variant="body2" color="textSecondary" sx={{ textAlign: 'center', py: 4 }}>
            No concept data available for this student.
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default LearningPathFlow;