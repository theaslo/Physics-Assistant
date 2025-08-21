import React from 'react';
import { Box, Typography } from '@mui/material';
import { 
  ResponsiveContainer, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  Radar, 
  Tooltip,
  Legend
} from 'recharts';
import { ConceptMastery } from '../../types/api';

interface ConceptMasteryRadarProps {
  concepts: ConceptMastery[];
}

const ConceptMasteryRadar: React.FC<ConceptMasteryRadarProps> = ({ concepts }) => {
  // Transform data for radar chart
  const radarData = concepts.map(concept => ({
    concept: concept.concept.replace(/_/g, ' ').toUpperCase(),
    mastery: Math.round(concept.mastery_score * 100),
    confidence: Math.round(concept.confidence[0] * 100), // Lower bound of confidence interval
  }));

  return (
    <Box sx={{ width: '100%', height: '320px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={radarData} margin={{ top: 40, right: 40, bottom: 40, left: 40 }}>
          <PolarGrid />
          <PolarAngleAxis 
            dataKey="concept" 
            tick={{ fontSize: 10 }}
            className="text-xs"
          />
          <PolarRadiusAxis 
            angle={90} 
            domain={[0, 100]} 
            tick={{ fontSize: 8 }}
          />
          <Radar
            name="Mastery Score"
            dataKey="mastery"
            stroke="#1976d2"
            fill="#1976d2"
            fillOpacity={0.3}
            strokeWidth={2}
          />
          <Radar
            name="Confidence"
            dataKey="confidence"
            stroke="#4caf50"
            fill="none"
            strokeWidth={1}
            strokeDasharray="5 5"
          />
          <Tooltip 
            formatter={(value: any, name: string) => [`${value}%`, name]}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ConceptMasteryRadar;