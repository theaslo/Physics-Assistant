import React from 'react';
import { Box, Typography } from '@mui/material';
import { 
  ResponsiveContainer, 
  ComposedChart, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Cell,
  Rectangle
} from 'recharts';

interface ProgressHeatmapProps {
  data: Array<{
    date: string;
    score: number;
  }>;
  studentId: string;
}

const ProgressHeatmap: React.FC<ProgressHeatmapProps> = ({ data, studentId }) => {
  // Transform data for heatmap visualization
  const heatmapData = data.map((item, index) => ({
    ...item,
    day: new Date(item.date).getDay(),
    week: Math.floor(index / 7),
    intensity: item.score / 100,
  }));

  const getColor = (intensity: number) => {
    const alpha = Math.max(0.1, intensity);
    return `rgba(25, 118, 210, ${alpha})`;
  };

  return (
    <Box sx={{ width: '100%', height: '300px' }}>
      <Typography variant="subtitle2" gutterBottom>
        Progress Activity for {studentId}
      </Typography>
      
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={heatmapData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tickFormatter={(value) => new Date(value).toLocaleDateString()}
          />
          <YAxis 
            dataKey="score"
            domain={[0, 100]}
          />
          <Tooltip 
            labelFormatter={(value) => `Date: ${new Date(value).toLocaleDateString()}`}
            formatter={(value: any) => [`${value}%`, 'Progress Score']}
          />
          {heatmapData.map((entry, index) => (
            <Rectangle
              key={index}
              x={index * 20}
              y={300 - (entry.score * 3)}
              width={18}
              height={entry.score * 3}
              fill={getColor(entry.intensity)}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default ProgressHeatmap;