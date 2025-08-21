import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { 
  ResponsiveContainer, 
  ComposedChart, 
  Bar,
  Line,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip,
  Legend
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../services/api-client';
import { TimeRangeRequest } from '../../types/api';

interface ClassComparisonChartProps {
  classes: string[];
  timeRange: TimeRangeRequest;
}

const ClassComparisonChart: React.FC<ClassComparisonChartProps> = ({ classes, timeRange }) => {
  // Mock data for class comparison
  const mockData = React.useMemo(() => {
    return classes.map(className => ({
      class: className,
      averageScore: 65 + Math.random() * 30,
      totalStudents: 20 + Math.floor(Math.random() * 15),
      activeStudents: 15 + Math.floor(Math.random() * 10),
      completionRate: 70 + Math.random() * 25,
      engagementScore: 60 + Math.random() * 35,
    }));
  }, [classes]);

  // In a real implementation, this would fetch actual comparison data
  const { data: comparisonData, isLoading } = useQuery({
    queryKey: ['class-comparison', classes, timeRange],
    queryFn: async () => {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      return mockData;
    },
    enabled: classes.length > 0,
  });

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="300px">
        <CircularProgress size={40} />
      </Box>
    );
  }

  const chartData = comparisonData || mockData;

  return (
    <Box sx={{ width: '100%', height: '400px' }}>
      <Typography variant="subtitle2" gutterBottom>
        Class Performance Comparison
      </Typography>
      
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="class" 
            angle={-45}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 10 }}
          />
          <YAxis 
            yAxisId="left"
            tick={{ fontSize: 10 }}
            domain={[0, 100]}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right"
            tick={{ fontSize: 10 }}
            domain={[0, 'dataMax']}
          />
          <Tooltip 
            formatter={(value: any, name: string) => {
              if (name.includes('Score') || name.includes('Rate')) {
                return [`${Math.round(value)}%`, name];
              }
              return [Math.round(value), name];
            }}
          />
          <Legend />
          
          <Bar 
            yAxisId="left"
            dataKey="averageScore" 
            fill="#1976d2" 
            name="Average Score"
            radius={[2, 2, 0, 0]}
          />
          <Bar 
            yAxisId="left"
            dataKey="completionRate" 
            fill="#4caf50" 
            name="Completion Rate"
            radius={[2, 2, 0, 0]}
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="totalStudents" 
            stroke="#ff9800" 
            strokeWidth={2}
            name="Total Students"
            dot={{ r: 4 }}
          />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="engagementScore" 
            stroke="#9c27b0" 
            strokeWidth={2}
            strokeDasharray="5 5"
            name="Engagement Score"
            dot={{ r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
      
      <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
        Comparing {classes.length} classes over the selected time period
      </Typography>
    </Box>
  );
};

export default ClassComparisonChart;