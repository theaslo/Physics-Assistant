import React from 'react';
import { Box, Typography } from '@mui/material';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip,
  Cell,
  ReferenceLine
} from 'recharts';
import { PerformanceDistribution } from '../../types/api';

interface PerformanceDistributionChartProps {
  distribution: any; // Using any since the interface varies between API types
  showPercentiles?: boolean;
}

const PerformanceDistributionChart: React.FC<PerformanceDistributionChartProps> = ({ 
  distribution, 
  showPercentiles = false 
}) => {
  // Handle different distribution data structures
  const chartData = React.useMemo(() => {
    if (!distribution) return [];

    // If it's the API PerformanceDistribution format
    if (distribution.percentiles) {
      return [
        { name: '25th Percentile', value: distribution.percentiles['25th'], color: '#ff9800' },
        { name: '50th Percentile', value: distribution.percentiles['50th'], color: '#2196f3' },
        { name: '75th Percentile', value: distribution.percentiles['75th'], color: '#4caf50' },
        { name: '90th Percentile', value: distribution.percentiles['90th'], color: '#9c27b0' },
      ];
    }

    // If it's the mock data format with performance categories
    if (typeof distribution.excellent === 'number') {
      return [
        { name: 'Excellent', value: distribution.excellent, color: '#4caf50' },
        { name: 'Good', value: distribution.good, color: '#8bc34a' },
        { name: 'Average', value: distribution.average, color: '#ff9800' },
        { name: 'Needs Help', value: distribution.needs_help, color: '#f44336' },
        { name: 'At Risk', value: distribution.at_risk, color: '#d32f2f' },
      ];
    }

    return [];
  }, [distribution]);

  const maxValue = Math.max(...chartData.map(item => item.value));

  return (
    <Box sx={{ width: '100%', height: '320px' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="name" 
            angle={-45}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 10 }}
          />
          <YAxis 
            tick={{ fontSize: 10 }}
            domain={[0, showPercentiles ? 100 : 'dataMax']}
          />
          <Tooltip 
            formatter={(value: any) => [
              showPercentiles ? `${value}%` : `${value} students`, 
              'Value'
            ]}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
          
          {showPercentiles && (
            <>
              <ReferenceLine y={50} stroke="#666" strokeDasharray="2 2" label="Median" />
              <ReferenceLine y={75} stroke="#4caf50" strokeDasharray="2 2" label="Target" />
            </>
          )}
        </BarChart>
      </ResponsiveContainer>
      
      {distribution?.student_count && (
        <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
          Total Students: {distribution.student_count}
        </Typography>
      )}
    </Box>
  );
};

export default PerformanceDistributionChart;