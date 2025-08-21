import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Chip,
  CircularProgress,
  Alert,
  Tooltip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Group as GroupIcon,
  TrendingUp as TrendingUpIcon,
  Timer as TimerIcon,
  CheckCircle as CheckCircleIcon,
  Psychology as PsychologyIcon,
  School as SchoolIcon,
} from '@mui/icons-material';
import { useQuery, useQueryClient } from '@tanstack/react-query';

// Components
import TimeSeriesChart from '../components/charts/TimeSeriesChart';
import BarChart from '../components/charts/BarChart';
import MetricCard from '../components/widgets/MetricCard';
import PerformanceDistributionChart from '../components/charts/PerformanceDistributionChart';
import ClassComparisonChart from '../components/charts/ClassComparisonChart';

// Services and Types
import { apiClient } from '../services/api-client';
import { useDashboardStore } from '../stores/dashboard-store';
import {
  ClassOverview as ClassOverviewType,
  TimeRangeRequest,
  TopPerformer,
  PerformanceDistribution,
  MockClassAnalytics,
} from '../types/api';

// Utilities
import { formatPercentage, formatNumber, formatDuration } from '../utils/formatters';
import { exportClassData } from '../utils/export-helpers';

const ClassOverview: React.FC = () => {
  const queryClient = useQueryClient();
  const { filters, setFilters, loading, error, setLoading, setError } = useDashboardStore();
  
  // Local state
  const [selectedClasses, setSelectedClasses] = useState<string[]>(['class_001']);
  const [availableClasses, setAvailableClasses] = useState<string[]>([]);
  const [exportLoading, setExportLoading] = useState(false);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Queries
  const { data: classOverview, isLoading: overviewLoading, error: overviewError } = useQuery({
    queryKey: ['class-overview', selectedClasses, filters.timeRange],
    queryFn: () => apiClient.getClassOverview({
      class_ids: selectedClasses,
      preset: filters.timeRange.preset,
      include_comparisons: true,
    }),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: mockClassAnalytics, isLoading: mockLoading } = useQuery({
    queryKey: ['mock-class-analytics'],
    queryFn: () => apiClient.getMockClassAnalytics(),
    staleTime: 60000, // Cache for 1 minute
  });

  // Effects
  useEffect(() => {
    // Load available classes (in real implementation, this would come from API)
    const classes = [
      'class_001', 'class_002', 'class_003', 'class_004', 'class_005',
      'Physics 101A', 'Physics 101B', 'Physics 102A', 'Physics 102B', 'Advanced Physics'
    ];
    setAvailableClasses(classes);
  }, []);

  // Handlers
  const handleRefresh = async () => {
    setLoading(true);
    try {
      await queryClient.invalidateQueries({ queryKey: ['class-overview'] });
      await queryClient.invalidateQueries({ queryKey: ['mock-class-analytics'] });
    } catch (err) {
      setError('Failed to refresh data');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!classOverview) return;
    
    setExportLoading(true);
    try {
      await exportClassData(classOverview, selectedClasses);
    } catch (err) {
      setError('Failed to export data');
    } finally {
      setExportLoading(false);
    }
  };

  const handleTimeRangeChange = (newTimeRange: Partial<TimeRangeRequest>) => {
    setFilters({
      ...filters,
      timeRange: { ...filters.timeRange, ...newTimeRange }
    });
  };

  const handleClassChange = (classes: string[]) => {
    setSelectedClasses(classes);
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Render loading state
  if (overviewLoading || mockLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading class overview...
        </Typography>
      </Box>
    );
  }

  // Render error state
  if (overviewError || error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Error loading class data: {overviewError?.message || error}
      </Alert>
    );
  }

  const classStats = classOverview?.class_statistics || mockClassAnalytics?.summary;
  const perfDistribution = classOverview?.performance_distribution || mockClassAnalytics?.performance_distribution;
  const topPerformers = classOverview?.top_performers || [];

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Class Overview
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={!classOverview || exportLoading}
          >
            {exportLoading ? 'Exporting...' : 'Export'}
          </Button>
        </Box>
      </Box>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth>
              <InputLabel>Classes</InputLabel>
              <Select
                multiple
                value={selectedClasses}
                onChange={(e) => handleClassChange(e.target.value as string[])}
                label="Classes"
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {availableClasses.map((className) => (
                  <MenuItem key={className} value={className}>
                    {className}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={filters.timeRange.preset || '7d'}
                onChange={(e) => handleTimeRangeChange({ preset: e.target.value as any })}
                label="Time Range"
              >
                <MenuItem value="1h">Last Hour</MenuItem>
                <MenuItem value="6h">Last 6 Hours</MenuItem>
                <MenuItem value="24h">Last 24 Hours</MenuItem>
                <MenuItem value="7d">Last 7 Days</MenuItem>
                <MenuItem value="30d">Last 30 Days</MenuItem>
                <MenuItem value="90d">Last 90 Days</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Students"
            value={classStats?.total_students || 0}
            unit=""
            icon={<GroupIcon />}
            trend={4.2}
            color="primary"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Students"
            value={classStats?.active_students || 0}
            unit=""
            icon={<SchoolIcon />}
            trend={2.1}
            color="success"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Average Progress"
            value={classStats?.avg_progress || 0}
            unit="%"
            icon={<TrendingUpIcon />}
            trend={-1.5}
            color="warning"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Completion Rate"
            value={classStats?.completion_rate || 0}
            unit="%"
            icon={<CheckCircleIcon />}
            trend={6.8}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Performance Distribution */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Class Performance Distribution
            </Typography>
            <PerformanceDistributionChart 
              distribution={perfDistribution}
              showPercentiles={true}
            />
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: 400 }}>
            <Typography variant="h6" gutterBottom>
              Performance Categories
            </Typography>
            <BarChart
              data={[
                { name: 'Excellent', value: perfDistribution?.excellent || 0, color: '#4caf50' },
                { name: 'Good', value: perfDistribution?.good || 0, color: '#8bc34a' },
                { name: 'Average', value: perfDistribution?.average || 0, color: '#ff9800' },
                { name: 'Needs Help', value: perfDistribution?.needs_help || 0, color: '#f44336' },
                { name: 'At Risk', value: perfDistribution?.at_risk || 0, color: '#d32f2f' },
              ]}
              xKey="name"
              yKey="value"
              showLabels={true}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Top Performers Table */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Top Performers
            </Typography>
            
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Student ID</TableCell>
                    <TableCell align="right">Success Rate</TableCell>
                    <TableCell align="right">Interactions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {topPerformers
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((performer) => (
                    <TableRow key={performer.user_id}>
                      <TableCell component="th" scope="row">
                        {performer.user_id}
                      </TableCell>
                      <TableCell align="right">
                        {formatPercentage(performer.success_rate)}
                      </TableCell>
                      <TableCell align="right">
                        {formatNumber(performer.interaction_count)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            
            <TablePagination
              rowsPerPageOptions={[5, 10, 25]}
              component="div"
              count={topPerformers.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </Paper>
        </Grid>
        
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Class Comparison
            </Typography>
            <ClassComparisonChart 
              classes={selectedClasses}
              timeRange={filters.timeRange}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Detailed Analytics */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Learning Analytics Summary
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Average Response Time
                    </Typography>
                    <Typography variant="h4" color="primary">
                      {formatDuration(classStats?.avg_response_time || 0)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      System performance metric
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Total Interactions
                    </Typography>
                    <Typography variant="h4" color="success.main">
                      {formatNumber(classStats?.total_interactions || 0)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Student engagement level
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Card variant="outlined">
                  <CardContent>
                    <Typography variant="subtitle2" color="textSecondary">
                      Class Success Rate
                    </Typography>
                    <Typography variant="h4" color="info.main">
                      {formatPercentage(classStats?.class_success_rate || 0)}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Overall performance indicator
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        
        {/* Learning Insights */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Learning Insights & Recommendations
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    High Engagement Detected
                  </Typography>
                  <Typography variant="body2">
                    Students are showing increased interaction rates this week. 
                    Consider introducing more challenging problems.
                  </Typography>
                </Alert>
                
                <Alert severity="warning" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Performance Gap Identified
                  </Typography>
                  <Typography variant="body2">
                    {perfDistribution?.at_risk || 0} students may need additional support. 
                    Review their learning paths for intervention opportunities.
                  </Typography>
                </Alert>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Alert severity="success" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Strong Concept Mastery
                  </Typography>
                  <Typography variant="body2">
                    The class shows excellent understanding of fundamental concepts. 
                    Ready for advanced topics in next module.
                  </Typography>
                </Alert>
                
                <Alert severity="info">
                  <Typography variant="subtitle2" gutterBottom>
                    Optimal Learning Pace
                  </Typography>
                  <Typography variant="body2">
                    Current pacing aligns well with curriculum goals. 
                    Maintain current instruction strategy.
                  </Typography>
                </Alert>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ClassOverview;