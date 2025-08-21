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
  TextField,
  Button,
  Chip,
  Autocomplete,
  CircularProgress,
  Alert,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  School as SchoolIcon,
  Psychology as PsychologyIcon,
  Speed as SpeedIcon,
  EmojiEvents as EmojiEventsIcon,
} from '@mui/icons-material';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';

// Components
import TimeSeriesChart from '../components/charts/TimeSeriesChart';
import BarChart from '../components/charts/BarChart';
import MetricCard from '../components/widgets/MetricCard';
import ProgressHeatmap from '../components/charts/ProgressHeatmap';
import ConceptMasteryRadar from '../components/charts/ConceptMasteryRadar';
import LearningPathFlow from '../components/charts/LearningPathFlow';

// Services and Types
import { apiClient } from '../services/api-client';
import { useDashboardStore } from '../stores/dashboard-store';
import {
  StudentInsights,
  TimeRangeRequest,
  DashboardFilters,
  ConceptMastery,
  ProgressTracking,
} from '../types/api';

// Utilities
import { formatDate, formatPercentage, formatNumber } from '../utils/formatters';
import { exportStudentData } from '../utils/export-helpers';

const StudentsAnalytics: React.FC = () => {
  const queryClient = useQueryClient();
  const { filters, setFilters, loading, error, setLoading, setError } = useDashboardStore();
  
  // Local state
  const [selectedStudent, setSelectedStudent] = useState<string>('');
  const [availableStudents, setAvailableStudents] = useState<string[]>([]);
  const [exportLoading, setExportLoading] = useState(false);

  // Queries
  const { data: studentInsights, isLoading: insightsLoading, error: insightsError } = useQuery({
    queryKey: ['student-insights', selectedStudent, filters.timeRange],
    queryFn: () => selectedStudent ? apiClient.getStudentInsights(selectedStudent, {
      preset: filters.timeRange.preset,
    }) : null,
    enabled: !!selectedStudent,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: mockStudentProgress, isLoading: mockLoading } = useQuery({
    queryKey: ['mock-student-progress'],
    queryFn: () => apiClient.getMockStudentProgress(),
    staleTime: 60000, // Cache for 1 minute
  });

  // Effects
  useEffect(() => {
    // Load available students (in real implementation, this would come from API)
    const students = [
      'student_001', 'student_002', 'student_003', 'student_004', 'student_005',
      'student_006', 'student_007', 'student_008', 'student_009', 'student_010'
    ];
    setAvailableStudents(students);
    
    if (!selectedStudent && students.length > 0) {
      setSelectedStudent(students[0]);
    }
  }, [selectedStudent]);

  // Handlers
  const handleRefresh = async () => {
    setLoading(true);
    try {
      await queryClient.invalidateQueries({ queryKey: ['student-insights'] });
      await queryClient.invalidateQueries({ queryKey: ['mock-student-progress'] });
    } catch (err) {
      setError('Failed to refresh data');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!studentInsights) return;
    
    setExportLoading(true);
    try {
      await exportStudentData(studentInsights, selectedStudent);
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

  const handleStudentChange = (newStudent: string) => {
    setSelectedStudent(newStudent);
  };

  // Render loading state
  if (insightsLoading || mockLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading student analytics...
        </Typography>
      </Box>
    );
  }

  // Render error state
  if (insightsError || error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Error loading student data: {insightsError?.message || error}
      </Alert>
    );
  }

  const progressData = studentInsights?.progress_tracking || mockStudentProgress?.progress_metrics;
  const conceptData = studentInsights?.concept_mastery || mockStudentProgress?.concept_breakdown || [];

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Student Analytics
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
              disabled={!studentInsights || exportLoading}
            >
              {exportLoading ? 'Exporting...' : 'Export'}
            </Button>
          </Box>
        </Box>

        {/* Filters */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Student</InputLabel>
                <Select
                  value={selectedStudent}
                  onChange={(e) => handleStudentChange(e.target.value)}
                  label="Student"
                >
                  {availableStudents.map((student) => (
                    <MenuItem key={student} value={student}>
                      {student}
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
            
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {filters.selectedAgents.map((agent) => (
                  <Chip
                    key={agent}
                    label={agent}
                    onDelete={() => {
                      setFilters({
                        ...filters,
                        selectedAgents: filters.selectedAgents.filter(a => a !== agent)
                      });
                    }}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Key Metrics */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Overall Progress"
              value={progressData?.overall_score || 0}
              unit="%"
              icon={<SchoolIcon />}
              trend={progressData?.learning_velocity || 0}
              color="primary"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Learning Velocity"
              value={progressData?.learning_velocity || 0}
              unit="pts/day"
              icon={<SpeedIcon />}
              trend={5.2}
              color="success"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Engagement Score"
              value={progressData?.engagement_score || 0}
              unit="%"
              icon={<PsychologyIcon />}
              trend={-2.1}
              color="warning"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Concepts Mastered"
              value={progressData?.concepts_mastered || 0}
              unit={`/${progressData?.total_concepts || 0}`}
              icon={<EmojiEventsIcon />}
              trend={8.7}
              color="info"
            />
          </Grid>
        </Grid>

        {/* Progress Timeline */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 3, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Learning Progress Timeline
              </Typography>
              <TimeSeriesChart
                data={mockStudentProgress?.time_series || []}
                xKey="date"
                yKey="score"
                title="Progress Over Time"
                color="#1976d2"
              />
            </Paper>
          </Grid>
          
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Concept Mastery Distribution
              </Typography>
              <BarChart
                data={conceptData.map((concept) => ({
                  name: concept.concept,
                  value: concept.mastery_score * 100,
                }))}
                xKey="name"
                yKey="value"
                color="#4caf50"
              />
            </Paper>
          </Grid>
        </Grid>

        {/* Detailed Analytics */}
        <Grid container spacing={3}>
          <Grid item xs={12} lg={6}>
            <Paper sx={{ p: 3, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Concept Mastery Radar
              </Typography>
              <ConceptMasteryRadar concepts={conceptData} />
            </Paper>
          </Grid>
          
          <Grid item xs={12} lg={6}>
            <Paper sx={{ p: 3, height: 400 }}>
              <Typography variant="h6" gutterBottom>
                Progress Heatmap
              </Typography>
              <ProgressHeatmap 
                data={mockStudentProgress?.time_series || []}
                studentId={selectedStudent}
              />
            </Paper>
          </Grid>
          
          {/* Learning Path Visualization */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3, minHeight: 300 }}>
              <Typography variant="h6" gutterBottom>
                Learning Path Progress
              </Typography>
              <LearningPathFlow 
                concepts={conceptData}
                studentId={selectedStudent}
              />
            </Paper>
          </Grid>
        </Grid>

        {/* ML Predictions */}
        {studentInsights?.predictions && (
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  ML-Powered Predictions
                </Typography>
                
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Card variant="outlined">
                      <CardContent>
                        <Typography variant="subtitle2" color="textSecondary">
                          Predicted Success Rate
                        </Typography>
                        <Typography variant="h4" color="primary">
                          {formatPercentage(studentInsights.predictions.success_rate.predicted_value)}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Confidence: {formatPercentage(studentInsights.predictions.success_rate.confidence)}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  {studentInsights.predictions.learning_velocity && (
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle2" color="textSecondary">
                            Predicted Learning Velocity
                          </Typography>
                          <Typography variant="h4" color="success.main">
                            {formatNumber(studentInsights.predictions.learning_velocity.predicted_value)} pts/day
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Confidence: {formatPercentage(studentInsights.predictions.learning_velocity.confidence)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  )}
                  
                  {studentInsights.predictions.engagement_score && (
                    <Grid item xs={12} md={4}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle2" color="textSecondary">
                            Predicted Engagement
                          </Typography>
                          <Typography variant="h4" color="warning.main">
                            {formatPercentage(studentInsights.predictions.engagement_score.predicted_value)}
                          </Typography>
                          <Typography variant="body2" color="textSecondary">
                            Confidence: {formatPercentage(studentInsights.predictions.engagement_score.confidence)}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  )}
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
    </LocalizationProvider>
  );
};

export default StudentsAnalytics;