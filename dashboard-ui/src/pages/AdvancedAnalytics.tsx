/**
 * Advanced Analytics Page
 * Comprehensive interface for predictive analytics, comparative analysis, 
 * content effectiveness, statistical analysis, and automated insights
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Tab,
  Tabs,
  Chip,
  Button,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Psychology as PredictiveIcon,
  Compare as CompareIcon,
  School as ContentIcon,
  Analytics as StatisticalIcon,
  Lightbulb as InsightsIcon,
  Refresh as RefreshIcon,
  Download as ExportIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  CheckCircle as SuccessIcon,
  ModelTraining as ModelIcon,
} from '@mui/icons-material';

import { apiClient } from '../services/api-client';
import { useDashboardStore } from '../stores/dashboard-store';
import MetricCard from '../components/widgets/MetricCard';

// ============================================================================
// Types and Interfaces
// ============================================================================

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

interface PredictiveAnalysisRequest {
  student_ids?: string[];
  prediction_type: 'success' | 'engagement' | 'performance' | 'risk';
  horizon_days: number;
  include_confidence_intervals: boolean;
  include_contributing_factors: boolean;
}

interface ComparativeAnalysisRequest {
  analysis_type: 'cohort' | 'temporal' | 'ab_test' | 'benchmark';
  primary_entities: string[];
  comparison_entities: string[];
  metrics: string[];
  time_range: { preset: string };
  statistical_tests: boolean;
  effect_size_calculation: boolean;
}

interface ContentEffectivenessRequest {
  content_ids?: string[];
  content_types?: string[];
  analysis_depth: 'basic' | 'standard' | 'comprehensive';
  include_recommendations: boolean;
  time_window_days: number;
}

interface StatisticalAnalysisRequest {
  analysis_type: 'timeseries' | 'clustering' | 'correlation' | 'anomaly';
  metrics: string[];
  time_range: { preset: string };
  granularity: '1H' | '6H' | '1D' | '1W';
  advanced_options?: Record<string, any>;
}

interface InsightGenerationRequest {
  insight_types: string[];
  time_window_days: number;
  importance_threshold: 'low' | 'medium' | 'high' | 'critical';
  target_audience: 'educator' | 'administrator' | 'student';
  include_natural_language: boolean;
}

// ============================================================================
// Tab Panel Component
// ============================================================================

function TabPanel({ children, value, index, ...other }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`advanced-analytics-tabpanel-${index}`}
      aria-labelledby={`advanced-analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

const AdvancedAnalytics: React.FC = () => {
  const theme = useTheme();
  
  // Store state
  const {
    loading,
    errors,
    ui: { selectedTimeRange },
    setLoading,
    setError,
    addAlert,
  } = useDashboardStore();

  // Local state
  const [currentTab, setCurrentTab] = useState(0);
  const [analysisResults, setAnalysisResults] = useState<Record<string, any>>({});
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogType, setDialogType] = useState<string>('');
  const [modelTrainingStatus, setModelTrainingStatus] = useState<string>('idle');

  // Analysis request states
  const [predictiveRequest, setPredictiveRequest] = useState<PredictiveAnalysisRequest>({
    prediction_type: 'success',
    horizon_days: 7,
    include_confidence_intervals: true,
    include_contributing_factors: true,
  });

  const [comparativeRequest, setComparativeRequest] = useState<ComparativeAnalysisRequest>({
    analysis_type: 'cohort',
    primary_entities: [],
    comparison_entities: [],
    metrics: ['success_rate', 'engagement_score'],
    time_range: { preset: selectedTimeRange },
    statistical_tests: true,
    effect_size_calculation: true,
  });

  const [contentRequest, setContentRequest] = useState<ContentEffectivenessRequest>({
    analysis_depth: 'standard',
    include_recommendations: true,
    time_window_days: 30,
  });

  const [statisticalRequest, setStatisticalRequest] = useState<StatisticalAnalysisRequest>({
    analysis_type: 'timeseries',
    metrics: ['interaction_count', 'success_rate'],
    time_range: { preset: selectedTimeRange },
    granularity: '1D',
  });

  const [insightRequest, setInsightRequest] = useState<InsightGenerationRequest>({
    insight_types: ['trend', 'anomaly', 'performance', 'recommendation'],
    time_window_days: 7,
    importance_threshold: 'medium',
    target_audience: 'educator',
    include_natural_language: true,
  });

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleRunAnalysis = async (analysisType: string) => {
    try {
      setLoading(analysisType, true);
      setError(analysisType, null);

      let result;

      switch (analysisType) {
        case 'predictive':
          result = await apiClient.post('/dashboard/analytics/predictive', predictiveRequest);
          break;
        case 'comparative':
          result = await apiClient.post('/dashboard/analytics/comparative', comparativeRequest);
          break;
        case 'content':
          result = await apiClient.post('/dashboard/analytics/content-effectiveness', contentRequest);
          break;
        case 'statistical':
          result = await apiClient.post('/dashboard/analytics/statistical', statisticalRequest);
          break;
        case 'insights':
          result = await apiClient.post('/dashboard/analytics/insights', insightRequest);
          break;
        default:
          throw new Error(`Unknown analysis type: ${analysisType}`);
      }

      setAnalysisResults(prev => ({
        ...prev,
        [analysisType]: result.data,
      }));

      addAlert({
        id: `analysis-${analysisType}-${Date.now()}`,
        type: 'success',
        message: `${analysisType.charAt(0).toUpperCase() + analysisType.slice(1)} analysis completed successfully`,
      });

    } catch (error: any) {
      console.error(`${analysisType} analysis error:`, error);
      setError(analysisType, error.response?.data?.detail || error.message || 'Analysis failed');
      
      addAlert({
        id: `analysis-error-${analysisType}-${Date.now()}`,
        type: 'error',
        message: `Failed to run ${analysisType} analysis: ${error.response?.data?.detail || error.message}`,
      });
    } finally {
      setLoading(analysisType, false);
    }
  };

  const handleModelTraining = async () => {
    try {
      setModelTrainingStatus('training');
      
      const result = await apiClient.post('/dashboard/analytics/train-models', {
        model_types: ['predictive', 'content_effectiveness'],
        force_retrain: false,
      });

      addAlert({
        id: `model-training-${Date.now()}`,
        type: 'info',
        message: 'Model training initiated. This process may take several minutes.',
      });

      setModelTrainingStatus('completed');
      
      setTimeout(() => {
        setModelTrainingStatus('idle');
      }, 5000);

    } catch (error: any) {
      console.error('Model training error:', error);
      setModelTrainingStatus('error');
      
      addAlert({
        id: `model-training-error-${Date.now()}`,
        type: 'error',
        message: `Model training failed: ${error.response?.data?.detail || error.message}`,
      });

      setTimeout(() => {
        setModelTrainingStatus('idle');
      }, 5000);
    }
  };

  const handleExportResults = (analysisType: string) => {
    const results = analysisResults[analysisType];
    if (!results) return;

    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${analysisType}_analysis_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
  };

  const openConfigDialog = (type: string) => {
    setDialogType(type);
    setDialogOpen(true);
  };

  // ============================================================================
  // Render Methods
  // ============================================================================

  const renderHeader = () => (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Advanced Analytics
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Comprehensive analytics suite with predictive modeling, statistical analysis, and automated insights
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<ModelIcon />}
            onClick={handleModelTraining}
            disabled={modelTrainingStatus === 'training'}
            color={modelTrainingStatus === 'completed' ? 'success' : modelTrainingStatus === 'error' ? 'error' : 'primary'}
          >
            {modelTrainingStatus === 'training' && <CircularProgress size={16} sx={{ mr: 1 }} />}
            {modelTrainingStatus === 'training' ? 'Training...' : 'Train Models'}
          </Button>
        </Box>
      </Box>
    </Box>
  );

  const renderTabNavigation = () => (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
      <Tabs value={currentTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
        <Tab 
          icon={<PredictiveIcon />} 
          label="Predictive Analytics" 
          id="advanced-analytics-tab-0"
          aria-controls="advanced-analytics-tabpanel-0"
        />
        <Tab 
          icon={<CompareIcon />} 
          label="Comparative Analysis" 
          id="advanced-analytics-tab-1"
          aria-controls="advanced-analytics-tabpanel-1"
        />
        <Tab 
          icon={<ContentIcon />} 
          label="Content Effectiveness" 
          id="advanced-analytics-tab-2"
          aria-controls="advanced-analytics-tabpanel-2"
        />
        <Tab 
          icon={<StatisticalIcon />} 
          label="Statistical Analysis" 
          id="advanced-analytics-tab-3"
          aria-controls="advanced-analytics-tabpanel-3"
        />
        <Tab 
          icon={<InsightsIcon />} 
          label="Automated Insights" 
          id="advanced-analytics-tab-4"
          aria-controls="advanced-analytics-tabpanel-4"
        />
      </Tabs>
    </Box>
  );

  const renderPredictiveAnalytics = () => (
    <TabPanel value={currentTab} index={0}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader 
              title="Predictive Analysis Configuration"
              action={
                <Button
                  variant="contained"
                  onClick={() => handleRunAnalysis('predictive')}
                  disabled={loading.predictive}
                  startIcon={loading.predictive ? <CircularProgress size={16} /> : <TrendingUpIcon />}
                >
                  {loading.predictive ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              }
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Prediction Type</InputLabel>
                  <Select
                    value={predictiveRequest.prediction_type}
                    onChange={(e) => setPredictiveRequest(prev => ({ 
                      ...prev, 
                      prediction_type: e.target.value as any 
                    }))}
                    label="Prediction Type"
                  >
                    <MenuItem value="success">Student Success</MenuItem>
                    <MenuItem value="engagement">Student Engagement</MenuItem>
                    <MenuItem value="performance">Academic Performance</MenuItem>
                    <MenuItem value="risk">At-Risk Students</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  label="Forecast Horizon (days)"
                  type="number"
                  value={predictiveRequest.horizon_days}
                  onChange={(e) => setPredictiveRequest(prev => ({ 
                    ...prev, 
                    horizon_days: parseInt(e.target.value) 
                  }))}
                  inputProps={{ min: 1, max: 30 }}
                  fullWidth
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={predictiveRequest.include_confidence_intervals}
                      onChange={(e) => setPredictiveRequest(prev => ({ 
                        ...prev, 
                        include_confidence_intervals: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Confidence Intervals"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={predictiveRequest.include_contributing_factors}
                      onChange={(e) => setPredictiveRequest(prev => ({ 
                        ...prev, 
                        include_contributing_factors: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Contributing Factors"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader 
              title="Predictive Analysis Results"
              action={
                analysisResults.predictive && (
                  <IconButton onClick={() => handleExportResults('predictive')}>
                    <ExportIcon />
                  </IconButton>
                )
              }
            />
            <CardContent>
              {errors.predictive && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {errors.predictive}
                </Alert>
              )}
              
              {analysisResults.predictive ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    {analysisResults.predictive.prediction_type?.charAt(0).toUpperCase() + 
                     analysisResults.predictive.prediction_type?.slice(1)} Predictions
                  </Typography>
                  
                  {analysisResults.predictive.predictions ? (
                    <Grid container spacing={2}>
                      {analysisResults.predictive.predictions.slice(0, 6).map((prediction: any, index: number) => (
                        <Grid item xs={12} sm={6} md={4} key={index}>
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" color="textSecondary">
                                Student {prediction.student_id}
                              </Typography>
                              <Typography variant="h6">
                                {(prediction.predicted_value * 100).toFixed(1)}%
                              </Typography>
                              <Chip 
                                label={prediction.risk_level} 
                                size="small"
                                color={
                                  prediction.risk_level === 'low' ? 'success' :
                                  prediction.risk_level === 'medium' ? 'warning' : 'error'
                                }
                              />
                              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                                Confidence: {(prediction.confidence_score * 100).toFixed(1)}%
                              </Typography>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  ) : analysisResults.predictive.alerts ? (
                    <Box>
                      <Typography variant="body1" gutterBottom>
                        {analysisResults.predictive.alerts_generated} alerts generated
                      </Typography>
                      {analysisResults.predictive.alerts.slice(0, 5).map((alert: any, index: number) => (
                        <Alert 
                          key={index}
                          severity={alert.severity === 'critical' ? 'error' : 'warning'}
                          sx={{ mb: 1 }}
                        >
                          <Typography variant="subtitle2">{alert.alert_type}</Typography>
                          <Typography variant="body2">{alert.predicted_outcome}</Typography>
                        </Alert>
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body1" color="textSecondary">
                      No prediction results available
                    </Typography>
                  )}
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary" textAlign="center">
                  Run predictive analysis to see results here
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </TabPanel>
  );

  const renderComparativeAnalysis = () => (
    <TabPanel value={currentTab} index={1}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader 
              title="Comparative Analysis Configuration"
              action={
                <Button
                  variant="contained"
                  onClick={() => handleRunAnalysis('comparative')}
                  disabled={loading.comparative}
                  startIcon={loading.comparative ? <CircularProgress size={16} /> : <CompareIcon />}
                >
                  {loading.comparative ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              }
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Analysis Type</InputLabel>
                  <Select
                    value={comparativeRequest.analysis_type}
                    onChange={(e) => setComparativeRequest(prev => ({ 
                      ...prev, 
                      analysis_type: e.target.value as any 
                    }))}
                    label="Analysis Type"
                  >
                    <MenuItem value="cohort">Cohort Comparison</MenuItem>
                    <MenuItem value="temporal">Temporal Analysis</MenuItem>
                    <MenuItem value="ab_test">A/B Testing</MenuItem>
                    <MenuItem value="benchmark">Benchmark Analysis</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  label="Primary Entities (comma-separated)"
                  value={comparativeRequest.primary_entities.join(', ')}
                  onChange={(e) => setComparativeRequest(prev => ({ 
                    ...prev, 
                    primary_entities: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                  }))}
                  fullWidth
                  placeholder="high_performers, group_a"
                />

                <TextField
                  label="Comparison Entities (comma-separated)"
                  value={comparativeRequest.comparison_entities.join(', ')}
                  onChange={(e) => setComparativeRequest(prev => ({ 
                    ...prev, 
                    comparison_entities: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                  }))}
                  fullWidth
                  placeholder="average_performers, group_b"
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={comparativeRequest.statistical_tests}
                      onChange={(e) => setComparativeRequest(prev => ({ 
                        ...prev, 
                        statistical_tests: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Statistical Tests"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader 
              title="Comparative Analysis Results"
              action={
                analysisResults.comparative && (
                  <IconButton onClick={() => handleExportResults('comparative')}>
                    <ExportIcon />
                  </IconButton>
                )
              }
            />
            <CardContent>
              {errors.comparative && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {errors.comparative}
                </Alert>
              )}
              
              {analysisResults.comparative ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    {analysisResults.comparative.analysis_type?.replace('_', ' ')} Results
                  </Typography>
                  
                  {analysisResults.comparative.recommendations && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        Key Recommendations:
                      </Typography>
                      {analysisResults.comparative.recommendations.map((rec: string, index: number) => (
                        <Alert key={index} severity="info" sx={{ mb: 1 }}>
                          {rec}
                        </Alert>
                      ))}
                    </Box>
                  )}

                  {analysisResults.comparative.benchmarks && (
                    <Grid container spacing={2}>
                      {analysisResults.comparative.benchmarks.map((benchmark: any, index: number) => (
                        <Grid item xs={12} sm={6} key={index}>
                          <Card variant="outlined">
                            <CardContent>
                              <Typography variant="subtitle2" color="textSecondary">
                                {benchmark.metric_name}
                              </Typography>
                              <Typography variant="h6">
                                {benchmark.current_value?.toFixed(2)}
                              </Typography>
                              <Typography variant="body2" color="textSecondary">
                                Baseline: {benchmark.historical_baseline?.toFixed(2)}
                              </Typography>
                              <Chip 
                                label={benchmark.trend_direction} 
                                size="small"
                                color={
                                  benchmark.trend_direction === 'improving' ? 'success' :
                                  benchmark.trend_direction === 'declining' ? 'error' : 'default'
                                }
                                sx={{ mt: 1 }}
                              />
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  )}
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary" textAlign="center">
                  Run comparative analysis to see results here
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </TabPanel>
  );

  const renderContentEffectiveness = () => (
    <TabPanel value={currentTab} index={2}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader 
              title="Content Analysis Configuration"
              action={
                <Button
                  variant="contained"
                  onClick={() => handleRunAnalysis('content')}
                  disabled={loading.content}
                  startIcon={loading.content ? <CircularProgress size={16} /> : <ContentIcon />}
                >
                  {loading.content ? 'Analyzing...' : 'Run Analysis'}
                </Button>
              }
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Content IDs (comma-separated)"
                  value={contentRequest.content_ids?.join(', ') || ''}
                  onChange={(e) => setContentRequest(prev => ({ 
                    ...prev, 
                    content_ids: e.target.value.split(',').map(s => s.trim()).filter(Boolean)
                  }))}
                  fullWidth
                  placeholder="kinematics, forces, energy"
                />

                <FormControl fullWidth>
                  <InputLabel>Analysis Depth</InputLabel>
                  <Select
                    value={contentRequest.analysis_depth}
                    onChange={(e) => setContentRequest(prev => ({ 
                      ...prev, 
                      analysis_depth: e.target.value as any 
                    }))}
                    label="Analysis Depth"
                  >
                    <MenuItem value="basic">Basic</MenuItem>
                    <MenuItem value="standard">Standard</MenuItem>
                    <MenuItem value="comprehensive">Comprehensive</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  label="Time Window (days)"
                  type="number"
                  value={contentRequest.time_window_days}
                  onChange={(e) => setContentRequest(prev => ({ 
                    ...prev, 
                    time_window_days: parseInt(e.target.value) 
                  }))}
                  inputProps={{ min: 7, max: 90 }}
                  fullWidth
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={contentRequest.include_recommendations}
                      onChange={(e) => setContentRequest(prev => ({ 
                        ...prev, 
                        include_recommendations: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Recommendations"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader 
              title="Content Effectiveness Results"
              action={
                analysisResults.content && (
                  <IconButton onClick={() => handleExportResults('content')}>
                    <ExportIcon />
                  </IconButton>
                )
              }
            />
            <CardContent>
              {errors.content && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {errors.content}
                </Alert>
              )}
              
              {analysisResults.content && analysisResults.content.content_analyses ? (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Content Analysis Results ({analysisResults.content.total_content_analyzed} items)
                  </Typography>
                  
                  <Grid container spacing={2}>
                    {analysisResults.content.content_analyses.map((content: any, index: number) => (
                      <Grid item xs={12} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6" gutterBottom>
                              {content.content_id || 'Unknown Content'}
                            </Typography>
                            
                            <Grid container spacing={2}>
                              <Grid item xs={6} sm={3}>
                                <MetricCard
                                  title="Engagement Score"
                                  value={`${(content.engagement_score * 100).toFixed(1)}%`}
                                  color={content.engagement_score > 0.7 ? 'success' : content.engagement_score > 0.5 ? 'warning' : 'error'}
                                />
                              </Grid>
                              <Grid item xs={6} sm={3}>
                                <MetricCard
                                  title="Success Rate"
                                  value={`${(content.success_rate * 100).toFixed(1)}%`}
                                  color={content.success_rate > 0.8 ? 'success' : content.success_rate > 0.6 ? 'warning' : 'error'}
                                />
                              </Grid>
                              <Grid item xs={6} sm={3}>
                                <MetricCard
                                  title="Difficulty Rating"
                                  value={`${(content.difficulty_rating * 100).toFixed(0)}%`}
                                  color={content.difficulty_rating < 0.4 ? 'success' : content.difficulty_rating < 0.7 ? 'warning' : 'error'}
                                />
                              </Grid>
                              <Grid item xs={6} sm={3}>
                                <MetricCard
                                  title="Time to Mastery"
                                  value={`${content.time_to_mastery?.toFixed(1) || 'N/A'} hrs`}
                                  color="info"
                                />
                              </Grid>
                            </Grid>

                            {content.recommendations && content.recommendations.length > 0 && (
                              <Box sx={{ mt: 2 }}>
                                <Typography variant="subtitle1" gutterBottom>
                                  Recommendations:
                                </Typography>
                                {content.recommendations.slice(0, 3).map((rec: any, recIndex: number) => (
                                  <Alert key={recIndex} severity="info" sx={{ mb: 1 }}>
                                    <Typography variant="body2" fontWeight="medium">
                                      {rec.recommendation_type}: {rec.rationale}
                                    </Typography>
                                  </Alert>
                                ))}
                              </Box>
                            )}
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary" textAlign="center">
                  Run content effectiveness analysis to see results here
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </TabPanel>
  );

  const renderStatisticalAnalysis = () => (
    <TabPanel value={currentTab} index={3}>
      <Typography variant="h6">Statistical Analysis</Typography>
      <Typography variant="body1" color="textSecondary">
        Advanced statistical analysis tools including time-series analysis, clustering, and correlation analysis.
      </Typography>
      {/* TODO: Implement statistical analysis interface */}
    </TabPanel>
  );

  const renderAutomatedInsights = () => (
    <TabPanel value={currentTab} index={4}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardHeader 
              title="Insight Generation Configuration"
              action={
                <Button
                  variant="contained"
                  onClick={() => handleRunAnalysis('insights')}
                  disabled={loading.insights}
                  startIcon={loading.insights ? <CircularProgress size={16} /> : <InsightsIcon />}
                >
                  {loading.insights ? 'Generating...' : 'Generate Insights'}
                </Button>
              }
            />
            <CardContent>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Importance Threshold</InputLabel>
                  <Select
                    value={insightRequest.importance_threshold}
                    onChange={(e) => setInsightRequest(prev => ({ 
                      ...prev, 
                      importance_threshold: e.target.value as any 
                    }))}
                    label="Importance Threshold"
                  >
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                    <MenuItem value="critical">Critical</MenuItem>
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>Target Audience</InputLabel>
                  <Select
                    value={insightRequest.target_audience}
                    onChange={(e) => setInsightRequest(prev => ({ 
                      ...prev, 
                      target_audience: e.target.value as any 
                    }))}
                    label="Target Audience"
                  >
                    <MenuItem value="educator">Educator</MenuItem>
                    <MenuItem value="administrator">Administrator</MenuItem>
                    <MenuItem value="student">Student</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  label="Time Window (days)"
                  type="number"
                  value={insightRequest.time_window_days}
                  onChange={(e) => setInsightRequest(prev => ({ 
                    ...prev, 
                    time_window_days: parseInt(e.target.value) 
                  }))}
                  inputProps={{ min: 1, max: 30 }}
                  fullWidth
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={insightRequest.include_natural_language}
                      onChange={(e) => setInsightRequest(prev => ({ 
                        ...prev, 
                        include_natural_language: e.target.checked 
                      }))}
                    />
                  }
                  label="Include Natural Language Summary"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardHeader 
              title="Automated Insights"
              action={
                analysisResults.insights && (
                  <IconButton onClick={() => handleExportResults('insights')}>
                    <ExportIcon />
                  </IconButton>
                )
              }
            />
            <CardContent>
              {errors.insights && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {errors.insights}
                </Alert>
              )}
              
              {analysisResults.insights ? (
                <Box>
                  {analysisResults.insights.natural_language_summary && (
                    <Card variant="outlined" sx={{ mb: 3, backgroundColor: alpha(theme.palette.primary.main, 0.05) }}>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          {analysisResults.insights.natural_language_summary.headline}
                        </Typography>
                        <Typography variant="body1" paragraph>
                          {analysisResults.insights.natural_language_summary.narrative_text}
                        </Typography>
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Key Points:
                          </Typography>
                          <ul>
                            {analysisResults.insights.natural_language_summary.key_points?.map((point: string, index: number) => (
                              <li key={index}>
                                <Typography variant="body2">{point}</Typography>
                              </li>
                            ))}
                          </ul>
                        </Box>
                      </CardContent>
                    </Card>
                  )}

                  <Typography variant="h6" gutterBottom>
                    Generated Insights ({analysisResults.insights.insights_generated})
                  </Typography>
                  
                  {analysisResults.insights.insights?.map((insight: any, index: number) => (
                    <Card key={index} variant="outlined" sx={{ mb: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'between', mb: 1 }}>
                          <Typography variant="h6">
                            {insight.title}
                          </Typography>
                          <Chip 
                            label={insight.importance_level} 
                            size="small"
                            color={
                              insight.importance_level === 'critical' ? 'error' :
                              insight.importance_level === 'high' ? 'warning' :
                              insight.importance_level === 'medium' ? 'info' : 'default'
                            }
                          />
                        </Box>
                        
                        <Typography variant="body1" paragraph>
                          {insight.detailed_explanation}
                        </Typography>

                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          Confidence: {(insight.confidence_score * 100).toFixed(1)}%
                        </Typography>

                        {insight.actionable_recommendations && insight.actionable_recommendations.length > 0 && (
                          <Box>
                            <Typography variant="subtitle2" gutterBottom>
                              Recommendations:
                            </Typography>
                            <ul>
                              {insight.actionable_recommendations.map((rec: string, recIndex: number) => (
                                <li key={recIndex}>
                                  <Typography variant="body2">{rec}</Typography>
                                </li>
                              ))}
                            </ul>
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </Box>
              ) : (
                <Typography variant="body1" color="textSecondary" textAlign="center">
                  Generate automated insights to see results here
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </TabPanel>
  );

  // ============================================================================
  // Main Render
  // ============================================================================

  return (
    <Box>
      {renderHeader()}
      {renderTabNavigation()}
      {renderPredictiveAnalytics()}
      {renderComparativeAnalysis()}
      {renderContentEffectiveness()}
      {renderStatisticalAnalysis()}
      {renderAutomatedInsights()}
    </Box>
  );
};

export default AdvancedAnalytics;