import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Chip,
  TextField,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  Schedule as ScheduleIcon,
  GetApp as GetAppIcon,
  CloudDownload as CloudDownloadIcon,
  TableChart as TableChartIcon,
  Description as DescriptionIcon,
  PictureAsPdf as PdfIcon,
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

// Services and Types
import { apiClient } from '../services/api-client';
import { useDashboardStore } from '../stores/dashboard-store';
import {
  ExportFormat,
  DataType,
  ExportRequest,
  TimeRangeRequest,
} from '../types/api';

// Utilities
import { formatBytes, formatDate } from '../utils/formatters';
import { generatePDFReport, generateExcelReport } from '../utils/export-helpers';

interface ExportJob {
  id: string;
  name: string;
  format: ExportFormat;
  dataType: DataType;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  fileSize?: number;
  downloadUrl?: string;
  createdAt: string;
  completedAt?: string;
  error?: string;
}

const DataExports: React.FC = () => {
  const queryClient = useQueryClient();
  const { setError } = useDashboardStore();
  
  // Local state
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([]);
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('json');
  const [selectedDataType, setSelectedDataType] = useState<DataType>('interactions');
  const [dateRange, setDateRange] = useState<{
    start: Date | null;
    end: Date | null;
  }>({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
    end: new Date(),
  });
  const [includeHeaders, setIncludeHeaders] = useState(true);
  const [customFileName, setCustomFileName] = useState('');
  const [selectedFilters, setSelectedFilters] = useState({
    agents: [] as string[],
    users: [] as string[],
    successOnly: false,
  });

  // Mock data for available filters
  const availableAgents = ['kinematics', 'forces', 'energy', 'momentum', 'angular_motion'];
  const availableUsers = ['student_001', 'student_002', 'student_003', 'student_004'];

  // Export mutation
  const exportMutation = useMutation({
    mutationFn: async (exportRequest: ExportRequest) => {
      return apiClient.exportData(exportRequest);
    },
    onSuccess: (data, variables) => {
      if (variables.export_format === 'json') {
        // Handle JSON response
        handleExportSuccess(data, variables);
      } else {
        // Handle blob response
        handleBlobDownload(data as Blob, variables);
      }
    },
    onError: (error) => {
      setError(`Export failed: ${error.message}`);
    },
  });

  // Handlers
  const handleExportClick = () => {
    setExportDialogOpen(true);
  };

  const handleExportSubmit = async () => {
    if (!dateRange.start || !dateRange.end) {
      setError('Please select a valid date range');
      return;
    }

    const exportRequest: ExportRequest = {
      export_format: selectedFormat,
      data_type: selectedDataType,
      time_range: {
        start_date: dateRange.start.toISOString(),
        end_date: dateRange.end.toISOString(),
      },
      filters: {
        agent_types: selectedFilters.agents,
        user_ids: selectedFilters.users,
        success_only: selectedFilters.successOnly,
      },
      include_headers: includeHeaders,
    };

    // Create export job
    const job: ExportJob = {
      id: Date.now().toString(),
      name: customFileName || `${selectedDataType}_${selectedFormat}_${formatDate(new Date())}`,
      format: selectedFormat,
      dataType: selectedDataType,
      status: 'pending',
      progress: 0,
      createdAt: new Date().toISOString(),
    };

    setExportJobs(prev => [job, ...prev]);
    setExportDialogOpen(false);

    // Simulate export progress
    simulateExportProgress(job.id);

    // Execute export
    exportMutation.mutate(exportRequest);
  };

  const handleExportSuccess = (data: any, request: ExportRequest) => {
    // Update job status
    setExportJobs(prev => prev.map(job => 
      job.status === 'processing' 
        ? { 
            ...job, 
            status: 'completed', 
            progress: 100,
            completedAt: new Date().toISOString(),
            fileSize: JSON.stringify(data).length,
          }
        : job
    ));
  };

  const handleBlobDownload = (blob: Blob, request: ExportRequest) => {
    const url = URL.createObjectURL(blob);
    const filename = `${request.data_type}_export_${formatDate(new Date())}.${request.export_format}`;
    
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    // Update job status
    setExportJobs(prev => prev.map(job => 
      job.status === 'processing' 
        ? { 
            ...job, 
            status: 'completed', 
            progress: 100,
            completedAt: new Date().toISOString(),
            fileSize: blob.size,
            downloadUrl: url,
          }
        : job
    ));
  };

  const simulateExportProgress = (jobId: string) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 20;
      if (progress >= 90) {
        progress = 90;
        clearInterval(interval);
      }
      
      setExportJobs(prev => prev.map(job => 
        job.id === jobId 
          ? { ...job, status: 'processing', progress: Math.min(progress, 90) }
          : job
      ));
    }, 500);
  };

  const handleDeleteJob = (jobId: string) => {
    setExportJobs(prev => prev.filter(job => job.id !== jobId));
  };

  const handleDownloadJob = (job: ExportJob) => {
    if (job.downloadUrl) {
      const link = document.createElement('a');
      link.href = job.downloadUrl;
      link.download = `${job.name}.${job.format}`;
      link.click();
    }
  };

  const getFormatIcon = (format: ExportFormat) => {
    switch (format) {
      case 'json': return <DescriptionIcon />;
      case 'csv': return <TableChartIcon />;
      case 'excel': return <GetAppIcon />;
      default: return <DownloadIcon />;
    }
  };

  const getStatusColor = (status: ExportJob['status']) => {
    switch (status) {
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'processing': return 'warning';
      default: return 'default';
    }
  };

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Data Exports
          </Typography>
          
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleExportClick}
          >
            New Export
          </Button>
        </Box>

        {/* Export Templates */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Quick Export Templates
            </Typography>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TableChartIcon color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">
                    Student Analytics Report
                  </Typography>
                </Box>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Complete student progress and performance data including concept mastery and learning paths.
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip label="CSV" size="small" />
                  <Chip label="Excel" size="small" />
                  <Chip label="PDF" size="small" />
                </Box>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => generateExcelReport('students')}>
                  Download Excel
                </Button>
                <Button size="small" onClick={() => generatePDFReport('students')}>
                  Download PDF
                </Button>
              </CardActions>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <CloudDownloadIcon color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="h6">
                    System Metrics Export
                  </Typography>
                </Box>
                <Typography variant="body2" color="textSecondary" paragraph>
                  System performance data, cache statistics, and health monitoring information.
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip label="JSON" size="small" />
                  <Chip label="CSV" size="small" />
                </Box>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => setSelectedDataType('analytics')}>
                  Quick Export
                </Button>
              </CardActions>
            </Card>
          </Grid>
          
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <ScheduleIcon color="success" sx={{ mr: 1 }} />
                  <Typography variant="h6">
                    Interaction Summary
                  </Typography>
                </Box>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Detailed interaction logs with agent usage patterns and response analytics.
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Chip label="JSON" size="small" />
                  <Chip label="CSV" size="small" />
                </Box>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => setSelectedDataType('interactions')}>
                  Quick Export
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>

        {/* Export Jobs */}
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Export History
              </Typography>
              
              {exportJobs.length === 0 ? (
                <Alert severity="info">
                  No export jobs yet. Create your first export using the "New Export" button above.
                </Alert>
              ) : (
                <List>
                  {exportJobs.map((job, index) => (
                    <React.Fragment key={job.id}>
                      <ListItem>
                        <ListItemIcon>
                          {getFormatIcon(job.format)}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="subtitle1">
                                {job.name}
                              </Typography>
                              <Chip
                                label={job.status}
                                size="small"
                                color={getStatusColor(job.status) as any}
                                variant="outlined"
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" color="textSecondary">
                                {job.dataType} • {job.format.toUpperCase()} • Created: {formatDate(job.createdAt)}
                                {job.fileSize && ` • Size: ${formatBytes(job.fileSize)}`}
                              </Typography>
                              {job.status === 'processing' && (
                                <LinearProgress 
                                  variant="determinate" 
                                  value={job.progress} 
                                  sx={{ mt: 1, width: '200px' }}
                                />
                              )}
                              {job.error && (
                                <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                                  Error: {job.error}
                                </Typography>
                              )}
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            {job.status === 'completed' && (
                              <IconButton
                                edge="end"
                                onClick={() => handleDownloadJob(job)}
                                color="primary"
                              >
                                <DownloadIcon />
                              </IconButton>
                            )}
                            <IconButton
                              edge="end"
                              onClick={() => handleDeleteJob(job.id)}
                              color="error"
                            >
                              <DeleteIcon />
                            </IconButton>
                          </Box>
                        </ListItemSecondaryAction>
                      </ListItem>
                      {index < exportJobs.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              )}
            </Paper>
          </Grid>
        </Grid>

        {/* Export Dialog */}
        <Dialog
          open={exportDialogOpen}
          onClose={() => setExportDialogOpen(false)}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>Create New Export</DialogTitle>
          <DialogContent>
            <Grid container spacing={3} sx={{ mt: 1 }}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Export Format</InputLabel>
                  <Select
                    value={selectedFormat}
                    onChange={(e) => setSelectedFormat(e.target.value as ExportFormat)}
                    label="Export Format"
                  >
                    <MenuItem value="json">JSON</MenuItem>
                    <MenuItem value="csv">CSV</MenuItem>
                    <MenuItem value="excel">Excel</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Data Type</InputLabel>
                  <Select
                    value={selectedDataType}
                    onChange={(e) => setSelectedDataType(e.target.value as DataType)}
                    label="Data Type"
                  >
                    <MenuItem value="interactions">Interactions</MenuItem>
                    <MenuItem value="analytics">Analytics</MenuItem>
                    <MenuItem value="summary">Summary</MenuItem>
                    <MenuItem value="students">Students</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <DatePicker
                  label="Start Date"
                  value={dateRange.start}
                  onChange={(newValue) => setDateRange(prev => ({ ...prev, start: newValue }))}
                  slotProps={{ textField: { fullWidth: true } }}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <DatePicker
                  label="End Date"
                  value={dateRange.end}
                  onChange={(newValue) => setDateRange(prev => ({ ...prev, end: newValue }))}
                  slotProps={{ textField: { fullWidth: true } }}
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Custom File Name (optional)"
                  value={customFileName}
                  onChange={(e) => setCustomFileName(e.target.value)}
                  placeholder="Leave empty for auto-generated name"
                />
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Filters
                </Typography>
                
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={selectedFilters.successOnly}
                      onChange={(e) => setSelectedFilters(prev => ({ 
                        ...prev, 
                        successOnly: e.target.checked 
                      }))}
                    />
                  }
                  label="Success only"
                />
                
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={includeHeaders}
                      onChange={(e) => setIncludeHeaders(e.target.checked)}
                    />
                  }
                  label="Include headers"
                />
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setExportDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleExportSubmit}
              variant="contained"
              disabled={exportMutation.isPending}
            >
              {exportMutation.isPending ? 'Creating...' : 'Create Export'}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </LocalizationProvider>
  );
};

export default DataExports;