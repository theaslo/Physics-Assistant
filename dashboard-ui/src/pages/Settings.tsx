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
  Switch,
  FormControlLabel,
  TextField,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Save as SaveIcon,
  Refresh as RefreshIcon,
  CloudSync as CloudSyncIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
  Palette as PaletteIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material';

// Services and Types
import { useDashboardStore } from '../stores/dashboard-store';
import { apiClient } from '../services/api-client';

interface DashboardSettings {
  theme: 'light' | 'dark' | 'auto';
  autoRefresh: boolean;
  refreshInterval: number;
  notifications: boolean;
  emailNotifications: boolean;
  realTimeUpdates: boolean;
  cacheSettings: {
    enabled: boolean;
    ttl: number;
    maxSize: number;
  };
  exportSettings: {
    defaultFormat: 'json' | 'csv' | 'excel';
    includeHeaders: boolean;
    maxExportSize: number;
  };
}

interface APIEndpoint {
  id: string;
  name: string;
  url: string;
  enabled: boolean;
  timeout: number;
}

const Settings: React.FC = () => {
  const { setError, addAlert } = useDashboardStore();
  
  // Local state
  const [settings, setSettings] = useState<DashboardSettings>({
    theme: 'light',
    autoRefresh: true,
    refreshInterval: 30000,
    notifications: true,
    emailNotifications: false,
    realTimeUpdates: true,
    cacheSettings: {
      enabled: true,
      ttl: 300,
      maxSize: 1000,
    },
    exportSettings: {
      defaultFormat: 'json',
      includeHeaders: true,
      maxExportSize: 100000,
    },
  });
  
  const [apiEndpoints, setApiEndpoints] = useState<APIEndpoint[]>([
    {
      id: '1',
      name: 'Dashboard API',
      url: 'http://localhost:8002',
      enabled: true,
      timeout: 30000,
    },
    {
      id: '2',
      name: 'Analytics API',
      url: 'http://localhost:8003',
      enabled: false,
      timeout: 60000,
    },
  ]);
  
  const [endpointDialogOpen, setEndpointDialogOpen] = useState(false);
  const [editingEndpoint, setEditingEndpoint] = useState<APIEndpoint | null>(null);
  const [unsavedChanges, setUnsavedChanges] = useState(false);
  const [testResults, setTestResults] = useState<Record<string, boolean>>({});

  // Handlers
  const handleSettingsChange = (
    category: keyof DashboardSettings,
    field: string,
    value: any
  ) => {
    setSettings(prev => ({
      ...prev,
      [category]: typeof prev[category] === 'object'
        ? { ...prev[category], [field]: value }
        : value,
    }));
    setUnsavedChanges(true);
  };

  const handleSaveSettings = async () => {
    try {
      // In a real implementation, this would save to backend
      localStorage.setItem('dashboardSettings', JSON.stringify(settings));
      setUnsavedChanges(false);
      addAlert('Settings saved successfully', 'success');
    } catch (error) {
      setError('Failed to save settings');
    }
  };

  const handleResetSettings = () => {
    setSettings({
      theme: 'light',
      autoRefresh: true,
      refreshInterval: 30000,
      notifications: true,
      emailNotifications: false,
      realTimeUpdates: true,
      cacheSettings: {
        enabled: true,
        ttl: 300,
        maxSize: 1000,
      },
      exportSettings: {
        defaultFormat: 'json',
        includeHeaders: true,
        maxExportSize: 100000,
      },
    });
    setUnsavedChanges(true);
  };

  const handleTestEndpoint = async (endpoint: APIEndpoint) => {
    try {
      // Test the endpoint connection
      const client = new (await import('../services/api-client')).default(endpoint.url, {
        timeout: endpoint.timeout,
      });
      
      const result = await client.testConnection();
      setTestResults(prev => ({ ...prev, [endpoint.id]: result }));
      
      if (result) {
        addAlert(`Connection to ${endpoint.name} successful`, 'success');
      } else {
        addAlert(`Connection to ${endpoint.name} failed`, 'error');
      }
    } catch (error) {
      setTestResults(prev => ({ ...prev, [endpoint.id]: false }));
      setError(`Failed to test ${endpoint.name}: ${error.message}`);
    }
  };

  const handleAddEndpoint = () => {
    setEditingEndpoint({
      id: Date.now().toString(),
      name: '',
      url: '',
      enabled: true,
      timeout: 30000,
    });
    setEndpointDialogOpen(true);
  };

  const handleEditEndpoint = (endpoint: APIEndpoint) => {
    setEditingEndpoint(endpoint);
    setEndpointDialogOpen(true);
  };

  const handleSaveEndpoint = () => {
    if (!editingEndpoint) return;
    
    setApiEndpoints(prev => {
      const existing = prev.find(e => e.id === editingEndpoint.id);
      if (existing) {
        return prev.map(e => e.id === editingEndpoint.id ? editingEndpoint : e);
      } else {
        return [...prev, editingEndpoint];
      }
    });
    
    setEndpointDialogOpen(false);
    setEditingEndpoint(null);
    setUnsavedChanges(true);
  };

  const handleDeleteEndpoint = (endpointId: string) => {
    setApiEndpoints(prev => prev.filter(e => e.id !== endpointId));
    setUnsavedChanges(true);
  };

  const handleClearCache = async () => {
    try {
      await apiClient.invalidateCache();
      addAlert('Cache cleared successfully', 'success');
    } catch (error) {
      setError('Failed to clear cache');
    }
  };

  const handleWarmCache = async () => {
    try {
      await apiClient.warmCache();
      addAlert('Cache warmed successfully', 'success');
    } catch (error) {
      setError('Failed to warm cache');
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard Settings
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleResetSettings}
          >
            Reset
          </Button>
          
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSaveSettings}
            disabled={!unsavedChanges}
          >
            Save Changes
          </Button>
        </Box>
      </Box>

      {/* Unsaved Changes Alert */}
      {unsavedChanges && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          You have unsaved changes. Don't forget to save your settings.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Appearance Settings */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <PaletteIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Appearance</Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Theme</InputLabel>
                  <Select
                    value={settings.theme}
                    onChange={(e) => handleSettingsChange('theme', '', e.target.value)}
                    label="Theme"
                  >
                    <MenuItem value="light">Light</MenuItem>
                    <MenuItem value="dark">Dark</MenuItem>
                    <MenuItem value="auto">Auto (System)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Performance Settings */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <SpeedIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Performance</Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.autoRefresh}
                      onChange={(e) => handleSettingsChange('autoRefresh', '', e.target.checked)}
                    />
                  }
                  label="Auto Refresh"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth disabled={!settings.autoRefresh}>
                  <InputLabel>Refresh Interval</InputLabel>
                  <Select
                    value={settings.refreshInterval}
                    onChange={(e) => handleSettingsChange('refreshInterval', '', e.target.value)}
                    label="Refresh Interval"
                  >
                    <MenuItem value={5000}>5 seconds</MenuItem>
                    <MenuItem value={10000}>10 seconds</MenuItem>
                    <MenuItem value={30000}>30 seconds</MenuItem>
                    <MenuItem value={60000}>1 minute</MenuItem>
                    <MenuItem value={300000}>5 minutes</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.realTimeUpdates}
                      onChange={(e) => handleSettingsChange('realTimeUpdates', '', e.target.checked)}
                    />
                  }
                  label="Real-time Updates"
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <NotificationsIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Notifications</Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.notifications}
                      onChange={(e) => handleSettingsChange('notifications', '', e.target.checked)}
                    />
                  }
                  label="Browser Notifications"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.emailNotifications}
                      onChange={(e) => handleSettingsChange('emailNotifications', '', e.target.checked)}
                    />
                  }
                  label="Email Notifications"
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Cache Settings */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <StorageIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Cache Settings</Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.cacheSettings.enabled}
                      onChange={(e) => handleSettingsChange('cacheSettings', 'enabled', e.target.checked)}
                    />
                  }
                  label="Enable Caching"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="TTL (seconds)"
                  type="number"
                  value={settings.cacheSettings.ttl}
                  onChange={(e) => handleSettingsChange('cacheSettings', 'ttl', parseInt(e.target.value))}
                  disabled={!settings.cacheSettings.enabled}
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Max Size (MB)"
                  type="number"
                  value={settings.cacheSettings.maxSize}
                  onChange={(e) => handleSettingsChange('cacheSettings', 'maxSize', parseInt(e.target.value))}
                  disabled={!settings.cacheSettings.enabled}
                />
              </Grid>
              
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={handleClearCache}
                    disabled={!settings.cacheSettings.enabled}
                  >
                    Clear Cache
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={handleWarmCache}
                    disabled={!settings.cacheSettings.enabled}
                  >
                    Warm Cache
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Export Settings */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <CloudSyncIcon color="primary" sx={{ mr: 1 }} />
              <Typography variant="h6">Export Settings</Typography>
            </Box>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControl fullWidth>
                  <InputLabel>Default Format</InputLabel>
                  <Select
                    value={settings.exportSettings.defaultFormat}
                    onChange={(e) => handleSettingsChange('exportSettings', 'defaultFormat', e.target.value)}
                    label="Default Format"
                  >
                    <MenuItem value="json">JSON</MenuItem>
                    <MenuItem value="csv">CSV</MenuItem>
                    <MenuItem value="excel">Excel</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.exportSettings.includeHeaders}
                      onChange={(e) => handleSettingsChange('exportSettings', 'includeHeaders', e.target.checked)}
                    />
                  }
                  label="Include Headers by Default"
                />
              </Grid>
              
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Max Export Size (rows)"
                  type="number"
                  value={settings.exportSettings.maxExportSize}
                  onChange={(e) => handleSettingsChange('exportSettings', 'maxExportSize', parseInt(e.target.value))}
                />
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* API Endpoints */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3, mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <SecurityIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">API Endpoints</Typography>
              </Box>
              <Button
                size="small"
                startIcon={<AddIcon />}
                onClick={handleAddEndpoint}
              >
                Add Endpoint
              </Button>
            </Box>
            
            <List>
              {apiEndpoints.map((endpoint, index) => (
                <React.Fragment key={endpoint.id}>
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle1">
                            {endpoint.name}
                          </Typography>
                          <Chip
                            label={endpoint.enabled ? 'Enabled' : 'Disabled'}
                            size="small"
                            color={endpoint.enabled ? 'success' : 'default'}
                            variant="outlined"
                          />
                          {testResults[endpoint.id] !== undefined && (
                            <Chip
                              label={testResults[endpoint.id] ? 'Connected' : 'Failed'}
                              size="small"
                              color={testResults[endpoint.id] ? 'success' : 'error'}
                            />
                          )}
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="textSecondary">
                            {endpoint.url}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            Timeout: {endpoint.timeout}ms
                          </Typography>
                        </Box>
                      }
                    />
                    <ListItemSecondaryAction>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Tooltip title="Test Connection">
                          <IconButton
                            size="small"
                            onClick={() => handleTestEndpoint(endpoint)}
                            color="primary"
                          >
                            <RefreshIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
                          <IconButton
                            size="small"
                            onClick={() => handleEditEndpoint(endpoint)}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            onClick={() => handleDeleteEndpoint(endpoint.id)}
                            color="error"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </ListItemSecondaryAction>
                  </ListItem>
                  {index < apiEndpoints.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </Paper>
        </Grid>
      </Grid>

      {/* Endpoint Dialog */}
      <Dialog
        open={endpointDialogOpen}
        onClose={() => setEndpointDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {editingEndpoint?.name ? 'Edit Endpoint' : 'Add New Endpoint'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Name"
                value={editingEndpoint?.name || ''}
                onChange={(e) => setEditingEndpoint(prev => 
                  prev ? { ...prev, name: e.target.value } : null
                )}
              />
            </Grid>
            
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="URL"
                value={editingEndpoint?.url || ''}
                onChange={(e) => setEditingEndpoint(prev => 
                  prev ? { ...prev, url: e.target.value } : null
                )}
                placeholder="http://localhost:8002"
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Timeout (ms)"
                type="number"
                value={editingEndpoint?.timeout || 30000}
                onChange={(e) => setEditingEndpoint(prev => 
                  prev ? { ...prev, timeout: parseInt(e.target.value) } : null
                )}
              />
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editingEndpoint?.enabled || false}
                    onChange={(e) => setEditingEndpoint(prev => 
                      prev ? { ...prev, enabled: e.target.checked } : null
                    )}
                  />
                }
                label="Enabled"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEndpointDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveEndpoint}
            variant="contained"
            disabled={!editingEndpoint?.name || !editingEndpoint?.url}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Settings;