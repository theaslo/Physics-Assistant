import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Theme and Layout
import { dashboardTheme } from './themes/dashboard-theme';
import DashboardLayout from './layouts/DashboardLayout';

// Pages
import DashboardOverview from './pages/DashboardOverview';
import StudentsAnalytics from './pages/StudentsAnalytics';
import ClassOverview from './pages/ClassOverview';
import SystemMetrics from './pages/SystemMetrics';
import RealTimeDashboard from './pages/RealTimeDashboard';
import DataExports from './pages/DataExports';
import Settings from './pages/Settings';
import AdvancedAnalytics from './pages/AdvancedAnalytics';

// Components
import AlertSnackbar from './components/common/AlertSnackbar';
import ConnectionStatus from './components/common/ConnectionStatus';

// Stores and Services
import { useDashboardStore } from './stores/dashboard-store';
import './App.css';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
      staleTime: 30000, // 30 seconds
      cacheTime: 300000, // 5 minutes
    },
    mutations: {
      retry: 1,
    },
  },
});

function App() {
  const { error } = useDashboardStore();

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={dashboardTheme}>
        <CssBaseline />
        <Router>
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            {/* Connection Status */}
            <ConnectionStatus />
            
            {/* Main Application */}
            <DashboardLayout>
              <Routes>
                {/* Dashboard Routes */}
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<DashboardOverview />} />
                <Route path="/students" element={<StudentsAnalytics />} />
                <Route path="/classes" element={<ClassOverview />} />
                <Route path="/system" element={<SystemMetrics />} />
                <Route path="/realtime" element={<RealTimeDashboard />} />
                <Route path="/exports" element={<DataExports />} />
                <Route path="/settings" element={<Settings />} />
                
                {/* Analytics Routes */}
                <Route path="/analytics" element={<DashboardOverview />} />
                <Route path="/analytics/performance" element={<SystemMetrics />} />
                <Route path="/analytics/trends" element={<RealTimeDashboard />} />
                <Route path="/analytics/charts" element={<DataExports />} />
                <Route path="/analytics/advanced" element={<AdvancedAnalytics />} />
                
                {/* Catch all route */}
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </DashboardLayout>
            
            {/* Global Alert System */}
            <AlertSnackbar
              open={!!error}
              message={error || ''}
              severity="error"
              onClose={() => useDashboardStore.getState().clearError()}
            />
          </Box>
        </Router>
        
        {/* React Query DevTools */}
        <ReactQueryDevtools initialIsOpen={false} />
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
