/**
 * Zustand Store for Dashboard State Management
 * Handles global state for the analytics dashboard
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { 
  DashboardSummary, 
  TimeSeriesResponse, 
  StudentInsights, 
  ClassOverview, 
  DashboardFilters, 
  TimePreset,
  Alert,
  WebSocketMessage,
  MetricsUpdate,
  StudentProgressUpdate
} from '../types/api';

// ============================================================================
// Dashboard State Interface
// ============================================================================

export interface DashboardState {
  // ---- Loading States ----
  loading: {
    summary: boolean;
    timeSeries: boolean;
    studentInsights: boolean;
    classOverview: boolean;
    export: boolean;
  };

  // ---- Error States ----
  errors: {
    summary: string | null;
    timeSeries: string | null;
    studentInsights: string | null;
    classOverview: string | null;
    export: string | null;
    connection: string | null;
  };

  // ---- Data States ----
  data: {
    summary: DashboardSummary | null;
    timeSeries: TimeSeriesResponse | null;
    studentInsights: { [userId: string]: StudentInsights };
    classOverview: ClassOverview | null;
    lastUpdated: string | null;
  };

  // ---- UI States ----
  ui: {
    selectedTimeRange: TimePreset;
    selectedStudents: string[];
    selectedAgents: string[];
    showSuccessOnly: boolean;
    selectedView: 'overview' | 'students' | 'classes' | 'analytics';
    sidebarOpen: boolean;
    theme: 'light' | 'dark';
    autoRefresh: boolean;
    refreshInterval: number; // seconds
  };

  // ---- Real-time States ----
  realtime: {
    connected: boolean;
    connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
    alerts: Alert[];
    unreadAlerts: number;
    lastHeartbeat: string | null;
  };

  // ---- Cache States ----
  cache: {
    enabled: boolean;
    hitRate: number;
    lastCleared: string | null;
  };
}

// ============================================================================
// Dashboard Actions Interface
// ============================================================================

export interface DashboardActions {
  // ---- Data Actions ----
  setSummary: (summary: DashboardSummary) => void;
  setTimeSeries: (timeSeries: TimeSeriesResponse) => void;
  setStudentInsights: (userId: string, insights: StudentInsights) => void;
  setClassOverview: (overview: ClassOverview) => void;
  clearData: () => void;

  // ---- Loading Actions ----
  setLoading: (key: keyof DashboardState['loading'], loading: boolean) => void;
  setAllLoading: (loading: boolean) => void;

  // ---- Error Actions ----
  setError: (key: keyof DashboardState['errors'], error: string | null) => void;
  clearErrors: () => void;

  // ---- UI Actions ----
  setTimeRange: (range: TimePreset) => void;
  setSelectedStudents: (students: string[]) => void;
  setSelectedAgents: (agents: string[]) => void;
  setShowSuccessOnly: (show: boolean) => void;
  setSelectedView: (view: DashboardState['ui']['selectedView']) => void;
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: 'light' | 'dark') => void;
  setAutoRefresh: (enabled: boolean) => void;
  setRefreshInterval: (interval: number) => void;

  // ---- Real-time Actions ----
  setConnectionStatus: (status: DashboardState['realtime']['connectionStatus']) => void;
  addAlert: (alert: Alert) => void;
  removeAlert: (alertId: string) => void;
  clearAlerts: () => void;
  markAlertsRead: () => void;
  updateHeartbeat: () => void;

  // ---- Cache Actions ----
  setCacheEnabled: (enabled: boolean) => void;
  updateCacheHitRate: (hitRate: number) => void;
  markCacheCleared: () => void;

  // ---- Utility Actions ----
  resetStore: () => void;
  updateLastUpdated: () => void;
  getFilters: () => DashboardFilters;
}

// ============================================================================
// Initial State
// ============================================================================

const initialState: DashboardState = {
  loading: {
    summary: false,
    timeSeries: false,
    studentInsights: false,
    classOverview: false,
    export: false,
  },

  errors: {
    summary: null,
    timeSeries: null,
    studentInsights: null,
    classOverview: null,
    export: null,
    connection: null,
  },

  data: {
    summary: null,
    timeSeries: null,
    studentInsights: {},
    classOverview: null,
    lastUpdated: null,
  },

  ui: {
    selectedTimeRange: '7d',
    selectedStudents: [],
    selectedAgents: [],
    showSuccessOnly: false,
    selectedView: 'overview',
    sidebarOpen: true,
    theme: 'light',
    autoRefresh: true,
    refreshInterval: 30, // 30 seconds
  },

  realtime: {
    connected: false,
    connectionStatus: 'disconnected',
    alerts: [],
    unreadAlerts: 0,
    lastHeartbeat: null,
  },

  cache: {
    enabled: true,
    hitRate: 0,
    lastCleared: null,
  },
};

// ============================================================================
// Zustand Store
// ============================================================================

export const useDashboardStore = create<DashboardState & DashboardActions>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    // ---- Data Actions ----
    setSummary: (summary) => 
      set((state) => ({ 
        data: { ...state.data, summary },
      })),

    setTimeSeries: (timeSeries) => 
      set((state) => ({ 
        data: { ...state.data, timeSeries },
      })),

    setStudentInsights: (userId, insights) => 
      set((state) => ({ 
        data: { 
          ...state.data, 
          studentInsights: { ...state.data.studentInsights, [userId]: insights }
        },
      })),

    setClassOverview: (overview) => 
      set((state) => ({ 
        data: { ...state.data, classOverview: overview },
      })),

    clearData: () => 
      set((state) => ({
        data: {
          summary: null,
          timeSeries: null,
          studentInsights: {},
          classOverview: null,
          lastUpdated: null,
        },
      })),

    // ---- Loading Actions ----
    setLoading: (key, loading) => 
      set((state) => ({
        loading: { ...state.loading, [key]: loading },
      })),

    setAllLoading: (loading) => 
      set((state) => ({
        loading: {
          summary: loading,
          timeSeries: loading,
          studentInsights: loading,
          classOverview: loading,
          export: loading,
        },
      })),

    // ---- Error Actions ----
    setError: (key, error) => 
      set((state) => ({
        errors: { ...state.errors, [key]: error },
      })),

    clearErrors: () => 
      set((state) => ({
        errors: {
          summary: null,
          timeSeries: null,
          studentInsights: null,
          classOverview: null,
          export: null,
          connection: null,
        },
      })),

    // ---- UI Actions ----
    setTimeRange: (range) => 
      set((state) => ({
        ui: { ...state.ui, selectedTimeRange: range },
      })),

    setSelectedStudents: (students) => 
      set((state) => ({
        ui: { ...state.ui, selectedStudents: students },
      })),

    setSelectedAgents: (agents) => 
      set((state) => ({
        ui: { ...state.ui, selectedAgents: agents },
      })),

    setShowSuccessOnly: (show) => 
      set((state) => ({
        ui: { ...state.ui, showSuccessOnly: show },
      })),

    setSelectedView: (view) => 
      set((state) => ({
        ui: { ...state.ui, selectedView: view },
      })),

    setSidebarOpen: (open) => 
      set((state) => ({
        ui: { ...state.ui, sidebarOpen: open },
      })),

    setTheme: (theme) => 
      set((state) => ({
        ui: { ...state.ui, theme },
      })),

    setAutoRefresh: (enabled) => 
      set((state) => ({
        ui: { ...state.ui, autoRefresh: enabled },
      })),

    setRefreshInterval: (interval) => 
      set((state) => ({
        ui: { ...state.ui, refreshInterval: interval },
      })),

    // ---- Real-time Actions ----
    setConnectionStatus: (status) => 
      set((state) => ({
        realtime: { 
          ...state.realtime, 
          connectionStatus: status,
          connected: status === 'connected',
        },
      })),

    addAlert: (alert) => 
      set((state) => ({
        realtime: {
          ...state.realtime,
          alerts: [alert, ...state.realtime.alerts].slice(0, 50), // Keep last 50 alerts
          unreadAlerts: state.realtime.unreadAlerts + 1,
        },
      })),

    removeAlert: (alertId) => 
      set((state) => ({
        realtime: {
          ...state.realtime,
          alerts: state.realtime.alerts.filter((alert, index) => 
            `${alert.timestamp}-${index}` !== alertId
          ),
        },
      })),

    clearAlerts: () => 
      set((state) => ({
        realtime: {
          ...state.realtime,
          alerts: [],
          unreadAlerts: 0,
        },
      })),

    markAlertsRead: () => 
      set((state) => ({
        realtime: {
          ...state.realtime,
          unreadAlerts: 0,
        },
      })),

    updateHeartbeat: () => 
      set((state) => ({
        realtime: {
          ...state.realtime,
          lastHeartbeat: new Date().toISOString(),
        },
      })),

    // ---- Cache Actions ----
    setCacheEnabled: (enabled) => 
      set((state) => ({
        cache: { ...state.cache, enabled },
      })),

    updateCacheHitRate: (hitRate) => 
      set((state) => ({
        cache: { ...state.cache, hitRate },
      })),

    markCacheCleared: () => 
      set((state) => ({
        cache: { ...state.cache, lastCleared: new Date().toISOString() },
      })),

    // ---- Utility Actions ----
    resetStore: () => set(initialState),

    updateLastUpdated: () => 
      set((state) => ({
        data: { ...state.data, lastUpdated: new Date().toISOString() },
      })),

    getFilters: (): DashboardFilters => {
      const state = get();
      return {
        timeRange: { preset: state.ui.selectedTimeRange },
        selectedUsers: state.ui.selectedStudents,
        selectedAgents: state.ui.selectedAgents,
        showSuccessOnly: state.ui.showSuccessOnly,
      };
    },
  }))
);

// ============================================================================
// Selectors
// ============================================================================

// Loading selectors
export const useIsLoading = () => 
  useDashboardStore((state) => Object.values(state.loading).some(Boolean));

export const useHasErrors = () => 
  useDashboardStore((state) => Object.values(state.errors).some(Boolean));

// Data selectors
export const useSummaryData = () => 
  useDashboardStore((state) => state.data.summary);

export const useTimeSeriesData = () => 
  useDashboardStore((state) => state.data.timeSeries);

export const useStudentInsights = (userId?: string) => 
  useDashboardStore((state) => 
    userId ? state.data.studentInsights[userId] : state.data.studentInsights
  );

export const useClassOverview = () => 
  useDashboardStore((state) => state.data.classOverview);

// UI selectors
export const useUIState = () => 
  useDashboardStore((state) => state.ui);

export const useCurrentFilters = () => 
  useDashboardStore((state) => state.getFilters());

// Real-time selectors
export const useRealtimeState = () => 
  useDashboardStore((state) => state.realtime);

export const useUnreadAlerts = () => 
  useDashboardStore((state) => state.realtime.unreadAlerts);

export const useConnectionStatus = () => 
  useDashboardStore((state) => state.realtime.connectionStatus);

// ============================================================================
// Store Subscriptions
// ============================================================================

// Auto-save preferences to localStorage
useDashboardStore.subscribe(
  (state) => state.ui,
  (ui) => {
    try {
      localStorage.setItem('dashboard-ui-preferences', JSON.stringify({
        theme: ui.theme,
        sidebarOpen: ui.sidebarOpen,
        autoRefresh: ui.autoRefresh,
        refreshInterval: ui.refreshInterval,
      }));
    } catch (error) {
      console.warn('Failed to save UI preferences:', error);
    }
  }
);

// Load preferences on initialization
try {
  const savedPreferences = localStorage.getItem('dashboard-ui-preferences');
  if (savedPreferences) {
    const preferences = JSON.parse(savedPreferences);
    useDashboardStore.getState().setTheme(preferences.theme || 'light');
    useDashboardStore.getState().setSidebarOpen(preferences.sidebarOpen ?? true);
    useDashboardStore.getState().setAutoRefresh(preferences.autoRefresh ?? true);
    useDashboardStore.getState().setRefreshInterval(preferences.refreshInterval || 30);
  }
} catch (error) {
  console.warn('Failed to load UI preferences:', error);
}

export default useDashboardStore;