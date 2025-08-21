/**
 * API Client for Physics Assistant Dashboard
 * Handles all communication with the dashboard backend API
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import {
  DashboardSummary,
  TimeSeriesRequest,
  TimeSeriesResponse,
  AggregationRequest,
  AggregationResponse,
  ComparativeRequest,
  ComparativeResponse,
  StudentInsights,
  ClassOverview,
  ExportRequest,
  ExportResponse,
  CacheStatistics,
  HealthCheck,
  MockStudentProgress,
  MockClassAnalytics,
  APIResponse,
  APIError,
  TimeRangeRequest,
} from '../types/api';

class DashboardAPIClient {
  private api: AxiosInstance;
  private baseURL: string;
  private defaultTimeout: number = 30000; // 30 seconds

  constructor(baseURL: string = 'http://localhost:8002', options?: AxiosRequestConfig) {
    this.baseURL = baseURL;
    
    this.api = axios.create({
      baseURL: this.baseURL,
      timeout: this.defaultTimeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      ...options,
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        // Add authentication headers if available
        const token = this.getAuthToken();
        const userId = this.getUserId();
        
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        if (userId) {
          config.headers['X-User-ID'] = userId;
        }

        // Add request timestamp
        config.headers['X-Request-Time'] = new Date().toISOString();

        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`, {
          params: config.params,
          data: config.data,
        });

        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        console.log(`API Response: ${response.status} ${response.config.url}`, {
          data: response.data,
          headers: response.headers,
        });
        return response;
      },
      (error) => {
        const apiError: APIError = {
          error: error.response?.data?.error || error.message,
          detail: error.response?.data?.detail || 'Network or server error',
          code: error.response?.data?.code || error.code || 'UNKNOWN_ERROR',
          timestamp: new Date().toISOString(),
          request_id: error.response?.headers?.['x-request-id'],
        };

        console.error('API Response Error:', apiError);
        return Promise.reject(apiError);
      }
    );
  }

  private getAuthToken(): string | null {
    // Get from localStorage, sessionStorage, or context
    return localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
  }

  private getUserId(): string | null {
    // Get from localStorage, sessionStorage, or context
    return localStorage.getItem('user_id') || sessionStorage.getItem('user_id');
  }

  // ============================================================================
  // Authentication Methods
  // ============================================================================

  public setAuthToken(token: string): void {
    localStorage.setItem('auth_token', token);
  }

  public setUserId(userId: string): void {
    localStorage.setItem('user_id', userId);
  }

  public clearAuth(): void {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_id');
    sessionStorage.removeItem('auth_token');
    sessionStorage.removeItem('user_id');
  }

  // ============================================================================
  // Dashboard Summary API
  // ============================================================================

  public async getDashboardSummary(params?: {
    preset?: string;
    start_date?: string;
    end_date?: string;
    user_ids?: string[];
    agent_types?: string[];
    force_refresh?: boolean;
  }): Promise<DashboardSummary> {
    const response = await this.api.get<DashboardSummary>('/dashboard/summary', {
      params: {
        ...params,
        user_ids: params?.user_ids?.join(','),
        agent_types: params?.agent_types?.join(','),
      },
    });
    return response.data;
  }

  // ============================================================================
  // Time Series API
  // ============================================================================

  public async getTimeSeriesData(request: TimeSeriesRequest): Promise<TimeSeriesResponse> {
    const response = await this.api.post<TimeSeriesResponse>('/dashboard/timeseries', request);
    return response.data;
  }

  // ============================================================================
  // Aggregation API
  // ============================================================================

  public async getAggregationData(request: AggregationRequest): Promise<AggregationResponse> {
    const response = await this.api.post<AggregationResponse>('/dashboard/aggregation', request);
    return response.data;
  }

  // ============================================================================
  // Comparative Analysis API
  // ============================================================================

  public async getComparativeAnalysis(request: ComparativeRequest): Promise<ComparativeResponse> {
    const response = await this.api.post<ComparativeResponse>('/dashboard/comparative', request);
    return response.data;
  }

  // ============================================================================
  // Student Insights API
  // ============================================================================

  public async getStudentInsights(
    userId: string, 
    params?: {
      preset?: string;
      include_predictions?: boolean;
    }
  ): Promise<StudentInsights> {
    const response = await this.api.get<StudentInsights>(`/dashboard/student-insights/${userId}`, {
      params,
    });
    return response.data;
  }

  // ============================================================================
  // Class Overview API
  // ============================================================================

  public async getClassOverview(params?: {
    class_ids?: string[];
    preset?: string;
    include_comparisons?: boolean;
  }): Promise<ClassOverview> {
    const response = await this.api.get<ClassOverview>('/dashboard/class-overview', {
      params: {
        ...params,
        class_ids: params?.class_ids?.join(','),
      },
    });
    return response.data;
  }

  // ============================================================================
  // Data Export API
  // ============================================================================

  public async exportData(request: ExportRequest): Promise<ExportResponse | Blob> {
    const response = await this.api.post('/dashboard/export', request, {
      responseType: request.export_format === 'json' ? 'json' : 'blob',
      params: {
        export_format: request.export_format,
        data_type: request.data_type,
      },
    });

    if (request.export_format === 'json') {
      return response.data as ExportResponse;
    } else {
      return response.data as Blob;
    }
  }

  // ============================================================================
  // Cache Management API
  // ============================================================================

  public async getCacheStatistics(): Promise<CacheStatistics> {
    const response = await this.api.get<CacheStatistics>('/dashboard/cache/stats');
    return response.data;
  }

  public async invalidateCache(pattern?: string, cacheLayer: string = 'all'): Promise<void> {
    await this.api.post('/dashboard/cache/invalidate', {
      pattern,
      cache_layer: cacheLayer,
    });
  }

  public async warmCache(cacheTypes?: string[]): Promise<void> {
    await this.api.post('/dashboard/cache/warm', {
      cache_types: cacheTypes,
    });
  }

  // ============================================================================
  // Health Check API
  // ============================================================================

  public async getHealthCheck(): Promise<HealthCheck> {
    const response = await this.api.get<HealthCheck>('/dashboard/health');
    return response.data;
  }

  // ============================================================================
  // Mock Data APIs (for development)
  // ============================================================================

  public async getMockStudentProgress(): Promise<MockStudentProgress> {
    const response = await this.api.get<MockStudentProgress>('/dashboard/mock/student-progress');
    return response.data;
  }

  public async getMockClassAnalytics(): Promise<MockClassAnalytics> {
    const response = await this.api.get<MockClassAnalytics>('/dashboard/mock/class-analytics');
    return response.data;
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  public async testConnection(): Promise<boolean> {
    try {
      await this.getHealthCheck();
      return true;
    } catch (error) {
      console.error('Connection test failed:', error);
      return false;
    }
  }

  public getBaseURL(): string {
    return this.baseURL;
  }

  public updateBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL;
    this.api.defaults.baseURL = newBaseURL;
  }

  public updateTimeout(timeout: number): void {
    this.defaultTimeout = timeout;
    this.api.defaults.timeout = timeout;
  }

  // ============================================================================
  // Batch Operations
  // ============================================================================

  public async batchRequest<T>(requests: Array<() => Promise<T>>): Promise<T[]> {
    try {
      const results = await Promise.allSettled(requests.map(req => req()));
      return results.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          console.error(`Batch request ${index} failed:`, result.reason);
          throw result.reason;
        }
      });
    } catch (error) {
      console.error('Batch request failed:', error);
      throw error;
    }
  }

  // ============================================================================
  // Server-Sent Events (SSE) Support
  // ============================================================================

  public createEventSource(endpoint: string, params?: Record<string, string>): EventSource {
    const url = new URL(endpoint, this.baseURL);
    
    // Add query parameters
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.append(key, value);
      });
    }

    // Add auth parameters
    const userId = this.getUserId();
    if (userId) {
      url.searchParams.append('user_id', userId);
    }

    return new EventSource(url.toString());
  }

  // ============================================================================
  // WebSocket Connection Helper
  // ============================================================================

  public createWebSocketURL(endpoint: string, userId?: string): string {
    const wsProtocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
    const baseWsURL = this.baseURL.replace(/^https?/, wsProtocol);
    const finalUserId = userId || this.getUserId() || 'anonymous';
    
    return `${baseWsURL}${endpoint}/${finalUserId}`;
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

// Create a default instance
export const apiClient = new DashboardAPIClient();

// Export the class for custom instances
export default DashboardAPIClient;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Helper function to handle API errors consistently
 */
export const handleAPIError = (error: any): APIError => {
  if (error.error && error.detail && error.code) {
    // Already an APIError
    return error as APIError;
  }

  // Convert generic error to APIError
  return {
    error: error.message || 'Unknown error',
    detail: error.response?.data?.detail || error.toString(),
    code: error.code || 'UNKNOWN_ERROR',
    timestamp: new Date().toISOString(),
    request_id: error.response?.headers?.['x-request-id'],
  };
};

/**
 * Helper function to format time range for API requests
 */
export const formatTimeRange = (range: TimeRangeRequest): TimeRangeRequest => {
  if (range.preset) {
    return { preset: range.preset };
  }
  
  return {
    start_date: range.start_date,
    end_date: range.end_date,
  };
};

/**
 * Helper function to download blob data
 */
export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};