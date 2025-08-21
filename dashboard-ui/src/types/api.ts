/**
 * TypeScript definitions for Physics Assistant Dashboard API
 * Based on API documentation v2.0.0
 */

// ============================================================================
// Time Range Types
// ============================================================================

export type TimePreset = '1h' | '6h' | '24h' | '7d' | '30d' | '90d' | '1y';

export interface TimeRange {
  start: string; // ISO format
  end: string;   // ISO format
  preset?: TimePreset;
}

export interface TimeRangeRequest {
  preset?: TimePreset;
  start_date?: string;
  end_date?: string;
}

// ============================================================================
// Dashboard Summary Types
// ============================================================================

export interface DashboardSummary {
  time_range: TimeRange;
  summary: {
    total_interactions: number;
    active_users: number;
    agents_used: number;
    avg_response_time: number;
    success_rate: number;
    successful_interactions: number;
  };
  agent_breakdown: Array<{
    agent_type: string;
    interaction_count: number;
  }>;
  hourly_activity: Array<{
    hour: string;
    interactions: number;
    avg_response_time: number;
  }>;
  cache_info: {
    generated_at: string;
    cache_key: string;
  };
}

// ============================================================================
// Time Series Types
// ============================================================================

export type MetricType = 
  | 'interaction_count'
  | 'success_rate'
  | 'avg_response_time'
  | 'unique_users'
  | 'agent_usage'
  | 'error_rate';

export type GranularityType = '5m' | '15m' | '1h' | '6h' | '1d' | '1w';

export type AggregationType = 'sum' | 'avg' | 'min' | 'max' | 'count';

export interface TimeSeriesFilters {
  user_ids?: string[];
  agent_types?: string[];
  success_only?: boolean;
  min_response_time?: number;
  max_response_time?: number;
}

export interface TimeSeriesRequest {
  metrics: MetricType[];
  time_range: TimeRangeRequest;
  granularity: GranularityType;
  filters?: TimeSeriesFilters;
  aggregation?: AggregationType;
}

export interface TimeSeriesDataPoint {
  timestamp: string;
  interaction_count?: number;
  success_rate?: number;
  avg_response_time?: number;
  unique_users?: number;
  agent_usage?: number;
  error_rate?: number;
}

export interface TimeSeriesResponse {
  metrics: MetricType[];
  granularity: GranularityType;
  time_range: TimeRange;
  data: TimeSeriesDataPoint[];
  generated_at: string;
}

// ============================================================================
// Aggregation Types
// ============================================================================

export type DimensionType = 'agent_type' | 'date' | 'hour' | 'user_id' | 'success_status';

export interface AggregationRequest {
  dimensions: DimensionType[];
  metrics: MetricType[];
  time_range: TimeRangeRequest;
  filters?: TimeSeriesFilters;
  limit?: number;
  offset?: number;
}

export interface AggregationDataPoint {
  [dimension: string]: string | number;
}

export interface AggregationResponse {
  dimensions: DimensionType[];
  metrics: MetricType[];
  data: AggregationDataPoint[];
  total_records: number;
  generated_at: string;
}

// ============================================================================
// Comparative Analysis Types
// ============================================================================

export type ComparisonType = 'students' | 'agents' | 'time_periods' | 'classes';

export interface ComparativeRequest {
  comparison_type: ComparisonType;
  primary_entities: string[];
  comparison_entities: string[];
  metrics: MetricType[];
  time_range: TimeRangeRequest;
}

export interface ComparativeMetrics {
  success_rate: number;
  interaction_count: number;
  avg_response_time: number;
  unique_users?: number;
}

export interface ComparativeEntity {
  entity_id: string;
  entity_type: 'primary' | 'comparison';
  metrics: ComparativeMetrics;
}

export interface ComparativeResponse {
  comparison_type: ComparisonType;
  primary_entities: string[];
  comparison_entities: string[];
  comparisons: ComparativeEntity[];
  generated_at: string;
}

// ============================================================================
// Student Analytics Types
// ============================================================================

export interface ProgressTracking {
  overall_score: number;
  learning_velocity: number;
  engagement_score: number;
  concepts_mastered: number;
  total_concepts: number;
}

export interface ConceptMastery {
  concept: string;
  mastery_score: number;
  confidence: [number, number]; // confidence interval
}

export interface MLPrediction {
  predicted_value: number;
  confidence: number;
  factors: string[];
}

export interface StudentInsights {
  user_id: string;
  time_range: TimeRange;
  progress_tracking: ProgressTracking;
  concept_mastery: ConceptMastery[];
  predictions: {
    success_rate: MLPrediction;
    learning_velocity?: MLPrediction;
    engagement_score?: MLPrediction;
  };
  generated_at: string;
}

// ============================================================================
// Class Overview Types
// ============================================================================

export interface ClassStatistics {
  total_students: number;
  total_interactions: number;
  avg_response_time: number;
  class_success_rate: number;
  most_used_agent: string;
}

export interface PerformanceDistribution {
  percentiles: {
    '25th': number;
    '50th': number;
    '75th': number;
    '90th': number;
  };
  student_count: number;
}

export interface TopPerformer {
  user_id: string;
  success_rate: number;
  interaction_count: number;
}

export interface ClassOverview {
  time_range: TimeRange;
  class_statistics: ClassStatistics;
  performance_distribution: PerformanceDistribution;
  top_performers: TopPerformer[];
  generated_at: string;
}

// ============================================================================
// Export Types
// ============================================================================

export type ExportFormat = 'json' | 'csv' | 'excel';
export type DataType = 'interactions' | 'analytics' | 'summary' | 'students';

export interface ExportRequest {
  export_format: ExportFormat;
  data_type: DataType;
  time_range: TimeRangeRequest;
  filters?: TimeSeriesFilters;
  include_headers?: boolean;
}

export interface ExportResponse {
  format: ExportFormat;
  data_type: DataType;
  time_range: TimeRange;
  data: any[]; // Dynamic based on data_type
  export_timestamp: string;
  download_url?: string; // For file formats
}

// ============================================================================
// Real-time WebSocket Types
// ============================================================================

export type WebSocketMessageType = 
  | 'metrics_update'
  | 'alert'
  | 'student_progress'
  | 'heartbeat'
  | 'connection_status';

export interface WebSocketMessage {
  type: WebSocketMessageType;
  data: any;
  timestamp: string;
  user_id?: string;
}

export interface MetricsUpdate {
  summary: Partial<DashboardSummary['summary']>;
  recent_interactions: number;
  active_users_count: number;
}

export interface Alert {
  severity: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  entity_id?: string;
}

export interface StudentProgressUpdate {
  user_id: string;
  progress_change: Partial<ProgressTracking>;
  concept_updates: ConceptMastery[];
}

// ============================================================================
// API Response Wrapper Types
// ============================================================================

export interface APIResponse<T = any> {
  data: T;
  success: boolean;
  timestamp: string;
  request_id?: string;
}

export interface APIError {
  error: string;
  detail: string;
  code: string;
  timestamp: string;
  request_id?: string;
}

// ============================================================================
// Cache and Performance Types
// ============================================================================

export interface CacheInfo {
  hit_rate: number;
  size?: number;
  max_size?: number;
  memory_usage?: string;
  connected?: boolean;
}

export interface CacheStatistics {
  cache_statistics: {
    memory: CacheInfo;
    redis: CacheInfo;
  };
  total_cache_requests: number;
}

export interface PerformanceMetrics {
  response_time: number;
  cache_status: 'hit' | 'miss' | 'stale';
  processing_time: number;
}

// ============================================================================
// Health Check Types
// ============================================================================

export interface ServiceStatus {
  status: 'healthy' | 'degraded' | 'down';
  details?: string;
}

export interface HealthCheck {
  status: 'healthy' | 'degraded' | 'down';
  timestamp: string;
  services: {
    redis: string;
    cache: string;
    websockets: string;
    background_processor: string;
  };
  cache_stats: {
    memory: CacheInfo;
    redis: CacheInfo;
  };
}

// ============================================================================
// Mock Data Types
// ============================================================================

export interface MockStudentProgress {
  user_id: string;
  progress_metrics: ProgressTracking;
  concept_breakdown: ConceptMastery[];
  time_series: Array<{
    date: string;
    score: number;
  }>;
  mock_data: true;
}

export interface MockClassAnalytics {
  class_id: string;
  summary: {
    total_students: number;
    active_students: number;
    avg_progress: number;
    completion_rate: number;
  };
  performance_distribution: {
    excellent: number;
    good: number;
    average: number;
    needs_help: number;
    at_risk: number;
  };
  mock_data: true;
}

// ============================================================================
// UI State Types
// ============================================================================

export interface DashboardFilters {
  timeRange: TimeRangeRequest;
  selectedUsers: string[];
  selectedAgents: string[];
  showSuccessOnly: boolean;
}

export interface UIState {
  loading: boolean;
  error: string | null;
  lastUpdated: string | null;
  filters: DashboardFilters;
}

// ============================================================================
// Chart Configuration Types
// ============================================================================

export interface ChartConfig {
  title: string;
  type: 'line' | 'bar' | 'area' | 'pie' | 'scatter' | 'heatmap';
  dataKey: string;
  xAxisKey?: string;
  yAxisKey?: string;
  colors?: string[];
  showLegend?: boolean;
  showTooltip?: boolean;
  responsive?: boolean;
}

export interface WidgetConfig {
  id: string;
  title: string;
  type: 'metric' | 'chart' | 'table' | 'alert';
  size: 'small' | 'medium' | 'large';
  refreshInterval?: number; // seconds
  config: ChartConfig | any;
}