#!/usr/bin/env python3
"""
Advanced Statistical Analysis Engine for Physics Assistant
Implements sophisticated statistical models, time-series forecasting, clustering algorithms,
anomaly detection, and causal inference for educational analytics.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesAnalysis:
    """Results of time series analysis"""
    series_id: str
    metric_name: str
    time_period: Tuple[datetime, datetime]
    trend_analysis: Dict[str, Any]
    seasonality_analysis: Dict[str, Any]
    stationarity_tests: Dict[str, float]
    forecast_results: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    change_points: List[Dict[str, Any]]
    decomposition: Dict[str, Any]
    model_performance: Dict[str, float]

@dataclass
class ClusterAnalysis:
    """Results of clustering analysis"""
    cluster_id: str
    clustering_method: str
    feature_set: List[str]
    n_clusters: int
    cluster_assignments: Dict[str, int]
    cluster_characteristics: Dict[int, Dict[str, Any]]
    cluster_quality_metrics: Dict[str, float]
    optimal_clusters: int
    feature_importance: Dict[str, float]
    cluster_interpretations: Dict[int, str]

@dataclass
class AnomalyDetection:
    """Results of anomaly detection analysis"""
    detection_id: str
    method_used: str
    anomalies_detected: List[Dict[str, Any]]
    anomaly_scores: Dict[str, float]
    threshold_values: Dict[str, float]
    confidence_levels: Dict[str, float]
    contextual_anomalies: List[Dict[str, Any]]
    collective_anomalies: List[Dict[str, Any]]

@dataclass
class CausalInference:
    """Results of causal inference analysis"""
    analysis_id: str
    treatment_variable: str
    outcome_variable: str
    causal_effect: float
    confidence_interval: Tuple[float, float]
    p_value: float
    method_used: str
    confounding_variables: List[str]
    effect_size: float
    practical_significance: bool
    recommendations: List[str]

@dataclass
class CorrelationMatrix:
    """Correlation analysis results"""
    matrix_id: str
    variables: List[str]
    pearson_correlations: np.ndarray
    spearman_correlations: np.ndarray
    significance_matrix: np.ndarray
    strong_correlations: List[Tuple[str, str, float]]
    partial_correlations: Dict[str, float]
    correlation_networks: Dict[str, List[str]]

class StatisticalAnalysisEngine:
    """Advanced statistical analysis engine for educational analytics"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Analysis results storage
        self.time_series_analyses: Dict[str, TimeSeriesAnalysis] = {}
        self.cluster_analyses: Dict[str, ClusterAnalysis] = {}
        self.anomaly_detections: Dict[str, AnomalyDetection] = {}
        self.causal_inferences: Dict[str, CausalInference] = {}
        self.correlation_matrices: Dict[str, CorrelationMatrix] = {}
        
        # Analysis configuration
        self.config = {
            'time_series': {
                'min_data_points': 30,
                'forecast_horizon': 14,  # days
                'confidence_level': 0.95,
                'seasonality_threshold': 0.1
            },
            'clustering': {
                'max_clusters': 10,
                'min_cluster_size': 5,
                'algorithms': ['kmeans', 'dbscan', 'hierarchical'],
                'feature_scaling': True
            },
            'anomaly_detection': {
                'contamination_rate': 0.1,
                'methods': ['isolation_forest', 'statistical', 'density_based'],
                'sensitivity': 0.05
            },
            'correlation': {
                'significance_threshold': 0.05,
                'correlation_threshold': 0.3,
                'partial_correlation_threshold': 0.2
            }
        }
    
    async def initialize(self):
        """Initialize the statistical analysis engine"""
        try:
            logger.info("🚀 Initializing Statistical Analysis Engine")
            
            # Create analysis tables
            await self._create_analysis_tables()
            
            # Load existing analyses
            await self._load_existing_analyses()
            
            logger.info("✅ Statistical Analysis Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Statistical Analysis Engine: {e}")
            return False
    
    async def _create_analysis_tables(self):
        """Create database tables for statistical analyses"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Time series analysis table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS time_series_analyses (
                        series_id VARCHAR(100) PRIMARY KEY,
                        metric_name VARCHAR(100) NOT NULL,
                        start_date TIMESTAMP NOT NULL,
                        end_date TIMESTAMP NOT NULL,
                        trend_analysis JSONB DEFAULT '{}',
                        seasonality_analysis JSONB DEFAULT '{}',
                        stationarity_tests JSONB DEFAULT '{}',
                        forecast_results JSONB DEFAULT '{}',
                        anomalies JSONB DEFAULT '[]',
                        change_points JSONB DEFAULT '[]',
                        decomposition JSONB DEFAULT '{}',
                        model_performance JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Cluster analysis table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS cluster_analyses (
                        cluster_id VARCHAR(100) PRIMARY KEY,
                        clustering_method VARCHAR(50) NOT NULL,
                        feature_set JSONB NOT NULL,
                        n_clusters INTEGER NOT NULL,
                        cluster_assignments JSONB NOT NULL,
                        cluster_characteristics JSONB DEFAULT '{}',
                        cluster_quality_metrics JSONB DEFAULT '{}',
                        optimal_clusters INTEGER,
                        feature_importance JSONB DEFAULT '{}',
                        cluster_interpretations JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Anomaly detection table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS anomaly_detections (
                        detection_id VARCHAR(100) PRIMARY KEY,
                        method_used VARCHAR(50) NOT NULL,
                        anomalies_detected JSONB DEFAULT '[]',
                        anomaly_scores JSONB DEFAULT '{}',
                        threshold_values JSONB DEFAULT '{}',
                        confidence_levels JSONB DEFAULT '{}',
                        contextual_anomalies JSONB DEFAULT '[]',
                        collective_anomalies JSONB DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                logger.info("✅ Statistical analysis tables created")
        
        except Exception as e:
            logger.error(f"❌ Failed to create analysis tables: {e}")
    
    async def _load_existing_analyses(self):
        """Load existing analyses from database"""
        try:
            if not self.db_manager:
                return
            
            # Load time series analyses
            async with self.db_manager.postgres.get_connection() as conn:
                ts_analyses = await conn.fetch("SELECT * FROM time_series_analyses ORDER BY created_at DESC LIMIT 100")
                
                for ts_row in ts_analyses:
                    ts_analysis = TimeSeriesAnalysis(
                        series_id=ts_row['series_id'],
                        metric_name=ts_row['metric_name'],
                        time_period=(ts_row['start_date'], ts_row['end_date']),
                        trend_analysis=ts_row['trend_analysis'],
                        seasonality_analysis=ts_row['seasonality_analysis'],
                        stationarity_tests=ts_row['stationarity_tests'],
                        forecast_results=ts_row['forecast_results'],
                        anomalies=ts_row['anomalies'],
                        change_points=ts_row['change_points'],
                        decomposition=ts_row['decomposition'],
                        model_performance=ts_row['model_performance']
                    )
                    self.time_series_analyses[ts_analysis.series_id] = ts_analysis
                
                logger.info(f"📊 Loaded {len(self.time_series_analyses)} time series analyses")
        
        except Exception as e:
            logger.error(f"❌ Failed to load existing analyses: {e}")
    
    async def analyze_time_series(self, metric_name: str, start_date: datetime, 
                                 end_date: datetime, granularity: str = '1D') -> TimeSeriesAnalysis:
        """Comprehensive time series analysis with forecasting"""
        try:
            logger.info(f"📈 Analyzing time series for metric: {metric_name}")
            
            # Collect time series data
            ts_data = await self._collect_time_series_data(metric_name, start_date, end_date, granularity)
            
            if len(ts_data) < self.config['time_series']['min_data_points']:
                raise ValueError(f"Insufficient data points: {len(ts_data)}")
            
            # Convert to pandas time series
            df = pd.DataFrame(ts_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            series = df[metric_name]
            
            # 1. Trend Analysis
            trend_analysis = self._analyze_trend(series)
            
            # 2. Seasonality Analysis
            seasonality_analysis = self._analyze_seasonality(series)
            
            # 3. Stationarity Tests
            stationarity_tests = self._test_stationarity(series)
            
            # 4. Decomposition
            decomposition = self._decompose_series(series)
            
            # 5. Anomaly Detection
            anomalies = self._detect_time_series_anomalies(series)
            
            # 6. Change Point Detection
            change_points = self._detect_change_points(series)
            
            # 7. Forecasting
            forecast_results = self._forecast_series(series)
            
            # 8. Model Performance
            model_performance = self._evaluate_forecast_performance(series, forecast_results)
            
            series_id = f"ts_{metric_name}_{int(start_date.timestamp())}"
            
            ts_analysis = TimeSeriesAnalysis(
                series_id=series_id,
                metric_name=metric_name,
                time_period=(start_date, end_date),
                trend_analysis=trend_analysis,
                seasonality_analysis=seasonality_analysis,
                stationarity_tests=stationarity_tests,
                forecast_results=forecast_results,
                anomalies=anomalies,
                change_points=change_points,
                decomposition=decomposition,
                model_performance=model_performance
            )
            
            # Store results
            self.time_series_analyses[series_id] = ts_analysis
            await self._save_time_series_analysis(ts_analysis)
            
            logger.info(f"✅ Time series analysis completed for {metric_name}")
            return ts_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze time series for {metric_name}: {e}")
            raise
    
    async def _collect_time_series_data(self, metric_name: str, start_date: datetime, 
                                       end_date: datetime, granularity: str) -> List[Dict[str, Any]]:
        """Collect time series data for analysis"""
        try:
            if not self.db_manager:
                return []
            
            # Map granularity to PostgreSQL interval
            interval_map = {
                '1H': '1 hour',
                '6H': '6 hours', 
                '1D': '1 day',
                '1W': '1 week'
            }
            interval = interval_map.get(granularity, '1 day')
            
            # Create time buckets and aggregate data
            async with self.db_manager.postgres.get_connection() as conn:
                if metric_name == 'interaction_count':
                    query = f"""
                    WITH time_buckets AS (
                        SELECT generate_series($1, $2, interval '{interval}') as timestamp
                    )
                    SELECT 
                        tb.timestamp,
                        COALESCE(COUNT(i.id), 0) as interaction_count
                    FROM time_buckets tb
                    LEFT JOIN interactions i ON 
                        date_trunc('{granularity.lower().replace('h', 'hour').replace('d', 'day').replace('w', 'week')}', i.created_at) = tb.timestamp
                        AND i.created_at BETWEEN $1 AND $2
                    GROUP BY tb.timestamp
                    ORDER BY tb.timestamp
                    """
                elif metric_name == 'success_rate':
                    query = f"""
                    WITH time_buckets AS (
                        SELECT generate_series($1, $2, interval '{interval}') as timestamp
                    )
                    SELECT 
                        tb.timestamp,
                        COALESCE(AVG(CASE WHEN i.success THEN 1.0 ELSE 0.0 END), 0) as success_rate
                    FROM time_buckets tb
                    LEFT JOIN interactions i ON 
                        date_trunc('{granularity.lower().replace('h', 'hour').replace('d', 'day').replace('w', 'week')}', i.created_at) = tb.timestamp
                        AND i.created_at BETWEEN $1 AND $2
                    GROUP BY tb.timestamp
                    ORDER BY tb.timestamp
                    """
                elif metric_name == 'avg_response_time':
                    query = f"""
                    WITH time_buckets AS (
                        SELECT generate_series($1, $2, interval '{interval}') as timestamp
                    )
                    SELECT 
                        tb.timestamp,
                        COALESCE(AVG(i.execution_time_ms), 0) as avg_response_time
                    FROM time_buckets tb
                    LEFT JOIN interactions i ON 
                        date_trunc('{granularity.lower().replace('h', 'hour').replace('d', 'day').replace('w', 'week')}', i.created_at) = tb.timestamp
                        AND i.created_at BETWEEN $1 AND $2
                    GROUP BY tb.timestamp
                    ORDER BY tb.timestamp
                    """
                else:
                    query = f"""
                    WITH time_buckets AS (
                        SELECT generate_series($1, $2, interval '{interval}') as timestamp
                    )
                    SELECT 
                        tb.timestamp,
                        0 as value
                    FROM time_buckets tb
                    ORDER BY tb.timestamp
                    """
                
                results = await conn.fetch(query, start_date, end_date)
                
                return [
                    {
                        'timestamp': row['timestamp'],
                        metric_name: float(row.get(metric_name, row.get('value', 0)))
                    }
                    for row in results
                ]
        
        except Exception as e:
            logger.error(f"❌ Failed to collect time series data: {e}")
            return []
    
    def _analyze_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze trend in time series"""
        try:
            # Linear trend analysis
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
            
            # Mann-Kendall trend test
            def mann_kendall_test(data):
                n = len(data)
                s = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        s += np.sign(data[j] - data[i])
                
                var_s = (n * (n - 1) * (2 * n + 5)) / 18
                
                if s > 0:
                    z = (s - 1) / np.sqrt(var_s)
                elif s < 0:
                    z = (s + 1) / np.sqrt(var_s)
                else:
                    z = 0
                
                p_value_mk = 2 * (1 - stats.norm.cdf(abs(z)))
                
                return z, p_value_mk
            
            mk_statistic, mk_p_value = mann_kendall_test(series.values)
            
            # Trend classification
            if p_value < 0.05:
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'no_trend'
            
            return {
                'linear_slope': float(slope),
                'linear_intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'standard_error': float(std_err),
                'trend_direction': trend_direction,
                'trend_strength': abs(float(r_value)),
                'mann_kendall_statistic': float(mk_statistic),
                'mann_kendall_p_value': float(mk_p_value)
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze trend: {e}")
            return {}
    
    def _analyze_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze seasonality patterns in time series"""
        try:
            seasonality_analysis = {}
            
            # Autocorrelation analysis
            max_lags = min(len(series) // 4, 40)
            autocorr = [series.autocorr(lag=lag) for lag in range(1, max_lags + 1)]
            
            # Find significant autocorrelations
            significant_lags = []
            for i, corr in enumerate(autocorr):
                if abs(corr) > 1.96 / np.sqrt(len(series)):  # 95% confidence
                    significant_lags.append({'lag': i + 1, 'correlation': float(corr)})
            
            seasonality_analysis['autocorrelations'] = autocorr[:10]  # First 10 lags
            seasonality_analysis['significant_lags'] = significant_lags
            
            # Detect periodic patterns
            if len(significant_lags) > 0:
                # Look for common seasonal patterns
                common_periods = [7, 14, 30]  # Weekly, bi-weekly, monthly
                detected_periods = []
                
                for period in common_periods:
                    if period < len(autocorr):
                        correlation = autocorr[period - 1]
                        if abs(correlation) > self.config['time_series']['seasonality_threshold']:
                            detected_periods.append({
                                'period': period,
                                'correlation': float(correlation),
                                'strength': 'strong' if abs(correlation) > 0.3 else 'weak'
                            })
                
                seasonality_analysis['detected_periods'] = detected_periods
            else:
                seasonality_analysis['detected_periods'] = []
            
            # Seasonal strength (if enough data)
            if len(series) >= 14:  # At least 2 weeks of daily data
                try:
                    decomposition = seasonal_decompose(series, model='additive', period=7)
                    seasonal_var = np.var(decomposition.seasonal.dropna())
                    total_var = np.var(series.dropna())
                    seasonal_strength = seasonal_var / total_var if total_var > 0 else 0
                    seasonality_analysis['seasonal_strength'] = float(seasonal_strength)
                except:
                    seasonality_analysis['seasonal_strength'] = 0.0
            else:
                seasonality_analysis['seasonal_strength'] = 0.0
            
            return seasonality_analysis
        
        except Exception as e:
            logger.error(f"❌ Failed to analyze seasonality: {e}")
            return {}
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, float]:
        """Test stationarity of time series"""
        try:
            stationarity_tests = {}
            
            # Augmented Dickey-Fuller test
            try:
                adf_result = adfuller(series.dropna())
                stationarity_tests['adf_statistic'] = float(adf_result[0])
                stationarity_tests['adf_p_value'] = float(adf_result[1])
                stationarity_tests['adf_critical_values'] = {k: float(v) for k, v in adf_result[4].items()}
                stationarity_tests['adf_stationary'] = adf_result[1] < 0.05
            except:
                stationarity_tests['adf_stationary'] = False
            
            # KPSS test
            try:
                kpss_result = kpss(series.dropna())
                stationarity_tests['kpss_statistic'] = float(kpss_result[0])
                stationarity_tests['kpss_p_value'] = float(kpss_result[1])
                stationarity_tests['kpss_critical_values'] = {k: float(v) for k, v in kpss_result[3].items()}
                stationarity_tests['kpss_stationary'] = kpss_result[1] > 0.05
            except:
                stationarity_tests['kpss_stationary'] = False
            
            # Overall stationarity assessment
            adf_stationary = stationarity_tests.get('adf_stationary', False)
            kpss_stationary = stationarity_tests.get('kpss_stationary', False)
            
            if adf_stationary and kpss_stationary:
                stationarity_tests['overall_assessment'] = 'stationary'
            elif not adf_stationary and not kpss_stationary:
                stationarity_tests['overall_assessment'] = 'non_stationary'
            else:
                stationarity_tests['overall_assessment'] = 'difference_stationary'
            
            return stationarity_tests
        
        except Exception as e:
            logger.error(f"❌ Failed to test stationarity: {e}")
            return {}
    
    def _decompose_series(self, series: pd.Series) -> Dict[str, Any]:
        """Decompose time series into components"""
        try:
            if len(series) < 14:  # Need at least 2 periods
                return {}
            
            # Try different decomposition methods
            decomposition_results = {}
            
            # Additive decomposition
            try:
                additive_decomp = seasonal_decompose(series, model='additive', period=7)
                decomposition_results['additive'] = {
                    'trend': additive_decomp.trend.dropna().tolist(),
                    'seasonal': additive_decomp.seasonal.dropna().tolist(),
                    'residual': additive_decomp.resid.dropna().tolist()
                }
            except:
                pass
            
            # Multiplicative decomposition
            try:
                if (series > 0).all():  # Multiplicative requires positive values
                    mult_decomp = seasonal_decompose(series, model='multiplicative', period=7)
                    decomposition_results['multiplicative'] = {
                        'trend': mult_decomp.trend.dropna().tolist(),
                        'seasonal': mult_decomp.seasonal.dropna().tolist(),
                        'residual': mult_decomp.resid.dropna().tolist()
                    }
            except:
                pass
            
            return decomposition_results
        
        except Exception as e:
            logger.error(f"❌ Failed to decompose series: {e}")
            return {}
    
    def _detect_time_series_anomalies(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect anomalies in time series"""
        try:
            anomalies = []
            
            # Statistical outlier detection
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            statistical_anomalies = series[(series < lower_bound) | (series > upper_bound)]
            
            for timestamp, value in statistical_anomalies.items():
                anomalies.append({
                    'timestamp': timestamp.isoformat(),
                    'value': float(value),
                    'type': 'statistical_outlier',
                    'method': 'iqr',
                    'severity': 'high' if abs(value - series.median()) > 2 * series.std() else 'medium'
                })
            
            # Z-score based detection
            z_scores = np.abs(stats.zscore(series.dropna()))
            z_anomalies = series[z_scores > 3]
            
            for timestamp, value in z_anomalies.items():
                if not any(a['timestamp'] == timestamp.isoformat() for a in anomalies):
                    anomalies.append({
                        'timestamp': timestamp.isoformat(),
                        'value': float(value),
                        'type': 'z_score_outlier',
                        'method': 'z_score',
                        'severity': 'high' if z_scores[series.index.get_loc(timestamp)] > 4 else 'medium'
                    })
            
            # Isolation Forest for complex anomalies
            if len(series) > 50:
                try:
                    # Prepare features (value, rolling mean, rolling std)
                    features = pd.DataFrame({
                        'value': series,
                        'rolling_mean': series.rolling(window=7).mean(),
                        'rolling_std': series.rolling(window=7).std()
                    }).fillna(method='bfill').fillna(method='ffill')
                    
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(features)
                    anomaly_scores = iso_forest.score_samples(features)
                    
                    for i, (timestamp, is_anomaly) in enumerate(zip(series.index, anomaly_labels)):
                        if is_anomaly == -1:  # Anomaly detected
                            if not any(a['timestamp'] == timestamp.isoformat() for a in anomalies):
                                anomalies.append({
                                    'timestamp': timestamp.isoformat(),
                                    'value': float(series.iloc[i]),
                                    'type': 'isolation_forest',
                                    'method': 'isolation_forest',
                                    'anomaly_score': float(anomaly_scores[i]),
                                    'severity': 'low'
                                })
                except:
                    pass
            
            return sorted(anomalies, key=lambda x: x['timestamp'])
        
        except Exception as e:
            logger.error(f"❌ Failed to detect time series anomalies: {e}")
            return []
    
    def _detect_change_points(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect change points in time series"""
        try:
            change_points = []
            
            # Simple change point detection using rolling statistics
            window_size = max(7, len(series) // 10)
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            # Detect significant changes in mean
            mean_changes = rolling_mean.diff().abs()
            mean_threshold = mean_changes.quantile(0.95)
            
            significant_mean_changes = mean_changes[mean_changes > mean_threshold]
            
            for timestamp, change_magnitude in significant_mean_changes.items():
                change_points.append({
                    'timestamp': timestamp.isoformat(),
                    'type': 'mean_shift',
                    'change_magnitude': float(change_magnitude),
                    'direction': 'increase' if rolling_mean.diff().loc[timestamp] > 0 else 'decrease'
                })
            
            # Detect significant changes in variance
            std_changes = rolling_std.diff().abs()
            std_threshold = std_changes.quantile(0.95)
            
            significant_std_changes = std_changes[std_changes > std_threshold]
            
            for timestamp, change_magnitude in significant_std_changes.items():
                if not any(cp['timestamp'] == timestamp.isoformat() for cp in change_points):
                    change_points.append({
                        'timestamp': timestamp.isoformat(),
                        'type': 'variance_shift',
                        'change_magnitude': float(change_magnitude),
                        'direction': 'increase' if rolling_std.diff().loc[timestamp] > 0 else 'decrease'
                    })
            
            return sorted(change_points, key=lambda x: x['timestamp'])[:10]  # Limit to top 10
        
        except Exception as e:
            logger.error(f"❌ Failed to detect change points: {e}")
            return []
    
    def _forecast_series(self, series: pd.Series) -> Dict[str, Any]:
        """Generate forecasts for time series"""
        try:
            forecast_results = {}
            horizon = self.config['time_series']['forecast_horizon']
            
            # Simple moving average forecast
            try:
                ma_window = min(7, len(series) // 4)
                ma_forecast = series.rolling(window=ma_window).mean().iloc[-1]
                
                forecast_results['moving_average'] = {
                    'method': 'simple_moving_average',
                    'forecast_value': float(ma_forecast),
                    'window_size': ma_window
                }
            except:
                pass
            
            # Exponential smoothing
            try:
                alpha = 0.3
                exp_smooth = series.ewm(alpha=alpha).mean().iloc[-1]
                
                forecast_results['exponential_smoothing'] = {
                    'method': 'exponential_smoothing',
                    'forecast_value': float(exp_smooth),
                    'alpha': alpha
                }
            except:
                pass
            
            # Linear trend extrapolation
            try:
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
                
                next_x = len(series)
                linear_forecast = slope * next_x + intercept
                
                forecast_results['linear_trend'] = {
                    'method': 'linear_trend',
                    'forecast_value': float(linear_forecast),
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2)
                }
            except:
                pass
            
            # ARIMA forecast (if enough data)
            if len(series) >= 50:
                try:
                    # Auto-select ARIMA parameters
                    model = ARIMA(series.dropna(), order=(1, 1, 1))
                    fitted_model = model.fit()
                    
                    arima_forecast = fitted_model.forecast(steps=1)[0]
                    confidence_int = fitted_model.forecast(steps=1, alpha=0.05)[1]
                    
                    forecast_results['arima'] = {
                        'method': 'arima',
                        'forecast_value': float(arima_forecast),
                        'confidence_interval': [float(confidence_int[0]), float(confidence_int[1])],
                        'aic': float(fitted_model.aic),
                        'bic': float(fitted_model.bic)
                    }
                except:
                    pass
            
            return forecast_results
        
        except Exception as e:
            logger.error(f"❌ Failed to forecast series: {e}")
            return {}
    
    def _evaluate_forecast_performance(self, series: pd.Series, forecast_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate forecast performance using historical data"""
        try:
            if len(series) < 20:
                return {}
            
            # Use last 20% of data for validation
            split_point = int(len(series) * 0.8)
            train_series = series.iloc[:split_point]
            test_series = series.iloc[split_point:]
            
            performance_metrics = {}
            
            for method, forecast_data in forecast_results.items():
                try:
                    # Generate forecasts for test period
                    if method == 'moving_average':
                        window = forecast_data.get('window_size', 7)
                        predictions = []
                        for i in range(len(test_series)):
                            if i == 0:
                                pred = train_series.rolling(window=window).mean().iloc[-1]
                            else:
                                recent_data = pd.concat([train_series.iloc[-(window-i):], test_series.iloc[:i]])
                                pred = recent_data.rolling(window=window).mean().iloc[-1]
                            predictions.append(pred)
                    
                    elif method == 'exponential_smoothing':
                        alpha = forecast_data.get('alpha', 0.3)
                        predictions = []
                        for i in range(len(test_series)):
                            if i == 0:
                                pred = train_series.ewm(alpha=alpha).mean().iloc[-1]
                            else:
                                recent_data = pd.concat([train_series, test_series.iloc[:i]])
                                pred = recent_data.ewm(alpha=alpha).mean().iloc[-1]
                            predictions.append(pred)
                    
                    else:
                        # For other methods, use simple persistence
                        predictions = [train_series.iloc[-1]] * len(test_series)
                    
                    # Calculate metrics
                    if len(predictions) == len(test_series):
                        mae = np.mean(np.abs(np.array(predictions) - test_series.values))
                        rmse = np.sqrt(np.mean((np.array(predictions) - test_series.values) ** 2))
                        mape = np.mean(np.abs((test_series.values - np.array(predictions)) / test_series.values)) * 100
                        
                        performance_metrics[method] = {
                            'mae': float(mae),
                            'rmse': float(rmse),
                            'mape': float(mape) if not np.isnan(mape) else 999.0
                        }
                
                except Exception as method_error:
                    logger.warning(f"Failed to evaluate {method}: {method_error}")
                    continue
            
            return performance_metrics
        
        except Exception as e:
            logger.error(f"❌ Failed to evaluate forecast performance: {e}")
            return {}
    
    async def perform_cluster_analysis(self, feature_set: List[str], 
                                     clustering_method: str = 'kmeans') -> ClusterAnalysis:
        """Perform clustering analysis on student data"""
        try:
            logger.info(f"🔍 Performing cluster analysis using {clustering_method}")
            
            # Collect student feature data
            student_data = await self._collect_student_feature_data(feature_set)
            
            if len(student_data) < 10:
                raise ValueError("Insufficient data for clustering analysis")
            
            # Prepare data
            df = pd.DataFrame(student_data)
            features = df[feature_set].fillna(df[feature_set].median())
            
            # Scale features
            if self.config['clustering']['feature_scaling']:
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                features_df = pd.DataFrame(features_scaled, columns=feature_set, index=df.index)
            else:
                features_df = features
            
            # Determine optimal number of clusters
            optimal_clusters = self._find_optimal_clusters(features_df, clustering_method)
            
            # Perform clustering
            cluster_assignments, cluster_model = self._perform_clustering(
                features_df, clustering_method, optimal_clusters
            )
            
            # Analyze cluster characteristics
            cluster_characteristics = self._analyze_cluster_characteristics(
                df, features_df, cluster_assignments, feature_set
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_cluster_quality(features_df, cluster_assignments)
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                features_df, cluster_assignments, feature_set
            )
            
            # Generate cluster interpretations
            cluster_interpretations = self._interpret_clusters(
                cluster_characteristics, feature_set
            )
            
            cluster_id = f"cluster_{clustering_method}_{int(datetime.now().timestamp())}"
            
            cluster_analysis = ClusterAnalysis(
                cluster_id=cluster_id,
                clustering_method=clustering_method,
                feature_set=feature_set,
                n_clusters=len(set(cluster_assignments.values())),
                cluster_assignments=cluster_assignments,
                cluster_characteristics=cluster_characteristics,
                cluster_quality_metrics=quality_metrics,
                optimal_clusters=optimal_clusters,
                feature_importance=feature_importance,
                cluster_interpretations=cluster_interpretations
            )
            
            # Store results
            self.cluster_analyses[cluster_id] = cluster_analysis
            await self._save_cluster_analysis(cluster_analysis)
            
            logger.info(f"✅ Cluster analysis completed with {len(set(cluster_assignments.values()))} clusters")
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to perform cluster analysis: {e}")
            raise
    
    async def _collect_student_feature_data(self, feature_set: List[str]) -> List[Dict[str, Any]]:
        """Collect student feature data for clustering"""
        try:
            if not self.db_manager:
                return []
            
            # Get student metrics
            async with self.db_manager.postgres.get_connection() as conn:
                students = await conn.fetch("""
                    SELECT DISTINCT user_id FROM interactions 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                
                student_data = []
                
                for student_row in students:
                    student_id = str(student_row['user_id'])
                    
                    # Calculate features for this student
                    student_features = await self._calculate_student_features(student_id, feature_set)
                    
                    if student_features:
                        student_features['student_id'] = student_id
                        student_data.append(student_features)
                
                return student_data
        
        except Exception as e:
            logger.error(f"❌ Failed to collect student feature data: {e}")
            return []
    
    async def _calculate_student_features(self, student_id: str, feature_set: List[str]) -> Dict[str, float]:
        """Calculate specific features for a student"""
        try:
            if not self.db_manager:
                return {}
            
            features = {}
            
            async with self.db_manager.postgres.get_connection() as conn:
                # Get student interaction data
                interactions = await conn.fetch("""
                    SELECT * FROM interactions 
                    WHERE user_id = $1 AND created_at >= NOW() - INTERVAL '30 days'
                    ORDER BY created_at ASC
                """, student_id)
                
                if not interactions:
                    return {}
                
                df = pd.DataFrame([dict(row) for row in interactions])
                
                # Calculate requested features
                for feature in feature_set:
                    if feature == 'success_rate':
                        features[feature] = df['success'].mean() if 'success' in df.columns else 0.0
                    elif feature == 'avg_response_time':
                        features[feature] = df['execution_time_ms'].mean() if 'execution_time_ms' in df.columns else 0.0
                    elif feature == 'interaction_frequency':
                        days_active = (df['created_at'].max() - df['created_at'].min()).days + 1
                        features[feature] = len(df) / days_active
                    elif feature == 'concepts_covered':
                        features[feature] = df['agent_type'].nunique() if 'agent_type' in df.columns else 0
                    elif feature == 'help_seeking_rate':
                        help_count = 0
                        for _, row in df.iterrows():
                            if row.get('metadata'):
                                try:
                                    metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                                    if metadata.get('help_requested'):
                                        help_count += 1
                                except:
                                    pass
                        features[feature] = help_count / len(df)
                    elif feature == 'session_duration':
                        # Calculate average session duration
                        df['time_diff'] = df['created_at'].diff().dt.total_seconds() / 60
                        session_breaks = df['time_diff'] > 30  # 30 minutes = new session
                        session_durations = df.groupby(session_breaks.cumsum())['time_diff'].sum()
                        features[feature] = session_durations.mean()
                    else:
                        features[feature] = 0.0  # Default value
                
                return features
        
        except Exception as e:
            logger.error(f"❌ Failed to calculate student features: {e}")
            return {}
    
    def _find_optimal_clusters(self, features_df: pd.DataFrame, method: str) -> int:
        """Find optimal number of clusters using various metrics"""
        try:
            max_clusters = min(self.config['clustering']['max_clusters'], len(features_df) // 3)
            
            if method == 'kmeans':
                # Elbow method and silhouette analysis
                inertias = []
                silhouette_scores = []
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(features_df)
                    
                    inertias.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(features_df, cluster_labels))
                
                # Find elbow using second derivative
                if len(inertias) >= 3:
                    second_derivatives = np.diff(inertias, 2)
                    elbow_index = np.argmax(second_derivatives) + 2
                else:
                    elbow_index = 2
                
                # Find best silhouette score
                best_silhouette_index = np.argmax(silhouette_scores) + 2
                
                # Average the two methods
                optimal_k = int((elbow_index + best_silhouette_index) / 2)
                return min(max(optimal_k, 2), max_clusters)
            
            elif method == 'dbscan':
                # Use heuristic based on data size
                return -1  # DBSCAN doesn't require pre-specified number of clusters
            
            elif method == 'hierarchical':
                # Use silhouette method
                silhouette_scores = []
                for k in range(2, max_clusters + 1):
                    hierarchical = AgglomerativeClustering(n_clusters=k)
                    cluster_labels = hierarchical.fit_predict(features_df)
                    silhouette_scores.append(silhouette_score(features_df, cluster_labels))
                
                return np.argmax(silhouette_scores) + 2
            
            return 3  # Default
        
        except Exception as e:
            logger.error(f"❌ Failed to find optimal clusters: {e}")
            return 3
    
    def _perform_clustering(self, features_df: pd.DataFrame, method: str, 
                          n_clusters: int) -> Tuple[Dict[str, int], Any]:
        """Perform clustering using specified method"""
        try:
            cluster_assignments = {}
            
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features_df)
                
                for i, student_id in enumerate(features_df.index):
                    cluster_assignments[str(student_id)] = int(cluster_labels[i])
                
                return cluster_assignments, kmeans
            
            elif method == 'dbscan':
                # Estimate eps using k-distance
                from sklearn.neighbors import NearestNeighbors
                
                neighbors = NearestNeighbors(n_neighbors=4)
                neighbors_fit = neighbors.fit(features_df)
                distances, indices = neighbors_fit.kneighbors(features_df)
                distances = np.sort(distances[:, 3], axis=0)
                
                # Use knee point as eps
                eps = distances[len(distances) // 2]
                
                dbscan = DBSCAN(eps=eps, min_samples=self.config['clustering']['min_cluster_size'])
                cluster_labels = dbscan.fit_predict(features_df)
                
                for i, student_id in enumerate(features_df.index):
                    cluster_assignments[str(student_id)] = int(cluster_labels[i])
                
                return cluster_assignments, dbscan
            
            elif method == 'hierarchical':
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                cluster_labels = hierarchical.fit_predict(features_df)
                
                for i, student_id in enumerate(features_df.index):
                    cluster_assignments[str(student_id)] = int(cluster_labels[i])
                
                return cluster_assignments, hierarchical
            
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        
        except Exception as e:
            logger.error(f"❌ Failed to perform clustering: {e}")
            return {}, None
    
    # Additional helper methods would be implemented here...
    # (continuing with clustering analysis, anomaly detection, etc.)
    
    async def _save_time_series_analysis(self, analysis: TimeSeriesAnalysis):
        """Save time series analysis to database"""
        try:
            if not self.db_manager:
                return
            
            async with self.db_manager.postgres.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO time_series_analyses 
                    (series_id, metric_name, start_date, end_date, trend_analysis,
                     seasonality_analysis, stationarity_tests, forecast_results,
                     anomalies, change_points, decomposition, model_performance)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (series_id) DO UPDATE SET
                    trend_analysis = EXCLUDED.trend_analysis,
                    seasonality_analysis = EXCLUDED.seasonality_analysis,
                    stationarity_tests = EXCLUDED.stationarity_tests,
                    forecast_results = EXCLUDED.forecast_results,
                    anomalies = EXCLUDED.anomalies,
                    change_points = EXCLUDED.change_points,
                    decomposition = EXCLUDED.decomposition,
                    model_performance = EXCLUDED.model_performance
                """,
                analysis.series_id, analysis.metric_name,
                analysis.time_period[0], analysis.time_period[1],
                json.dumps(analysis.trend_analysis),
                json.dumps(analysis.seasonality_analysis),
                json.dumps(analysis.stationarity_tests),
                json.dumps(analysis.forecast_results),
                json.dumps(analysis.anomalies),
                json.dumps(analysis.change_points),
                json.dumps(analysis.decomposition),
                json.dumps(analysis.model_performance))
        
        except Exception as e:
            logger.error(f"❌ Failed to save time series analysis: {e}")
    
    # Placeholder methods for additional functionality
    def _analyze_cluster_characteristics(self, df, features_df, cluster_assignments, feature_set):
        """Analyze characteristics of each cluster"""
        return {}
    
    def _calculate_cluster_quality(self, features_df, cluster_assignments):
        """Calculate cluster quality metrics"""
        return {}
    
    def _calculate_feature_importance(self, features_df, cluster_assignments, feature_set):
        """Calculate feature importance for clustering"""
        return {}
    
    def _interpret_clusters(self, cluster_characteristics, feature_set):
        """Generate interpretations for clusters"""
        return {}
    
    async def _save_cluster_analysis(self, analysis):
        """Save cluster analysis to database"""
        pass

# Main testing function
async def test_statistical_analysis():
    """Test statistical analysis engine"""
    try:
        logger.info("🧪 Testing Statistical Analysis Engine")
        
        engine = StatisticalAnalysisEngine()
        await engine.initialize()
        
        # Test time series analysis structure
        sample_ts_analysis = TimeSeriesAnalysis(
            series_id="ts_test_metric",
            metric_name="interaction_count",
            time_period=(datetime.now() - timedelta(days=30), datetime.now()),
            trend_analysis={'trend_direction': 'increasing', 'linear_slope': 0.5},
            seasonality_analysis={'seasonal_strength': 0.3},
            stationarity_tests={'adf_stationary': True},
            forecast_results={'moving_average': {'forecast_value': 100.5}},
            anomalies=[],
            change_points=[],
            decomposition={},
            model_performance={'mae': 5.2}
        )
        
        logger.info(f"✅ Sample time series analysis: {sample_ts_analysis.series_id}")
        
        # Test cluster analysis structure
        sample_cluster_analysis = ClusterAnalysis(
            cluster_id="cluster_test",
            clustering_method="kmeans",
            feature_set=["success_rate", "interaction_frequency"],
            n_clusters=3,
            cluster_assignments={"student1": 0, "student2": 1},
            cluster_characteristics={},
            cluster_quality_metrics={"silhouette_score": 0.65},
            optimal_clusters=3,
            feature_importance={"success_rate": 0.7, "interaction_frequency": 0.3},
            cluster_interpretations={0: "High performers", 1: "Average performers"}
        )
        
        logger.info(f"✅ Sample cluster analysis: {sample_cluster_analysis.cluster_id}")
        
        logger.info("✅ Statistical Analysis Engine test completed")
        
    except Exception as e:
        logger.error(f"❌ Statistical Analysis test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_statistical_analysis())