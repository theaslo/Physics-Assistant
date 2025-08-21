#!/usr/bin/env python3
"""
Comprehensive Model Monitoring and Alerting System - Phase 6
Monitors ML model performance, data drift, concept drift, and system health
for educational ML models with automated alerting and remediation.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    SYSTEM_HEALTH = "system_health"
    BIAS_DETECTION = "bias_detection"
    RESOURCE_USAGE = "resource_usage"
    MODEL_STALENESS = "model_staleness"

class ModelStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"

class DriftType(Enum):
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_name: str
    timestamp: datetime
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    
    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Distribution metrics
    prediction_mean: Optional[float] = None
    prediction_std: Optional[float] = None
    prediction_min: Optional[float] = None
    prediction_max: Optional[float] = None
    
    # Business metrics
    conversion_rate: Optional[float] = None
    engagement_score: Optional[float] = None
    learning_effectiveness: Optional[float] = None
    
    # System metrics
    inference_time_p50: Optional[float] = None
    inference_time_p95: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Data quality
    missing_values_rate: Optional[float] = None
    data_completeness: Optional[float] = None
    sample_size: Optional[int] = None

@dataclass
class DriftReport:
    """Data/concept drift detection report"""
    model_id: str
    drift_type: DriftType
    timestamp: datetime
    
    # Statistical test results
    test_statistic: float
    p_value: float
    drift_score: float  # 0-1 scale
    is_significant: bool
    
    # Affected features/aspects
    affected_features: List[str]
    drift_magnitude: Dict[str, float]
    
    # Recommendations
    severity: AlertSeverity
    recommended_actions: List[str]
    
    # Additional context
    baseline_period: Tuple[datetime, datetime]
    comparison_period: Tuple[datetime, datetime]
    sample_sizes: Dict[str, int]

@dataclass
class Alert:
    """System alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    model_id: str
    title: str
    description: str
    
    # Context
    triggered_at: datetime
    metric_values: Dict[str, Any]
    threshold_values: Dict[str, Any]
    
    # Resolution
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Escalation
    escalation_level: int = 0
    last_escalated: Optional[datetime] = None
    assigned_to: Optional[str] = None

@dataclass
class MonitoringThresholds:
    """Monitoring thresholds for alerts"""
    model_id: str
    
    # Performance thresholds
    min_accuracy: Optional[float] = 0.7
    min_precision: Optional[float] = 0.7
    min_recall: Optional[float] = 0.7
    min_f1_score: Optional[float] = 0.7
    max_rmse: Optional[float] = None
    min_r2_score: Optional[float] = 0.6
    
    # System thresholds
    max_inference_time_p95: float = 200.0  # milliseconds
    max_memory_usage_mb: float = 2048.0
    max_cpu_usage_percent: float = 80.0
    
    # Data quality thresholds
    max_missing_values_rate: float = 0.1
    min_data_completeness: float = 0.9
    min_sample_size: int = 100
    
    # Drift thresholds
    drift_significance_level: float = 0.05
    drift_score_threshold: float = 0.3
    
    # Business metrics
    min_conversion_rate: Optional[float] = None
    min_engagement_score: Optional[float] = None

class DataDriftDetector:
    """Detect data drift in features"""
    
    def __init__(self):
        self.baseline_stats = {}
        self.detection_methods = {
            'ks_test': self._kolmogorov_smirnov_test,
            'chi2_test': self._chi_square_test,
            'psi': self._population_stability_index,
            'jensen_shannon': self._jensen_shannon_divergence
        }
    
    async def set_baseline(self, model_id: str, baseline_data: pd.DataFrame):
        """Set baseline data distribution"""
        try:
            self.baseline_stats[model_id] = {}
            
            for column in baseline_data.select_dtypes(include=[np.number]).columns:
                self.baseline_stats[model_id][column] = {
                    'mean': baseline_data[column].mean(),
                    'std': baseline_data[column].std(),
                    'min': baseline_data[column].min(),
                    'max': baseline_data[column].max(),
                    'quantiles': baseline_data[column].quantile([0.25, 0.5, 0.75]).to_dict(),
                    'distribution': baseline_data[column].values
                }
            
            # For categorical columns
            for column in baseline_data.select_dtypes(include=['object', 'category']).columns:
                self.baseline_stats[model_id][column] = {
                    'value_counts': baseline_data[column].value_counts(normalize=True).to_dict(),
                    'unique_values': set(baseline_data[column].unique())
                }
            
            logger.info(f"📊 Set baseline for model {model_id} with {len(baseline_data)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to set baseline: {e}")
    
    async def detect_drift(self, model_id: str, current_data: pd.DataFrame,
                         method: str = 'ks_test') -> DriftReport:
        """Detect data drift compared to baseline"""
        try:
            if model_id not in self.baseline_stats:
                raise ValueError(f"No baseline set for model {model_id}")
            
            baseline_stats = self.baseline_stats[model_id]
            drift_results = {}
            affected_features = []
            
            detection_func = self.detection_methods.get(method, self._kolmogorov_smirnov_test)
            
            # Test numerical features
            for column in current_data.select_dtypes(include=[np.number]).columns:
                if column in baseline_stats:
                    try:
                        baseline_dist = baseline_stats[column]['distribution']
                        current_dist = current_data[column].dropna().values
                        
                        if len(current_dist) > 0:
                            test_stat, p_value = detection_func(baseline_dist, current_dist)
                            
                            drift_results[column] = {
                                'test_statistic': test_stat,
                                'p_value': p_value,
                                'drift_score': 1 - p_value  # Simple drift score
                            }
                            
                            if p_value < 0.05:  # Significant drift
                                affected_features.append(column)
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Drift test failed for {column}: {e}")
            
            # Test categorical features
            for column in current_data.select_dtypes(include=['object', 'category']).columns:
                if column in baseline_stats:
                    try:
                        baseline_dist = baseline_stats[column]['value_counts']
                        current_dist = current_data[column].value_counts(normalize=True).to_dict()
                        
                        # Calculate Population Stability Index for categorical
                        psi_score = self._calculate_psi_categorical(baseline_dist, current_dist)
                        
                        drift_results[column] = {
                            'test_statistic': psi_score,
                            'p_value': 1 - min(1.0, psi_score / 0.25),  # Convert PSI to p-value-like
                            'drift_score': min(1.0, psi_score / 0.25)
                        }
                        
                        if psi_score > 0.1:  # PSI > 0.1 indicates drift
                            affected_features.append(column)
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Categorical drift test failed for {column}: {e}")
            
            # Calculate overall drift metrics
            overall_drift_score = np.mean([r['drift_score'] for r in drift_results.values()]) if drift_results else 0.0
            max_p_value = max([r['p_value'] for r in drift_results.values()]) if drift_results else 1.0
            
            # Determine severity
            severity = AlertSeverity.LOW
            if overall_drift_score > 0.7:
                severity = AlertSeverity.CRITICAL
            elif overall_drift_score > 0.5:
                severity = AlertSeverity.HIGH
            elif overall_drift_score > 0.3:
                severity = AlertSeverity.MEDIUM
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(affected_features, overall_drift_score)
            
            drift_report = DriftReport(
                model_id=model_id,
                drift_type=DriftType.FEATURE_DRIFT,
                timestamp=datetime.now(),
                test_statistic=np.mean([r['test_statistic'] for r in drift_results.values()]) if drift_results else 0.0,
                p_value=max_p_value,
                drift_score=overall_drift_score,
                is_significant=len(affected_features) > 0,
                affected_features=affected_features,
                drift_magnitude={f: drift_results[f]['drift_score'] for f in affected_features},
                severity=severity,
                recommended_actions=recommendations,
                baseline_period=(datetime.now() - timedelta(days=30), datetime.now() - timedelta(days=7)),
                comparison_period=(datetime.now() - timedelta(days=7), datetime.now()),
                sample_sizes={'baseline': len(baseline_stats), 'current': len(current_data)}
            )
            
            return drift_report
            
        except Exception as e:
            logger.error(f"❌ Drift detection failed: {e}")
            raise
    
    def _kolmogorov_smirnov_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Kolmogorov-Smirnov test for distribution comparison"""
        return ks_2samp(baseline, current)
    
    def _chi_square_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Chi-square test for distribution comparison"""
        try:
            # Create bins
            combined = np.concatenate([baseline, current])
            bins = np.histogram_bin_edges(combined, bins='auto')
            
            baseline_hist, _ = np.histogram(baseline, bins=bins)
            current_hist, _ = np.histogram(current, bins=bins)
            
            # Ensure no zero counts for chi-square test
            baseline_hist = baseline_hist + 1
            current_hist = current_hist + 1
            
            contingency_table = np.array([baseline_hist, current_hist])
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            return chi2, p_value
            
        except Exception as e:
            logger.error(f"❌ Chi-square test failed: {e}")
            return 0.0, 1.0
    
    def _population_stability_index(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Population Stability Index calculation"""
        try:
            # Create bins based on baseline data
            bins = np.percentile(baseline, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            bins = np.unique(bins)  # Remove duplicates
            
            baseline_hist, _ = np.histogram(baseline, bins=bins)
            current_hist, _ = np.histogram(current, bins=bins)
            
            # Normalize to probabilities
            baseline_prob = baseline_hist / baseline_hist.sum()
            current_prob = current_hist / current_hist.sum()
            
            # Avoid division by zero
            baseline_prob = np.where(baseline_prob == 0, 1e-10, baseline_prob)
            current_prob = np.where(current_prob == 0, 1e-10, current_prob)
            
            # Calculate PSI
            psi = np.sum((current_prob - baseline_prob) * np.log(current_prob / baseline_prob))
            
            # Convert to p-value-like score
            p_value = max(0.0, 1.0 - min(1.0, psi / 0.25))
            
            return psi, p_value
            
        except Exception as e:
            logger.error(f"❌ PSI calculation failed: {e}")
            return 0.0, 1.0
    
    def _jensen_shannon_divergence(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Jensen-Shannon divergence calculation"""
        try:
            # Create probability distributions
            combined = np.concatenate([baseline, current])
            bins = np.histogram_bin_edges(combined, bins='auto')
            
            baseline_hist, _ = np.histogram(baseline, bins=bins)
            current_hist, _ = np.histogram(current, bins=bins)
            
            # Normalize to probabilities
            p = baseline_hist / baseline_hist.sum()
            q = current_hist / current_hist.sum()
            
            # Avoid log(0)
            p = np.where(p == 0, 1e-10, p)
            q = np.where(q == 0, 1e-10, q)
            
            # Calculate JS divergence
            m = 0.5 * (p + q)
            divergence = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            
            # Convert to p-value-like score
            p_value = max(0.0, 1.0 - min(1.0, divergence))
            
            return divergence, p_value
            
        except Exception as e:
            logger.error(f"❌ JS divergence calculation failed: {e}")
            return 0.0, 1.0
    
    def _calculate_psi_categorical(self, baseline_dist: Dict[str, float], 
                                 current_dist: Dict[str, float]) -> float:
        """Calculate PSI for categorical variables"""
        try:
            psi = 0.0
            all_categories = set(baseline_dist.keys()) | set(current_dist.keys())
            
            for category in all_categories:
                baseline_prop = baseline_dist.get(category, 1e-10)
                current_prop = current_dist.get(category, 1e-10)
                
                psi += (current_prop - baseline_prop) * np.log(current_prop / baseline_prop)
            
            return psi
            
        except Exception as e:
            logger.error(f"❌ Categorical PSI calculation failed: {e}")
            return 0.0
    
    def _generate_drift_recommendations(self, affected_features: List[str], 
                                      drift_score: float) -> List[str]:
        """Generate recommendations for handling drift"""
        recommendations = []
        
        if drift_score > 0.7:
            recommendations.append("Critical drift detected - immediate model retraining recommended")
            recommendations.append("Review data collection process for potential issues")
        elif drift_score > 0.5:
            recommendations.append("Significant drift detected - schedule model retraining")
            recommendations.append("Investigate changes in data sources or user behavior")
        elif drift_score > 0.3:
            recommendations.append("Moderate drift detected - monitor closely")
            recommendations.append("Consider incremental model updates")
        
        if affected_features:
            recommendations.append(f"Focus retraining on features: {', '.join(affected_features[:5])}")
        
        if not recommendations:
            recommendations.append("No significant drift detected - continue monitoring")
        
        return recommendations

class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self):
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics = {}
    
    async def record_metrics(self, metrics: ModelMetrics):
        """Record performance metrics"""
        try:
            self.metric_history[metrics.model_id].append(metrics)
            logger.info(f"📊 Recorded metrics for model {metrics.model_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record metrics: {e}")
    
    async def detect_performance_degradation(self, model_id: str, 
                                           window_size: int = 100) -> Optional[Alert]:
        """Detect performance degradation using statistical tests"""
        try:
            if model_id not in self.metric_history:
                return None
            
            history = list(self.metric_history[model_id])
            if len(history) < window_size * 2:
                return None
            
            # Compare recent window vs baseline window
            recent_metrics = history[-window_size:]
            baseline_metrics = history[-window_size*2:-window_size]
            
            # Extract performance values (use accuracy as primary metric)
            recent_performance = [m.accuracy for m in recent_metrics if m.accuracy is not None]
            baseline_performance = [m.accuracy for m in baseline_metrics if m.accuracy is not None]
            
            if len(recent_performance) < 10 or len(baseline_performance) < 10:
                return None
            
            # Perform statistical test
            statistic, p_value = stats.ttest_ind(baseline_performance, recent_performance)
            
            # Check for significant degradation
            recent_mean = np.mean(recent_performance)
            baseline_mean = np.mean(baseline_performance)
            degradation_percent = (baseline_mean - recent_mean) / baseline_mean * 100
            
            if p_value < 0.05 and degradation_percent > 5:  # Significant degradation > 5%
                severity = AlertSeverity.CRITICAL if degradation_percent > 20 else \
                          AlertSeverity.HIGH if degradation_percent > 10 else AlertSeverity.MEDIUM
                
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.PERFORMANCE_DEGRADATION,
                    severity=severity,
                    model_id=model_id,
                    title=f"Performance Degradation Detected",
                    description=f"Model accuracy decreased by {degradation_percent:.1f}% "
                               f"(from {baseline_mean:.3f} to {recent_mean:.3f})",
                    triggered_at=datetime.now(),
                    metric_values={
                        'recent_accuracy': recent_mean,
                        'baseline_accuracy': baseline_mean,
                        'degradation_percent': degradation_percent,
                        'p_value': p_value
                    },
                    threshold_values={'max_degradation_percent': 5}
                )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Performance degradation detection failed: {e}")
            return None
    
    async def get_model_health_summary(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model health summary"""
        try:
            if model_id not in self.metric_history:
                return {'status': ModelStatus.UNKNOWN.value, 'metrics': {}}
            
            recent_metrics = list(self.metric_history[model_id])[-50:]  # Last 50 records
            
            if not recent_metrics:
                return {'status': ModelStatus.UNKNOWN.value, 'metrics': {}}
            
            # Calculate aggregated metrics
            metrics_summary = {}
            
            # Performance metrics
            accuracies = [m.accuracy for m in recent_metrics if m.accuracy is not None]
            if accuracies:
                metrics_summary['accuracy'] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'trend': self._calculate_trend(accuracies)
                }
            
            # System metrics
            inference_times = [m.inference_time_p95 for m in recent_metrics if m.inference_time_p95 is not None]
            if inference_times:
                metrics_summary['inference_time_p95'] = {
                    'mean': np.mean(inference_times),
                    'max': np.max(inference_times),
                    'trend': self._calculate_trend(inference_times)
                }
            
            # Determine overall health status
            status = self._determine_health_status(metrics_summary)
            
            return {
                'status': status.value,
                'metrics': metrics_summary,
                'last_updated': recent_metrics[-1].timestamp.isoformat(),
                'sample_count': len(recent_metrics)
            }
            
        except Exception as e:
            logger.error(f"❌ Health summary generation failed: {e}")
            return {'status': ModelStatus.UNKNOWN.value, 'metrics': {}}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.001:
            return 'improving'
        elif slope < -0.001:
            return 'degrading'
        else:
            return 'stable'
    
    def _determine_health_status(self, metrics_summary: Dict[str, Any]) -> ModelStatus:
        """Determine overall model health status"""
        try:
            # Check accuracy
            if 'accuracy' in metrics_summary:
                acc_mean = metrics_summary['accuracy']['mean']
                acc_trend = metrics_summary['accuracy']['trend']
                
                if acc_mean < 0.6:
                    return ModelStatus.FAILED
                elif acc_mean < 0.7 or acc_trend == 'degrading':
                    return ModelStatus.DEGRADED
                elif acc_trend == 'degrading':
                    return ModelStatus.WARNING
            
            # Check inference time
            if 'inference_time_p95' in metrics_summary:
                inf_time = metrics_summary['inference_time_p95']['max']
                if inf_time > 500:  # > 500ms
                    return ModelStatus.WARNING
            
            return ModelStatus.HEALTHY
            
        except Exception:
            return ModelStatus.UNKNOWN

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.notification_channels = []
        
        # Alert escalation rules
        self.escalation_rules = {
            AlertSeverity.CRITICAL: timedelta(minutes=5),
            AlertSeverity.HIGH: timedelta(minutes=15),
            AlertSeverity.MEDIUM: timedelta(hours=1),
            AlertSeverity.LOW: timedelta(hours=4)
        }
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel"""
        self.notification_channels.append({
            'type': channel_type,
            'config': config
        })
    
    async def trigger_alert(self, alert: Alert) -> bool:
        """Trigger a new alert"""
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            # Store in Redis for persistence
            if self.redis_client:
                alert_data = {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'model_id': alert.model_id,
                    'title': alert.title,
                    'description': alert.description,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'metric_values': alert.metric_values,
                    'threshold_values': alert.threshold_values
                }
                
                self.redis_client.setex(
                    f"alert:{alert.alert_id}",
                    timedelta(days=7),
                    json.dumps(alert_data)
                )
            
            logger.info(f"🚨 Alert triggered: {alert.title} ({alert.severity.value})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to trigger alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve an active alert"""
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.is_resolved = True
            alert.resolved_at = datetime.now()
            alert.resolution_notes = resolution_notes
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"✅ Alert resolved: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to resolve alert: {e}")
            return False
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through configured channels"""
        try:
            for channel in self.notification_channels:
                try:
                    if channel['type'] == 'email':
                        await self._send_email_notification(alert, channel['config'])
                    elif channel['type'] == 'slack':
                        await self._send_slack_notification(alert, channel['config'])
                    elif channel['type'] == 'webhook':
                        await self._send_webhook_notification(alert, channel['config'])
                        
                except Exception as e:
                    logger.error(f"❌ Failed to send {channel['type']} notification: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Notification sending failed: {e}")
    
    async def _send_email_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = config['to_email']
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Alert ID: {alert.alert_id}
            - Model: {alert.model_id}
            - Severity: {alert.severity.value}
            - Description: {alert.description}
            - Triggered At: {alert.triggered_at}
            
            Metric Values: {json.dumps(alert.metric_values, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls'):
                server.starttls()
            if config.get('username'):
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"❌ Email notification failed: {e}")
    
    async def _send_slack_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack notification"""
        try:
            webhook_url = config['webhook_url']
            
            color_map = {
                AlertSeverity.CRITICAL: '#FF0000',
                AlertSeverity.HIGH: '#FF8C00',
                AlertSeverity.MEDIUM: '#FFD700',
                AlertSeverity.LOW: '#32CD32'
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, '#808080'),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {"title": "Model", "value": alert.model_id, "short": True},
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Time", "value": alert.triggered_at.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"❌ Slack notification failed: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook notification"""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'model_id': alert.model_id,
                'title': alert.title,
                'description': alert.description,
                'triggered_at': alert.triggered_at.isoformat(),
                'metric_values': alert.metric_values,
                'threshold_values': alert.threshold_values
            }
            
            response = requests.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"❌ Webhook notification failed: {e}")

class ModelMonitoringSystem:
    """Comprehensive model monitoring system"""
    
    def __init__(self, db_manager=None, redis_client=None):
        self.db_manager = db_manager
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        
        # Core components
        self.performance_monitor = PerformanceMonitor()
        self.drift_detector = DataDriftDetector()
        self.alert_manager = AlertManager(redis_client)
        
        # Monitoring configuration
        self.thresholds = {}
        self.monitoring_enabled = {}
        
        # Background monitoring
        self._monitoring_tasks = []
        self._monitoring_active = False
    
    async def initialize(self):
        """Initialize the monitoring system"""
        try:
            logger.info("🚀 Initializing Model Monitoring System")
            
            # Set up default notification channels
            await self._setup_default_notifications()
            
            # Load existing configurations
            await self._load_monitoring_configurations()
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            logger.info("✅ Model Monitoring System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Model Monitoring System: {e}")
            return False
    
    async def register_model(self, model_id: str, model_name: str,
                           thresholds: MonitoringThresholds = None):
        """Register a model for monitoring"""
        try:
            # Set default thresholds if not provided
            if thresholds is None:
                thresholds = MonitoringThresholds(model_id=model_id)
            
            self.thresholds[model_id] = thresholds
            self.monitoring_enabled[model_id] = True
            
            logger.info(f"📝 Registered model for monitoring: {model_name} ({model_id})")
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
    
    async def record_model_metrics(self, metrics: ModelMetrics):
        """Record model performance metrics"""
        try:
            # Store metrics
            await self.performance_monitor.record_metrics(metrics)
            
            # Check for alerts
            await self._check_metric_thresholds(metrics)
            
            # Check for performance degradation
            degradation_alert = await self.performance_monitor.detect_performance_degradation(
                metrics.model_id
            )
            if degradation_alert:
                await self.alert_manager.trigger_alert(degradation_alert)
            
        except Exception as e:
            logger.error(f"❌ Failed to record model metrics: {e}")
    
    async def check_data_drift(self, model_id: str, current_data: pd.DataFrame) -> DriftReport:
        """Check for data drift"""
        try:
            drift_report = await self.drift_detector.detect_drift(model_id, current_data)
            
            # Trigger alert if significant drift detected
            if drift_report.is_significant and drift_report.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    alert_type=AlertType.DATA_DRIFT,
                    severity=drift_report.severity,
                    model_id=model_id,
                    title="Data Drift Detected",
                    description=f"Significant drift detected in {len(drift_report.affected_features)} features",
                    triggered_at=datetime.now(),
                    metric_values={
                        'drift_score': drift_report.drift_score,
                        'affected_features': drift_report.affected_features,
                        'p_value': drift_report.p_value
                    },
                    threshold_values={'drift_threshold': 0.3}
                )
                
                await self.alert_manager.trigger_alert(alert)
            
            return drift_report
            
        except Exception as e:
            logger.error(f"❌ Data drift check failed: {e}")
            raise
    
    async def get_monitoring_dashboard(self, model_id: str = None) -> Dict[str, Any]:
        """Get monitoring dashboard data"""
        try:
            if model_id:
                # Single model dashboard
                model_health = await self.performance_monitor.get_model_health_summary(model_id)
                active_alerts = [a for a in self.alert_manager.active_alerts.values() 
                               if a.model_id == model_id]
                
                return {
                    'model_id': model_id,
                    'health_status': model_health,
                    'active_alerts': len(active_alerts),
                    'recent_alerts': active_alerts[-5:],
                    'monitoring_enabled': self.monitoring_enabled.get(model_id, False)
                }
            else:
                # System-wide dashboard
                all_models = list(self.monitoring_enabled.keys())
                model_statuses = {}
                
                for mid in all_models:
                    health = await self.performance_monitor.get_model_health_summary(mid)
                    model_statuses[mid] = health['status']
                
                return {
                    'total_models': len(all_models),
                    'model_statuses': model_statuses,
                    'total_active_alerts': len(self.alert_manager.active_alerts),
                    'alert_summary': self._get_alert_summary(),
                    'system_health': self._get_system_health_status()
                }
                
        except Exception as e:
            logger.error(f"❌ Dashboard generation failed: {e}")
            return {}
    
    async def _check_metric_thresholds(self, metrics: ModelMetrics):
        """Check if metrics violate thresholds"""
        try:
            if metrics.model_id not in self.thresholds:
                return
            
            thresholds = self.thresholds[metrics.model_id]
            alerts_to_trigger = []
            
            # Check performance thresholds
            if metrics.accuracy is not None and thresholds.min_accuracy is not None:
                if metrics.accuracy < thresholds.min_accuracy:
                    alerts_to_trigger.append(
                        self._create_threshold_alert(
                            metrics.model_id, "Accuracy Below Threshold",
                            f"Accuracy {metrics.accuracy:.3f} is below threshold {thresholds.min_accuracy}",
                            AlertType.PERFORMANCE_DEGRADATION, AlertSeverity.HIGH,
                            {'current_accuracy': metrics.accuracy, 'threshold': thresholds.min_accuracy}
                        )
                    )
            
            # Check system thresholds
            if metrics.inference_time_p95 is not None:
                if metrics.inference_time_p95 > thresholds.max_inference_time_p95:
                    severity = AlertSeverity.CRITICAL if metrics.inference_time_p95 > thresholds.max_inference_time_p95 * 2 else AlertSeverity.HIGH
                    alerts_to_trigger.append(
                        self._create_threshold_alert(
                            metrics.model_id, "High Inference Time",
                            f"P95 inference time {metrics.inference_time_p95:.1f}ms exceeds threshold",
                            AlertType.SYSTEM_HEALTH, severity,
                            {'current_time': metrics.inference_time_p95, 'threshold': thresholds.max_inference_time_p95}
                        )
                    )
            
            # Trigger alerts
            for alert in alerts_to_trigger:
                await self.alert_manager.trigger_alert(alert)
                
        except Exception as e:
            logger.error(f"❌ Threshold checking failed: {e}")
    
    def _create_threshold_alert(self, model_id: str, title: str, description: str,
                              alert_type: AlertType, severity: AlertSeverity,
                              metric_values: Dict[str, Any]) -> Alert:
        """Create threshold violation alert"""
        return Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            model_id=model_id,
            title=title,
            description=description,
            triggered_at=datetime.now(),
            metric_values=metric_values,
            threshold_values={}
        )
    
    def _get_alert_summary(self) -> Dict[str, int]:
        """Get alert summary by severity"""
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.alert_manager.active_alerts.values():
            summary[alert.severity.value] += 1
        
        return summary
    
    def _get_system_health_status(self) -> str:
        """Get overall system health status"""
        active_alerts = self.alert_manager.active_alerts.values()
        
        if any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            return "critical"
        elif any(a.severity == AlertSeverity.HIGH for a in active_alerts):
            return "degraded"
        elif any(a.severity == AlertSeverity.MEDIUM for a in active_alerts):
            return "warning"
        else:
            return "healthy"
    
    async def _setup_default_notifications(self):
        """Setup default notification channels"""
        # In production, these would be configured from environment variables
        logger.info("📧 Setting up default notification channels")
        
        # Example webhook channel
        self.alert_manager.add_notification_channel('webhook', {
            'url': 'http://localhost:8080/alerts',  # Example webhook
            'headers': {'Content-Type': 'application/json'}
        })
    
    async def _load_monitoring_configurations(self):
        """Load monitoring configurations from database"""
        # In production, would load from database
        logger.info("⚙️ Loading monitoring configurations")
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        self._monitoring_active = True
        
        # Create background task for periodic checks
        task = asyncio.create_task(self._background_monitoring_loop())
        self._monitoring_tasks.append(task)
    
    async def _background_monitoring_loop(self):
        """Background monitoring loop"""
        try:
            while self._monitoring_active:
                # Perform periodic health checks
                await self._periodic_health_check()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
        except Exception as e:
            logger.error(f"❌ Background monitoring failed: {e}")
    
    async def _periodic_health_check(self):
        """Perform periodic health checks"""
        try:
            # Check for stale models (no metrics in last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for model_id in self.monitoring_enabled.keys():
                if model_id in self.performance_monitor.metric_history:
                    history = self.performance_monitor.metric_history[model_id]
                    if history and history[-1].timestamp < cutoff_time:
                        # Model is stale
                        alert = Alert(
                            alert_id=str(uuid.uuid4()),
                            alert_type=AlertType.MODEL_STALENESS,
                            severity=AlertSeverity.MEDIUM,
                            model_id=model_id,
                            title="Model Staleness Detected",
                            description=f"No metrics received for {model_id} in the last hour",
                            triggered_at=datetime.now(),
                            metric_values={'last_seen': history[-1].timestamp.isoformat()},
                            threshold_values={'max_staleness_hours': 1}
                        )
                        
                        await self.alert_manager.trigger_alert(alert)
                        
        except Exception as e:
            logger.error(f"❌ Periodic health check failed: {e}")
    
    async def cleanup(self):
        """Cleanup monitoring system"""
        try:
            self._monitoring_active = False
            
            # Cancel background tasks
            for task in self._monitoring_tasks:
                task.cancel()
            
            logger.info("✅ Model monitoring system cleaned up")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Testing function
async def test_model_monitoring_system():
    """Test model monitoring system"""
    try:
        logger.info("🧪 Testing Model Monitoring System")
        
        # Initialize system
        monitoring_system = ModelMonitoringSystem()
        await monitoring_system.initialize()
        
        # Register a test model
        model_id = "test_model_001"
        await monitoring_system.register_model(
            model_id, 
            "Test Physics Recommendation Model",
            MonitoringThresholds(
                model_id=model_id,
                min_accuracy=0.75,
                max_inference_time_p95=150.0
            )
        )
        
        # Simulate some metrics
        for i in range(50):
            # Simulate degrading performance
            accuracy = 0.9 - (i * 0.01)  # Decreasing accuracy
            inference_time = 100 + (i * 2)  # Increasing inference time
            
            metrics = ModelMetrics(
                model_id=model_id,
                model_name="Test Model",
                timestamp=datetime.now() - timedelta(hours=50-i),
                accuracy=accuracy,
                precision=accuracy + 0.02,
                recall=accuracy - 0.01,
                f1_score=accuracy,
                inference_time_p95=inference_time,
                memory_usage_mb=512 + i * 10,
                sample_size=1000
            )
            
            await monitoring_system.record_model_metrics(metrics)
        
        # Test data drift detection with sample data
        np.random.seed(42)
        baseline_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        # Set baseline
        await monitoring_system.drift_detector.set_baseline(model_id, baseline_data)
        
        # Current data with drift
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0.5, 1.2, 500),  # Mean shift and variance change
            'feature2': np.random.normal(5.5, 2.5, 500),  # Mean and variance shift
            'feature3': np.random.choice(['A', 'B', 'C', 'D'], 500, p=[0.3, 0.3, 0.3, 0.1])  # New category
        })
        
        # Check for drift
        drift_report = await monitoring_system.check_data_drift(model_id, current_data)
        logger.info(f"📊 Drift Detection - Score: {drift_report.drift_score:.3f}, Significant: {drift_report.is_significant}")
        
        # Get monitoring dashboard
        dashboard = await monitoring_system.get_monitoring_dashboard(model_id)
        logger.info(f"📋 Model Status: {dashboard['health_status']['status']}")
        logger.info(f"🚨 Active Alerts: {dashboard['active_alerts']}")
        
        # System-wide dashboard
        system_dashboard = await monitoring_system.get_monitoring_dashboard()
        logger.info(f"🌐 System Health: {system_dashboard['system_health']}")
        logger.info(f"📊 Total Models: {system_dashboard['total_models']}")
        
        # Cleanup
        await monitoring_system.cleanup()
        
        logger.info("✅ Model Monitoring System test completed")
        
    except Exception as e:
        logger.error(f"❌ Model Monitoring System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_model_monitoring_system())