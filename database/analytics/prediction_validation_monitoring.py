#!/usr/bin/env python3
"""
Prediction Validation and Monitoring System for Physics Assistant Phase 6.3
Implements comprehensive model validation, performance monitoring, drift detection,
and continuous learning capabilities for educational prediction systems.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy import stats
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
from collections import defaultdict, deque
import math
import statistics
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import related modules
from .predictive_analytics import PredictiveAnalyticsEngine, PredictionResult
from .ensemble_prediction_system import EnsemblePrediction, EnsemblePredictionSystem

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationMethod(Enum):
    HOLDOUT = "holdout"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    BOOTSTRAP = "bootstrap"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"

class DriftType(Enum):
    CONCEPT_DRIFT = "concept_drift"        # Change in P(y|X)
    COVARIATE_DRIFT = "covariate_drift"    # Change in P(X)
    PRIOR_DRIFT = "prior_drift"            # Change in P(y)
    VIRTUAL_DRIFT = "virtual_drift"        # False positive drift

class MonitoringMetric(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    CALIBRATION_ERROR = "calibration_error"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ValidationResult:
    """Results from model validation"""
    validation_id: str
    model_id: str
    validation_method: ValidationMethod
    dataset_size: int
    train_size: int
    test_size: int
    metrics: Dict[str, float]
    confusion_matrix: Optional[List[List[int]]]
    feature_importance: Dict[str, float]
    cross_val_scores: List[float]
    validation_curve_data: Dict[str, Any]
    overfitting_indicators: Dict[str, float]
    validation_confidence: float
    validation_warnings: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DriftDetectionResult:
    """Results from drift detection analysis"""
    detection_id: str
    model_id: str
    drift_type: DriftType
    drift_detected: bool
    drift_score: float
    confidence: float
    affected_features: List[str]
    statistical_tests: Dict[str, Dict[str, float]]
    drift_timeline: List[Tuple[datetime, float]]
    severity: AlertLevel
    recommended_actions: List[str]
    detection_method: str
    p_value: float
    threshold: float
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PerformanceMonitoringResult:
    """Results from performance monitoring"""
    monitoring_id: str
    model_id: str
    monitoring_period: Tuple[datetime, datetime]
    metrics_tracked: List[MonitoringMetric]
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    metric_trends: Dict[str, List[Tuple[datetime, float]]]
    performance_degradation: Dict[str, float]
    alerts_triggered: List[Dict[str, Any]]
    model_stability_score: float
    prediction_consistency_score: float
    user_feedback_integration: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelCalibrationResult:
    """Results from model calibration analysis"""
    calibration_id: str
    model_id: str
    calibration_score: float
    reliability_diagram_data: Dict[str, List[float]]
    calibration_curve_data: Dict[str, List[float]]
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    is_well_calibrated: bool
    calibration_method: str
    confidence_bins: List[Tuple[float, float]]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    created_at: datetime = field(default_factory=datetime.now)

class StatisticalDriftDetector:
    """Detects statistical drift in data and model performance"""
    
    def __init__(self):
        self.drift_tests = {
            'ks_test': self._kolmogorov_smirnov_test,
            'chi_square_test': self._chi_square_test,
            'psi_test': self._population_stability_index,
            'kl_divergence': self._kl_divergence_test,
            'wasserstein_distance': self._wasserstein_distance_test
        }
        self.drift_history = defaultdict(list)
        self.baseline_distributions = {}
    
    async def detect_drift(self, reference_data: np.ndarray, current_data: np.ndarray,
                         feature_names: List[str], model_id: str,
                         drift_threshold: float = 0.05) -> DriftDetectionResult:
        """Detect drift between reference and current data"""
        try:
            logger.info(f"🔍 Detecting drift for model {model_id}")
            
            drift_detected = False
            overall_drift_score = 0.0
            affected_features = []
            statistical_tests = {}
            
            # Test each feature for drift
            for i, feature_name in enumerate(feature_names):
                if i < reference_data.shape[1] and i < current_data.shape[1]:
                    ref_feature = reference_data[:, i]
                    cur_feature = current_data[:, i]
                    
                    # Run multiple drift tests
                    feature_tests = {}
                    feature_drift_scores = []
                    
                    for test_name, test_func in self.drift_tests.items():
                        try:
                            test_result = await test_func(ref_feature, cur_feature)
                            feature_tests[test_name] = test_result
                            feature_drift_scores.append(test_result['drift_score'])
                        except Exception as e:
                            logger.warning(f"Drift test {test_name} failed for {feature_name}: {e}")
                    
                    statistical_tests[feature_name] = feature_tests
                    
                    # Aggregate drift score for this feature
                    if feature_drift_scores:
                        feature_drift_score = np.mean(feature_drift_scores)
                        overall_drift_score += feature_drift_score
                        
                        # Check if drift threshold exceeded
                        if feature_drift_score > drift_threshold:
                            affected_features.append(feature_name)
            
            # Calculate overall drift score
            if len(feature_names) > 0:
                overall_drift_score /= len(feature_names)
            
            drift_detected = overall_drift_score > drift_threshold
            
            # Determine drift type
            drift_type = await self._classify_drift_type(
                reference_data, current_data, statistical_tests
            )
            
            # Determine severity
            if overall_drift_score > 0.3:
                severity = AlertLevel.CRITICAL
            elif overall_drift_score > 0.2:
                severity = AlertLevel.WARNING
            elif overall_drift_score > 0.1:
                severity = AlertLevel.INFO
            else:
                severity = AlertLevel.INFO
            
            # Generate recommendations
            recommendations = await self._generate_drift_recommendations(
                drift_type, affected_features, overall_drift_score
            )
            
            result = DriftDetectionResult(
                detection_id=str(uuid.uuid4()),
                model_id=model_id,
                drift_type=drift_type,
                drift_detected=drift_detected,
                drift_score=overall_drift_score,
                confidence=min(0.95, 0.5 + overall_drift_score),
                affected_features=affected_features,
                statistical_tests=statistical_tests,
                drift_timeline=[],  # Would be populated with historical data
                severity=severity,
                recommended_actions=recommendations,
                detection_method='multi_test_ensemble',
                p_value=min([test['p_value'] for tests in statistical_tests.values() 
                           for test in tests.values() if 'p_value' in test]),
                threshold=drift_threshold
            )
            
            # Store in history
            self.drift_history[model_id].append((datetime.now(), overall_drift_score))
            
            logger.info(f"✅ Drift detection completed: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'} (score: {overall_drift_score:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to detect drift: {e}")
            return self._create_default_drift_result(model_id)
    
    async def _kolmogorov_smirnov_test(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        """Kolmogorov-Smirnov test for distribution difference"""
        try:
            statistic, p_value = stats.ks_2samp(ref_data, cur_data)
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_score': float(statistic),
                'significant': p_value < 0.05
            }
        except Exception as e:
            logger.error(f"KS test failed: {e}")
            return {'statistic': 0.0, 'p_value': 1.0, 'drift_score': 0.0, 'significant': False}
    
    async def _chi_square_test(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        """Chi-square test for categorical data drift"""
        try:
            # Create histograms for comparison
            bins = np.linspace(min(ref_data.min(), cur_data.min()), 
                             max(ref_data.max(), cur_data.max()), 20)
            
            ref_hist, _ = np.histogram(ref_data, bins=bins)
            cur_hist, _ = np.histogram(cur_data, bins=bins)
            
            # Avoid zero frequencies
            ref_hist = ref_hist + 1
            cur_hist = cur_hist + 1
            
            # Chi-square test
            statistic, p_value = stats.chisquare(cur_hist, ref_hist)
            
            # Normalize statistic to 0-1 range
            normalized_stat = min(1.0, statistic / (len(bins) * 10))
            
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'drift_score': float(normalized_stat),
                'significant': p_value < 0.05
            }
        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return {'statistic': 0.0, 'p_value': 1.0, 'drift_score': 0.0, 'significant': False}
    
    async def _population_stability_index(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        """Population Stability Index calculation"""
        try:
            # Create quantile-based bins
            quantiles = np.linspace(0, 1, 11)  # 10 bins
            bin_edges = np.quantile(ref_data, quantiles)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate proportions
            ref_props = []
            cur_props = []
            
            for i in range(len(bin_edges) - 1):
                ref_count = np.sum((ref_data >= bin_edges[i]) & (ref_data < bin_edges[i + 1]))
                cur_count = np.sum((cur_data >= bin_edges[i]) & (cur_data < bin_edges[i + 1]))
                
                ref_prop = ref_count / len(ref_data)
                cur_prop = cur_count / len(cur_data)
                
                # Avoid zero proportions
                ref_prop = max(ref_prop, 1e-6)
                cur_prop = max(cur_prop, 1e-6)
                
                ref_props.append(ref_prop)
                cur_props.append(cur_prop)
            
            # Calculate PSI
            psi = sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop) 
                     for ref_prop, cur_prop in zip(ref_props, cur_props))
            
            return {
                'psi': float(psi),
                'drift_score': min(1.0, float(psi) / 0.25),  # Normalize using 0.25 as high drift threshold
                'p_value': 0.01 if psi > 0.25 else 0.1,  # Approximate p-value
                'significant': psi > 0.1
            }
        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return {'psi': 0.0, 'drift_score': 0.0, 'p_value': 1.0, 'significant': False}
    
    async def _kl_divergence_test(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        """Kullback-Leibler divergence test"""
        try:
            # Create probability distributions
            bins = np.linspace(min(ref_data.min(), cur_data.min()), 
                             max(ref_data.max(), cur_data.max()), 50)
            
            ref_hist, _ = np.histogram(ref_data, bins=bins, density=True)
            cur_hist, _ = np.histogram(cur_data, bins=bins, density=True)
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            cur_prob = cur_hist / np.sum(cur_hist)
            
            # Avoid zero probabilities
            ref_prob = np.maximum(ref_prob, 1e-10)
            cur_prob = np.maximum(cur_prob, 1e-10)
            
            # Calculate KL divergence
            kl_div = stats.entropy(cur_prob, ref_prob)
            
            return {
                'kl_divergence': float(kl_div),
                'drift_score': min(1.0, float(kl_div) / 2.0),  # Normalize
                'p_value': 0.01 if kl_div > 1.0 else 0.1,  # Approximate
                'significant': kl_div > 0.5
            }
        except Exception as e:
            logger.error(f"KL divergence test failed: {e}")
            return {'kl_divergence': 0.0, 'drift_score': 0.0, 'p_value': 1.0, 'significant': False}
    
    async def _wasserstein_distance_test(self, ref_data: np.ndarray, cur_data: np.ndarray) -> Dict[str, float]:
        """Wasserstein distance test"""
        try:
            distance = stats.wasserstein_distance(ref_data, cur_data)
            
            # Normalize by data range
            data_range = max(ref_data.max(), cur_data.max()) - min(ref_data.min(), cur_data.min())
            normalized_distance = distance / data_range if data_range > 0 else 0.0
            
            return {
                'wasserstein_distance': float(distance),
                'normalized_distance': float(normalized_distance),
                'drift_score': min(1.0, float(normalized_distance) * 2),
                'p_value': 0.01 if normalized_distance > 0.1 else 0.1,
                'significant': normalized_distance > 0.05
            }
        except Exception as e:
            logger.error(f"Wasserstein distance test failed: {e}")
            return {'wasserstein_distance': 0.0, 'drift_score': 0.0, 'p_value': 1.0, 'significant': False}
    
    async def _classify_drift_type(self, ref_data: np.ndarray, cur_data: np.ndarray,
                                 statistical_tests: Dict[str, Dict[str, Any]]) -> DriftType:
        """Classify the type of drift detected"""
        try:
            # Simple heuristic based on which features are affected
            affected_count = sum(1 for tests in statistical_tests.values() 
                               if any(test.get('significant', False) for test in tests.values()))
            
            total_features = len(statistical_tests)
            
            if affected_count > total_features * 0.7:
                return DriftType.CONCEPT_DRIFT  # Most features affected
            elif affected_count > total_features * 0.3:
                return DriftType.COVARIATE_DRIFT  # Some features affected
            else:
                return DriftType.VIRTUAL_DRIFT  # Few features affected
                
        except Exception as e:
            logger.error(f"Failed to classify drift type: {e}")
            return DriftType.VIRTUAL_DRIFT
    
    async def _generate_drift_recommendations(self, drift_type: DriftType,
                                            affected_features: List[str],
                                            drift_score: float) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        
        try:
            if drift_type == DriftType.CONCEPT_DRIFT:
                recommendations.extend([
                    "Consider retraining the model with recent data",
                    "Investigate changes in the underlying relationship between features and target",
                    "Implement online learning or model adaptation techniques"
                ])
            
            elif drift_type == DriftType.COVARIATE_DRIFT:
                recommendations.extend([
                    "Update data preprocessing pipelines",
                    "Retrain model with new feature distributions",
                    "Consider feature selection or engineering"
                ])
            
            elif drift_type == DriftType.PRIOR_DRIFT:
                recommendations.extend([
                    "Adjust model output calibration",
                    "Update class weights or sampling strategies",
                    "Monitor target variable distribution"
                ])
            
            else:  # VIRTUAL_DRIFT
                recommendations.extend([
                    "Monitor the situation but no immediate action needed",
                    "Verify data quality and collection processes",
                    "Continue regular monitoring"
                ])
            
            # Add feature-specific recommendations
            if affected_features:
                recommendations.append(f"Pay special attention to features: {', '.join(affected_features[:5])}")
            
            # Add severity-based recommendations
            if drift_score > 0.5:
                recommendations.append("High drift detected - immediate model update recommended")
            elif drift_score > 0.2:
                recommendations.append("Moderate drift detected - schedule model evaluation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Monitor model performance and consider retraining if necessary"]
    
    def _create_default_drift_result(self, model_id: str) -> DriftDetectionResult:
        """Create default drift result when detection fails"""
        return DriftDetectionResult(
            detection_id=str(uuid.uuid4()),
            model_id=model_id,
            drift_type=DriftType.VIRTUAL_DRIFT,
            drift_detected=False,
            drift_score=0.0,
            confidence=0.0,
            affected_features=[],
            statistical_tests={},
            drift_timeline=[],
            severity=AlertLevel.INFO,
            recommended_actions=["Drift detection failed - manual review required"],
            detection_method='failed',
            p_value=1.0,
            threshold=0.05
        )

class ModelValidator:
    """Comprehensive model validation system"""
    
    def __init__(self):
        self.validation_methods = {
            ValidationMethod.HOLDOUT: self._holdout_validation,
            ValidationMethod.CROSS_VALIDATION: self._cross_validation,
            ValidationMethod.TIME_SERIES_SPLIT: self._time_series_validation,
            ValidationMethod.BOOTSTRAP: self._bootstrap_validation
        }
        
        self.regression_metrics = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2_score
        }
        
        self.classification_metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
        }
    
    async def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           model_id: str, task_type: str = 'regression',
                           validation_method: ValidationMethod = ValidationMethod.CROSS_VALIDATION) -> ValidationResult:
        """Comprehensive model validation"""
        try:
            logger.info(f"🔬 Validating model {model_id} using {validation_method.value}")
            
            # Choose validation method
            if validation_method in self.validation_methods:
                validation_func = self.validation_methods[validation_method]
            else:
                validation_func = self._cross_validation
            
            # Run validation
            results = await validation_func(model, X, y, task_type)
            
            # Calculate additional metrics
            feature_importance = await self._calculate_feature_importance(model, X, y)
            overfitting_indicators = await self._check_overfitting(model, X, y, task_type)
            
            # Generate warnings
            warnings = await self._generate_validation_warnings(results, overfitting_indicators)
            
            return ValidationResult(
                validation_id=str(uuid.uuid4()),
                model_id=model_id,
                validation_method=validation_method,
                dataset_size=len(X),
                train_size=results.get('train_size', 0),
                test_size=results.get('test_size', 0),
                metrics=results['metrics'],
                confusion_matrix=results.get('confusion_matrix'),
                feature_importance=feature_importance,
                cross_val_scores=results.get('cv_scores', []),
                validation_curve_data=results.get('validation_curve', {}),
                overfitting_indicators=overfitting_indicators,
                validation_confidence=results.get('confidence', 0.8),
                validation_warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return self._create_default_validation_result(model_id)
    
    async def _cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                              task_type: str, cv_folds: int = 5) -> Dict[str, Any]:
        """Cross-validation with comprehensive metrics"""
        try:
            # Choose metrics based on task type
            if task_type == 'classification':
                metrics = self.classification_metrics
            else:
                metrics = self.regression_metrics
            
            # Perform cross-validation
            cv_results = {}
            cv_scores = []
            
            from sklearn.model_selection import StratifiedKFold, KFold
            
            if task_type == 'classification':
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            all_predictions = []
            all_true_values = []
            
            for train_idx, test_idx in cv_splitter.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                all_predictions.extend(y_pred)
                all_true_values.extend(y_test)
                
                # Calculate primary metric
                if task_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)
                
                cv_scores.append(score)
            
            # Calculate final metrics
            final_metrics = {}
            for metric_name, metric_func in metrics.items():
                try:
                    if task_type == 'classification' and metric_name in ['precision', 'recall', 'f1']:
                        # Handle potential zero division
                        score = metric_func(all_true_values, all_predictions)
                    else:
                        score = metric_func(all_true_values, all_predictions)
                    final_metrics[metric_name] = float(score)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    final_metrics[metric_name] = 0.0
            
            # Generate confusion matrix for classification
            confusion_matrix_data = None
            if task_type == 'classification':
                try:
                    cm = confusion_matrix(all_true_values, all_predictions)
                    confusion_matrix_data = cm.tolist()
                except Exception as e:
                    logger.warning(f"Failed to generate confusion matrix: {e}")
            
            return {
                'metrics': final_metrics,
                'cv_scores': cv_scores,
                'confusion_matrix': confusion_matrix_data,
                'train_size': int(len(X) * (cv_folds - 1) / cv_folds),
                'test_size': int(len(X) / cv_folds),
                'confidence': float(np.mean(cv_scores))
            }
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            return {'metrics': {}, 'cv_scores': [], 'confidence': 0.0}
    
    async def _holdout_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                task_type: str, test_size: float = 0.2) -> Dict[str, Any]:
        """Holdout validation"""
        try:
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y if task_type == 'classification' else None
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if task_type == 'classification':
                metrics = self.classification_metrics
            else:
                metrics = self.regression_metrics
            
            final_metrics = {}
            for metric_name, metric_func in metrics.items():
                try:
                    score = metric_func(y_test, y_pred)
                    final_metrics[metric_name] = float(score)
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    final_metrics[metric_name] = 0.0
            
            return {
                'metrics': final_metrics,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'confidence': final_metrics.get('accuracy' if task_type == 'classification' else 'r2', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Holdout validation failed: {e}")
            return {'metrics': {}, 'confidence': 0.0}
    
    async def _time_series_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                    task_type: str, n_splits: int = 5) -> Dict[str, Any]:
        """Time series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_scores = []
            all_predictions = []
            all_true_values = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                all_predictions.extend(y_pred)
                all_true_values.extend(y_test)
                
                # Calculate score
                if task_type == 'classification':
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = r2_score(y_test, y_pred)
                
                cv_scores.append(score)
            
            # Final metrics
            if task_type == 'classification':
                metrics = self.classification_metrics
            else:
                metrics = self.regression_metrics
            
            final_metrics = {}
            for metric_name, metric_func in metrics.items():
                try:
                    score = metric_func(all_true_values, all_predictions)
                    final_metrics[metric_name] = float(score)
                except:
                    final_metrics[metric_name] = 0.0
            
            return {
                'metrics': final_metrics,
                'cv_scores': cv_scores,
                'confidence': float(np.mean(cv_scores))
            }
            
        except Exception as e:
            logger.error(f"❌ Time series validation failed: {e}")
            return {'metrics': {}, 'cv_scores': [], 'confidence': 0.0}
    
    async def _bootstrap_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                                  task_type: str, n_bootstrap: int = 100) -> Dict[str, Any]:
        """Bootstrap validation"""
        try:
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot, y_boot = X[indices], y[indices]
                
                # Out-of-bag sample
                oob_indices = np.setdiff1d(np.arange(len(X)), indices)
                if len(oob_indices) == 0:
                    continue
                
                X_oob, y_oob = X[oob_indices], y[oob_indices]
                
                # Train and evaluate
                model.fit(X_boot, y_boot)
                y_pred = model.predict(X_oob)
                
                if task_type == 'classification':
                    score = accuracy_score(y_oob, y_pred)
                else:
                    score = r2_score(y_oob, y_pred)
                
                bootstrap_scores.append(score)
            
            # Calculate confidence interval
            confidence_interval = np.percentile(bootstrap_scores, [2.5, 97.5])
            
            return {
                'metrics': {'bootstrap_score': float(np.mean(bootstrap_scores))},
                'cv_scores': bootstrap_scores,
                'confidence': float(np.mean(bootstrap_scores)),
                'confidence_interval': confidence_interval.tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ Bootstrap validation failed: {e}")
            return {'metrics': {}, 'cv_scores': [], 'confidence': 0.0}
    
    async def _calculate_feature_importance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            feature_importance = {}
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    feature_importance[f'feature_{i}'] = float(importance)
            
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) == 1:
                    coefficients = model.coef_
                else:
                    coefficients = model.coef_[0]
                
                for i, coef in enumerate(coefficients):
                    feature_importance[f'feature_{i}'] = float(abs(coef))
            
            else:
                # Use permutation importance as fallback
                from sklearn.inspection import permutation_importance
                
                perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42)
                for i, importance in enumerate(perm_importance.importances_mean):
                    feature_importance[f'feature_{i}'] = float(importance)
            
            return feature_importance
            
        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
            return {}
    
    async def _check_overfitting(self, model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, float]:
        """Check for overfitting indicators"""
        try:
            indicators = {}
            
            # Train-test performance gap
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            if task_type == 'classification':
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
            else:
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
            
            indicators['train_test_gap'] = float(train_score - test_score)
            indicators['train_score'] = float(train_score)
            indicators['test_score'] = float(test_score)
            
            # Learning curve analysis (simplified)
            train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
            train_scores = []
            val_scores = []
            
            for size in train_sizes:
                subset_size = int(len(X_train) * size)
                X_subset = X_train[:subset_size]
                y_subset = y_train[:subset_size]
                
                model.fit(X_subset, y_subset)
                
                train_pred_subset = model.predict(X_subset)
                val_pred_subset = model.predict(X_test)
                
                if task_type == 'classification':
                    train_score_subset = accuracy_score(y_subset, train_pred_subset)
                    val_score_subset = accuracy_score(y_test, val_pred_subset)
                else:
                    train_score_subset = r2_score(y_subset, train_pred_subset)
                    val_score_subset = r2_score(y_test, val_pred_subset)
                
                train_scores.append(train_score_subset)
                val_scores.append(val_score_subset)
            
            # Calculate convergence
            if len(train_scores) > 2:
                train_trend = np.polyfit(range(len(train_scores)), train_scores, 1)[0]
                val_trend = np.polyfit(range(len(val_scores)), val_scores, 1)[0]
                
                indicators['train_trend'] = float(train_trend)
                indicators['val_trend'] = float(val_trend)
                indicators['converging'] = float(abs(train_trend - val_trend) < 0.1)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Failed to check overfitting: {e}")
            return {}
    
    async def _generate_validation_warnings(self, results: Dict[str, Any], 
                                          overfitting_indicators: Dict[str, float]) -> List[str]:
        """Generate validation warnings"""
        warnings = []
        
        try:
            # Check for overfitting
            gap = overfitting_indicators.get('train_test_gap', 0.0)
            if gap > 0.1:
                warnings.append(f"Potential overfitting detected (train-test gap: {gap:.3f})")
            
            # Check for poor performance
            cv_scores = results.get('cv_scores', [])
            if cv_scores and np.mean(cv_scores) < 0.6:
                warnings.append(f"Low cross-validation score: {np.mean(cv_scores):.3f}")
            
            # Check for high variance
            if cv_scores and np.std(cv_scores) > 0.1:
                warnings.append(f"High variance in CV scores (std: {np.std(cv_scores):.3f})")
            
            # Check convergence
            if not overfitting_indicators.get('converging', True):
                warnings.append("Learning curves show poor convergence")
            
            return warnings
            
        except Exception as e:
            logger.warning(f"Failed to generate warnings: {e}")
            return []
    
    def _create_default_validation_result(self, model_id: str) -> ValidationResult:
        """Create default validation result when validation fails"""
        return ValidationResult(
            validation_id=str(uuid.uuid4()),
            model_id=model_id,
            validation_method=ValidationMethod.CROSS_VALIDATION,
            dataset_size=0,
            train_size=0,
            test_size=0,
            metrics={},
            confusion_matrix=None,
            feature_importance={},
            cross_val_scores=[],
            validation_curve_data={},
            overfitting_indicators={},
            validation_confidence=0.0,
            validation_warnings=["Validation failed - manual review required"]
        )

class PredictionValidationMonitoringSystem:
    """Main system for prediction validation and monitoring"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Core components
        self.drift_detector = StatisticalDriftDetector()
        self.model_validator = ModelValidator()
        
        # Monitoring data
        self.performance_history = defaultdict(list)
        self.validation_results = {}
        self.drift_results = {}
        self.monitoring_results = {}
        
        # Configuration
        self.config = {
            'drift_check_frequency_hours': 24,
            'performance_monitoring_frequency_hours': 1,
            'validation_frequency_days': 7,
            'alert_thresholds': {
                'performance_drop': 0.1,
                'drift_score': 0.2,
                'validation_score': 0.6
            },
            'retention_days': 90
        }
        
        # Monitoring active flag
        self.monitoring_active = False
    
    async def initialize(self):
        """Initialize the validation and monitoring system"""
        try:
            logger.info("🚀 Initializing Prediction Validation and Monitoring System")
            
            # Load historical data
            await self._load_historical_monitoring_data()
            
            # Start monitoring processes
            await self._start_monitoring_processes()
            
            logger.info("✅ Prediction Validation and Monitoring System initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize validation and monitoring system: {e}")
            return False
    
    async def validate_prediction_model(self, model: Any, training_data: pd.DataFrame,
                                      target_column: str, model_id: str,
                                      validation_method: ValidationMethod = ValidationMethod.CROSS_VALIDATION) -> ValidationResult:
        """Comprehensive model validation"""
        try:
            logger.info(f"🔬 Validating prediction model {model_id}")
            
            # Prepare data
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].values
            y = training_data[target_column].values
            
            # Determine task type
            task_type = 'classification' if len(np.unique(y)) < 10 else 'regression'
            
            # Run validation
            result = await self.model_validator.validate_model(
                model, X, y, model_id, task_type, validation_method
            )
            
            # Store result
            self.validation_results[model_id] = result
            
            # Generate alerts if needed
            await self._check_validation_alerts(result)
            
            logger.info(f"✅ Model validation completed for {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return self.model_validator._create_default_validation_result(model_id)
    
    async def monitor_drift(self, model_id: str, reference_data: pd.DataFrame,
                          current_data: pd.DataFrame, feature_columns: List[str]) -> DriftDetectionResult:
        """Monitor for data drift"""
        try:
            logger.info(f"🔍 Monitoring drift for model {model_id}")
            
            # Extract feature arrays
            ref_features = reference_data[feature_columns].values
            cur_features = current_data[feature_columns].values
            
            # Detect drift
            result = await self.drift_detector.detect_drift(
                ref_features, cur_features, feature_columns, model_id
            )
            
            # Store result
            self.drift_results[model_id] = result
            
            # Generate alerts if needed
            await self._check_drift_alerts(result)
            
            logger.info(f"✅ Drift monitoring completed for {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Drift monitoring failed: {e}")
            return self.drift_detector._create_default_drift_result(model_id)
    
    async def monitor_performance(self, model_id: str, predictions: List[float],
                                actual_values: List[float], prediction_metadata: List[Dict[str, Any]]) -> PerformanceMonitoringResult:
        """Monitor model performance over time"""
        try:
            logger.info(f"📊 Monitoring performance for model {model_id}")
            
            current_time = datetime.now()
            
            # Calculate current metrics
            current_metrics = await self._calculate_performance_metrics(
                predictions, actual_values
            )
            
            # Get baseline metrics
            baseline_metrics = await self._get_baseline_metrics(model_id)
            
            # Calculate performance degradation
            performance_degradation = {}
            for metric_name, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric_name, current_value)
                if baseline_value != 0:
                    degradation = (baseline_value - current_value) / baseline_value
                    performance_degradation[metric_name] = degradation
            
            # Update performance history
            for metric_name, value in current_metrics.items():
                self.performance_history[f"{model_id}_{metric_name}"].append((current_time, value))
            
            # Generate metric trends
            metric_trends = {}
            for metric_name in current_metrics.keys():
                history_key = f"{model_id}_{metric_name}"
                if history_key in self.performance_history:
                    metric_trends[metric_name] = list(self.performance_history[history_key])
            
            # Check for alerts
            alerts = await self._check_performance_alerts(model_id, current_metrics, baseline_metrics)
            
            # Calculate stability scores
            model_stability_score = await self._calculate_model_stability(model_id, current_metrics)
            prediction_consistency_score = await self._calculate_prediction_consistency(
                predictions, prediction_metadata
            )
            
            result = PerformanceMonitoringResult(
                monitoring_id=str(uuid.uuid4()),
                model_id=model_id,
                monitoring_period=(current_time - timedelta(hours=1), current_time),
                metrics_tracked=[MonitoringMetric.ACCURACY, MonitoringMetric.MSE, MonitoringMetric.MAE],
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                metric_trends=metric_trends,
                performance_degradation=performance_degradation,
                alerts_triggered=alerts,
                model_stability_score=model_stability_score,
                prediction_consistency_score=prediction_consistency_score,
                user_feedback_integration={}  # Would integrate actual user feedback
            )
            
            # Store result
            self.monitoring_results[model_id] = result
            
            logger.info(f"✅ Performance monitoring completed for {model_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {e}")
            return self._create_default_monitoring_result(model_id)
    
    async def _calculate_performance_metrics(self, predictions: List[float],
                                           actual_values: List[float]) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            metrics = {}
            
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)
            
            # Regression metrics
            metrics['mse'] = float(mean_squared_error(actual_values, predictions))
            metrics['mae'] = float(mean_absolute_error(actual_values, predictions))
            metrics['r2'] = float(r2_score(actual_values, predictions))
            
            # Classification metrics (if applicable)
            if len(np.unique(actual_values)) < 10:
                # Binary threshold at 0.5
                binary_predictions = (predictions > 0.5).astype(int)
                binary_actual = (actual_values > 0.5).astype(int)
                
                metrics['accuracy'] = float(accuracy_score(binary_actual, binary_predictions))
                metrics['precision'] = float(precision_score(binary_actual, binary_predictions, average='weighted'))
                metrics['recall'] = float(recall_score(binary_actual, binary_predictions, average='weighted'))
                metrics['f1'] = float(f1_score(binary_actual, binary_predictions, average='weighted'))
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance metrics: {e}")
            return {}
    
    async def _get_baseline_metrics(self, model_id: str) -> Dict[str, float]:
        """Get baseline metrics for comparison"""
        try:
            # Look up stored baseline or calculate from validation results
            if model_id in self.validation_results:
                return self.validation_results[model_id].metrics
            
            # Use historical average as baseline
            baseline = {}
            for metric_name in ['mse', 'mae', 'r2', 'accuracy']:
                history_key = f"{model_id}_{metric_name}"
                if history_key in self.performance_history:
                    values = [value for _, value in self.performance_history[history_key]]
                    if values:
                        baseline[metric_name] = np.mean(values)
            
            return baseline
            
        except Exception as e:
            logger.error(f"❌ Failed to get baseline metrics: {e}")
            return {}
    
    async def _check_performance_alerts(self, model_id: str, current_metrics: Dict[str, float],
                                      baseline_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance-based alerts"""
        alerts = []
        
        try:
            threshold = self.config['alert_thresholds']['performance_drop']
            
            for metric_name, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric_name)
                
                if baseline_value is not None and baseline_value != 0:
                    change = (baseline_value - current_value) / baseline_value
                    
                    # For metrics where higher is better (accuracy, r2)
                    if metric_name in ['accuracy', 'r2', 'precision', 'recall', 'f1']:
                        if change > threshold:
                            alerts.append({
                                'type': 'performance_degradation',
                                'metric': metric_name,
                                'current_value': current_value,
                                'baseline_value': baseline_value,
                                'change': change,
                                'severity': 'high' if change > 0.2 else 'medium',
                                'timestamp': datetime.now()
                            })
                    
                    # For metrics where lower is better (mse, mae)
                    elif metric_name in ['mse', 'mae']:
                        if change < -threshold:  # Negative change means increase (worse)
                            alerts.append({
                                'type': 'performance_degradation',
                                'metric': metric_name,
                                'current_value': current_value,
                                'baseline_value': baseline_value,
                                'change': abs(change),
                                'severity': 'high' if abs(change) > 0.2 else 'medium',
                                'timestamp': datetime.now()
                            })
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to check performance alerts: {e}")
            return []
    
    async def _calculate_model_stability(self, model_id: str, current_metrics: Dict[str, float]) -> float:
        """Calculate model stability score"""
        try:
            stability_scores = []
            
            for metric_name, current_value in current_metrics.items():
                history_key = f"{model_id}_{metric_name}"
                
                if history_key in self.performance_history:
                    recent_values = [value for _, value in self.performance_history[history_key][-10:]]
                    
                    if len(recent_values) > 3:
                        # Calculate coefficient of variation
                        cv = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
                        stability = 1.0 / (1.0 + cv)  # Higher stability for lower variation
                        stability_scores.append(stability)
            
            return float(np.mean(stability_scores)) if stability_scores else 0.5
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate model stability: {e}")
            return 0.5
    
    async def _calculate_prediction_consistency(self, predictions: List[float],
                                              metadata: List[Dict[str, Any]]) -> float:
        """Calculate prediction consistency score"""
        try:
            if len(predictions) < 2:
                return 1.0
            
            # Calculate variance in predictions
            pred_variance = np.var(predictions)
            
            # Lower variance indicates higher consistency
            consistency = 1.0 / (1.0 + pred_variance)
            
            return float(consistency)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate prediction consistency: {e}")
            return 0.5
    
    async def _check_validation_alerts(self, result: ValidationResult):
        """Check for validation-based alerts"""
        try:
            threshold = self.config['alert_thresholds']['validation_score']
            
            # Check primary validation score
            primary_score = result.validation_confidence
            if primary_score < threshold:
                logger.warning(f"⚠️ Low validation score for model {result.model_id}: {primary_score:.3f}")
            
            # Check for overfitting warnings
            if result.validation_warnings:
                for warning in result.validation_warnings:
                    logger.warning(f"⚠️ Validation warning for model {result.model_id}: {warning}")
            
        except Exception as e:
            logger.error(f"❌ Failed to check validation alerts: {e}")
    
    async def _check_drift_alerts(self, result: DriftDetectionResult):
        """Check for drift-based alerts"""
        try:
            if result.drift_detected and result.severity in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                logger.warning(f"⚠️ Drift detected for model {result.model_id}: {result.drift_type.value} (score: {result.drift_score:.3f})")
                
                # Log recommendations
                for recommendation in result.recommended_actions:
                    logger.info(f"💡 Recommendation: {recommendation}")
            
        except Exception as e:
            logger.error(f"❌ Failed to check drift alerts: {e}")
    
    def _create_default_monitoring_result(self, model_id: str) -> PerformanceMonitoringResult:
        """Create default monitoring result when monitoring fails"""
        return PerformanceMonitoringResult(
            monitoring_id=str(uuid.uuid4()),
            model_id=model_id,
            monitoring_period=(datetime.now() - timedelta(hours=1), datetime.now()),
            metrics_tracked=[],
            current_metrics={},
            baseline_metrics={},
            metric_trends={},
            performance_degradation={},
            alerts_triggered=[],
            model_stability_score=0.0,
            prediction_consistency_score=0.0,
            user_feedback_integration={}
        )
    
    async def _load_historical_monitoring_data(self):
        """Load historical monitoring data"""
        try:
            # In practice, would load from database
            logger.info("📊 Historical monitoring data loading not implemented yet")
            
        except Exception as e:
            logger.error(f"❌ Failed to load historical monitoring data: {e}")
    
    async def _start_monitoring_processes(self):
        """Start background monitoring processes"""
        try:
            self.monitoring_active = True
            
            # Start periodic monitoring tasks
            asyncio.create_task(self._periodic_monitoring_task())
            
            logger.info("🔄 Monitoring processes started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring processes: {e}")
    
    async def _periodic_monitoring_task(self):
        """Periodic monitoring task"""
        try:
            while self.monitoring_active:
                # Clean up old data
                await self._cleanup_old_data()
                
                # Could add other periodic tasks here
                
                # Sleep for an hour
                await asyncio.sleep(3600)
                
        except Exception as e:
            logger.error(f"❌ Periodic monitoring task error: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
            
            # Clean performance history
            for key, history in self.performance_history.items():
                self.performance_history[key] = [
                    (timestamp, value) for timestamp, value in history
                    if timestamp > cutoff_date
                ]
            
            logger.debug("🧹 Cleaned up old monitoring data")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            return {
                'monitoring_active': self.monitoring_active,
                'models_validated': len(self.validation_results),
                'models_drift_monitored': len(self.drift_results),
                'models_performance_monitored': len(self.monitoring_results),
                'total_performance_data_points': sum(len(history) for history in self.performance_history.values()),
                'config': self.config,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Shutdown the monitoring system"""
        try:
            logger.info("🛑 Shutting down Prediction Validation and Monitoring System")
            
            self.monitoring_active = False
            
            logger.info("✅ Prediction Validation and Monitoring System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")

# Testing function
async def test_prediction_validation_monitoring():
    """Test the prediction validation and monitoring system"""
    try:
        logger.info("🧪 Testing Prediction Validation and Monitoring System")
        
        system = PredictionValidationMonitoringSystem()
        await system.initialize()
        
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        X_data = np.random.randn(n_samples, n_features)
        y_data = X_data[:, 0] + 0.5 * X_data[:, 1] + np.random.randn(n_samples) * 0.1
        
        training_df = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(n_features)])
        training_df['target'] = y_data
        
        # Test model validation
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        validation_result = await system.validate_prediction_model(
            model, training_df, 'target', 'test_model_1'
        )
        
        logger.info(f"✅ Validation completed: confidence={validation_result.validation_confidence:.3f}")
        
        # Test drift detection
        # Create slightly different data for drift simulation
        X_drift = np.random.randn(n_samples, n_features) + 0.2  # Slight shift
        drift_df = pd.DataFrame(X_drift, columns=[f'feature_{i}' for i in range(n_features)])
        
        drift_result = await system.monitor_drift(
            'test_model_1', training_df, drift_df, [f'feature_{i}' for i in range(n_features)]
        )
        
        logger.info(f"✅ Drift detection completed: drift_detected={drift_result.drift_detected}, score={drift_result.drift_score:.3f}")
        
        # Test performance monitoring
        test_predictions = np.random.randn(50) + y_data[:50]
        test_actuals = y_data[:50].tolist()
        
        performance_result = await system.monitor_performance(
            'test_model_1', test_predictions.tolist(), test_actuals, [{}] * 50
        )
        
        logger.info(f"✅ Performance monitoring completed: stability={performance_result.model_stability_score:.3f}")
        
        # Test system status
        status = await system.get_system_status()
        logger.info(f"✅ System status: {status['models_validated']} models validated")
        
        await system.shutdown()
        logger.info("✅ Prediction Validation and Monitoring System test completed")
        
    except Exception as e:
        logger.error(f"❌ Prediction Validation and Monitoring System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_prediction_validation_monitoring())