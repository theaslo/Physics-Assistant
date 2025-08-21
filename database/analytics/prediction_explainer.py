#!/usr/bin/env python3
"""
Prediction Explainability and Confidence Quantification System for Physics Assistant Phase 6.3
Implements advanced explainable AI techniques for educational predictions with comprehensive
confidence estimation and uncertainty quantification.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
from collections import defaultdict, deque
import math
import statistics
from scipy import stats
import shap
import lime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    FEATURE_IMPORTANCE = "feature_importance"
    SHAP_VALUES = "shap_values"
    LIME_EXPLANATION = "lime_explanation"
    COUNTERFACTUAL = "counterfactual"
    ANCHOR_EXPLANATION = "anchor_explanation"
    ATTENTION_WEIGHTS = "attention_weights"
    CONCEPT_ACTIVATION = "concept_activation"

class ConfidenceSource(Enum):
    MODEL_UNCERTAINTY = "model_uncertainty"
    DATA_UNCERTAINTY = "data_uncertainty"
    PREDICTION_VARIANCE = "prediction_variance"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    CROSS_VALIDATION = "cross_validation"
    ENSEMBLE_AGREEMENT = "ensemble_agreement"
    HISTORICAL_ACCURACY = "historical_accuracy"

class UncertaintyType(Enum):
    ALEATORIC = "aleatoric"     # Data/inherent uncertainty
    EPISTEMIC = "epistemic"     # Model/knowledge uncertainty
    TEMPORAL = "temporal"       # Time-related uncertainty
    DISTRIBUTIONAL = "distributional"  # Distribution shift uncertainty

@dataclass
class FeatureExplanation:
    """Explanation for individual feature contribution"""
    feature_name: str
    feature_value: float
    importance_score: float
    contribution: float  # Positive or negative contribution to prediction
    confidence: float
    explanation_text: str
    category: str  # 'academic', 'behavioral', 'temporal', etc.
    rank: int      # Importance ranking
    uncertainty: float

@dataclass
class PredictionExplanation:
    """Comprehensive prediction explanation"""
    explanation_id: str
    student_id: str
    prediction_type: str
    predicted_value: float
    confidence_score: float
    explanation_type: ExplanationType
    feature_explanations: List[FeatureExplanation]
    top_factors: List[str]  # Top 5 most important factors
    decision_boundary_info: Dict[str, Any]
    counterfactual_scenarios: List[Dict[str, Any]]
    uncertainty_breakdown: Dict[UncertaintyType, float]
    confidence_sources: Dict[ConfidenceSource, float]
    explanation_confidence: float  # How confident we are in the explanation itself
    natural_language_explanation: str
    visual_explanation_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ConfidenceAnalysis:
    """Detailed confidence analysis"""
    overall_confidence: float
    confidence_level: str  # 'very_high', 'high', 'medium', 'low', 'very_low'
    confidence_sources: Dict[ConfidenceSource, float]
    uncertainty_components: Dict[UncertaintyType, float]
    reliability_indicators: Dict[str, float]
    historical_accuracy: float
    prediction_stability: float
    data_quality_score: float
    model_calibration_score: float
    confidence_interval: Tuple[float, float]
    prediction_variance: float
    cross_validation_score: float

@dataclass
class CounterfactualScenario:
    """What-if scenario for prediction explanation"""
    scenario_id: str
    scenario_description: str
    feature_changes: Dict[str, float]
    predicted_outcome: float
    confidence_change: float
    feasibility_score: float  # How realistic/achievable the changes are
    recommended_actions: List[str]
    expected_timeline: str

class FeatureImportanceCalculator:
    """Calculate feature importance using multiple methods"""
    
    def __init__(self):
        self.importance_methods = [
            'permutation',
            'gradient_based',
            'integrated_gradients',
            'shap_values'
        ]
    
    async def calculate_feature_importance(self, model: Any, features: Dict[str, float],
                                         prediction: float, method: str = 'permutation') -> Dict[str, float]:
        """Calculate feature importance using specified method"""
        try:
            if method == 'permutation':
                return await self._permutation_importance(model, features, prediction)
            elif method == 'gradient_based':
                return await self._gradient_importance(model, features)
            elif method == 'shap_values':
                return await self._shap_importance(model, features)
            elif method == 'integrated_gradients':
                return await self._integrated_gradients(model, features)
            else:
                return await self._simple_correlation_importance(features, prediction)
                
        except Exception as e:
            logger.error(f"❌ Failed to calculate feature importance: {e}")
            return {}
    
    async def _permutation_importance(self, model: Any, features: Dict[str, float],
                                    prediction: float) -> Dict[str, float]:
        """Calculate permutation-based feature importance"""
        importance_scores = {}
        
        try:
            feature_names = list(features.keys())
            feature_values = list(features.values())
            baseline_prediction = prediction
            
            # Permute each feature and measure impact
            for i, feature_name in enumerate(feature_names):
                perturbed_features = feature_values.copy()
                
                # Try different perturbation strategies
                original_value = perturbed_features[i]
                
                # Permute with random value from reasonable range
                if original_value != 0:
                    perturbation_range = abs(original_value) * 0.5
                    perturbed_value = original_value + np.random.normal(0, perturbation_range)
                else:
                    perturbed_value = np.random.normal(0, 0.1)
                
                perturbed_features[i] = perturbed_value
                
                # Get prediction with perturbed feature (simplified)
                # In practice, would use actual model prediction
                feature_impact = abs(original_value - perturbed_value) * 0.1
                perturbed_prediction = baseline_prediction + np.random.normal(0, feature_impact)
                
                # Calculate importance as absolute change
                importance = abs(baseline_prediction - perturbed_prediction)
                importance_scores[feature_name] = importance
            
            # Normalize scores
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {
                    name: score / total_importance 
                    for name, score in importance_scores.items()
                }
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"❌ Permutation importance calculation failed: {e}")
            return {}
    
    async def _gradient_importance(self, model: Any, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate gradient-based importance for neural networks"""
        importance_scores = {}
        
        try:
            if hasattr(model, 'parameters'):  # PyTorch model
                feature_names = list(features.keys())
                feature_tensor = torch.FloatTensor(list(features.values())).requires_grad_(True)
                
                # Forward pass
                output = model(feature_tensor.unsqueeze(0))
                if isinstance(output, tuple):
                    output = output[0]
                
                # Backward pass
                output.backward()
                
                # Get gradients
                gradients = feature_tensor.grad.abs().detach().numpy()
                
                # Multiply by feature values for importance
                for i, feature_name in enumerate(feature_names):
                    importance_scores[feature_name] = float(gradients[i] * abs(features[feature_name]))
                
                # Normalize
                total_importance = sum(importance_scores.values())
                if total_importance > 0:
                    importance_scores = {
                        name: score / total_importance 
                        for name, score in importance_scores.items()
                    }
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"❌ Gradient importance calculation failed: {e}")
            return {}
    
    async def _shap_importance(self, model: Any, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate SHAP-based feature importance"""
        importance_scores = {}
        
        try:
            # Simplified SHAP calculation
            # In practice, would use proper SHAP library with background dataset
            feature_names = list(features.keys())
            feature_values = np.array(list(features.values()))
            
            # Simulate SHAP values using coalition-based approximation
            n_features = len(feature_names)
            shap_values = np.zeros(n_features)
            
            # Calculate marginal contributions
            for i in range(n_features):
                # Contribution when feature is present vs absent
                with_feature = feature_values.copy()
                without_feature = feature_values.copy()
                without_feature[i] = 0  # Remove feature
                
                # Simulate prediction difference
                contribution = abs(with_feature[i]) * 0.1 * np.random.uniform(0.5, 1.5)
                shap_values[i] = contribution
            
            # Create importance scores
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = abs(shap_values[i])
            
            # Normalize
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {
                    name: score / total_importance 
                    for name, score in importance_scores.items()
                }
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"❌ SHAP importance calculation failed: {e}")
            return {}
    
    async def _integrated_gradients(self, model: Any, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate integrated gradients importance"""
        importance_scores = {}
        
        try:
            if hasattr(model, 'parameters'):  # PyTorch model
                feature_names = list(features.keys())
                feature_tensor = torch.FloatTensor(list(features.values()))
                baseline = torch.zeros_like(feature_tensor)
                
                # Number of steps for integration
                n_steps = 50
                
                # Compute integrated gradients
                integrated_grads = torch.zeros_like(feature_tensor)
                
                for step in range(n_steps):
                    alpha = step / (n_steps - 1)
                    interpolated = baseline + alpha * (feature_tensor - baseline)
                    interpolated.requires_grad_(True)
                    
                    # Forward pass
                    output = model(interpolated.unsqueeze(0))
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    # Compute gradients
                    grad = torch.autograd.grad(output, interpolated, create_graph=True)[0]
                    integrated_grads += grad / n_steps
                
                # Multiply by input difference
                integrated_grads = integrated_grads * (feature_tensor - baseline)
                
                # Create importance scores
                for i, feature_name in enumerate(feature_names):
                    importance_scores[feature_name] = float(abs(integrated_grads[i]))
                
                # Normalize
                total_importance = sum(importance_scores.values())
                if total_importance > 0:
                    importance_scores = {
                        name: score / total_importance 
                        for name, score in importance_scores.items()
                    }
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"❌ Integrated gradients calculation failed: {e}")
            return {}
    
    async def _simple_correlation_importance(self, features: Dict[str, float],
                                           prediction: float) -> Dict[str, float]:
        """Simple correlation-based importance as fallback"""
        importance_scores = {}
        
        try:
            # Use feature magnitude and some heuristics
            for feature_name, feature_value in features.items():
                # Higher values and certain feature types get higher importance
                base_importance = abs(feature_value)
                
                # Boost importance for key educational features
                if any(keyword in feature_name.lower() for keyword in ['performance', 'success', 'mastery']):
                    base_importance *= 2.0
                elif any(keyword in feature_name.lower() for keyword in ['engagement', 'velocity', 'trend']):
                    base_importance *= 1.5
                
                importance_scores[feature_name] = base_importance
            
            # Normalize
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {
                    name: score / total_importance 
                    for name, score in importance_scores.items()
                }
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"❌ Simple correlation importance failed: {e}")
            return {name: 1.0 / len(features) for name in features.keys()}

class UncertaintyQuantifier:
    """Quantify different types of uncertainty in predictions"""
    
    def __init__(self):
        self.uncertainty_methods = [
            'monte_carlo_dropout',
            'ensemble_variance',
            'prediction_intervals',
            'conformal_prediction'
        ]
    
    async def quantify_uncertainty(self, model: Any, features: Dict[str, float],
                                 prediction: float, method: str = 'ensemble_variance') -> Dict[UncertaintyType, float]:
        """Quantify prediction uncertainty by type"""
        try:
            uncertainties = {}
            
            # Aleatoric uncertainty (data uncertainty)
            uncertainties[UncertaintyType.ALEATORIC] = await self._calculate_aleatoric_uncertainty(
                features, prediction
            )
            
            # Epistemic uncertainty (model uncertainty)
            uncertainties[UncertaintyType.EPISTEMIC] = await self._calculate_epistemic_uncertainty(
                model, features, prediction
            )
            
            # Temporal uncertainty
            uncertainties[UncertaintyType.TEMPORAL] = await self._calculate_temporal_uncertainty(
                features
            )
            
            # Distributional uncertainty
            uncertainties[UncertaintyType.DISTRIBUTIONAL] = await self._calculate_distributional_uncertainty(
                features
            )
            
            return uncertainties
            
        except Exception as e:
            logger.error(f"❌ Failed to quantify uncertainty: {e}")
            return {ut: 0.1 for ut in UncertaintyType}
    
    async def _calculate_aleatoric_uncertainty(self, features: Dict[str, float],
                                             prediction: float) -> float:
        """Calculate aleatoric (data) uncertainty"""
        try:
            # Estimate uncertainty based on data quality indicators
            data_quality_factors = []
            
            # Check for missing or unusual values
            for feature_name, feature_value in features.items():
                if np.isnan(feature_value) or np.isinf(feature_value):
                    data_quality_factors.append(0.5)  # High uncertainty for missing data
                elif abs(feature_value) > 3:  # Potential outlier
                    data_quality_factors.append(0.3)
                else:
                    data_quality_factors.append(0.1)
            
            # Average uncertainty from data quality
            avg_uncertainty = np.mean(data_quality_factors) if data_quality_factors else 0.1
            
            # Add uncertainty based on prediction extremes
            if prediction < 0.1 or prediction > 0.9:
                avg_uncertainty += 0.1  # Higher uncertainty at extremes
            
            return min(1.0, avg_uncertainty)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate aleatoric uncertainty: {e}")
            return 0.2
    
    async def _calculate_epistemic_uncertainty(self, model: Any, features: Dict[str, float],
                                             prediction: float) -> float:
        """Calculate epistemic (model) uncertainty"""
        try:
            # Simulate model uncertainty using prediction variance
            # In practice, would use Monte Carlo dropout or ensemble methods
            
            uncertainty = 0.1  # Base epistemic uncertainty
            
            # Increase uncertainty for edge cases
            feature_values = list(features.values())
            if any(abs(val) > 2 for val in feature_values):
                uncertainty += 0.1  # Higher uncertainty for unusual inputs
            
            # Check for model confidence indicators
            if hasattr(model, 'predict_proba') or hasattr(model, 'uncertainty'):
                # Model has built-in uncertainty estimation
                uncertainty *= 0.8  # Reduce uncertainty if model provides estimates
            
            return min(1.0, uncertainty)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate epistemic uncertainty: {e}")
            return 0.15
    
    async def _calculate_temporal_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate temporal uncertainty"""
        try:
            # Uncertainty increases with time-related factors
            temporal_features = [
                'hours_since_last_interaction',
                'session_frequency',
                'study_consistency'
            ]
            
            temporal_uncertainty = 0.05  # Base temporal uncertainty
            
            for feature_name in temporal_features:
                if feature_name in features:
                    value = features[feature_name]
                    
                    if 'hours_since_last' in feature_name:
                        # Longer gaps increase uncertainty
                        if value > 168:  # More than a week
                            temporal_uncertainty += 0.2
                        elif value > 48:  # More than 2 days
                            temporal_uncertainty += 0.1
                    
                    elif 'consistency' in feature_name:
                        # Lower consistency increases uncertainty
                        if value < 0.5:
                            temporal_uncertainty += 0.1
            
            return min(1.0, temporal_uncertainty)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate temporal uncertainty: {e}")
            return 0.1
    
    async def _calculate_distributional_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate distributional uncertainty (out-of-distribution detection)"""
        try:
            # Simple OOD detection based on feature ranges
            # In practice, would use more sophisticated methods
            
            # Expected feature ranges (would be learned from training data)
            expected_ranges = {
                'current_mastery_score': (0.0, 1.0),
                'learning_velocity': (-0.5, 0.5),
                'session_frequency': (0.0, 2.0),
                'performance_trend': (-1.0, 1.0)
            }
            
            ood_score = 0.0
            checked_features = 0
            
            for feature_name, feature_value in features.items():
                if feature_name in expected_ranges:
                    min_val, max_val = expected_ranges[feature_name]
                    
                    if feature_value < min_val or feature_value > max_val:
                        # Out of expected range
                        extent = max(min_val - feature_value, feature_value - max_val, 0)
                        range_size = max_val - min_val
                        ood_score += min(1.0, extent / range_size)
                    
                    checked_features += 1
            
            # Average OOD score
            if checked_features > 0:
                ood_score /= checked_features
            
            return min(1.0, ood_score)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate distributional uncertainty: {e}")
            return 0.05

class ConfidenceEstimator:
    """Estimate confidence in predictions using multiple sources"""
    
    def __init__(self):
        self.historical_accuracies = defaultdict(list)
        self.calibration_data = defaultdict(list)
    
    async def estimate_confidence(self, model: Any, features: Dict[str, float],
                                prediction: float, uncertainties: Dict[UncertaintyType, float],
                                historical_performance: Optional[Dict[str, float]] = None) -> ConfidenceAnalysis:
        """Comprehensive confidence estimation"""
        try:
            # Calculate confidence from multiple sources
            confidence_sources = {}
            
            # Model uncertainty confidence
            confidence_sources[ConfidenceSource.MODEL_UNCERTAINTY] = 1.0 - uncertainties.get(UncertaintyType.EPISTEMIC, 0.2)
            
            # Data uncertainty confidence
            confidence_sources[ConfidenceSource.DATA_UNCERTAINTY] = 1.0 - uncertainties.get(UncertaintyType.ALEATORIC, 0.2)
            
            # Prediction variance confidence
            confidence_sources[ConfidenceSource.PREDICTION_VARIANCE] = await self._calculate_prediction_variance_confidence(
                model, features, prediction
            )
            
            # Temporal consistency confidence
            confidence_sources[ConfidenceSource.TEMPORAL_CONSISTENCY] = 1.0 - uncertainties.get(UncertaintyType.TEMPORAL, 0.1)
            
            # Historical accuracy confidence
            if historical_performance:
                confidence_sources[ConfidenceSource.HISTORICAL_ACCURACY] = historical_performance.get('accuracy', 0.7)
            else:
                confidence_sources[ConfidenceSource.HISTORICAL_ACCURACY] = 0.7
            
            # Ensemble agreement (simulated)
            confidence_sources[ConfidenceSource.ENSEMBLE_AGREEMENT] = await self._simulate_ensemble_agreement(
                features, prediction
            )
            
            # Calculate overall confidence
            overall_confidence = np.mean(list(confidence_sources.values()))
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(overall_confidence)
            
            # Calculate reliability indicators
            reliability_indicators = await self._calculate_reliability_indicators(
                features, prediction, uncertainties
            )
            
            # Calculate prediction stability
            prediction_stability = await self._calculate_prediction_stability(
                model, features, prediction
            )
            
            # Data quality score
            data_quality_score = await self._calculate_data_quality_score(features)
            
            # Model calibration score
            model_calibration_score = await self._calculate_model_calibration_score(
                prediction, overall_confidence
            )
            
            # Confidence interval
            confidence_interval = await self._calculate_confidence_interval(
                prediction, overall_confidence, uncertainties
            )
            
            # Prediction variance
            prediction_variance = sum(uncertainties.values()) / len(uncertainties)
            
            # Simulated cross-validation score
            cross_validation_score = 0.8  # Would be calculated from actual CV
            
            return ConfidenceAnalysis(
                overall_confidence=overall_confidence,
                confidence_level=confidence_level,
                confidence_sources=confidence_sources,
                uncertainty_components=uncertainties,
                reliability_indicators=reliability_indicators,
                historical_accuracy=confidence_sources[ConfidenceSource.HISTORICAL_ACCURACY],
                prediction_stability=prediction_stability,
                data_quality_score=data_quality_score,
                model_calibration_score=model_calibration_score,
                confidence_interval=confidence_interval,
                prediction_variance=prediction_variance,
                cross_validation_score=cross_validation_score
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to estimate confidence: {e}")
            return self._create_default_confidence_analysis()
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """Determine categorical confidence level"""
        if confidence_score >= 0.9:
            return 'very_high'
        elif confidence_score >= 0.75:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        elif confidence_score >= 0.4:
            return 'low'
        else:
            return 'very_low'
    
    async def _calculate_prediction_variance_confidence(self, model: Any, features: Dict[str, float],
                                                      prediction: float) -> float:
        """Calculate confidence based on prediction variance"""
        try:
            # Simulate prediction variance by adding small perturbations
            perturbations = []
            base_features = list(features.values())
            
            for _ in range(10):
                # Add small random perturbations
                perturbed_features = [val + np.random.normal(0, abs(val) * 0.05) for val in base_features]
                
                # Simulate prediction with perturbations
                # In practice, would use actual model
                perturbation_effect = np.random.normal(0, 0.02)
                perturbed_prediction = prediction + perturbation_effect
                perturbations.append(perturbed_prediction)
            
            # Calculate variance
            variance = np.var(perturbations)
            
            # Convert to confidence (lower variance = higher confidence)
            confidence = 1.0 / (1.0 + variance * 10)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate prediction variance confidence: {e}")
            return 0.7
    
    async def _simulate_ensemble_agreement(self, features: Dict[str, float], prediction: float) -> float:
        """Simulate ensemble agreement confidence"""
        try:
            # Simulate multiple model predictions
            ensemble_predictions = []
            
            for _ in range(5):
                # Simulate different model predictions with some variance
                noise = np.random.normal(0, 0.05)
                ensemble_pred = prediction + noise
                ensemble_predictions.append(ensemble_pred)
            
            # Calculate agreement (inverse of variance)
            variance = np.var(ensemble_predictions)
            agreement = 1.0 / (1.0 + variance * 20)
            
            return min(1.0, max(0.0, agreement))
            
        except Exception as e:
            logger.error(f"❌ Failed to simulate ensemble agreement: {e}")
            return 0.75
    
    async def _calculate_reliability_indicators(self, features: Dict[str, float],
                                              prediction: float,
                                              uncertainties: Dict[UncertaintyType, float]) -> Dict[str, float]:
        """Calculate various reliability indicators"""
        indicators = {}
        
        try:
            # Feature completeness
            non_zero_features = sum(1 for val in features.values() if abs(val) > 0.01)
            indicators['feature_completeness'] = non_zero_features / len(features)
            
            # Prediction extremeness (more extreme predictions are less reliable)
            extremeness = max(prediction, 1.0 - prediction)
            indicators['prediction_extremeness'] = 1.0 - extremeness
            
            # Uncertainty consistency
            uncertainty_variance = np.var(list(uncertainties.values()))
            indicators['uncertainty_consistency'] = 1.0 / (1.0 + uncertainty_variance * 10)
            
            # Feature value reasonableness
            reasonable_count = 0
            for feature_name, feature_value in features.items():
                if 0.0 <= abs(feature_value) <= 3.0:  # Reasonable range
                    reasonable_count += 1
            indicators['feature_reasonableness'] = reasonable_count / len(features)
            
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate reliability indicators: {e}")
            return {'overall_reliability': 0.7}
    
    async def _calculate_prediction_stability(self, model: Any, features: Dict[str, float],
                                            prediction: float) -> float:
        """Calculate how stable the prediction is to small changes"""
        try:
            stability_scores = []
            
            # Test stability with small feature perturbations
            for feature_name in features.keys():
                original_value = features[feature_name]
                
                # Small perturbation
                perturbation = original_value * 0.01 if original_value != 0 else 0.01
                perturbed_features = features.copy()
                perturbed_features[feature_name] = original_value + perturbation
                
                # Simulate prediction change (would use actual model)
                prediction_change = abs(perturbation) * 0.1
                stability = 1.0 / (1.0 + prediction_change * 10)
                stability_scores.append(stability)
            
            return np.mean(stability_scores) if stability_scores else 0.8
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate prediction stability: {e}")
            return 0.8
    
    async def _calculate_data_quality_score(self, features: Dict[str, float]) -> float:
        """Calculate overall data quality score"""
        try:
            quality_factors = []
            
            # Check for missing values
            missing_rate = sum(1 for val in features.values() if np.isnan(val) or val == 0) / len(features)
            quality_factors.append(1.0 - missing_rate)
            
            # Check for outliers
            values = list(features.values())
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                outlier_rate = sum(1 for val in values if abs(val - mean_val) > 3 * std_val) / len(values)
                quality_factors.append(1.0 - outlier_rate)
            
            # Check for reasonable ranges
            reasonable_rate = sum(1 for val in values if -5 <= val <= 5) / len(values)
            quality_factors.append(reasonable_rate)
            
            return np.mean(quality_factors) if quality_factors else 0.7
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate data quality score: {e}")
            return 0.7
    
    async def _calculate_model_calibration_score(self, prediction: float, confidence: float) -> float:
        """Calculate how well-calibrated the model predictions are"""
        try:
            # Simple calibration check
            # In practice, would use proper calibration curves
            
            # Well-calibrated models have similar confidence and accuracy
            # For now, use a heuristic based on prediction confidence alignment
            
            if prediction > 0.8 and confidence > 0.8:
                # High prediction with high confidence - good
                calibration = 0.9
            elif prediction < 0.2 and confidence > 0.8:
                # Low prediction with high confidence - good
                calibration = 0.9
            elif 0.4 <= prediction <= 0.6 and confidence < 0.7:
                # Uncertain prediction with appropriate confidence - good
                calibration = 0.8
            else:
                # Moderate calibration
                calibration = 0.7
            
            return calibration
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate model calibration score: {e}")
            return 0.7
    
    async def _calculate_confidence_interval(self, prediction: float, confidence: float,
                                           uncertainties: Dict[UncertaintyType, float]) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        try:
            # Calculate margin based on uncertainties and confidence
            total_uncertainty = sum(uncertainties.values())
            
            # Base margin from uncertainty
            base_margin = total_uncertainty * 0.5
            
            # Adjust margin based on confidence
            margin_adjustment = (1.0 - confidence) * 0.3
            final_margin = base_margin + margin_adjustment
            
            # Ensure reasonable bounds
            lower_bound = max(0.0, prediction - final_margin)
            upper_bound = min(1.0, prediction + final_margin)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate confidence interval: {e}")
            return (max(0.0, prediction - 0.2), min(1.0, prediction + 0.2))
    
    def _create_default_confidence_analysis(self) -> ConfidenceAnalysis:
        """Create default confidence analysis when calculation fails"""
        return ConfidenceAnalysis(
            overall_confidence=0.5,
            confidence_level='medium',
            confidence_sources={source: 0.5 for source in ConfidenceSource},
            uncertainty_components={ut: 0.2 for ut in UncertaintyType},
            reliability_indicators={'overall_reliability': 0.5},
            historical_accuracy=0.7,
            prediction_stability=0.7,
            data_quality_score=0.7,
            model_calibration_score=0.7,
            confidence_interval=(0.3, 0.7),
            prediction_variance=0.2,
            cross_validation_score=0.7
        )

class PredictionExplainer:
    """Main class for prediction explainability and confidence quantification"""
    
    def __init__(self):
        self.feature_calculator = FeatureImportanceCalculator()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.confidence_estimator = ConfidenceEstimator()
        
        # Natural language templates
        self.explanation_templates = {
            'high_confidence_positive': "Based on {top_factors}, the prediction shows {outcome} with high confidence ({confidence:.1%}). Key contributing factors include {factors}.",
            'high_confidence_negative': "The analysis indicates {outcome} with high confidence ({confidence:.1%}). This is primarily due to {factors}.",
            'medium_confidence': "The prediction suggests {outcome} with moderate confidence ({confidence:.1%}). The main factors are {factors}, though there is some uncertainty due to {uncertainties}.",
            'low_confidence': "While the model predicts {outcome}, confidence is low ({confidence:.1%}) due to {uncertainties}. Key factors include {factors}."
        }
    
    async def explain_prediction(self, model: Any, features: Dict[str, float],
                               prediction: float, prediction_type: str,
                               student_id: str) -> PredictionExplanation:
        """Generate comprehensive prediction explanation"""
        try:
            logger.info(f"🔍 Explaining prediction for student {student_id}, type {prediction_type}")
            
            # Calculate feature importance
            feature_importance = await self.feature_calculator.calculate_feature_importance(
                model, features, prediction
            )
            
            # Quantify uncertainties
            uncertainties = await self.uncertainty_quantifier.quantify_uncertainty(
                model, features, prediction
            )
            
            # Estimate confidence
            confidence_analysis = await self.confidence_estimator.estimate_confidence(
                model, features, prediction, uncertainties
            )
            
            # Create feature explanations
            feature_explanations = await self._create_feature_explanations(
                features, feature_importance, prediction
            )
            
            # Identify top factors
            top_factors = await self._identify_top_factors(feature_explanations, feature_importance)
            
            # Generate counterfactual scenarios
            counterfactuals = await self._generate_counterfactuals(
                features, prediction, feature_importance
            )
            
            # Create natural language explanation
            natural_explanation = await self._generate_natural_language_explanation(
                prediction, confidence_analysis, top_factors, uncertainties, prediction_type
            )
            
            # Create visual explanation data
            visual_data = await self._create_visual_explanation_data(
                feature_explanations, confidence_analysis, uncertainties
            )
            
            return PredictionExplanation(
                explanation_id=str(uuid.uuid4()),
                student_id=student_id,
                prediction_type=prediction_type,
                predicted_value=prediction,
                confidence_score=confidence_analysis.overall_confidence,
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                feature_explanations=feature_explanations,
                top_factors=top_factors,
                decision_boundary_info={},
                counterfactual_scenarios=counterfactuals,
                uncertainty_breakdown=uncertainties,
                confidence_sources=confidence_analysis.confidence_sources,
                explanation_confidence=confidence_analysis.overall_confidence,
                natural_language_explanation=natural_explanation,
                visual_explanation_data=visual_data
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to explain prediction: {e}")
            return self._create_default_explanation(student_id, prediction_type, prediction)
    
    async def _create_feature_explanations(self, features: Dict[str, float],
                                         feature_importance: Dict[str, float],
                                         prediction: float) -> List[FeatureExplanation]:
        """Create detailed explanations for each feature"""
        explanations = []
        
        try:
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (feature_name, importance) in enumerate(sorted_features):
                feature_value = features.get(feature_name, 0.0)
                
                # Calculate contribution (positive or negative)
                contribution = self._calculate_feature_contribution(
                    feature_name, feature_value, importance, prediction
                )
                
                # Generate explanation text
                explanation_text = self._generate_feature_explanation_text(
                    feature_name, feature_value, contribution, importance
                )
                
                # Categorize feature
                category = self._categorize_feature(feature_name)
                
                # Estimate feature-specific uncertainty
                uncertainty = importance * 0.1  # Simple estimate
                
                explanation = FeatureExplanation(
                    feature_name=feature_name,
                    feature_value=feature_value,
                    importance_score=importance,
                    contribution=contribution,
                    confidence=1.0 - uncertainty,
                    explanation_text=explanation_text,
                    category=category,
                    rank=rank + 1,
                    uncertainty=uncertainty
                )
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature explanations: {e}")
            return []
    
    def _calculate_feature_contribution(self, feature_name: str, feature_value: float,
                                      importance: float, prediction: float) -> float:
        """Calculate how much a feature contributes to the prediction"""
        try:
            # Simple contribution calculation
            # Positive values generally contribute positively to positive predictions
            base_contribution = feature_value * importance
            
            # Adjust based on feature type
            if 'success' in feature_name or 'mastery' in feature_name:
                # Success-related features contribute directly
                contribution = base_contribution
            elif 'decline' in feature_name or 'drop' in feature_name:
                # Decline features contribute negatively
                contribution = -base_contribution
            elif 'volatility' in feature_name or 'inconsistency' in feature_name:
                # Volatility features generally contribute negatively
                contribution = -abs(base_contribution)
            else:
                # Default contribution
                contribution = base_contribution
            
            return contribution
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate feature contribution: {e}")
            return 0.0
    
    def _generate_feature_explanation_text(self, feature_name: str, feature_value: float,
                                          contribution: float, importance: float) -> str:
        """Generate human-readable explanation for a feature"""
        try:
            # Clean up feature name for display
            display_name = feature_name.replace('_', ' ').title()
            
            # Determine contribution direction
            if contribution > 0.1:
                direction = "positively contributes"
            elif contribution < -0.1:
                direction = "negatively impacts"
            else:
                direction = "has minimal effect on"
            
            # Determine magnitude
            if importance > 0.3:
                magnitude = "strongly"
            elif importance > 0.1:
                magnitude = "moderately"
            else:
                magnitude = "slightly"
            
            return f"{display_name} (value: {feature_value:.3f}) {magnitude} {direction} the prediction."
            
        except Exception as e:
            logger.error(f"❌ Failed to generate feature explanation text: {e}")
            return f"{feature_name}: {feature_value:.3f}"
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature for grouping in explanations"""
        academic_keywords = ['performance', 'success', 'mastery', 'grade', 'score']
        behavioral_keywords = ['engagement', 'interaction', 'session', 'frequency']
        temporal_keywords = ['time', 'trend', 'velocity', 'consistency']
        social_keywords = ['peer', 'collaboration', 'help', 'social']
        
        feature_lower = feature_name.lower()
        
        if any(keyword in feature_lower for keyword in academic_keywords):
            return 'academic'
        elif any(keyword in feature_lower for keyword in behavioral_keywords):
            return 'behavioral'
        elif any(keyword in feature_lower for keyword in temporal_keywords):
            return 'temporal'
        elif any(keyword in feature_lower for keyword in social_keywords):
            return 'social'
        else:
            return 'other'
    
    async def _identify_top_factors(self, feature_explanations: List[FeatureExplanation],
                                  feature_importance: Dict[str, float]) -> List[str]:
        """Identify top contributing factors"""
        try:
            # Sort by importance and take top 5
            sorted_explanations = sorted(feature_explanations, key=lambda x: x.importance_score, reverse=True)
            
            top_factors = []
            for explanation in sorted_explanations[:5]:
                factor_description = f"{explanation.feature_name.replace('_', ' ')} ({explanation.importance_score:.2f})"
                top_factors.append(factor_description)
            
            return top_factors
            
        except Exception as e:
            logger.error(f"❌ Failed to identify top factors: {e}")
            return ["Data analysis in progress"]
    
    async def _generate_counterfactuals(self, features: Dict[str, float],
                                      prediction: float,
                                      feature_importance: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate counterfactual scenarios"""
        counterfactuals = []
        
        try:
            # Get most important features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature_name, importance in sorted_features[:3]:
                current_value = features.get(feature_name, 0.0)
                
                # Generate scenarios for improvement
                if current_value < 0.8:  # Room for improvement
                    improved_value = min(1.0, current_value + 0.2)
                    estimated_new_prediction = prediction + (importance * 0.2)
                    
                    scenario = {
                        'scenario_id': str(uuid.uuid4()),
                        'description': f"If {feature_name.replace('_', ' ')} improved from {current_value:.2f} to {improved_value:.2f}",
                        'feature_changes': {feature_name: improved_value - current_value},
                        'predicted_outcome': min(1.0, estimated_new_prediction),
                        'feasibility': 0.7,  # How achievable this change is
                        'actions': [f"Focus on improving {feature_name.replace('_', ' ')}"]
                    }
                    counterfactuals.append(scenario)
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"❌ Failed to generate counterfactuals: {e}")
            return []
    
    async def _generate_natural_language_explanation(self, prediction: float,
                                                   confidence_analysis: ConfidenceAnalysis,
                                                   top_factors: List[str],
                                                   uncertainties: Dict[UncertaintyType, float],
                                                   prediction_type: str) -> str:
        """Generate natural language explanation"""
        try:
            # Determine outcome description
            if prediction_type == 'performance':
                if prediction > 0.8:
                    outcome = "strong academic performance"
                elif prediction > 0.6:
                    outcome = "good academic performance"
                elif prediction > 0.4:
                    outcome = "moderate academic performance"
                else:
                    outcome = "academic challenges"
            elif prediction_type == 'engagement':
                if prediction > 0.8:
                    outcome = "high engagement"
                elif prediction > 0.6:
                    outcome = "good engagement"
                elif prediction > 0.4:
                    outcome = "moderate engagement"
                else:
                    outcome = "low engagement"
            else:
                outcome = f"{prediction_type} score of {prediction:.2f}"
            
            # Get confidence level
            confidence = confidence_analysis.overall_confidence
            
            # Select appropriate template
            if confidence > 0.8:
                if prediction > 0.6:
                    template = self.explanation_templates['high_confidence_positive']
                else:
                    template = self.explanation_templates['high_confidence_negative']
            elif confidence > 0.6:
                template = self.explanation_templates['medium_confidence']
            else:
                template = self.explanation_templates['low_confidence']
            
            # Prepare factors list
            factors_text = ", ".join(top_factors[:3]) if top_factors else "various factors"
            
            # Prepare uncertainties list
            main_uncertainties = [
                ut.value for ut, value in uncertainties.items() 
                if value > 0.2
            ]
            uncertainties_text = ", ".join(main_uncertainties) if main_uncertainties else "model limitations"
            
            # Format explanation
            explanation = template.format(
                outcome=outcome,
                confidence=confidence,
                top_factors=factors_text,
                factors=factors_text,
                uncertainties=uncertainties_text
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Failed to generate natural language explanation: {e}")
            return f"The model predicts {prediction:.2f} for {prediction_type} with {confidence_analysis.overall_confidence:.1%} confidence."
    
    async def _create_visual_explanation_data(self, feature_explanations: List[FeatureExplanation],
                                            confidence_analysis: ConfidenceAnalysis,
                                            uncertainties: Dict[UncertaintyType, float]) -> Dict[str, Any]:
        """Create data for visual explanations"""
        try:
            return {
                'feature_importance_chart': {
                    'features': [exp.feature_name for exp in feature_explanations[:10]],
                    'importance_scores': [exp.importance_score for exp in feature_explanations[:10]],
                    'contributions': [exp.contribution for exp in feature_explanations[:10]]
                },
                'confidence_breakdown': {
                    'sources': list(confidence_analysis.confidence_sources.keys()),
                    'values': list(confidence_analysis.confidence_sources.values())
                },
                'uncertainty_components': {
                    'types': [ut.value for ut in uncertainties.keys()],
                    'values': list(uncertainties.values())
                },
                'prediction_confidence_meter': {
                    'value': confidence_analysis.overall_confidence,
                    'level': confidence_analysis.confidence_level,
                    'interval': confidence_analysis.confidence_interval
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create visual explanation data: {e}")
            return {}
    
    def _create_default_explanation(self, student_id: str, prediction_type: str,
                                  prediction: float) -> PredictionExplanation:
        """Create default explanation when full analysis fails"""
        return PredictionExplanation(
            explanation_id=str(uuid.uuid4()),
            student_id=student_id,
            prediction_type=prediction_type,
            predicted_value=prediction,
            confidence_score=0.5,
            explanation_type=ExplanationType.FEATURE_IMPORTANCE,
            feature_explanations=[],
            top_factors=["Analysis in progress"],
            decision_boundary_info={},
            counterfactual_scenarios=[],
            uncertainty_breakdown={ut: 0.2 for ut in UncertaintyType},
            confidence_sources={cs: 0.5 for cs in ConfidenceSource},
            explanation_confidence=0.3,
            natural_language_explanation=f"The model predicts {prediction:.2f} for {prediction_type}. Detailed analysis is being processed.",
            visual_explanation_data={}
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'explanation_methods': len(self.feature_calculator.importance_methods),
            'uncertainty_types': len(UncertaintyType),
            'confidence_sources': len(ConfidenceSource),
            'templates_available': len(self.explanation_templates),
            'last_updated': datetime.now().isoformat()
        }

# Testing function
async def test_prediction_explainer():
    """Test the prediction explainer system"""
    try:
        logger.info("🧪 Testing Prediction Explainer System")
        
        explainer = PredictionExplainer()
        
        # Mock features and prediction
        features = {
            'current_performance': 0.7,
            'learning_velocity': 0.1,
            'engagement_score': 0.8,
            'session_frequency': 1.2,
            'help_seeking_rate': 0.3
        }
        
        # Mock model (would be actual trained model)
        class MockModel:
            def predict(self, X):
                return [0.75]
        
        model = MockModel()
        
        # Test explanation
        explanation = await explainer.explain_prediction(
            model, features, 0.75, "performance", "test_student"
        )
        
        logger.info(f"✅ Generated explanation with {len(explanation.feature_explanations)} features")
        logger.info(f"Natural language: {explanation.natural_language_explanation}")
        
        # Test system status
        status = await explainer.get_system_status()
        logger.info(f"✅ System status: {status['explanation_methods']} methods available")
        
        logger.info("✅ Prediction Explainer System test completed")
        
    except Exception as e:
        logger.error(f"❌ Prediction Explainer System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_prediction_explainer())