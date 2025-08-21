#!/usr/bin/env python3
"""
Ensemble Prediction System for Physics Assistant Phase 6.3
Implements robust ensemble methods for educational predictions with adaptive
model selection, dynamic weighting, and meta-learning capabilities.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
    BaggingRegressor, BaggingClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import uuid
from collections import defaultdict, deque
import math
import statistics
from scipy import stats
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    SIMPLE_AVERAGING = "simple_averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    STACKING = "stacking"
    VOTING = "voting"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    DYNAMIC_SELECTION = "dynamic_selection"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"

class ModelType(Enum):
    TREE_BASED = "tree_based"
    LINEAR = "linear"
    NEURAL_NETWORK = "neural_network"
    SVM = "svm"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"

class PredictionTask(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    RANKING = "ranking"
    MULTI_OUTPUT = "multi_output"

@dataclass
class BaseModel:
    """Individual model in the ensemble"""
    model_id: str
    model_name: str
    model_type: ModelType
    model_instance: Any
    scaler: Optional[Any] = None
    feature_selector: Optional[Any] = None
    training_score: float = 0.0
    validation_score: float = 0.0
    cross_val_score: float = 0.0
    prediction_variance: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    stability_score: float = 0.0
    complexity_score: float = 0.0
    is_trained: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class EnsembleMetrics:
    """Performance metrics for ensemble"""
    ensemble_score: float
    individual_scores: Dict[str, float]
    diversity_score: float
    stability_score: float
    prediction_variance: float
    confidence_score: float
    bias_score: float
    variance_score: float
    ensemble_gain: float  # Improvement over best individual model
    training_time: float
    prediction_time: float
    memory_usage: float
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction_id: str
    student_id: str
    prediction_type: str
    ensemble_prediction: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    confidence_score: float
    prediction_variance: float
    uncertainty_estimates: Dict[str, float]
    model_agreements: Dict[str, float]
    ensemble_method: EnsembleMethod
    contributing_models: List[str]
    prediction_metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

class MetaLearner(nn.Module):
    """Neural network meta-learner for ensemble combination"""
    
    def __init__(self, num_models: int, feature_dim: int = 10, hidden_dim: int = 64):
        super(MetaLearner, self).__init__()
        
        self.num_models = num_models
        
        # Input: [model_predictions, model_confidences, base_features]
        input_dim = num_models * 2 + feature_dim
        
        self.meta_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=1)  # Output weights for each model
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
    
    def forward(self, model_predictions, model_confidences, base_features):
        # Concatenate all inputs
        meta_input = torch.cat([model_predictions, model_confidences, base_features], dim=1)
        
        # Get intermediate representation
        features = self.meta_network[:-1](meta_input)
        
        # Get model weights
        weights = self.meta_network[-1](features)
        
        # Get uncertainty estimate
        uncertainty = self.uncertainty_head(features)
        
        return weights, uncertainty

class DynamicModelSelector:
    """Dynamically select best models for each prediction"""
    
    def __init__(self):
        self.model_performance_history = defaultdict(list)
        self.context_model_mapping = defaultdict(dict)
        self.selection_strategy = 'performance_based'
    
    async def select_models(self, available_models: List[BaseModel],
                          context: Dict[str, Any],
                          num_select: int = 5) -> List[BaseModel]:
        """Select best models for current context"""
        try:
            if len(available_models) <= num_select:
                return available_models
            
            # Calculate selection scores for each model
            model_scores = {}
            
            for model in available_models:
                score = await self._calculate_selection_score(model, context)
                model_scores[model.model_id] = score
            
            # Sort by score and select top models
            sorted_models = sorted(
                available_models,
                key=lambda m: model_scores[m.model_id],
                reverse=True
            )
            
            selected = sorted_models[:num_select]
            
            logger.info(f"📊 Selected {len(selected)} models from {len(available_models)} available")
            return selected
            
        except Exception as e:
            logger.error(f"❌ Failed to select models: {e}")
            return available_models[:num_select]
    
    async def _calculate_selection_score(self, model: BaseModel, context: Dict[str, Any]) -> float:
        """Calculate selection score for a model given context"""
        try:
            score = 0.0
            
            # Base performance score
            score += model.validation_score * 0.4
            
            # Cross-validation score
            score += model.cross_val_score * 0.3
            
            # Stability score
            score += model.stability_score * 0.15
            
            # Efficiency score (inverse of complexity)
            efficiency = 1.0 / (1.0 + model.complexity_score)
            score += efficiency * 0.1
            
            # Context-specific adjustments
            student_level = context.get('student_level', 'intermediate')
            prediction_urgency = context.get('urgency', 'normal')
            
            # Prefer simpler models for beginners
            if student_level == 'beginner' and model.complexity_score < 0.5:
                score += 0.1
            
            # Prefer faster models for urgent predictions
            if prediction_urgency == 'high' and model.prediction_time < 0.1:
                score += 0.05
            
            # Historical performance in similar contexts
            historical_score = self._get_historical_performance(model.model_id, context)
            score += historical_score * 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate selection score: {e}")
            return 0.5
    
    def _get_historical_performance(self, model_id: str, context: Dict[str, Any]) -> float:
        """Get historical performance for model in similar contexts"""
        try:
            context_key = self._create_context_key(context)
            
            if context_key in self.context_model_mapping:
                return self.context_model_mapping[context_key].get(model_id, 0.5)
            
            return 0.5  # Default score
            
        except Exception as e:
            return 0.5
    
    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create hashable key from context"""
        try:
            # Use key context factors to create identifier
            key_factors = [
                context.get('student_level', 'unknown'),
                context.get('prediction_type', 'unknown'),
                context.get('time_horizon', 'unknown')
            ]
            return "_".join(key_factors)
        except:
            return "default"
    
    async def update_performance(self, model_id: str, context: Dict[str, Any], 
                               performance: float):
        """Update model performance for given context"""
        try:
            context_key = self._create_context_key(context)
            self.context_model_mapping[context_key][model_id] = performance
            
        except Exception as e:
            logger.error(f"❌ Failed to update performance: {e}")

class EnsemblePredictionSystem:
    """Comprehensive ensemble prediction system"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        
        # Core components
        self.base_models: Dict[str, BaseModel] = {}
        self.meta_learners: Dict[str, MetaLearner] = {}
        self.model_selector = DynamicModelSelector()
        
        # Ensemble configurations
        self.ensemble_configs = {
            EnsembleMethod.SIMPLE_AVERAGING: {'weights': None},
            EnsembleMethod.WEIGHTED_AVERAGING: {'weight_strategy': 'performance'},
            EnsembleMethod.STACKING: {'meta_model': 'ridge'},
            EnsembleMethod.VOTING: {'voting_strategy': 'soft'},
            EnsembleMethod.BAGGING: {'n_estimators': 10},
            EnsembleMethod.BOOSTING: {'learning_rate': 0.1},
            EnsembleMethod.DYNAMIC_SELECTION: {'selection_metric': 'accuracy'},
            EnsembleMethod.MIXTURE_OF_EXPERTS: {'gating_network': 'neural'}
        }
        
        # Performance tracking
        self.ensemble_metrics: Dict[str, EnsembleMetrics] = {}
        self.prediction_history: Dict[str, List[EnsemblePrediction]] = defaultdict(list)
        
        # Model libraries
        self.model_library = self._initialize_model_library()
        
        # Scalers and preprocessors
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Configuration
        self.config = {
            'max_models_per_ensemble': 10,
            'min_models_per_ensemble': 3,
            'retraining_frequency_hours': 24,
            'performance_history_length': 1000,
            'ensemble_diversity_weight': 0.2,
            'stability_weight': 0.3,
            'accuracy_weight': 0.5
        }
    
    def _initialize_model_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize library of available models"""
        return {
            # Tree-based models
            'random_forest_reg': {
                'class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.REGRESSION
            },
            'random_forest_clf': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.CLASSIFICATION
            },
            'gradient_boost_reg': {
                'class': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.REGRESSION
            },
            'gradient_boost_clf': {
                'class': GradientBoostingClassifier,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.CLASSIFICATION
            },
            'xgboost_reg': {
                'class': xgb.XGBRegressor,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.REGRESSION
            },
            'xgboost_clf': {
                'class': xgb.XGBClassifier,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.CLASSIFICATION
            },
            'lightgbm_reg': {
                'class': lgb.LGBMRegressor,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.REGRESSION
            },
            'lightgbm_clf': {
                'class': lgb.LGBMClassifier,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42, 'verbose': -1},
                'type': ModelType.TREE_BASED,
                'task': PredictionTask.CLASSIFICATION
            },
            
            # Linear models
            'ridge_reg': {
                'class': Ridge,
                'params': {'alpha': 1.0, 'random_state': 42},
                'type': ModelType.LINEAR,
                'task': PredictionTask.REGRESSION
            },
            'logistic_reg': {
                'class': LogisticRegression,
                'params': {'random_state': 42, 'max_iter': 1000},
                'type': ModelType.LINEAR,
                'task': PredictionTask.CLASSIFICATION
            },
            
            # Neural networks
            'mlp_reg': {
                'class': MLPRegressor,
                'params': {'hidden_layer_sizes': (100, 50), 'random_state': 42, 'max_iter': 500},
                'type': ModelType.NEURAL_NETWORK,
                'task': PredictionTask.REGRESSION
            },
            'mlp_clf': {
                'class': MLPClassifier,
                'params': {'hidden_layer_sizes': (100, 50), 'random_state': 42, 'max_iter': 500},
                'type': ModelType.NEURAL_NETWORK,
                'task': PredictionTask.CLASSIFICATION
            },
            
            # SVM models
            'svr': {
                'class': SVR,
                'params': {'kernel': 'rbf', 'C': 1.0},
                'type': ModelType.SVM,
                'task': PredictionTask.REGRESSION
            },
            'svc': {
                'class': SVC,
                'params': {'kernel': 'rbf', 'C': 1.0, 'probability': True, 'random_state': 42},
                'type': ModelType.SVM,
                'task': PredictionTask.CLASSIFICATION
            }
        }
    
    async def initialize(self):
        """Initialize the ensemble prediction system"""
        try:
            logger.info("🚀 Initializing Ensemble Prediction System")
            
            # Create initial ensemble of models
            await self._create_base_models()
            
            # Initialize meta-learners
            await self._initialize_meta_learners()
            
            # Load any pre-trained models
            await self._load_pretrained_models()
            
            logger.info(f"✅ Ensemble system initialized with {len(self.base_models)} base models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ensemble system: {e}")
            return False
    
    async def _create_base_models(self):
        """Create base models for the ensemble"""
        try:
            for model_name, model_config in self.model_library.items():
                model_id = str(uuid.uuid4())
                
                # Create model instance
                model_class = model_config['class']
                model_params = model_config['params']
                model_instance = model_class(**model_params)
                
                # Create BaseModel wrapper
                base_model = BaseModel(
                    model_id=model_id,
                    model_name=model_name,
                    model_type=model_config['type'],
                    model_instance=model_instance,
                    scaler=StandardScaler() if model_config['type'] in [ModelType.NEURAL_NETWORK, ModelType.SVM] else None
                )
                
                self.base_models[model_id] = base_model
                
                logger.debug(f"Created base model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create base models: {e}")
    
    async def _initialize_meta_learners(self):
        """Initialize meta-learning models"""
        try:
            for task in [PredictionTask.REGRESSION, PredictionTask.CLASSIFICATION]:
                # Count models for this task
                task_models = [
                    model for model in self.base_models.values()
                    if self._get_model_task(model.model_name) == task
                ]
                
                if len(task_models) > 0:
                    meta_learner = MetaLearner(
                        num_models=len(task_models),
                        feature_dim=10,  # Base feature dimension
                        hidden_dim=64
                    )
                    
                    self.meta_learners[task.value] = meta_learner
                    
            logger.info(f"Initialized {len(self.meta_learners)} meta-learners")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize meta-learners: {e}")
    
    def _get_model_task(self, model_name: str) -> PredictionTask:
        """Get the task type for a model"""
        if model_name in self.model_library:
            return self.model_library[model_name]['task']
        
        # Infer from name
        if 'clf' in model_name:
            return PredictionTask.CLASSIFICATION
        else:
            return PredictionTask.REGRESSION
    
    async def train_ensemble(self, training_data: pd.DataFrame, target_column: str,
                           task_type: PredictionTask = PredictionTask.REGRESSION):
        """Train the ensemble on provided data"""
        try:
            logger.info(f"🎯 Training ensemble on {len(training_data)} samples for {task_type.value}")
            
            # Prepare features and targets
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].values
            y = training_data[target_column].values
            
            # Filter models for this task
            task_models = {
                model_id: model for model_id, model in self.base_models.items()
                if self._get_model_task(model.model_name) == task_type
            }
            
            if not task_models:
                logger.warning(f"No models available for task {task_type.value}")
                return False
            
            # Train each base model
            trained_models = []
            for model_id, base_model in task_models.items():
                try:
                    success = await self._train_base_model(base_model, X, y)
                    if success:
                        trained_models.append(base_model)
                except Exception as e:
                    logger.warning(f"Failed to train model {base_model.model_name}: {e}")
            
            if not trained_models:
                logger.error("No models trained successfully")
                return False
            
            # Train meta-learner if we have enough models
            if len(trained_models) >= 3:
                await self._train_meta_learner(trained_models, X, y, task_type)
            
            # Calculate ensemble metrics
            await self._calculate_ensemble_metrics(trained_models, X, y, task_type)
            
            logger.info(f"✅ Ensemble training completed with {len(trained_models)} models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to train ensemble: {e}")
            return False
    
    async def _train_base_model(self, base_model: BaseModel, X: np.ndarray, y: np.ndarray) -> bool:
        """Train individual base model"""
        try:
            start_time = datetime.now()
            
            # Apply scaling if needed
            X_processed = X.copy()
            if base_model.scaler is not None:
                X_processed = base_model.scaler.fit_transform(X_processed)
            
            # Train model
            base_model.model_instance.fit(X_processed, y)
            base_model.is_trained = True
            
            # Calculate training metrics
            train_pred = base_model.model_instance.predict(X_processed)
            
            if len(np.unique(y)) > 10:  # Regression
                base_model.training_score = 1.0 - mean_squared_error(y, train_pred)
            else:  # Classification
                base_model.training_score = accuracy_score(y, (train_pred > 0.5).astype(int))
            
            # Cross-validation score
            cv_scores = cross_val_score(base_model.model_instance, X_processed, y, cv=3)
            base_model.cross_val_score = np.mean(cv_scores)
            
            # Calculate stability score
            base_model.stability_score = 1.0 - np.std(cv_scores)
            
            # Training time
            base_model.training_time = (datetime.now() - start_time).total_seconds()
            base_model.last_updated = datetime.now()
            
            logger.debug(f"Trained {base_model.model_name}: score={base_model.training_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to train base model {base_model.model_name}: {e}")
            return False
    
    async def _train_meta_learner(self, trained_models: List[BaseModel], 
                                X: np.ndarray, y: np.ndarray, task_type: PredictionTask):
        """Train meta-learner to combine base model predictions"""
        try:
            if task_type.value not in self.meta_learners:
                return
            
            meta_learner = self.meta_learners[task_type.value]
            
            # Generate base model predictions
            base_predictions = []
            base_confidences = []
            
            for model in trained_models:
                X_processed = X.copy()
                if model.scaler is not None:
                    X_processed = model.scaler.transform(X_processed)
                
                pred = model.model_instance.predict(X_processed)
                base_predictions.append(pred)
                
                # Estimate confidence (simplified)
                if hasattr(model.model_instance, 'predict_proba'):
                    proba = model.model_instance.predict_proba(X_processed)
                    confidence = np.max(proba, axis=1)
                else:
                    confidence = np.ones(len(pred)) * 0.8  # Default confidence
                
                base_confidences.append(confidence)
            
            # Prepare meta-learning data
            meta_X_models = torch.FloatTensor(np.column_stack(base_predictions))
            meta_X_confidences = torch.FloatTensor(np.column_stack(base_confidences))
            
            # Simple base features (would be more sophisticated in practice)
            base_features = torch.FloatTensor(np.column_stack([
                np.mean(X, axis=1),  # Mean feature value
                np.std(X, axis=1),   # Feature variance
                np.sum(X > 0, axis=1) / X.shape[1]  # Fraction of positive features
            ]))
            
            # Pad base features to match expected dimension
            if base_features.shape[1] < 10:
                padding = torch.zeros(base_features.shape[0], 10 - base_features.shape[1])
                base_features = torch.cat([base_features, padding], dim=1)
            
            meta_y = torch.FloatTensor(y)
            
            # Train meta-learner
            optimizer = optim.Adam(meta_learner.parameters(), lr=0.001)
            criterion = nn.MSELoss() if task_type == PredictionTask.REGRESSION else nn.CrossEntropyLoss()
            
            meta_learner.train()
            for epoch in range(100):
                optimizer.zero_grad()
                
                weights, uncertainty = meta_learner(meta_X_models, meta_X_confidences, base_features)
                
                # Weighted ensemble prediction
                ensemble_pred = torch.sum(meta_X_models * weights, dim=1)
                
                loss = criterion(ensemble_pred, meta_y)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.debug(f"Meta-learner epoch {epoch}, loss: {loss.item():.4f}")
            
            logger.info(f"Meta-learner trained for {task_type.value}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train meta-learner: {e}")
    
    async def predict_ensemble(self, features: Dict[str, float], 
                             prediction_type: str,
                             student_id: str,
                             ensemble_method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGING) -> EnsemblePrediction:
        """Make ensemble prediction"""
        try:
            logger.info(f"🎯 Making ensemble prediction for {student_id} using {ensemble_method.value}")
            
            # Convert features to array
            feature_names = sorted(features.keys())
            feature_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
            
            # Determine task type
            task_type = PredictionTask.REGRESSION  # Default, would be determined from prediction_type
            
            # Select models for this prediction
            available_models = [
                model for model in self.base_models.values()
                if model.is_trained and self._get_model_task(model.model_name) == task_type
            ]
            
            if not available_models:
                logger.warning("No trained models available for prediction")
                return self._create_fallback_prediction(student_id, prediction_type)
            
            # Get predictions from base models
            individual_predictions = {}
            model_confidences = {}
            
            for model in available_models:
                try:
                    pred, conf = await self._get_model_prediction(model, feature_array)
                    individual_predictions[model.model_id] = pred
                    model_confidences[model.model_id] = conf
                except Exception as e:
                    logger.warning(f"Failed to get prediction from {model.model_name}: {e}")
            
            if not individual_predictions:
                logger.warning("No individual predictions obtained")
                return self._create_fallback_prediction(student_id, prediction_type)
            
            # Combine predictions using specified method
            ensemble_result = await self._combine_predictions(
                individual_predictions, model_confidences, available_models, 
                ensemble_method, feature_array, task_type
            )
            
            # Calculate ensemble metrics
            prediction_variance = np.var(list(individual_predictions.values()))
            model_agreements = await self._calculate_model_agreements(individual_predictions)
            
            return EnsemblePrediction(
                prediction_id=str(uuid.uuid4()),
                student_id=student_id,
                prediction_type=prediction_type,
                ensemble_prediction=ensemble_result['prediction'],
                individual_predictions=individual_predictions,
                model_weights=ensemble_result['weights'],
                confidence_score=ensemble_result['confidence'],
                prediction_variance=prediction_variance,
                uncertainty_estimates=ensemble_result['uncertainties'],
                model_agreements=model_agreements,
                ensemble_method=ensemble_method,
                contributing_models=list(individual_predictions.keys()),
                prediction_metadata={
                    'num_models': len(individual_predictions),
                    'feature_count': len(features),
                    'method_used': ensemble_method.value
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to make ensemble prediction: {e}")
            return self._create_fallback_prediction(student_id, prediction_type)
    
    async def _get_model_prediction(self, model: BaseModel, 
                                  feature_array: np.ndarray) -> Tuple[float, float]:
        """Get prediction and confidence from individual model"""
        try:
            # Apply scaling if needed
            X_processed = feature_array.copy()
            if model.scaler is not None:
                X_processed = model.scaler.transform(X_processed)
            
            # Get prediction
            prediction = model.model_instance.predict(X_processed)[0]
            
            # Estimate confidence
            if hasattr(model.model_instance, 'predict_proba'):
                proba = model.model_instance.predict_proba(X_processed)[0]
                confidence = np.max(proba)
            elif hasattr(model.model_instance, 'decision_function'):
                decision = model.model_instance.decision_function(X_processed)[0]
                confidence = 1.0 / (1.0 + np.exp(-abs(decision)))  # Sigmoid transformation
            else:
                # Use model's cross-validation score as confidence proxy
                confidence = model.cross_val_score
            
            return float(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"❌ Failed to get prediction from model: {e}")
            return 0.5, 0.5
    
    async def _combine_predictions(self, individual_predictions: Dict[str, float],
                                 model_confidences: Dict[str, float],
                                 models: List[BaseModel],
                                 method: EnsembleMethod,
                                 features: np.ndarray,
                                 task_type: PredictionTask) -> Dict[str, Any]:
        """Combine individual model predictions using specified method"""
        try:
            predictions = np.array(list(individual_predictions.values()))
            confidences = np.array(list(model_confidences.values()))
            
            if method == EnsembleMethod.SIMPLE_AVERAGING:
                weights = np.ones(len(predictions)) / len(predictions)
                ensemble_pred = np.average(predictions, weights=weights)
                
            elif method == EnsembleMethod.WEIGHTED_AVERAGING:
                # Weight by model performance and confidence
                model_scores = np.array([model.cross_val_score for model in models])
                weights = model_scores * confidences
                weights = weights / np.sum(weights)
                ensemble_pred = np.average(predictions, weights=weights)
                
            elif method == EnsembleMethod.MIXTURE_OF_EXPERTS:
                # Use meta-learner if available
                if task_type.value in self.meta_learners:
                    ensemble_pred, weights = await self._meta_learner_prediction(
                        predictions, confidences, features, task_type
                    )
                else:
                    # Fallback to weighted averaging
                    weights = confidences / np.sum(confidences)
                    ensemble_pred = np.average(predictions, weights=weights)
                    
            else:
                # Default to weighted averaging
                weights = confidences / np.sum(confidences)
                ensemble_pred = np.average(predictions, weights=weights)
            
            # Calculate ensemble confidence
            ensemble_confidence = await self._calculate_ensemble_confidence(
                predictions, confidences, weights
            )
            
            # Calculate uncertainties
            uncertainties = {
                'prediction_variance': float(np.var(predictions)),
                'model_disagreement': float(np.std(predictions)),
                'confidence_variance': float(np.var(confidences))
            }
            
            return {
                'prediction': float(ensemble_pred),
                'weights': {list(individual_predictions.keys())[i]: float(weights[i]) for i in range(len(weights))},
                'confidence': float(ensemble_confidence),
                'uncertainties': uncertainties
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to combine predictions: {e}")
            return {
                'prediction': float(np.mean(list(individual_predictions.values()))),
                'weights': {k: 1.0/len(individual_predictions) for k in individual_predictions.keys()},
                'confidence': 0.5,
                'uncertainties': {'error': 'calculation_failed'}
            }
    
    async def _meta_learner_prediction(self, predictions: np.ndarray, confidences: np.ndarray,
                                     features: np.ndarray, task_type: PredictionTask) -> Tuple[float, np.ndarray]:
        """Use meta-learner to combine predictions"""
        try:
            meta_learner = self.meta_learners[task_type.value]
            meta_learner.eval()
            
            with torch.no_grad():
                # Prepare inputs
                pred_tensor = torch.FloatTensor(predictions).unsqueeze(0)
                conf_tensor = torch.FloatTensor(confidences).unsqueeze(0)
                
                # Simple base features
                base_features = torch.FloatTensor([
                    np.mean(features),
                    np.std(features),
                    np.sum(features > 0) / len(features[0])
                ])
                
                # Pad to expected dimension
                if len(base_features) < 10:
                    padding = torch.zeros(10 - len(base_features))
                    base_features = torch.cat([base_features, padding])
                
                base_features = base_features.unsqueeze(0)
                
                # Get weights and uncertainty
                weights, uncertainty = meta_learner(pred_tensor, conf_tensor, base_features)
                
                # Calculate ensemble prediction
                ensemble_pred = torch.sum(pred_tensor * weights)
                
                return float(ensemble_pred), weights.squeeze().numpy()
                
        except Exception as e:
            logger.error(f"❌ Meta-learner prediction failed: {e}")
            # Fallback to equal weights
            weights = np.ones(len(predictions)) / len(predictions)
            return float(np.average(predictions, weights=weights)), weights
    
    async def _calculate_ensemble_confidence(self, predictions: np.ndarray,
                                           confidences: np.ndarray,
                                           weights: np.ndarray) -> float:
        """Calculate confidence in ensemble prediction"""
        try:
            # Weighted average of individual confidences
            weighted_confidence = np.average(confidences, weights=weights)
            
            # Adjust for prediction agreement
            prediction_std = np.std(predictions)
            agreement_factor = 1.0 / (1.0 + prediction_std)  # Higher agreement = higher confidence
            
            # Combine factors
            ensemble_confidence = weighted_confidence * agreement_factor
            
            return float(np.clip(ensemble_confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate ensemble confidence: {e}")
            return 0.5
    
    async def _calculate_model_agreements(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate pairwise agreements between models"""
        agreements = {}
        
        try:
            pred_values = list(predictions.values())
            pred_keys = list(predictions.keys())
            
            for i, key1 in enumerate(pred_keys):
                for j, key2 in enumerate(pred_keys[i+1:], i+1):
                    agreement = 1.0 - abs(pred_values[i] - pred_values[j])
                    agreements[f"{key1}_{key2}"] = agreement
            
            return agreements
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate model agreements: {e}")
            return {}
    
    async def _calculate_ensemble_metrics(self, models: List[BaseModel], 
                                        X: np.ndarray, y: np.ndarray, 
                                        task_type: PredictionTask):
        """Calculate comprehensive ensemble metrics"""
        try:
            # Get individual model predictions
            individual_preds = []
            individual_scores = {}
            
            for model in models:
                X_processed = X.copy()
                if model.scaler is not None:
                    X_processed = model.scaler.transform(X_processed)
                
                pred = model.model_instance.predict(X_processed)
                individual_preds.append(pred)
                
                if task_type == PredictionTask.REGRESSION:
                    score = 1.0 - mean_squared_error(y, pred)
                else:
                    score = accuracy_score(y, (pred > 0.5).astype(int))
                
                individual_scores[model.model_id] = score
            
            # Calculate ensemble prediction (simple average)
            ensemble_pred = np.mean(individual_preds, axis=0)
            
            if task_type == PredictionTask.REGRESSION:
                ensemble_score = 1.0 - mean_squared_error(y, ensemble_pred)
            else:
                ensemble_score = accuracy_score(y, (ensemble_pred > 0.5).astype(int))
            
            # Calculate diversity
            diversity_score = self._calculate_diversity(individual_preds)
            
            # Calculate other metrics
            prediction_variance = np.mean([np.var(pred) for pred in individual_preds])
            best_individual_score = max(individual_scores.values())
            ensemble_gain = ensemble_score - best_individual_score
            
            metrics = EnsembleMetrics(
                ensemble_score=ensemble_score,
                individual_scores=individual_scores,
                diversity_score=diversity_score,
                stability_score=np.mean([model.stability_score for model in models]),
                prediction_variance=prediction_variance,
                confidence_score=ensemble_score,  # Simplified
                bias_score=0.0,  # Would be calculated with proper validation
                variance_score=prediction_variance,
                ensemble_gain=ensemble_gain,
                training_time=sum(model.training_time for model in models),
                prediction_time=0.1,  # Estimated
                memory_usage=len(models) * 10.0  # Estimated MB
            )
            
            self.ensemble_metrics[task_type.value] = metrics
            
            logger.info(f"Ensemble metrics - Score: {ensemble_score:.3f}, Gain: {ensemble_gain:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate ensemble metrics: {e}")
    
    def _calculate_diversity(self, predictions: List[np.ndarray]) -> float:
        """Calculate diversity among model predictions"""
        try:
            if len(predictions) < 2:
                return 0.0
            
            # Calculate pairwise disagreements
            disagreements = []
            
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    disagreement = np.mean(np.abs(predictions[i] - predictions[j]))
                    disagreements.append(disagreement)
            
            return np.mean(disagreements)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate diversity: {e}")
            return 0.0
    
    def _create_fallback_prediction(self, student_id: str, prediction_type: str) -> EnsemblePrediction:
        """Create fallback prediction when ensemble fails"""
        return EnsemblePrediction(
            prediction_id=str(uuid.uuid4()),
            student_id=student_id,
            prediction_type=prediction_type,
            ensemble_prediction=0.5,
            individual_predictions={'fallback': 0.5},
            model_weights={'fallback': 1.0},
            confidence_score=0.3,
            prediction_variance=0.0,
            uncertainty_estimates={'fallback_used': 1.0},
            model_agreements={},
            ensemble_method=EnsembleMethod.SIMPLE_AVERAGING,
            contributing_models=['fallback'],
            prediction_metadata={'error': 'ensemble_failed'}
        )
    
    async def _load_pretrained_models(self):
        """Load any pre-trained models from storage"""
        try:
            # In practice, would load from files or database
            logger.info("Pre-trained model loading not implemented yet")
            
        except Exception as e:
            logger.error(f"❌ Failed to load pre-trained models: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            trained_models = sum(1 for model in self.base_models.values() if model.is_trained)
            
            return {
                'total_models': len(self.base_models),
                'trained_models': trained_models,
                'meta_learners': len(self.meta_learners),
                'ensemble_methods': len(self.ensemble_configs),
                'model_types': len(set(model.model_type for model in self.base_models.values())),
                'ensemble_metrics': len(self.ensemble_metrics),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}

# Testing function
async def test_ensemble_prediction_system():
    """Test the ensemble prediction system"""
    try:
        logger.info("🧪 Testing Ensemble Prediction System")
        
        system = EnsemblePredictionSystem()
        await system.initialize()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_data = np.random.randn(n_samples, n_features)
        y_data = X_data[:, 0] + 0.5 * X_data[:, 1] + np.random.randn(n_samples) * 0.1
        
        training_df = pd.DataFrame(X_data, columns=[f'feature_{i}' for i in range(n_features)])
        training_df['target'] = y_data
        
        # Train ensemble
        success = await system.train_ensemble(training_df, 'target', PredictionTask.REGRESSION)
        logger.info(f"Training successful: {success}")
        
        # Test prediction
        test_features = {f'feature_{i}': np.random.randn() for i in range(n_features)}
        
        prediction = await system.predict_ensemble(
            test_features, 'test_prediction', 'test_student'
        )
        
        logger.info(f"✅ Ensemble prediction: {prediction.ensemble_prediction:.3f}")
        logger.info(f"✅ Confidence: {prediction.confidence_score:.3f}")
        logger.info(f"✅ Contributing models: {len(prediction.contributing_models)}")
        
        # Test system status
        status = await system.get_system_status()
        logger.info(f"✅ System status: {status['trained_models']}/{status['total_models']} models trained")
        
        logger.info("✅ Ensemble Prediction System test completed")
        
    except Exception as e:
        logger.error(f"❌ Ensemble Prediction System test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ensemble_prediction_system())